import torch 
from datetime import datetime
from tqdm import tqdm 
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

import util_lob
import model_lob
import train_epoch

# A function to encapsulate the training loop
def get_optimizer(args, model):
    
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        
    elif args.optimizer == "momentum":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    print(
        f"Initialized {args.model.upper()} model with {sum(p.numel() for p in model.parameters())} "
        f"total parameters, of which {sum(p.numel() for p in model.parameters() if p.requires_grad)} are learnable."
    )
    
    return optimizer

def batch_gd(model, writer, criterion, optimizer, train_loader, val_loader, args):
    
    epochs = args.epochs

    train_losses = np.zeros(epochs)
    valid_losses = np.zeros(epochs)
    train_times = np.zeros(epochs)


    best_valid_loss = np.inf
    best_valid_epoch = 0

    for epoch in tqdm(range(epochs)):
        
        t0 = datetime.now()
        running_loss_train = 0
        running_loss_val = 0

        model.train()
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(args.device, dtype = torch.float, non_blocking=True) 
            targets = targets.to(args.device, dtype = torch.int64, non_blocking=True)
            
            #! DO NOT separate models like: embed = extractor(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
                     
            running_loss_train += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                  
            writer.add_scalar(f"train/loss", loss.item(), i)

            if i % 2000 == 0:
                print(f"the {i} round and the train loss is {running_loss_train / (i + 1):04f}")

        print(f"Evaluating")
        model.eval() 
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):

                inputs = inputs.to(args.device, dtype = torch.float,  non_blocking=True) 
                targets = targets.to(args.device, dtype = torch.int64, non_blocking=True)      
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss_val += loss.item()

                writer.add_scalar(f"valid/loss", loss.item(), i)

        # Save losses
        train_losses[epoch] = running_loss_train / len(train_loader) #* the it'th row of [epoch, K]
        valid_losses[epoch] = running_loss_val / len(val_loader)
        train_times[epoch] =  np.timedelta64(datetime.now() - t0, 'm')
        # https://numpy.org/doc/stable/reference/arrays.datetime.html
        
        if epoch % 5 == 0:
            util_lob.save_checkpoint(args,
                state={
                    'epoch': epoch + 1,
                    # 'arch': arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, 
                is_best=False,
                filename=f'{args.save_dir}/{args.exp_id}/checkpoint_{epoch:04d}.pth.zip'
            )
            
            print(f"{epoch} result is saved")
        
        if valid_losses[epoch] < best_valid_loss:
            util_lob.save_checkpoint(args,
                state={
                    'epoch': epoch + 1,
                    # 'arch': arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, 
                is_best=True,
                filename=f'{args.save_dir}/{args.exp_id}/model_best.pth.zip'
            )
            # torch.save(models, './models/best_val_model_pytorch.pt')
            # torch.save(models.state_dict(), './state_dict/best_val_model_pytorch.pt')
            # torch.save(models, './best_val_model_pytorch')
            best_valid_loss = valid_losses[epoch]
            best_valid_epoch = epoch
            print(f" best {epoch} result is saved")

        if epoch % 2 == 0:
            util_lob.save_train_logs(args,train_losses, valid_losses, train_times)
        
        print(f'Epoch {epoch+1}/{epochs}, \
            Train Loss: {train_losses[epoch]:.4f}, \
            Validation Loss: {valid_losses[epoch]:.4f},\
            Duration: {train_times[epoch]},\
            Best Val Epoch: {best_valid_epoch}')

    return train_losses, valid_losses, train_times


def metrics(model, test_loader, device="cuda:0" if torch.cuda.is_available() else "cpu"):
    model.eval()
    n_correct = 0.
    n_total = 0.

    all_targets = []
    all_predictions = []

    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs)
        
        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    test_acc = n_correct / n_total
    all_targets = np.concatenate(all_targets)    
    all_predictions = np.concatenate(all_predictions)    

    print(f"Test acc: {test_acc:.4f}")
    print('accuracy_score:', accuracy_score(all_targets, all_predictions))
    print(classification_report(all_targets, all_predictions, digits=4))

def load_checkpoint(args, model, optimizer, num, best = False, device="cuda:0" if torch.cuda.is_available() else "cpu"):
    model = model_lob.TransformerForSequenceClassfification(args).to(device=device)
    optimizer = train_epoch.get_optimizer(args, model)

    if not best:
        checkpoint = torch.load(f"./{args.save_dir}/{args.exp_id}/checkpoint_{num:04d}.pth.zip")
    else:
        checkpoint = torch.load(f"./{args.save_dir}/{args.exp_id}/model_best.pth.zip")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

# optim0 = torch.o
    # optimizer = torch.optim.AdamW(models.parameters(), lr=0.0001)

    # optimizerList = []
    # for n in range(K):  
    #   optimizer = torch.optim.AdamW(models.parameters(), lr=0.001)
    #   optimizerList.append(optimizer)

    # optim0 = torch.optim.AdamW(models.parameters(), lr=0.001)
    # optim1 = torch.optim.AdamW(models.parameters(), lr=0.0001)
    # optim2 = torch.optim.AdamW(models.parameters(), lr=0.0001)
    # optim3 = torch.optim.AdamW(models.parameters(), lr=0.0001)
    # optim4 = torch.optim.AdamW(models.parameters(), lr=0.0001)
    # optim5 = torch.optim.AdamW(models.parameters(), lr=0.0001)

    # optimizerList
    
    # Optimizer
    
    
# for k in range(K):  
            #     optimizerList[n].zero_grad()
                
            # for k in range(K): 
            #     loss = criterion(outputs[k], targets[:,k])
            #     lossList.append(loss)
            # loss = sum(lossList)
            # loss.backward()
            # loss = criterion(outputs, targets)
            
# for k in range(K-1):  
            #     lossList[k].backward(retain_graph = True)
            # lossList[K-1].backward()
            
            # optimizer.zero_grad()
            # loss.backward()
            # for k in range(K):  
            #     optimizerList[k].step()
            # optimizer.step()
            
            # train_loss.append(lossList)
            
# loss_avg_over_k = torch.mean(torch.stack(lossList).detach(), axis = 0).item() #* [K] -> [1]
            # print(f"{len(lossList)} losslist, {torch.stack(lossList).size()} the stacked list, and {loss_avg_over_k}, train_loss size {len(train_loss)}")

#TODO for debug 
            # if i==2:
            #     print(f"{outputs.size()} output, target {targets.size()}")
            #     train_loss_mean = np.mean(train_loss)
            #     train_losses = np.zeros((epochs,))
            #     train_losses[it] = train_loss_mean
                
            #     print(train_loss_mean, train_losses)
            #     break
            
# loss = criterion(outputs, targets)
                # for k in range(K): 
                #     loss = criterion(outputs[k], targets[:,k])
                #     lossList.append(loss)
                
# loss_avg_over_k = torch.mean(torch.stack(lossList).detach(), axis = 0)
        # train_loss_all = torch.stack(train_loss,axis =0) 
        # test_loss_all = torch.stack(test_loss,axis =0) 
