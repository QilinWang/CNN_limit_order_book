import os
import json
import torch
import shutil
import math 

def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
            
def save_checkpoint(args, state, is_best, filename='checkpoint.pth'):
    save_dir = os.path.join(args.save_dir, args.exp_id)
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

def save_train_logs(
        args,
        train_losses,
        valid_losses,
        train_times,
    ):
    log_dir = os.path.join(args.log_dir, args.exp_id)
    os.makedirs(log_dir, exist_ok=True)

    # Log arguments
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Log train, val, and test losses and perplexities
    with open(os.path.join(log_dir, "train_loss.txt"), "w") as f:
        f.write("\n".join(str(item) for item in train_losses))
    
    with open(os.path.join(log_dir, "valid_loss.txt"), "w") as f:
        f.write("\n".join(str(item) for item in valid_losses))
    with open(os.path.join(log_dir, "train_time.txt"), "w") as f:
        f.write("\n".join(str(item) for item in train_times))
    print("train logs saved")
        
def save_logs(
        args,
        train_losses,
        train_ppls,
        train_times,
        valid_losses,
        valid_ppls,
        valid_times,
        test_loss,
        test_ppl,
        test_time
    ):
    log_dir = os.path.join(args.log_dir, args.exp_id)
    os.makedirs(log_dir, exist_ok=True)

    # Log arguments
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Log train, val, and test losses and perplexities
    with open(os.path.join(log_dir, "train_loss.txt"), "w") as f:
        f.write("\n".join(str(item) for item in train_losses))
    with open(os.path.join(log_dir, "train_ppl.txt"), "w") as f:
        f.write("\n".join(str(item) for item in train_ppls))
    with open(os.path.join(log_dir, "train_time.txt"), "w") as f:
        f.write("\n".join(str(item) for item in train_times))
    with open(os.path.join(log_dir, "valid_loss.txt"), "w") as f:
        f.write("\n".join(str(item) for item in valid_losses))
    with open(os.path.join(log_dir, "valid_ppl.txt"), "w") as f:
        f.write("\n".join(str(item) for item in valid_ppls))
    with open(os.path.join(log_dir, "valid_time.txt"), "w") as f:
        f.write("\n".join(str(item) for item in valid_times))
    with open(os.path.join(log_dir, "test_loss.txt"), "w") as f:
        f.write(f"{test_loss}\n")
    with open(os.path.join(log_dir, "test_ppl.txt"), "w") as f:
        f.write(f"{test_ppl}\n")
    with open(os.path.join(log_dir, "test_time.txt"), "w") as f:
        f.write(f"{test_time}\n")