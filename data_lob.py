import torch
import numpy as np
import torch.nn.functional as F
from torch.utils import data

def prepare_x(data):
    df = data[:40, :].T
    return np.array(df)

def get_label(data):

    df = data[-5:, :].T
    return np.array(df)


def data_classification(X, Y, T): 
    
    #* Eg: N = 200, D = 40, T = 5, then in total we have 196 windows (N-T+1); for each window, time length is 5 and level length is 40
    
    [N, D] = X.shape
    dataY = Y[T - 1:N]
    dataX = np.zeros((N - T + 1, T, D))
    
    for end_point in range(T, N + 1): #* range([20, 201])
        dataX[end_point - T] = X[end_point - T:end_point, :]

    return dataX, dataY


def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization""" 
        self.k = k
        self.num_classes = num_classes
        self.T = T
            
        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        
        x = torch.from_numpy(x).unsqueeze_(1).to(dtype = torch.float)  # * eg [196, 5, 40]
        y = torch.from_numpy(y[:, self.k] - 1).to(dtype = torch.long)
        # self.xx = x
        self.y = y
        
        #* must be long tensor, https://stackoverflow.com/questions/66543659/one-hot-encoding-in-pytorch
        
        
        self.length = len(x)  # * eg 196, # of windows
        
        self.x = x
        # self.x = torch.empty(x.size(0), x.size(1)*2, x.size(2), x.size(3) // 2)
        # self.x[:,0,:,:] = x[:,0,:, 0::2]
        # self.x[:,1,:,:] = x[:,0,:, 1::2]
        
        # self.y = F.one_hot(y, num_classes=3) #* eg [196, 4, 3]
        
        #? Why does the author forget about onehot encoding altogether?
        #? In original code self.y = torch.from_numpy(y)
        #* Answer: https://discuss.pytorch.org/t/runtimeerror-expected-floating-point-type-for-target-with-class-probabilities-got-long/142098
        
        #* crossentropy loss does not require onehot

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]

def main(args):

    # dec_data = np.loadtxt('Train_Dst_NoAuction_DecPre_CF_7.txt')
    # dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    # dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

    # dec_test1 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_7.txt')
    # dec_test2 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_8.txt')
    # dec_test3 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_9.txt')
    # dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

    # print(dec_train.shape, dec_val.shape, dec_test.shape)
    # np.save("dec_train", dec_train)
    # np.save("dec_val",dec_val)
    # np.save("dec_test", dec_test)
    dec_train = np.load("dec_train.npy")
    dec_val = np.load("dec_val.npy")
    dec_test = np.load("dec_test.npy")

    dataset_train = Dataset(data=dec_train, k=args.horizon, num_classes=args.num_classes, T=args.time_length)
    dataset_val = Dataset(data=dec_val, k=args.horizon, num_classes=args.num_classes, T=args.time_length)
    dataset_test = Dataset(data=dec_test, k=args.horizon, num_classes=args.num_classes, T=args.time_length)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, ) # pin_memory = True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, ) # pin_memory = True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, ) # pin_memory = True)
    
    return dataset_train, dataset_val, dataset_test, train_loader,val_loader,test_loader
    

