import torch
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils import data
import os


def prepare_x(data):
    df = data[:40, :].T
    return np.array(df)

def get_label(data):
    
    #? why -5? what does this 5 label mean?
    
    df = data[-5:, :].T
    return np.array(df)

def data_classification(X, Y, T): 
    
    #* Eg: N = 200, D = 40, T = 5, then in total we have 196 windows (N-T+1); for each window, time length is 5 and level length is 40
    
    [N, D] = X.shape
    dataY = Y[T - 1:N] # Get Y[4:200]
    dataX = np.zeros((N - T + 1, T, D))
    
    for t in range(T, N + 1): #* range([5, 201])
        dataX[t - T] = X[t - T:t, :] # data[25-5] = X[25-5:25] ie data[20] = X[20:25] which is T by D
        
    #* end_point is 5,      index of dataX is 0,         corr.indices of X is timestep range(0:5), i.e 0-4
    #* ...
    #* end_point is 200,    index of dataX is 195,       corr.indices of X is timestep range(195:200), i.e 195-200

    return dataX, dataY

def data_classification_forPD(X, Y, returns_log, returns_ori, T):
    [N, D] = X.shape
    # df = np.array(X)
    # dY = np.array(Y)
    X = X.to_numpy()
    dataY = Y[T - 1:N]
    returns_log = returns_log[T - 1:N]
    returns_ori = returns_ori[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = X[i - T:i, :]

    #* end_point (i) is 5,   index of dataX is 0,     corr.indices of X is timestep range(0:5), i.e 0-4
    #* ...
    #* end_point is 200,     index of dataX is 195,   corr.indices of X is timestep range(195:200), i.e 195-200

    return dataX, dataY, returns_log, returns_ori


def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y




def get_y_and_logReturns_and_returns( d, span ):
    lab = np.zeros(len(d))

    midprice_ori = ( np.exp(d['L1-AskPrice']) + np.exp(d['L1-BidPrice']) ) / 2 
    midprice = np.log( ( np.exp(d['L1-AskPrice']) + np.exp(d['L1-BidPrice']) ) / 2 )

    # Correct index for m_plus - m_minus: 
    # a[position+1 : position+1+length] - a[position+1-length:position +1 ]
    # a.rolling(5).mean().shift(-5) - a.rolling(5).mean()

    # smooth_midprice = midprice.rolling(1).mean()
    # future_midprice = smooth_midprice.shift(-length)
    # targ = future_midprice - smooth_midprice


    # midprice_return = midprice.shift(-length) - midprice

    #! m_plus: mean of the next k prices
    # m_plus = midprice.rolling(span).mean().shift(-span)
    #! m_minus: mean of the past k prices (including current)
    # m_minus = midprice.rolling(span).mean()

    m_plus = midprice.ewm(alpha = 0.5, adjust=False, ignore_na=True).mean().shift(-span)
    m_minus = midprice.ewm(alpha = 0.5, adjust=False, ignore_na=True).mean()
    # targ = (m_plus - m_minus) / m_minus
    targ = (m_plus - m_minus)
    midprice_return_log = midprice.shift(-span) - midprice
    midprice_return_ori = midprice_ori.shift(-span) - midprice_ori
    
    q1 = -0.0004
    q2 = +0.0004

    # q1 = np.quantile( targ.dropna(), .33 )
    # q2 = np.quantile( targ.dropna(), .66 )
    # plt.hist( targ.dropna() )
    # plt.show()
    # print( q1, q2 )
    lab[ np.where( np.isnan(targ) ) ] = np.NaN
    lab[ np.where( (targ > q1) & (targ <= q2) ) ] = 1
    lab[ np.where( targ > q2 ) ] = 2
    return lab, midprice_return_log, midprice_return_ori

def get_y_box( d, max_length ):
    ticksize = 0.01
    target = 8 * ticksize

    d = d.iloc[::100]
    midprice = ( d['L1-AskPrice'] + d['L1-BidPrice'] ) / 2
    smooth_midprice = midprice.rolling(2).mean()

    lab = np.ones(len(d))
    mid_move = np.zeros(len(d))

    length = 1
    while length < max_length:
        future_midprice = smooth_midprice.shift(-length)
        targ = future_midprice - smooth_midprice
        mid_move[ np.where( ( targ >= target ) & ( lab == 1 ) ) ] = np.array(  midprice.shift(-length) - midprice )[ np.where( ( targ >= target ) & ( lab == 1 ) ) ]

        lab[ np.where( ( targ >= target ) & ( lab == 1 ) ) ] = 2

#         print(np.array(  midprice.shift(-length) - midprice )[ np.where( ( targ >= target ) & ( lab == 1 ) ) ])
#         print(lab[ np.where( ( targ >= target ) & ( lab == 1 ) ) ])

        mid_move[ np.where( ( targ <= -target ) & ( lab == 1 ) ) ] = np.array( midprice.shift(-length) - midprice )[ np.where( ( targ <= -target ) & ( lab == 1 ) ) ]

        lab[ np.where( ( targ <= -target ) & ( lab == 1 ) ) ] = 0
        length += 1

    mid_move[ np.where( ( lab == 1 ) ) ] = np.array( midprice.shift(-length) - midprice )[ np.where( ( lab == 1 ) ) ]
    # print( lab.shape, mid_move.shape )
    return np.expand_dims( lab, -1), np.expand_dims( mid_move, -1 )




# def compare_with_mid_price_chart(test_loader, chunksize = 1000):
#     chunksize = chunksize
#     test_array = test_loader.dataset.x.numpy()
#     # test_array[np.isnan(test_array)] = 0
#     test_array_length = len(test_array)
#     returns = np.zeros(test_array_length)
#     for i, out in enumerate(test_array):


def get_image_data(cnn_data, halfwidth=20):
    new_x = np.zeros((cnn_data.shape[0], cnn_data.shape[1], halfwidth*2, cnn_data.shape[3]))
    print(new_x.shape[0])
    for j in range(cnn_data.shape[0]):
        if np.min(cnn_data[j,:,::2,0]) == np.max(cnn_data[j,:,::2,0]):
            continue
        # print( np.min( new_x[j,:,::2,0] ), np.max( new_x[j,:,::2,0] ), np.min( np.abs( np.diff( new_x[j,0,::2,0] ) ) ) )
        ticksize = np.min( np.abs( np.diff( cnn_data[j,0,2::4,0] ) ) )
        if ticksize == 0:
            continue
        x_coords = np.arange(
            np.min(cnn_data[j,:,::2,0]), 
            np.max(cnn_data[j,:,::2,0]), 
            ticksize
        )
        new_image = np.zeros((100,len(x_coords)+1))
        
        for z in range(100):
            new_image[z, np.digitize( cnn_data[j,z,2::4,0], x_coords )] = cnn_data[j,z,3::4,0] #bid volumes
            new_image[z, np.digitize( cnn_data[j,z,::4,0], x_coords )] = -cnn_data[j,z,1::4,0] #ask volums

        anchor_position = np.digitize( cnn_data[j,-1,2,0], x_coords ) # bid price level 1 of last time period is anchor price
        while anchor_position < halfwidth:
            new_image = np.hstack([np.zeros((100,1)),new_image]) # stack from top 
            anchor_position += 1

        while new_image.shape[1] < anchor_position + halfwidth:
            new_image = np.hstack([new_image,np.zeros((100,1))])
            
        new_x[j,:,:,0] = new_image[:,range(anchor_position-halfwidth,anchor_position+halfwidth)]
        
        if j % 100 == 0:
            print( j, end="\r")
    return new_x

def get_mean_and_sd(ticker, subticker, year, month):
    npy_file = np.load(f"../data/{ticker}/npy/{subticker}/{ticker}_{year}-{month:02d}.npy",allow_pickle=True)
    # np.save( f'../data/{ticker}/npy/{subticker}/{ticker}_{year}-{month:02d}', trainX_CNN , allow_pickle=True)
    df = pd.DataFrame(npy_file)
    price_mean = df.iloc[:,::2].stack().mean()
    volume_mean = df.iloc[:,1::2].stack().mean()
    price_std = df.iloc[:,::2].stack().std()
    volume_std = df.iloc[:,1::2].stack().std()

    return price_mean, volume_mean, price_std, volume_std


def csv_to_npy(folder = "./data", ticker="SB", subticker = "1", year = 2021, span = 10, T = 100, downsample_rows = 10, valid_days=3, test_days=3):
    # url = f'../data/{ticker}/raw/{subticker}'
    os.makedirs(f'{folder}/{ticker}/npy/{subticker}', exist_ok=True)
    # [ x for x in os.listdir(url)  ]

    cols = []
    for k in range(1,11):
        cols += ['L%s-AskPrice'%k, 'L%s-AskSize'%k, 'L%s-BidPrice'%k, 'L%s-BidSize'%k]

    running_queue = deque([]) # 5 days of running average


    for month in range(3, 7):
        lst_train = []
        lst_valid = []
        lst_test = []

        all_train_X = []
        all_valid_X = []
        all_test_X = []

        all_train_Y = []
        all_valid_Y = []
        all_test_Y = []

        # all_train_returns = []
        # all_valid_returns = []
        # all_test_returns = []

        full_list = [x for x in os.listdir(f'{folder}/{ticker}/raw/{subticker}') for j in [month]  for k in range(1,32) if x in f'{ticker}_{year}-{j:02d}-{k:02d}.csv']
        total_days = len(full_list)

        assert valid_days + test_days + 1 < total_days

        for i, f in enumerate(full_list):
            d = pd.read_csv(f"{folder}/{ticker}/raw/{subticker}/{f}")
            d = d[cols].dropna()
            d = d.iloc[::downsample_rows,:]
            d = np.log(d)
            # train = d.to_numpy()[::downsample_rows]
            # running_queue.append( d )
            # if len(running_queue) > 5:
            #     running_queue.popleft()

            # q = pd.concat(running_queue)
            # price_mean = q.iloc[:,::2].stack().mean()
            # volume_mean = q.iloc[:,1::2].stack().mean()
            # price_std = q.iloc[:,::2].stack().std()
            # volume_std = q.iloc[:,1::2].stack().std()
            
            # # (*) Normalizing with 5 day average
            # d.iloc[:,::2] = (d.iloc[:,::2] - price_mean) / price_std
            # d.iloc[:,1::2] = (d.iloc[:,::2] - volume_mean) / volume_std

            # Exponential averaging of {span} days
            # d.ewm(span = 20, adjust=False, ignore_na=True).mean()

            # (*) Get Y
            # d["target"], d["returns"] = get_y_and_return(d, span)
            # d = d.dropna()
            # # (*) T=100 window
            # [N, D] = d.shape
            # if not N < (T-1):
            #     X, Y, returns= data_classification_forPD(d.drop(['target','returns'], axis=1), d["target"],d["returns"] T=100)
            # ret = adjust_Y(ret, T=100)

            if i < total_days - valid_days - test_days:
                lst_train.append( d )
                # all_train_X.append( X )
                # all_train_Y.append( Y )
                # all_train_returns.append( ret )
            elif i < total_days- test_days:
                lst_valid.append( d )
                # all_valid_X.append( X )
                # all_valid_Y.append( Y )
                # all_valid_returns.append( ret )
            else: 
                lst_test.append( d )
                # all_test_X.append( X )
                # all_test_Y.append( Y )
                # all_test_returns.append( ret )
        assert len(all_train_X) == len(all_train_Y)
        assert len(all_train_X) == len(all_train_Y)
        assert len(all_train_X) == len(all_train_Y)

        if all_train_X:
                # train = np.vstack( all_train_X )
                d = np.concatenate( lst_train, axis = 0)
                d["target"], d["returns"], d["ori_returns"] = get_y_and_logReturns_and_returns(d, span)
                d = d.dropna()
                [N, D] = d.shape
                if not N < (T-1):
                    X, Y, returns_log, returns_ori = data_classification_forPD(d.drop(['target','returns'], axis=1), d["target"],d["returns"] T=100)
                labels = np.concatenate( all_train_Y , axis = None)
                # rets = np.vstack(all_train_returns)
                np.save( f'{folder}/{ticker}/npy/{subticker}/{ticker}_{year}-{month:02d}_train_X', train , allow_pickle=True)
                np.save( f'{folder}/{ticker}/npy/{subticker}/{ticker}_{year}-{month:02d}_train_Y', labels , allow_pickle=True)
                # np.save( f'../data/{ticker}/npy/{subticker}/{ticker}_{year}-{month:02d}_train_returns', rets , allow_pickle=True)
        if all_valid_X:
                valid = np.vstack( all_valid_X )
                labels = np.concatenate( all_valid_Y , axis = None)
                # rets = np.vstack(all_valid_returns)
                np.save( f'{folder}/{ticker}/npy/{subticker}/{ticker}_{year}-{month:02d}_valid_X', valid , allow_pickle=True)
                np.save( f'{folder}/{ticker}/npy/{subticker}/{ticker}_{year}-{month:02d}_valid_Y', labels , allow_pickle=True)
                # np.save( f'../data/{ticker}/npy/{subticker}/{ticker}_{year}-{month:02d}_valid_returns', rets , allow_pickle=True)
        if all_test_X:
                test = np.vstack( all_test_X )
                labels = np.concatenate( all_test_Y , axis = None)
                # rets = np.vstack(all_test_returns)
                np.save( f'{folder}/{ticker}/npy/{subticker}/{ticker}_{year}-{month:02d}_test_X', test , allow_pickle=True)
                np.save( f'{folder}/{ticker}/npy/{subticker}/{ticker}_{year}-{month:02d}_test_Y', labels , allow_pickle=True)
                # np.save( f'../data/{ticker}/npy/{subticker}/{ticker}_{year}-{month:02d}_test_returns', rets , allow_pickle=True)
        # price_mean, volume_mean, price_std, volume_std = get_mean_and_sd(ticker, subticker, year, normalizeing_month)
        



class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization""" 
        self.k = k #? What is this? It seems to be 5 period -> 4 in index 0
        self.num_classes = num_classes
        self.T = T
            
        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        
        x = torch.from_numpy(x).unsqueeze_(1).to(dtype = torch.float) #* eg [196, 5, 40]
        y = torch.from_numpy(y[:,self.k] - 1).to(dtype = torch.long) 
        # self.xx = x
        self.y = y
        
        #* must be long tensor, https://stackoverflow.com/questions/66543659/one-hot-encoding-in-pytorch
        
        self.length = len(x) #* eg 196, # of windows
        
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
    # please change the data_path to your local path
    # data_path = '/nfs/home/zihaoz/limit_order_book/data'

    # dec_data = np.loadtxt('Train_Dst_NoAuction_DecPre_CF_7.txt')
    # dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    # dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

    # dec_test1 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_7.txt') # (149, 55478)
    # dec_test2 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_8.txt') # (149, 55478)
    # dec_test3 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_9.txt') # (149, 31937)
    # dec_test = np.hstack((dec_test1, dec_test2, dec_test3)) # (149, 139587)

    # print(dec_train.shape, dec_val.shape, dec_test.shape)
    # np.save("dec_train", dec_train)
    # np.save("dec_val",dec_val)
    # np.save("dec_test", dec_test)
    dec_train = np.load(os.path.join(args.data_folder, "dec_train.npy"))
    dec_val = np.load(os.path.join(args.data_folder, "dec_val.npy"))
    dec_test = np.load(os.path.join(args.data_folder, "dec_test.npy"))

    dataset_train = Dataset(data=dec_train, k=args.horizon, num_classes=args.num_classes, T=args.time_length)
    dataset_valid = Dataset(data=dec_val, k=args.horizon, num_classes=args.num_classes, T=args.time_length)
    dataset_test = Dataset(data=dec_test, k=args.horizon, num_classes=args.num_classes, T=args.time_length)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, ) # pin_memory = True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, ) # pin_memory = True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, ) # pin_memory = True)
    
    return train_loader,val_loader,test_loader # 
    

######## prepard from raw data


def sb(args):
    trainX_CNN = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.month:02d}_train_X.npy')
    trainY_CNN = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.month:02d}_train_Y.npy')
    
    trainX_CNN_2 = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.extra_month_train:02d}_train_X.npy')
    trainY_CNN_2 = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.extra_month_train:02d}_train_Y.npy')
    
    validX_CNN = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.month:02d}_valid_X.npy')
    validY_CNN = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.month:02d}_valid_Y.npy')

    validX_CNN_2 = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.extra_month_valid:02d}_valid_X.npy')
    validY_CNN_2 = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.extra_month_valid:02d}_valid_Y.npy')

    testX_CNN = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.month:02d}_test_X.npy')
    testY_CNN = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.month:02d}_test_Y.npy')

    testX_CNN_2 = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.extra_month_test:02d}_test_X.npy')
    testY_CNN_2 = np.load( f'./data/{args.ticker}/npy/{args.subticker}/{args.ticker}_{args.year}-{args.extra_month_test:02d}_test_Y.npy')
    # train_returns = np.load( f'./data/{args.ticker}/train_returns_{args.ticker}.npy')
    # valid_returns = np.load( f'./data/{args.ticker}/valid_returns_{args.ticker}.npy')
    # test_returns = np.load( f'./data/{args.ticker}/test_returns_{args.ticker}.npy')
    # trainX_CNN_AUG = np.load( f'./data/{args.ticker}/trainX_CNN_Augmented_{args.ticker}.npy')
    # validX_CNN_AUG = np.load( f'./data/{args.ticker}/validX_CNN_Augmented_{args.ticker}.npy')
    # testX_CNN_AUG = np.load( f'./data/{args.ticker}/testX_CNN_Augmented_{args.ticker}.npy')

    dataset_train = Dataset_private(np.concatenate([trainX_CNN, trainX_CNN_2],axis = 0), np.concatenate([trainY_CNN, trainY_CNN_2],axis = None))
    dataset_valid = Dataset_private(np.concatenate([validX_CNN, ],axis =0), np.concatenate([validY_CNN, ],axis = None)) # validX_CNN_2 validY_CNN_2
    dataset_test = Dataset_private(np.concatenate([testX_CNN, ],axis=0), np.concatenate([testY_CNN, ],axis = None)) # testX_CNN_2 testY_CNN_2

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, ) # pin_memory = True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, ) # pin_memory = True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, ) # pin_memory = True)

    return train_loader, val_loader, test_loader

class Dataset_private(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, x, y):
        """Initialization""" 
        # self.k = k #? What is this? It seems to be 5 period -> 4 in index 0
        # self.num_classes = num_classes
        # self.T = T
            
        x = torch.from_numpy(x)
        x = x.unsqueeze(dim=1).to(dtype = torch.float)
        # y = torch.argmax(torch.from_numpy(y), dim=1).to(dtype = torch.long) 
        self.y = y
        
        #* must be long tensor, https://stackoverflow.com/questions/66543659/one-hot-encoding-in-pytorch
        
        self.length = len(x) #* eg 196, # of windows
        self.x = x

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]