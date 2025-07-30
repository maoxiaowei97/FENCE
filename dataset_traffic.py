import numpy as np
from torch.utils.data import DataLoader, Dataset

class Traffic_Dataset(Dataset):
    def __init__(self,c_data,true_data,observed_masks,gt_mask,isTest=False,eval_length=12):
        self.eval_length = eval_length
        self.observed_masks = observed_masks
        self.gt_masks = gt_mask
        self.observed_data = c_data
        

        if isTest:  
            self.observed_data = true_data
            self.observed_masks = np.ones_like(self.gt_masks)

    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_data[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.observed_data)

def get_sample_by_overlaped_Sliding_window(X,Y,  mask, gt_mask,sample_len):
    #X,Y,mask: shape(T,N*F)
    X_window,Y_window ,mask_window ,gt_mask_window= [], [], [],[]
    for i in range(X.shape[0]-sample_len+1):
        X_window.append(X[i:i+sample_len])
        Y_window.append(Y[i:i+sample_len])
        mask_window.append(mask[i:i+sample_len])
        gt_mask_window.append(gt_mask[i:i+sample_len])

    X_window = np.array(X_window)
    Y_window = np.array(Y_window)
    mask_window = np.array(mask_window)
    gt_mask_window=np.array(gt_mask_window)
    return X_window,Y_window, mask_window,gt_mask_window

def get_dataloader(true_datapath,miss_datapath,val_ratio,test_ratio, batch_size=16,eval_length=12):
    # 缺失的位置为1 有值的地方为0
    miss = np.load(miss_datapath)
    true = np.load(true_datapath)
    # observed_masks = np.nan_to_num(true['mask'][:,:,0])
    # gt_masks= np.nan_to_num(miss['mask'][:,:,0])
    # true_data = np.nan_to_num(np.load(true_datapath)['data'][:,:,0].astype(np.float32))
    observed_masks = np.nan_to_num(true['mask'][:, :30, 0])
    gt_masks = np.nan_to_num(miss['mask'][:, :30, 0])
    true_data = np.nan_to_num(np.load(true_datapath)['data'][:, :30, 0].astype(np.float32))

    """
    [B, T, N, C] 
    [0.1, 0.2, 0.7]
    
    
    [0.08, 0.21, 0.35, 30]
    [0.07, 0.23, 0.46, 32]
    ...
    [0.07, 0.23, 0.46, 35]
    
    obs_mask
    cond_mask
    
    obs - cond
    """
    # obs_mask

    mean = np.mean(true_data[observed_masks==1])
    std = np.std(true_data[observed_masks==1])
    true_data = (true_data - mean)/std
    
    # observed_values = (observed_values - mean)/std
    # observed_values[(1-observed_masks).astype(np.bool_)] = 0
    c_data=np.copy(true_data)*observed_masks
    
    
    T,N = true_data.shape

    val_len = int(T * val_ratio)
    test_len = int(T * test_ratio)

    train_X, val_X, test_X = c_data[ :-(val_len+test_len)], \
                             c_data[ -(val_len+test_len):-(test_len)],\
                             c_data[-test_len:]
    
    train_Y, val_Y, test_Y = true_data[ :-(val_len+test_len)], \
                             true_data[ -(val_len+test_len):-(test_len)],\
                             true_data[-test_len:]

    train_mask, val_mask, test_mask = observed_masks[ :-(val_len+test_len)], \
                                      observed_masks[ -(val_len+test_len):-(test_len)],\
                                      observed_masks[-test_len:]
                                      
    train_gtmask, val_gtmask, test_gtmask = gt_masks[ :-(val_len+test_len)], \
                                      gt_masks[ -(val_len+test_len):-(test_len)],\
                                      gt_masks[-test_len:]
    train_X, train_Y,  train_mask,train_gtmask = get_sample_by_overlaped_Sliding_window(train_X, train_Y,  train_mask,train_gtmask, eval_length)
    val_X, val_Y,  val_mask,val_gtmask = get_sample_by_overlaped_Sliding_window( val_X, val_Y,  val_mask, val_gtmask,eval_length)
    test_X, test_Y, test_mask,test_gtmask = get_sample_by_overlaped_Sliding_window(test_X, test_Y, test_mask, test_gtmask,eval_length)
    


    dataset = Traffic_Dataset(
        train_X,train_Y,train_mask,train_gtmask,isTest=False,eval_length=12
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Traffic_Dataset(
        val_X,val_Y,val_mask,val_gtmask,isTest=False,eval_length=12
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Traffic_Dataset(
        test_X,test_Y,test_mask,test_gtmask,isTest=True,eval_length=12
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader,N,std,mean
