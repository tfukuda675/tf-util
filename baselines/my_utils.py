print("**Info :: CUDA available. Use AMP function.")# %% [code]
# %% [code]
# %% [code]
# %% [code]
import os
import re

import pandas as pd
import numpy as np
import tensorflow as tf
import torch


from tqdm import tqdm
tqdm.pandas()


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler, PowerTransformer, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, precision_recall_curve, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from xml.sax.saxutils import unescape
import emoji


scaler = torch.cuda.amp.GradScaler() 


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2 
    dfs = []
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dfs.append(df[col].astype(np.int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dfs.append(df[col].astype(np.int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dfs.append(df[col].astype(np.int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dfs.append(df[col].astype(np.int64) ) 
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dfs.append(df[col].astype(np.float16))
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dfs.append(df[col].astype(np.float32))
                else:
                    dfs.append(df[col].astype(np.float64))
        else:
            dfs.append(df[col])
    
    df_out = pd.concat(dfs, axis=1)
    if verbose:
        end_mem = df_out.memory_usage().sum() / 1024**2
        num_reduction = str(100 * (start_mem - end_mem) / start_mem)
        print(f'Mem. usage decreased to {str(end_mem)[:3]}Mb:  {num_reduction[:2]}% reduction')
        
    del df
    return df_out

def confirm_accelerator():
    tpu = None
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        return "tpu"

    elif tf.test.is_gpu_available():
        strategy = tf.distribute.get_strategy()
        print('Running on GPU')
        return "gpu"

    else:
        strategy = tf.distribute.get_strategy()
        print('Running on CPU')
        return "cpu"




def plot_acc(f_scores,val_loss_col, loss_col, val_acc_col, acc_col):
    for fold in range(f_scores['folds'].nunique()):
        print(f"\n**Info :: Plot fold {fold}")
        history_f = f_scores[f_scores['folds'] == fold]

        best_epoch = np.argmin(np.array(history_f[val_loss_col]))
        best_val_loss = history_f[val_loss_col][best_epoch]

        fig, ax1 = plt.subplots(1, 2, tight_layout=True, figsize=(15,4))

        fig.suptitle('Fold : '+ str(fold+1) +
                     " Validation Loss: {:0.4f}".format(history_f[val_loss_col].min()) +
                     " Validation Accuracy: {:0.4f}".format(history_f[val_acc_col].max()) +
                     " LR: {:0.8f}".format(history_f['lr'].min())
                     , fontsize=14)

        plt.subplot(1,2,1)
        plt.plot(history_f.loc[:, [loss_col, val_loss_col]], label= [loss_col, val_loss_col])
                
        from_epoch = 0
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_loss], c = 'r', label = f'Best val_loss = {best_val_loss:.5f}')
        if best_epoch > 0:
            almost_epoch = np.argmin(np.array(history_f[val_loss_col])[:best_epoch])
            almost_val_loss = history_f[val_loss_col][almost_epoch]
            if almost_epoch >= from_epoch:
                plt.scatter([almost_epoch], [almost_val_loss], c='orange', label = 'Second best val_loss')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')   
        
        ax2 = plt.gca().twinx()
        ax2.plot(history_f.loc[:, ['lr']], 'y:', label='lr' ) # default color is same as first ax
        ax2.set_ylabel('Learning rate')
        ax2.legend(loc = 'upper right')
        ax2.grid()

        best_epoch = np.argmax(np.array(history_f[val_acc_col]))
        best_val_acc = history_f[val_acc_col][best_epoch]
        
        plt.subplot(1,2,2)
        plt.plot(history_f.loc[:, [acc_col, val_acc_col]],label= [acc_col, val_acc_col])
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_acc], c = 'r', label = f'Best val_acc = {best_val_acc:.5f}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc = 'lower left')
        plt.legend(fontsize = 15)
        plt.grid(b = True, linestyle = '-')
    plt.show()


    
    
def plot_err(f_scores,val_loss_col, loss_col, val_err_col, err_col):
    for fold in range(f_scores['folds'].nunique()):
        print(f"\n**Info :: Plot fold {fold}")
        history_f = f_scores[f_scores['folds'] == fold]

        best_epoch = np.argmin(np.array(history_f[val_loss_col]))
        best_val_loss = history_f[val_loss_col][best_epoch]

        fig, ax1 = plt.subplots(1, 2, tight_layout=True, figsize=(15,4))

        fig.suptitle('Fold : '+ str(fold+1) +
                     " Validation Loss: {:0.4f}".format(history_f[val_loss_col].min()) +
                     " Validation Error: {:0.4f}".format(history_f[val_err_col].max()) +
                     " LR: {:0.8f}".format(history_f['lr'].min())
                     , fontsize=14)

        plt.subplot(1,2,1)
        plt.plot(history_f.loc[:, [loss_col, val_loss_col]], label= [loss_col, val_loss_col])
                
        from_epoch = 0
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_loss], c = 'r', label = f'Best val_loss = {best_val_loss:.5f}')
        if best_epoch > 0:
            almost_epoch = np.argmin(np.array(history_f[val_loss_col])[:best_epoch])
            almost_val_loss = history_f[val_loss_col][almost_epoch]
            if almost_epoch >= from_epoch:
                plt.scatter([almost_epoch], [almost_val_loss], c='orange', label = 'Second best val_loss')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')   
        
        ax2 = plt.gca().twinx()
        ax2.plot(history_f.loc[:, ['lr']], 'y:', label='lr' ) # default color is same as first ax
        ax2.set_ylabel('Learning rate')
        ax2.legend(loc = 'upper right')
        ax2.grid()

        best_epoch = np.argmin(np.array(history_f[val_err_col]))
        best_val_err = history_f[val_err_col][best_epoch]
        
        plt.subplot(1,2,2)
        plt.plot(history_f.loc[:, [err_col, val_err_col]],label= [err_col, val_err_col])
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_err], c = 'r', label = f'Best val_acc = {best_val_err:.5f}')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.yscale("log")
        plt.legend(loc = 'lower left')
        plt.legend(fontsize = 15)
        plt.grid(b = True, linestyle = '-')
    plt.show()
    
    



#    _____________________________________________________
#___/ For Bert                                            \______________
#

def bert_train(train_dataloader,model,optimizer,scheduler,device,loss_fn):
    '''
    BERTのTrainingを実行

    Parameters
    ----------
    T.D.B

    Returns
    -------
    T．D.B

    Notes
    -----
    AMP機能を使うため、Lossはmodel内部のものを使わず、外部から与える。
    '''
    model.train() # 訓練モードで実行
    scaler = torch.cuda.amp.GradScaler()
    #print(loss_fn)
    m = torch.nn.Sigmoid()
    
#    if device == torch.device("cuda"):
    if device == "cuda":

        print("**Info :: Activate CDUNN BenchMarck.")
        torch.backends.cudnn.benchmark = True # ネットワークの形が固定の場合、GPU側で最適化
    
    fin_targets = []
    fin_outputs = []
    train_loss = 0
    batch_count = 0
    loss = None
    sig_output = None
    #print(type(device))

    #print(f"**Info :: device {device}")
    
    for batch in tqdm(train_dataloader):# train_dataloaderはinput_ids, token_type_ids, attension_mask, labelを出力する点に注意
        batch_count += 1
        
        b_input_ids      = batch[0].to(device)
        b_token_type_ids = batch[1].to(device)
        b_attension_mask = batch[2].to(device)
        b_train_labels   = batch[3].to(device)
        b_train_logit    = torch.logit(batch[3]).to(device)
        
        optimizer.zero_grad()
        
#        if device == torch.device("cuda"):
        if device == "cuda":

            print("**Info :: CUDA available. Use AMP function.")
        
            with torch.cuda.amp.autocast():
                outputs = model(b_input_ids, 
                                token_type_ids=b_token_type_ids, 
                                attention_mask=b_attension_mask)
                output = outputs["logits"].squeeze(-1)
                sig_output = m(output)
                print("**Info :: Sigmoid output")
                print(sig_output)
                print("**Info :: targets")
                print(m(b_train_logit))
                loss = loss_fn(output,b_train_logit)

        ## loss は で算出するので、次は使わない。
        ## return_dict=Trueなので、.lossでlossを取得可能。
        #loss = outputs.loss
            scaler.scale(loss).backward() # ロスのバックワード
            scaler.step(optimizer) # オプティマイザーの更新
            scaler.update() # スケーラーの更新
            
            #del loss #ここでlossを消した方がGPUのメモリの無駄な部分を消せるらしい。
            # https://tma15.github.io/blog/2020/08/22/pytorch%E4%B8%8D%E8%A6%81%E3%81%AB%E3%81%AA%E3%81%A3%E3%81%9F%E8%A8%88%E7%AE%97%E3%82%B0%E3%83%A9%E3%83%95%E3%82%92%E5%89%8A%E9%99%A4%E3%81%97%E3%81%A6%E3%83%A1%E3%83%A2%E3%83%AA%E3%82%92%E7%AF%80%E7%B4%84/
            
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
        
        ## with CPU, TPU?
        else:
            #print("**Info :: None CUDA mode")
            outputs = model(b_input_ids, 
                            token_type_ids=b_token_type_ids, 
                            attention_mask=b_attension_mask)
            output = outputs["logits"].squeeze(-1)
            sig_output = m(output)
            #print("**Info :: Sigmoid output")
            #print(sig_output)
            #print("**Info :: targets")
            #print(m(b_train_logit))
            loss = loss_fn(output,b_train_logit)
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            
        #tf.softmax(x, axis=None)
        train_loss += loss.tolist()
        fin_targets.extend(b_train_labels.cpu().detach().numpy().tolist())
        # following is multi label
        #fin_outputs.extend(outputs.logits.softmax(axis=-1).cpu().detach().numpy().tolist())
        #print(m(output.cpu().detach()))
        fin_outputs.extend(sig_output)
    
    # following is multi label
    #train_accu = accuracy_score(fin_targets,np.argmax(fin_outputs, axis=1).tolist())
    train_accu = mean_squared_error(fin_targets,fin_outputs)
    
    return train_accu, train_loss / batch_count, lr, model




def bert_valid(valid_dataloader, model, device, loss_fn):
    model.eval() # 訓練モードで実行
    fin_targets = []
    fin_outputs = []
    valid_loss = 0
    batch_count = 0
    
    m = torch.nn.Sigmoid()
    
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            batch_count += 1
            
            b_input_ids      = batch[0].to(device)
            b_token_type_ids = batch[1].to(device)
            b_attension_mask = batch[2].to(device)
            b_train_labels   = batch[3].to(device)
            b_train_logit    = torch.logit(batch[3]).to(device)
        
            # Compute logits
            with torch.no_grad():
                outputs = model(b_input_ids, 
                        token_type_ids=b_token_type_ids, 
                        attention_mask=b_attension_mask)
                output = outputs["logits"].squeeze(-1)
            
            loss = loss_fn(output,b_train_logit)
            #loss = outputs.loss
            valid_loss += loss.tolist()
            fin_targets.extend(b_train_labels.cpu().detach().numpy().tolist())
            #fin_outputs.extend(outputs.logits.softmax(axis=-1).cpu().detach().numpy().tolist())
            fin_outputs.extend(m(output.cpu().detach()))
            
    #valid_accu = accuracy_score(fin_targets,np.argmax(fin_outputs, axis=1).tolist())
    valid_accu = mean_squared_error(fin_targets,fin_outputs)

    return valid_accu, valid_loss / batch_count
