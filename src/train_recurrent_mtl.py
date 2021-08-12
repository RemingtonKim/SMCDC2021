import pandas as pd
import os
import pickle
import numpy as np
import torch
import torch_geometric
import sklearn.metrics as metrics
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from models import MTLRecurrentGCN
from utils import MTLLoss, log_and_print

EPOCHS = 10
BATCH_SIZE = 4
VALIDATION_TEST_BATCH_SIZE = 2
NUM_NODE_FEATURES = 32
LEADTIME_RANGE = 4
LEARNING_RATE = 0.001
MODE = 'train'
STORAGE_PATH = '../models/rec_mtl/'
LOG_FILE = 'log_rec_mtl.txt'
EPOCH_FILE = 'epoch_rec_mtl.txt'

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Loading data.')
    with open('../data/sampled_graphs/index_dict.pickle', 'rb') as handle:
        index_dict = pickle.load(handle)
    with open('../data/sampled_graphs/node_names.pickle', 'rb') as handle:
        node_names = pickle.load(handle)
    with open('../data/sampled_graphs/node_features.pickle', 'rb') as handle:
        node_features = pickle.load(handle).type(torch.float32)


    date_data_pos = pd.read_csv('../data/sampled_graphs/date_data_pos.csv')
    date_data_neg = pd.read_csv('../data/sampled_graphs/date_data_neg.csv')

    date_data_pos['src'] = [index_dict[x] for x in date_data_pos['src']]
    date_data_pos['dst'] = [index_dict[x] for x in date_data_pos['dst']]
    date_data_neg['src'] = [index_dict[x] for x in date_data_neg['src']]
    date_data_neg['dst'] = [index_dict[x] for x in date_data_neg['dst']]

    # future_time means the same thing as leadtime
    date_data_pos = date_data_pos.assign(y1=1, y2=date_data_pos['future_time'].values+1)
    date_data_neg = date_data_neg.assign(y1=0, y2=0)

    # 1/2019 - 7/2020 = train; 8/2020 = val; 9/2020 - 5/2021 = test
    train_pos = date_data_pos.iloc[:date_data_pos[(date_data_pos['year']==2020) & (date_data_pos['month']==7)].index[-1]+1][['src', 'dst', 'start_index', 'end_index', 'y1', 'y2']].values
    val_pos = date_data_pos[date_data_pos[(date_data_pos['year']==2020) & (date_data_pos['month']==7)].index[-1]+1:date_data_pos[(date_data_pos['year']==2020) & (date_data_pos['month']==8)].index[-1]+1][['src', 'dst', 'start_index', 'end_index', 'y1', 'y2']].values
    test_pos = date_data_pos[date_data_pos[(date_data_pos['year']==2020) & (date_data_pos['month']==8)].index[-1]+1:][['src', 'dst', 'start_index', 'end_index', 'y1', 'y2']].values
    train_neg = date_data_neg.iloc[:date_data_pos[(date_data_pos['year']==2020) & (date_data_pos['month']==7)].index[-1]+1][['src', 'dst', 'start_index', 'end_index', 'y1', 'y2']].values
    val_neg = date_data_neg.iloc[date_data_pos[(date_data_pos['year']==2020) & (date_data_pos['month']==7)].index[-1]+1:date_data_pos[(date_data_pos['year']==2020) & (date_data_pos['month']==8)].index[-1]+1][['src', 'dst', 'start_index', 'end_index', 'y1', 'y2']].values
    test_neg = date_data_neg.iloc[date_data_pos[(date_data_pos['year']==2020) & (date_data_pos['month']==8)].index[-1]+1:][['src', 'dst', 'start_index', 'end_index', 'y1', 'y2']].values

    # shuffle 
    train = np.concatenate((train_pos, train_neg))
    val = np.concatenate((val_pos, val_neg))
    test = np.concatenate((test_pos, test_neg))
    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)
    
    print('Getting edge indices.')
    edge_indices = []
    for file in os.listdir('../data/sampled_graphs'):
        if file.startswith("graph"):
            with open('../data/sampled_graphs/' + file, 'rb') as handle:
                file = file.split('.')[0]
                # print(file)
                edge_indices.append((pickle.load(handle), int(file.split('_')[1]), int(file.split('_')[2])))
    
    # sort by date
    edge_indices.sort(key = lambda x: (x[1], x[2]))
    edge_indices = [x[0] for x in edge_indices]

    # remove self loops that were added during preprocessing 
    for i in range(len(edge_indices)):
        edge_indices[i] = torch_geometric.utils.remove_self_loops(edge_index=edge_indices[i])[0]

    if MODE == 'train':
        print('Building training model.')
        model = MTLRecurrentGCN(node_features=NUM_NODE_FEATURES, leadtime_range=LEADTIME_RANGE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        log_file = open(STORAGE_PATH + LOG_FILE, 'a')


        max_val_auc = 0
        max_val_auc_epoch = -1
        for epoch in range(EPOCHS):
            print('Epoch: {}.'.format(epoch))

            model.train()
            train_cost = []

            bin_yhats = []
            lead_yhats = []

            bin_ys = []
            lead_ys = []

            for index in tqdm(range(0, len(train), BATCH_SIZE)):

                torch.cuda.empty_cache()
                optimizer.zero_grad()

                current_segment = train[index: index + BATCH_SIZE]

                indices = []
                bin_y = []
                lead_y = []
                src = []
                dst = []
                offset = 0
                for sample in current_segment:
                    # append a range object with three values 
                    indices.append(range(sample[2]+12, sample[3]+1, 12))
                    bin_y.append(sample[4])
                    lead_y.append(sample[5])
                    src.append(sample[0]+offset)
                    dst.append(sample[1]+offset)
                    offset+=node_features.shape[0]

                bin_ys += bin_y
                lead_ys += lead_y

                bin_y = torch.tensor(bin_y, dtype=torch.float32).reshape(-1, 1).to(device)
                lead_y = torch.tensor(lead_y).to(device)
                src = torch.tensor(src)
                dst = torch.tensor(dst)
                indices = np.array(indices).T

                batch = []
                for t in indices:
                    batch.append(Batch.from_data_list([Data(edge_index=edge_indices[i], num_nodes=node_features.shape[0]) for i in t], exclude_keys=['x']).to(device))

                x = torch.cat([node_features]*BATCH_SIZE, axis = 0).to(device)

                bin_yhat, lead_yhat = model(x, batch, src, dst)
                
                bin_yhats += bin_yhat.detach().cpu().tolist()
                lead_yhats += torch.exp(lead_yhat).detach().cpu().tolist()

                loss = MTLLoss(bin_yhat, bin_y, lead_yhat, lead_y)

                train_cost.append(loss.item())

                loss.backward()
                optimizer.step()
            
            log_and_print('Training loss : {}'.format(sum(train_cost)/len(train_cost)), log_file)
            log_and_print('Training Bin ROC AUC : {}; training lead ROC AUC : {}'.format(metrics.roc_auc_score(bin_ys, bin_yhats), metrics.roc_auc_score(lead_ys, lead_yhats, multi_class='ovr')), log_file)

            torch.save(model.state_dict(), STORAGE_PATH + 'rec_mtl_{}.pth'.format(epoch))

            model.eval()
            val_cost = []

            bin_yhats = []
            lead_yhats = []

            bin_ys = []
            lead_ys = []

            for index in tqdm(range(0, len(val), VALIDATION_TEST_BATCH_SIZE)):

                torch.cuda.empty_cache()
                current_segment = val[index: index + VALIDATION_TEST_BATCH_SIZE]

                indices = []
                bin_y = []
                lead_y = []
                src = []
                dst = []
                offset = 0

                for sample in current_segment:
                    # append a range object with three values 
                    indices.append(range(sample[2]+12, sample[3]+1, 12))
                    bin_y.append(sample[4])
                    lead_y.append(sample[5])
                    src.append(sample[0]+offset)
                    dst.append(sample[1]+offset)
                    offset+=node_features.shape[0]

                bin_ys += bin_y
                lead_ys += lead_y

                bin_y = torch.tensor(bin_y, dtype=torch.float32).reshape(-1, 1).to(device)
                lead_y = torch.tensor(lead_y).to(device)
                src = torch.tensor(src)
                dst = torch.tensor(dst)
                indices = np.array(indices).T

                batch = []
                for t in indices:
                    batch.append(Batch.from_data_list([Data(edge_index=edge_indices[i], num_nodes=node_features.shape[0]) for i in t], exclude_keys=['x']).to(device))

                x = torch.cat([node_features]*VALIDATION_TEST_BATCH_SIZE, axis = 0).to(device)

                bin_yhat, lead_yhat = model(x, batch, src, dst)
                
                bin_yhats += bin_yhat.detach().cpu().tolist()
                lead_yhats += torch.exp(lead_yhat).detach().cpu().tolist()
                
                loss = MTLLoss(bin_yhat, bin_y, lead_yhat, lead_y)

                val_cost.append(loss.item())

            val_loss = sum(val_cost)/len(val_cost)
            val_auc = metrics.roc_auc_score(bin_ys, bin_yhats)
            
            # keep track of max val auc
            if(val_auc>=max_val_auc):
                max_val_auc = val_auc
                max_val_auc_epoch = epoch
            
            log_and_print('Validation loss : {}'.format(val_loss), log_file)
            log_and_print('Validation Bin ROC AUC : {}; validation lead ROC AUC : {}'.format(val_auc, metrics.roc_auc_score(lead_ys, lead_yhats, multi_class='ovr')), log_file)

        log_file.close()

        # write out max epoch
        epoch_file = open(STORAGE_PATH + EPOCH_FILE, 'w')
        epoch_file.write(max_val_auc_epoch)
        epoch_file.close()

    elif MODE == 'test':
        print('Building testing model.')
        model = MTLRecurrentGCN(node_features=NUM_NODE_FEATURES, leadtime_range=LEADTIME_RANGE).to(device)

        # get epoch with best val auc
        epoch_file = open(STORAGE_PATH + EPOCH_FILE, 'r')
        load_epoch = int(epoch_file.readline())
        epoch_file.close()

        print('Loading epoch: {}.'.format(load_epoch))

        model.load_state_dict(torch.load(STORAGE_PATH + 'rec_mtl_{}.pth'.format(load_epoch)))
        model.eval()

        test_cost = []

        bin_yhats = []
        lead_yhats = []

        bin_ys = []
        lead_ys = []

        for index in tqdm(range(0, len(test), VALIDATION_TEST_BATCH_SIZE)):
            torch.cuda.empty_cache()
            current_segment = test[index: index + VALIDATION_TEST_BATCH_SIZE]

            indices = []
            bin_y = []
            lead_y = []
            src = []
            dst = []
            offset = 0
            for sample in current_segment:
                # append a range object with three values 
                indices.append(range(sample[2]+12, sample[3]+1, 12))
                bin_y.append(sample[4])
                lead_y.append(sample[5])
                src.append(sample[0]+offset)
                dst.append(sample[1]+offset)
                offset+=node_features.shape[0]

            bin_ys += bin_y
            lead_ys += lead_y

            bin_y = torch.tensor(bin_y, dtype=torch.float32).reshape(-1, 1).to(device)
            lead_y = torch.tensor(lead_y).to(device)
            src = torch.tensor(src)
            dst = torch.tensor(dst)
            indices = np.array(indices).T

            batch = []
            for t in indices:
                batch.append(Batch.from_data_list([Data(edge_index=edge_indices[i], num_nodes=node_features.shape[0]) for i in t], exclude_keys=['x']).to(device))

            x = torch.cat([node_features]*VALIDATION_TEST_BATCH_SIZE, axis = 0).to(device)

            bin_yhat, lead_yhat = model(x, batch, src, dst)
            
            bin_yhats += bin_yhat.detach().cpu().tolist()
            lead_yhats += torch.exp(lead_yhat).detach().cpu().tolist()
            
            loss = MTLLoss(bin_yhat, bin_y, lead_yhat, lead_y)

            test_cost.append(loss.item())
        
        print('Test loss : {}'.format(sum(test_cost)/len(test_cost)))

        bin_auc = metrics.roc_auc_score(bin_ys, bin_yhats)
        bin_yhats_hard = (torch.tensor(bin_yhats)>0.5).float()
        bin_ accuracy = metrics.accuracy_score(bin_ys, bin_yhats_hard)
        precision = metrics.precision_score(bin_ys, bin_yhats_hard)
        recall = metrics.recall_score(bin_ys, bin_yhats_hard)
        f1 = metrics.f1_score(bin_ys, bin_yhats_hard)
        print('Test Bin ROC AUC: {}\nAccuracy : {}\nPrecision : {}\nRecall : {}\nF1 : {}'.format(bin_auc, accuracy, precision, recall, f1))

        lead_auc = metrics.roc_auc_score(lead_ys, lead_yhats, multi_class='ovr')
        lead_accuracy = metrics.accuracy_score(lead_ys, np.argmax(lead_yhats, axis=1))
        print('Test lead ROC AUC: {}\nAccuracy: {}'.format(lead_auc, lead_accuracy))

        

