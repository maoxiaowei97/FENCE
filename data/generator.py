import numpy as np
import random
import csv
import os
import pandas as pd
from communities.algorithms import louvain_method
import argparse
import configparser

# generate_miss_data
class Generator: 
    def __init__(self, dataPath, savedatapath, distancePath, miss_rate, miss_func, patch_size):              
        self.dataPath = dataPath
        self.savedatapath = savedatapath
        self.distancePath = distancePath
        self.miss_rate = miss_rate
        self.miss_func = miss_func
        self.patch_size = patch_size
    
    def load_data(self):  
        if os.path.isdir(self.dataPath): 
            files = os.listdir(self.dataPath)  
            data = []
            for file in files:
                position = os.path.join(self.dataPath, file)
                print("processing:", position)
                X = np.load(position)['data']
                data.append(X)
            data = np.concatenate(data, 0)
        else:  
            data = np.load(self.dataPath)['data']   
            print('data shape:',data.shape)
      
        return data
    
    def get_graph_classes(self,distance_of_Path, num_of_vertices, id_filename=None):
        if '.npy' in distance_of_Path:          
            adj_mx = np.load(distance_of_Path)
            return adj_mx, None

        else:
            A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                         dtype=np.float32)
            distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                                dtype=np.float32)

            if id_filename:
                with open(id_filename, 'r') as f:
                    id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  
   
                with open(distance_of_Path, 'r') as f:
                    f.readline() 
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) != 3:
                            continue
                        i, j, distance = int(row[0]), int(row[1]), float(row[2])
                        A[id_dict[i], id_dict[j]] = 1
                        A[id_dict[j], id_dict[i]] = 1
                        distaneA[id_dict[i], id_dict[j]] = distance
                        distaneA[id_dict[j], id_dict[i]] = distance
            else: 
                with open(distance_of_Path, 'r') as f:
                    f.readline()
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) != 3:
                            continue
                        i, j, distance = int(row[0]), int(row[1]), float(row[2])   
                        A[i, j] = 1
                        A[j, i] = 1
                        distaneA[i, j] = distance
                        distaneA[j, i] = distance
        communities, _ = louvain_method(A) 
        return communities

    def mask_and_save_data(self, ori_data, miss_func, miss_rate, mask):       
        true_data_savename = 'true_data_' + miss_func + '_' +str(miss_rate) +'.npz'
        save_path = os.path.join(self.savedatapath, true_data_savename)
        data = ori_data[:, :, :]
        np.savez_compressed(save_path, data=data, mask=mask)
        print('missing rate', miss_rate, miss_func, 'true data and mask save in:',save_path)

        miss_data_savename = 'miss_data_' + miss_func + '_' +str(miss_rate) +'.npz'
        save_path = os.path.join(self.savedatapath, miss_data_savename)
        miss_data = np.where(mask==1, ori_data, 0)
        np.savez_compressed(save_path, data=miss_data, mask=mask)
        print('missing rate',miss_rate, miss_func, 'miss data and mask save in:',save_path)

    def mask_array(self, ori_data, miss_rate, miss_func, patch_size): 
        T, N, F = ori_data.shape
        num_of_node = N
        mask = np. ones_like(ori_data)

        if miss_func == 'SR-TR':
            rm = np.random.rand(T,N,1) 
            rm = np.where(rm <= miss_rate, 0, 1)
            mask = np.where(rm == 1, mask, rm)
            self.mask_and_save_data(ori_data, miss_func, miss_rate, mask)

            return mask

        elif miss_func == 'SR-TC':
            rm = np.random.rand(round(T/patch_size),N,1)               
            rm = np.where(rm <= miss_rate, 0, 1)
            rm = rm.repeat(patch_size, axis=0) 
            mask = np.where(rm == 1, mask, rm)
            self.mask_and_save_data(ori_data, miss_func, miss_rate, mask)
            return mask
        
        elif miss_func == 'SC-TR':
            communities = self.get_graph_classes(self.distancePath,num_of_node)
            communities = [list(communities[i]) for i in range(len(communities))]
            community_save_name = 'communities of PEMS04' + '.npz'
            community_save_path = os.path.join(self.savedatapath,community_save_name)
            np.savez_compressed(community_save_path, length = len(communities), communities=communities)
            num_cluster = len(communities)
            rm = np.random.rand(T, num_cluster)
            x = round(T*N*miss_rate/num_of_node) 
            for j in range(num_cluster):              
                for i in range(T):
                    if rm[i][j] <= x/T:
                        t, n = i, j
                        mask_node = communities[n]
                        mask[t, mask_node, :] = 0 
            self.mask_and_save_data(ori_data, miss_func, miss_rate, mask)

            return mask
        
        elif miss_func == 'SC-TC':
            communities = self.get_graph_classes(self.distancePath,num_of_node)
            communities = [list(communities[i]) for i in range(len(communities))]
            num_cluster = len(communities)
            rm = np.random.rand(round(T/patch_size), num_cluster)
            x = round(T * N * miss_rate / (num_of_node * patch_size))
            for j in range(num_cluster):
                for i in range(round(T/patch_size)):
                    if rm[i][j] <= x * patch_size / T:
                        t,n = i,j
                        start = t * patch_size
                        end = (t + 1) * patch_size
                        mask[start:end,communities[n],:] = 0
            self.mask_and_save_data(ori_data, miss_func, miss_rate, mask)  

        else:
            print("miss_func type error!please select from  'SR-TR','SC-TR','SR-TC','SC-TC'")
            return -1             

        return mask
     
    def generate_miss_data(self):
        data = self.load_data()
        mask = self.mask_array(data, self.miss_rate, self.miss_func, self.patch_size)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="PEMS04", type=str,
                    help="Dataset name to process (e.g., PEMS04, PEMS07).")
  parser.add_argument("--patch", default=16, type=int,
                    help="time patch size")
  parser.add_argument("--missrate", default=0.8, type=float,
                    help="missrate")
  parser.add_argument("--misstype", default="SR-TC", type=str,
                    help="misstype: 'SR-TR','SC-TR','SR-TC','SC-TC' ")
  args = parser.parse_args()
  config_path = os.path.join(os.path.dirname(__file__), 'dataset_configs', f'{args.dataset}.conf')
  if not os.path.exists(config_path):
      print(f"Error: Config file not found at '{config_path}'")
      exit(1)
  config = configparser.ConfigParser()
  print('Read config file: %s' % (config_path))

  config.read(args.config)
  config = config["generator"]
  graph_signal_matrix_filename = config["graph_signal_matrix_filename"]
  save_filesdir = config["save_filesdir"]
  os.makedirs(save_filesdir, exist_ok=True)
  distancePath = config["distancePath"]
  patch_size = args.patch
  missrate = args.missrate
  misstype = args.misstype
      
  g = Generator(graph_signal_matrix_filename, save_filesdir, distancePath, missrate, misstype, patch_size)
  g.generate_miss_data()