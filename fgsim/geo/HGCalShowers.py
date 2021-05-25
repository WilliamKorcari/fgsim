import os
import time
import pickle
import yaml
import awkward as ak
import numpy as np
import uproot
import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
import pandas as pd
from HGCal_utils import idx_cluster, new_coord, summed_e, bool_mask_matched_indeces, side_indeces_bool_mask, filter_array


class HGCalShowers(InMemoryDataset):
    def __init__(self, root, raw_files = [], 
                 out_file = 'data.pt', 
                 transform=None, 
                 pre_transform=None,
                 load_on_gpu = False, 
                 include_labels = False):
        self.include_labels = include_labels
        #self.use_gpu = use_gpu
        self.raw_files = raw_files
        self.out_file = out_file
        self.device = torch.device("cuda:0" if load_on_gpu else "cpu")
        super(HGCalShowers, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location = self.device) #
        

    @property
    def raw_file_names(self):
            return self.raw_files    
    @property
    def raw_dir(self):
        return osp.join(self.root, 'ntuple')
    @property
    def processed_file_names(self):
        return [self.out_file]
    
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')
    @property
    def geometry_dir(self):
        return osp.join(self.root, 'geom')
    @property
    def tree(self):
        return "treeMaker/tree"

    
    def download(self):
        # Download to `self.raw_dir`.
        #download_url(url, self.raw_dir)
        #...
        pass

    def delete_data(self):
        if os.path.isdir(self.processed_dir):
            os.remove(self.processed_dir+self.processed_file_names)
            os.rmdir(self.processed_dir)
        else:
            print("Directory does not exist yet")

            
    def load_ntuple(self, path):
        with uproot.open(path)[self.tree] as ntuple:
            E = np.array(ntuple["simHit_E"].array(library = "np"))
            Z = np.array(ntuple["simHit_z"].array(library = "np"))
            detId = np.array(ntuple["simHit_detid"].array(library = "np")) 
            gen_E = np.array(ntuple["genPh_E"].array(library = "np")) 

            ##Might be needed for building on-the-fly the geometry file
            #gen_eta = np.array(ntuple["genPh_eta"].array(library = "np"))
            #gen_phi = np.array(ntuple["genPh_phi"].array(library = "np"))

            #select only z+ side of the calorimeter
            pos_idx, neg_idx = side_indeces_bool_mask(Z, (len(Z)))
            E      = filter_array(E , pos_idx, len(E))
            detId  = filter_array(detId , pos_idx, len(E))

            #clustering hits on the same cell
            idx = {}
            e = np.empty_like(detId, dtype=object)
            detid = np.empty_like(detId, dtype=object)

            for i in range(len(detId)):
                idx[i] = idx_cluster(detId[i])
                detid[i] = new_coord(detId[i], idx[i])
                e[i] = summed_e(E[i], idx[i])
        return detid, e, gen_E
            
    def cellid_adj_matrix(self): #pytorch_geometric adj_matrix format (tensor of connected edges with dim 2xnum_of_edges)
        fnlup = osp.join(self.geometry_dir, "DetIdLUT.root") #conf["luppath"]
        rf = uproot.open(fnlup)
        arr = rf["analyzer/tree"].arrays()
        keydf = ak.to_pandas(arr[0])
        keydf = keydf.set_index("globalid")

        # load the geometry
        geoyamlpath = osp.join(self.geometry_dir, "geometry.yaml")
        fngeopic = osp.join(self.geometry_dir, "geometry.pickle") #conf["geoyamlpath"].strip("yaml") + "pickle"
        if os.path.isfile(fngeopic):
            with open(fngeopic, "rb") as f:
                geoD = pickle.load(f)
        else:
            with open(geoyamlpath, "r") as f:
                geoD = yaml.load(f)
            with open(fngeopic, "wb") as f:
                pickle.dump(geoD, f)
        graphpath = osp.join(self.geometry_dir, "edge_index.pt")
        if os.path.isfile(graphpath):
            edge_index = torch.load(graphpath)
        else:
            # Instanciate array
            edgeA = np.empty((2,0), dtype=int)

            for originid, row in keydf.iterrows():
                for i in range(row.nneighbors + row.ngapneighbors):
                    edgeA = np.append(edgeA, [[originid], [row["n" + str(i)]]], axis=1)


            # Prune
            edgeA = edgeA[:, edgeA[0] != 0]

            edge_index = torch.tensor(edgeA, dtype=torch.long)
            torch.save(edge_index, graphpath)
        return keydf.index, edge_index            
            
    def idx_adj_matrix(self, index, edge_index):
        adjpath = osp.join(self.geometry_dir, "adj_matrix.pt")
        mapID = pd.DataFrame(index)

        mapID = mapID["globalid"]
        if os.path.isfile(adjpath):
            adj = torch.load(adjpath)
            
            return mapID, adj
        else:
            ## makes a copy of adj matrix but using indeces of detids instead of directly detids
            idx_pairs = [(np.where(mapID==i)[0][0], np.where(mapID==j)[0][0]) for i, j in zip(edge_index[0].numpy(), edge_index[1].numpy())]
            idx_pairs = np.array(idx_pairs).T

            #build and store actual adj matrix
            adj = torch.zeros([len(index), len(index)])
            adj[idx_pairs[0, :], idx_pairs[1, :]] = 1
            torch.save(adj, adjpath)
            return mapID, adj

        
    def process(self):
        # Read data into huge `Data` list.
        file_list = self.raw_paths
        if not os.path.isdir(self.processed_dir):
            os.mkdir(self.processed_dir)
        '''
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        '''
        #creating graph dataset
        data_list = []
        globalids, edge_indeces = self.cellid_adj_matrix()
        mapID, adj = self.idx_adj_matrix(globalids, edge_indeces)
        adj = adj.to(self.device)
        for iFile in range(len(self.raw_file_names)):
            print(f'Opening file {self.raw_file_names[iFile]}')
            path = osp.join(self.raw_dir, self.raw_file_names[iFile])
            detid, e, gen_E = self.load_ntuple(path)
                
            for j in range(len(detid)):
                fmx = np.array(detid[j], dtype = int)
                bad_idx = []
                for i in range(len(fmx)):
                    if fmx[i] not in globalids:
                        bad_idx.append(i)
                #remove hits outside the considered portion of calorimeter (window)
                fmx = np.delete(fmx, bad_idx, 0)
                e[j] = np.delete(e[j], bad_idx, 0)
                feat_idx = torch.tensor([np.where(mapID==k)[0][0] for k in fmx]).to(self.device)
                local_adj = torch.index_select(adj, 0, feat_idx)
                local_adj = torch.index_select(local_adj, 1, feat_idx)
                label = torch.tensor(gen_E[j][0]) if self.include_labels else None
            #for h in range(len(e)):
                e[j] = torch.tensor(e[j])
                e[j] = torch.reshape(e[j], (e[h].size()[0], 1))
                data_list.append(Data(x =e[j],
                                 edge_index = torch.tensor(local_adj.to_sparse()._indices(), dtype = torch.long), 
                                 y = label
                                )
                            )
        print(f"The dataset contains {len(data_list)} showers")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])#save(data, self.processed_paths[0])#
