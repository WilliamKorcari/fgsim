import os
import pickle
import yaml
import awkward as ak
import numpy as np
import uproot
import torch
#from config import conf
#from ..utils.logger import logger
from torch_geometric.data import Data

path = "/afs/desy.de/user/k/korcariw/geo_ex/src/output/"

# load the root table
fnlup = f"{path}DetIdLUT.root"
rf = uproot.open(fnlup)
arr = rf["analyzer/tree"].arrays()
keydf = ak.to_pandas(arr[0])
keydf = keydf.set_index("globalid")

# load the geometry

fngeopic = f"{path}geometry.pickle"
fngeoyam = f"{path}geometry.yaml"
graphpath = f"{path}edge_index.pt"
if os.path.isfile(fngeopic):
    with open(fngeopic, "rb") as f:
        geoD = pickle.load(f)
else:
    with open(fngeoyam, "r") as f:
        geoD = yaml.load(f, Loader = yaml.safe_load)
    with open(fngeopic, "wb") as f:
        pickle.dump(geoD, f)

if os.path.isfile(graphpath):
    graph = torch.load(graphpath)
else:
    # Instanciate empty array

    edgeA = np.empty((2,0), dtype=int)
    #logger.info(f"egdeA shape: {egdeA.shape}")
    print('\nBuilding edge_index...\n')
    for originid, row in keydf.iterrows():
        if row.detectorid==8:
            for i in range(12):
                edgeA = np.append(edgeA, [[originid], [row["n" + str(i)]]], axis=1)
            edgeA = np.append(edgeA, [[originid], [row["next"]]], axis=1)
            edgeA = np.append(edgeA, [[originid], [row["previous"]]], axis=1)


    # Prune
    print("Pruning...\n")
    edgeA = edgeA[:, edgeA[0] != 0]
    print("Saving the graph...\n")
    edge_index = torch.tensor(edgeA, dtype=torch.long)
    nodes = torch.tensor(keydf.index.to_numpy(dtype=int))
    graph = Data(x=nodes, edge_index=edge_index)
    torch.save(graph, graphpath)
    print("Done.")

