import os
import pickle
import yaml
import awkward as ak
import numpy as np
import uproot
import torch
from ..config import conf
from ..utils.logger import logger
from torch_geometric.data import Data


# load the root table
fnlup = conf["luppath"]
rf = uproot.open(fnlup)
arr = rf["analyzer/tree"].arrays()
keydf = ak.to_pandas(arr[0])
keydf = keydf.set_index("globalid")

# load the geometry

fngeopic = conf["geoyamlpath"].strip("yaml") + "pickle"
if os.path.isfile(fngeopic):
    with open(fngeopic, "rb") as f:
        geoD = pickle.load(f)
else:
    with open(conf["geoyamlpath"], "r") as f:
        geoD = yaml.load(f)
    with open(fngeopic, "wb") as f:
        pickle.dump(geoD, f)

if os.path.isfile(conf["graphpath"]):
    graph = torch.load(conf["graphpath"])
else:
    # Instanciate empty array
    edgeA = np.empty((2,0), dtype=int)
    logger.info(f"egdeA shape: {egdeA.shape}")

    for originid, row in keydf.iterrows():
        for i in range(row.nneighbors + row.ngapneighbors):
            edgeA = np.append(edgeA, [[originid], [row["n" + str(i)]]], axis=1)



    # Prune
    edgeA = edgeA[:, edgeA[0] != 0]

    edge_index = torch.tensor(edgeA, dtype=torch.long)
    nodes = torch.tensor(keydf.index.to_numpy(dtype=int))
    graph = Data(x=nodes, edge_index=edge_index)
    torch.save(graph, conf["graphpath"])
