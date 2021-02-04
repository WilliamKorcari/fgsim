from functools import reduce
from multiprocessing import Pool

import awkward as ak
import numba as nb
import numpy as np
import yaml
from tqdm import tqdm

from ..config import conf
from ..utils.istarmap import istarmap
from ..utils.timeit import timeit

threshold = np.float(conf.mapper.energy_threshold)
nvoxel = reduce(lambda a, b: a * b, conf.mapper.calo_img_shape)
xyzvars = nb.typed.List([str(e) for e in conf.mapper.xyz])
maxhits = 5000

cellposD = nb.typed.Dict.empty(
    key_type=nb.types.unicode_type,
    value_type=nb.types.float32[:],
)

with open(f"wd/{conf.tag}/cellpos.yaml", "r") as f:
    y = yaml.load(f, Loader=yaml.SafeLoader)
    for v in xyzvars:
        cellposD[v] = np.array(y[v], dtype=np.float32)


class mapBack:
    def __init__(self) -> None:
        pass

    def map_events(self, events: np.ndarray) -> ak.Array:
        nevents = len(events)
        events = np.array(events)

        if conf.debug:
            eventL = []
            for eventNumber, caloimg in zip(range(nevents), events):
                for fct in (map_calo_to_hitsB,):
                    arr = timeit(fct, n=10)(eventNumber, caloimg)
                    assert set(arr.fields) == {
                        "eventNumber",
                        "recHit_x",
                        "recHit_y",
                        "recHit_z",
                        "energy",
                    }
                    assert len(arr) == maxhits
                    eventL.append(arr)
                break

        else:
            with Pool(conf.cores) as p:
                eventL = list(
                    tqdm(
                        istarmap(
                            p,
                            map_calo_to_hitsB,
                            zip(range(nevents), events),
                        ),
                        total=nevents,
                    )
                )

        res = ak.concatenate(eventL)
        return res


@nb.njit()
def get_idxs_under_threshold(arr: np.ndarray):
    arr = np.argwhere(np.abs(arr) < threshold)
    return arr[: min(len(arr), maxhits)]


@nb.njit()
def coordinate_fieldB(idxs, nhits, cellposD, caloimg):
    ourA = np.empty((nhits, 4), dtype=np.float32)
    for i, idx in enumerate(idxs):
        ourA[i, 0] = caloimg[idx[0], idx[1], idx[2]]
        for j, coord in enumerate(cellposD):
            ourA[i, j + 1] = cellposD[coord][idx[j]]
    return ourA


@nb.njit(nb.types.int64[:](nb.types.int64, nb.types.int64))
def eventNumber_field(nhits, eventNumber):
    return np.repeat(eventNumber, nhits)


def map_calo_to_hitsB(eventNumber: int, caloimg: np.ndarray):
    idxs = get_idxs_under_threshold(caloimg)
    nhits = len(idxs)

    arrD = {}
    arrD["eventNumber"] = ak.layout.NumpyArray(eventNumber_field(nhits, eventNumber))

    coord_arr = coordinate_fieldB(idxs, nhits, cellposD, caloimg)

    for i, coordinate in enumerate(["energy", *xyzvars]):
        arrD[coordinate] = ak.layout.NumpyArray(coord_arr[:, i])
    arr = ak.Array(arrD)
    return arr


# @nb.njit()
# def energy_field(idxs, nhits, caloimg):
#     eV = np.empty(nhits, dtype=np.float32)
#     for i, idx in enumerate(idxs):
#         eV[i] = caloimg[idx[0], idx[1], idx[2]]
#     return eV


# def map_calo_to_hitsA(eventNumber: int, caloimg: np.ndarray):
#     idxs = get_idxs_under_threshold(caloimg)

#     nhits = len(idxs)

#     arrD = {}
#     arrD["eventNumber"] = ak.layout.NumpyArray(eventNumber_field(nhits, eventNumber))
#     arrD["energy"] = ak.layout.NumpyArray(energy_field(idxs, nhits, caloimg))

#     coord_arr = coordinate_field(idxs, nhits, cellposD)

#     for i, coordinate in enumerate(xyzvars):
#         arrD[coordinate] = ak.layout.NumpyArray(coord_arr[:, i])
#     arr = ak.Array(arrD)
#     return arr


# nbtypesD = {
#     "nhits": nb.types.int64[:],
#     "eventNumber": nb.types.int64,
#     "idxs": nb.types.int64[:, :],
#     "cellposD":
# nb.typed.Dict.empty(
#         key_type=nb.types.unicode_type,
#         value_type=nb.types.float32[:],
#     ),
#     "caloimg": nb.types.Array(nb.types.float32, "3", "C"),
# }


# @nb.njit()
# def coordinate_field(idxs, nhits, cellposD):
#     cV = np.empty((nhits, 3), dtype=np.float32)
#     for i, idx in enumerate(idxs):
#         for j, coord in enumerate(cellposD):
#             cV[i, j] = cellposD[coord][idx[j]]
#     return cV


# def map_calo_to_hitsC(eventNumber: int, caloimg: np.ndarray):
#     eventNumbers, nbout = map_calo_to_hitsC_jit(eventNumber, caloimg, cellposD, xyzvars)

#     arr = ak.Array({**nbout, "eventNumber": eventNumbers})
#     return arr


# @nb.njit(parallel=True)
# def map_calo_to_hitsC_jit(eventNumber: int, caloimg: np.ndarray, cellposD, xyzvars):
#     idxs = get_idxs_under_threshold(caloimg)
#     nhits = len(idxs)
#     eventNumbers = eventNumber_field(nhits, eventNumber)
#     arrD = {}
#     coord_arr = coordinate_fieldB(idxs, nhits, cellposD, caloimg)
#     for i in nb.prange(4):
#         if i == 0:
#             arrD["energy"] = coord_arr[:, i]
#         else:
#             arrD[xyzvars[i - 1]] = coord_arr[:, i]

#     return (eventNumbers, arrD)
