from functools import reduce
from logging import raiseExceptions
from multiprocessing import Pool

import awkward as ak
import numpy as np
import yaml

import numba as nb
from tqdm import tqdm

from ..config import conf
from ..utils import timeit


threshold = np.float(conf.mapper.energy_threshold)
nvoxel = reduce(lambda a, b: a * b, conf.mapper.calo_img_shape)
xyzvars = nb.typed.List([str(e) for e in conf.mapper.xyz])

cellposD = nb.typed.Dict.empty(
    key_type=nb.types.unicode_type,
    value_type=nb.types.float32[:],
)

with open(f"wd/{conf.tag}/cellpos.yaml", "r") as f:
    y = yaml.load(f, Loader=yaml.SafeLoader)
    for v in xyzvars:
        cellposD[v] = np.array(y[v], dtype=np.float32)

cellPosDType = nb.typeof(cellposD)


class mapBack:
    def __init__(self) -> None:
        pass

    def _map_calo_to_hitsMListDict(self, eventNumber: int, caloimg) -> ak.Array:
        return _map_calo_to_hitsMListDict_jit(eventNumber, caloimg)

    # @timing_val
    def _map_calo_to_hitsMbuilderList(self, eventNumber: int, caloimg) -> ak.Array:
        builder = ak.ArrayBuilder()
        _map_calo_to_hitsMbuilderList_jit(eventNumber, caloimg, builder)
        arr = builder.snapshot()
        return arr

    def map_events(self, events: np.ndarray) -> ak.Array:
        nevents = len(events)
        events = np.array(events)

        if True:
            eventL = []
            # for eventNumber, caloimg in tqdm(
            #     zip(range(nevents), events), total=nevents
            # ):
            for eventNumber, caloimg in zip(range(nevents), events):
                for fct in (
                    # self._map_calo_to_hitsMListDict,
                    # self._map_calo_to_hitsMbuilderList,
                    map_calo_to_hitsA,
                ):
                    print(fct.__name__)

                    arr = timeit(fct, n=10)(eventNumber, caloimg)
                    assert set(arr.fields) == {
                        "eventNumber",
                        "recHit_x",
                        "recHit_y",
                        "recHit_z",
                        "energy",
                    }
                    eventL.append(arr)
                break

        else:
            with Pool(5) as p:
                eventL = list(
                    tqdm(
                        p.starmap(
                            map_calo_to_hitsA,
                            zip(range(nevents), events),
                        ),
                        total=nevents,
                    )
                )

        res = ak.concatenate(eventL)
        return res


# ak.nb.register()

# @nb.njit(signature=(nb.types.int64,nb.types.float32[:,:,:]))
def _map_calo_to_hitsMListDict_jit(eventNumber: int, caloimg: np.ndarray) -> ak.Array:
    # Get the indices with entries over the energy threshold
    idxs = get_idxs_under_threshold(np.array(caloimg))
    assert len(idxs.shape) == 2 and idxs.shape[1] == 3
    outD = {}
    outD["eventNumber"] = np.repeat(eventNumber, len(idxs))
    for coordinate in xyzvars:
        outD[coordinate] = []
    outD["energy"] = []

    for idx in idxs:
        for coordinate, i in zip(xyzvars, idx):
            outD[coordinate].append(cellposD[coordinate][i])
        outD["energy"].append(caloimg[tuple(idx)].item())
    arr = ak.Array(outD)
    return arr


# @nb.njit(
#     (nb.types.int64, nb.types.float32[:, :, :], ak.ArrayBuilder.nb_type)
# )
def _map_calo_to_hitsMbuilderList_jit(
    eventNumber: int, caloimg: np.ndarray, builder: ak.ArrayBuilder
):
    idxs = get_idxs_under_threshold(caloimg)

    builder.begin_record()
    builder.field("eventNumber")
    builder.begin_list()
    for idx in idxs:
        builder.integer(eventNumber)
    builder.end_list()
    builder.end_record()

    for i, coordinate in enumerate(xyzvars):
        builder.begin_record()
        builder.field(coordinate)
        builder.begin_list()
        for idx in idxs:
            builder.append(cellposD[coordinate][idx[i]])
        builder.end_list()
        builder.end_record()

    builder.begin_record()
    builder.field("energy")
    builder.begin_list()
    for idx in idxs:
        builder.append(caloimg[tuple(idx)].item())
    builder.end_list()
    builder.end_record()

    return


# def map_calo_to_hitsMbuilderRec_jit(
#     eventNumber: int, caloimg: np.ndarray, builder: ak.ArrayBuilder
# ):
#     idxs = get_idxs_under_threshold(caloimg)

#     nhits = len(idxs)

#     builder.begin_record()
#     builder.field("eventNumber")
#     builder.append(ak.layout.NumpyArray(eventNumber_field(nhits, eventNumber)))
#     builder.field("energy")
#     builder.append(ak.layout.NumpyArray(energy_field(idxs, nhits, caloimg)))

#     coord_arr = coordinate_field(idxs, nhits, cellposD)

#     for i, coordinate in enumerate(xyzvars):
#         builder.field(coordinate)
#         builder.append(ak.layout.NumpyArray(coord_arr[:, i]))

#     return


def map_calo_to_hitsA(eventNumber: int, caloimg: np.ndarray):
    idxs = get_idxs_under_threshold(caloimg)

    nhits = len(idxs)

    arrD = {}
    arrD["eventNumber"] = ak.layout.NumpyArray(eventNumber_field(nhits, eventNumber))
    arrD["energy"] = ak.layout.NumpyArray(energy_field(idxs, nhits, caloimg))

    coord_arr = coordinate_field(idxs, nhits, cellposD)

    for i, coordinate in enumerate(xyzvars):
        arrD[coordinate] = ak.layout.NumpyArray(coord_arr[:, i])
    arr = ak.Array(arrD)
    return arr


# @nb.njit("int64[:,:](float32[:,:,:])")
@nb.njit()
def get_idxs_under_threshold(arr: np.ndarray):
    arr = np.argwhere(np.abs(arr) < threshold)
    return arr[: min(len(arr), 5000)]


@nb.njit()
def energy_field(idxs, nhits, caloimg):
    eV = np.empty(nhits, dtype=np.float32)
    for i, idx in enumerate(idxs):
        eV[i] = caloimg[idx[0], idx[1], idx[2]]
    return eV


@nb.njit(nb.types.int64[:](nb.types.int64, nb.types.int64))
def eventNumber_field(nhits, eventNumber):
    return np.repeat(eventNumber, nhits)


@nb.njit()
def coordinate_field(idxs, nhits, cellposD):
    cV = np.empty((nhits, 3), dtype=np.float32)
    for i, idx in enumerate(idxs):
        for j, coord in enumerate(cellposD):
            cV[i, j] = cellposD[coord][idx[j]]
    return cV
