from functools import reduce
from multiprocessing import Pool

import awkward as ak
import numpy as np
import yaml

import numba
from tqdm import tqdm

from ..config import conf


threshold = np.float(conf.mapper.energy_threshold)
nvoxel = reduce(lambda a, b: a * b, conf.mapper.calo_img_shape)
xyzvars = [str(e) for e in conf.mapper.xyz]

cellposD = numba.typed.Dict.empty(
    key_type=numba.types.unicode_type,
    value_type=numba.types.float32[:],
)

with open(f"wd/{conf.tag}/cellpos.yaml", "r") as f:
    y = yaml.load(f, Loader=yaml.SafeLoader)
    for v in xyzvars:
        cellposD[v] = np.array(y[v], dtype=np.float32)


@numba.njit("int64[:,:](float32[:,:,:])")
def get_idxs_under_threshold(arr: np.ndarray):
    arr = np.argwhere(np.abs(arr) < threshold)
    return arr[: min(len(arr), 5000)]


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

    def _map_calo_to_hitsMbuilderRec(self, eventNumber: int, caloimg) -> ak.Array:
        builder = ak.ArrayBuilder()
        _map_calo_to_hitsMbuilderRec_jit(eventNumber, caloimg, builder)
        arr = builder.snapshot()
        return arr

    def map_events(self, events: np.ndarray) -> ak.Array:
        nevents = len(events)
        events = np.array(events)

        if conf.debug:
            eventL = []
            for fct in (
                # self._map_calo_to_hitsMListDict,
                # self._map_calo_to_hitsMbuilderList,
                self._map_calo_to_hitsMbuilderRec,
            ):
                print(fct.__name__)
                for eventNumber, caloimg in tqdm(
                    zip(range(nevents), events), total=nevents
                ):
                    arr = fct(eventNumber, caloimg)
                    assert set(arr.fields) == {
                        "eventNumber",
                        "recHit_x",
                        "recHit_y",
                        "recHit_z",
                        "energy",
                    }
                    eventL.append(arr)
        else:
            with Pool(5) as p:
                eventL = list(
                    tqdm(
                        p.starmap(
                            self._map_calo_to_hitsMbuilderRec,
                            zip(range(nevents), events),
                        ),
                        total=nevents,
                    )
                )

        res = ak.concatenate(eventL)
        return res


# ak.numba.register()

# @numba.njit(signature=(numba.types.int64,numba.types.float32[:,:,:]))
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


# @numba.njit(
#     (numba.types.int64, numba.types.float32[:, :, :], ak.ArrayBuilder.numba_type)
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


# @numba.njit(
#     (numba.types.int64, numba.types.float32[:, :, :], ak.ArrayBuilder.numba_type)
# )
def _map_calo_to_hitsMbuilderRec_jit(
    eventNumber: int, caloimg: np.ndarray, builder: ak.ArrayBuilder
):
    idxs = get_idxs_under_threshold(caloimg)
    builder = ak.ArrayBuilder()
    for idx in idxs:
        builder.begin_record()
        builder.field("eventNumber")
        builder.integer(eventNumber)
        for coordinate, i in zip(xyzvars, idx):
            builder.field(coordinate)
            builder.real(cellposD[coordinate][i])

        # idx is a tensor here, to get the positon, it needs to be converted to a tuple
        builder.field("energy")
        builder.real(caloimg[tuple(idx)].item())
        builder.end_record()
    return
