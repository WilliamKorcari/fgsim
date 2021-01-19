import awkward as ak
import numpy as np


class Geomapper:
    """
    This class provides a geomapper object, that is used to map hits ton cells.
    """

    def __init__(self, posD: dict):
        self.xyz = posD.keys()
        # reduce the number of entries
        for v in self.xyz:
            posD[v] = self.__redArray(posD[v])
        self.flatposD = posD
        del posD
        for v in self.xyz:
            self._providemapping(v)

    def __redArray(self, arr):
        # limit to 10 events
        arr = arr[: min(len(arr), 10)]
        print(type(arr))
        return(ak.flatten(ak.Array(arr)))

    def _providemapping(self, var):
        print(self.flatposD[var])
