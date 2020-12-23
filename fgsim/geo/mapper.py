from typing import TextIO


class geomapper:
    """
    This class provides a geomapper object, that is used to map hits ton cells.
    """

    def __init__(self, configfile: TextIO):
        self.configfile = configfile
        print(f"Init mappber by {self.configfile}")
        with open(self.configfile, "r") as _:
            pass
