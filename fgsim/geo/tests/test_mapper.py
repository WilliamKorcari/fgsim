from fgsim.geo import mapper


def test_geomapper():
    """
    docstring
    """
    mymapper = mapper.geomapper("data/test.toml")
    print(mymapper.configfile)
