from collections import OrderedDict
from .config import conf


def most_freq_zval(posD: OrderedDict):
    from collections import Counter

    cnt = Counter(posD[conf.mapper.xyz[-1]])
    zval = cnt.most_common(1)[0][0]
    return(zval)
