import logging
from ..config import conf

logger = logging.getLogger(__name__)
format="%(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    filename=f"wd/{conf.tag}/log",
    filemode="w",
    format=format,
)
logger.setLevel(logging.INFO)


streamhandler = logging.StreamHandler()
formatter = logging.Formatter(format)
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)