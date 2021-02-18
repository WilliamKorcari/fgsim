from ..config import conf
from ..data_dumper import dump_generated_events
from ..geo.mapback import mapBack
from ..utils.logger import logger
from .holder import modelHolder
from .train import create_noise


def generation_procedure(c: modelHolder):
    # Create the noise vectors
    noise = create_noise(
        sample_size=conf["predict"]["nevents"], nz=conf["model"]["gan"]["nz"]
    )
    # create the final fake image for the epoch
    genEvents = c.generator(noise).cpu().detach()
    logger.info("Generation done")
    logger.debug(f"genEvents.shape {genEvents.shape}")
    # shape : sample_size * x * y *z
    mapper = mapBack()
    logger.info("Mapper setup done")
    arr = mapper.map_events(genEvents)
    logger.info("Mapping done")
    dump_generated_events(arr)
