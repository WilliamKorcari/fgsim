# from torchvision.utils import save_image

from ..config import conf
from ..data_dumper import dump_generated_events
from ..geo.mapback import mapBack
from .holder import modelHolder
from .train import create_noise


def generation_procedure(c: modelHolder):
    # Create the noise vectors
    noise = create_noise(
        sample_size=conf["predict"]["nevents"], nz=conf["model"]["gan"]["nz"]
    )
    # create the final fake image for the epoch
    genEvents = c.generator(noise).cpu().detach()
    print("Generation done")
    print(f"genEvents.shape {genEvents.shape}")
    # shape : sample_size * x * y *z
    mapper = mapBack()
    print("Mapper setup done")
    arr = mapper.map_events(genEvents)
    print("Mapping done")
    dump_generated_events(arr)
