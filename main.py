import hydra
import logging
import sys
from omegaconf import DictConfig, OmegaConf
from cai.application import Application

@hydra.main(config_path="config", config_name="main")
def main(cfg: DictConfig):
    print(cfg)
    logger = logging.getLogger(__name__)
    logger.info(f"{__name__} is running")
    logger.info(f"Initializing Application")
    app = Application(cfg)

    logger.info(f"Running Application")
    app.run()

if __name__ == "__main__":
    main()
