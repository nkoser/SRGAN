import wandb


class WandbUtils:
    def __init__(self, config):
        self.config = config

    def connect_to_wandb(self) -> None:
        wandb.init(project=self.config['exp_name'],
                   config=self.config,
                   name=self.config['run_name']
                   )

    def log_metrics(self, update_dict: dict) -> None:
        wandb.log(update_dict)

    def close_wandb(self) -> None:
        wandb.finish()
