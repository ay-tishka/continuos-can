from contgan1d.contgan import ContGAN
from contgan1d.utils import count_parameters
from contgan1d.train_utils import train

import os
import click
import wandb

@click.command()
@click.option("--device", default="cuda:0", help="Device.")
def fit(device):
    run = wandb.init(project="contgan1d", name=f"contgan1d")
    run.config.device = device

    path = os.path.join(f"out/contgan1d")
    if not os.path.exists(path):
        os.makedirs(path)

    model = ContGAN()

    num_params, top = count_parameters(model, top=5)
    print("num params:", num_params, "top:", top)
    print("num params direction:", count_parameters(model.direction))
    print("num params score:", count_parameters(model.score))

    train(
        model=model,
        device=device,
        run=run,
        path=path
    )

if __name__ == "__main__":
    fit()