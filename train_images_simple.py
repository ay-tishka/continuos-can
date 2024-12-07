from contgan_images_simple.contgan import ContGAN
from contgan_images_simple.utils import count_parameters, load_weights
from contgan_images_simple.train_utils import train
from contgan_images_simple.dataloaders import load_mnist, load_cifar

import os
import click
import wandb

def create_dirs(path):
    dirs = ["generations", "latent_trajectories"]
    for dir in dirs:
        new_path = os.path.join(path, dir)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

@click.command()
@click.option("--device", default="cuda:0", help="Device.")
@click.option("--score_steps", default=20, help="Number of score steps.")
@click.option("--dataset", default="cifar", help="Dataset.")
@click.option("--checkpoint", default=None, help="Checkpoint path.")
def fit(device, score_steps, dataset, checkpoint):
    run = wandb.init(project="contgan-images-simple", name=f"contgan-images-simple-{score_steps}-{dataset}-{checkpoint}")
    run.config.device = device
    run.config.score_steps = score_steps
    run.config.dataset = dataset

    path = os.path.join(f"out/contgan-images-simple-{score_steps}-{dataset}-{checkpoint}")
    if not os.path.exists(path):
        os.makedirs(path)
    create_dirs(path)

    if dataset == "mnist":
        image_shape = (1, 32, 32)
        train_loader, eval_loader = load_mnist(train_batch_size=64, eval_batch_size=64)
    elif dataset == "cifar":
        image_shape = (3, 32, 32)
        train_loader, eval_loader = load_cifar(train_batch_size=64, eval_batch_size=64)
    else:
        raise ValueError("No such dataset.")

    model = ContGAN(
        image_shape=image_shape
    )

    if checkpoint is not None:
        load_weights(model, checkpoint)

    num_params, top = count_parameters(model, return_top=True)
    print("num params:", num_params, "top:", top)
    print("num params encoder:", count_parameters(model.encoder))
    print("num params direction:", count_parameters(model.direction))
    print("num params score:", count_parameters(model.score))

    train(
        model=model,
        device=device,
        path=path,
        run=run,
        train_loader=train_loader,
        eval_loader=eval_loader,
        num_epoch=2000,
        score_steps=score_steps
    )

if __name__ == "__main__":
    fit()