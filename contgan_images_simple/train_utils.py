from torchvision.utils import save_image
import torch as th

from tqdm import tqdm
import os

def visualize(model, path, epoch, device, n=16):
    sample, latent_trajectories = model.generate(n=n, device=device)
    save_image((sample + 1.) / 2., os.path.join(path, f"generations/{epoch}.png"))

    images = th.concat([(h[0].unsqueeze(0) + 1.) / 2. for h in latent_trajectories], axis=0)
    save_image(images, os.path.join(path, f"latent_trajectories/{epoch}.png"))
    
def train(model, device, path, run, train_loader, eval_loader, num_epoch, score_steps):
    model.to(device)
    run.watch(model)

    encoder_direction_optimizer = th.optim.AdamW(model.direction.parameters(), lr=1e-4, weight_decay=0.)
    score_optimizer = th.optim.AdamW(model.score.parameters(), lr=1e-4, weight_decay=0.)

    model.eval()
    visualize(model, path=path, epoch=0, device=device)

    for epoch in range(1, num_epoch + 1):
        model.train()
        loss_accum = [0, 0]
        loss_count = [0, 0]

        epoch_stat = {"epoch": epoch}

        for step, (x_0, _) in enumerate(tqdm(train_loader), start=1):
            x_0 = x_0.to(device)
            x_1 = th.randn_like(x_0)
            t = th.rand(x_0.shape[0], device=device)

            if step % (score_steps + 1) != 0:
                # Score loss
                score_optimizer.zero_grad()
                score_loss = model.score_loss(x_0, x_1, t)
                score_loss.backward()
                score_optimizer.step()

                loss_accum[0] += score_loss.item() * x_0.shape[0]
                loss_count[0] += x_0.shape[0]
            else:
                # Encoder direction loss
                encoder_direction_optimizer.zero_grad()
                encoder_direction_loss = model.encoder_direction_loss(x_0, x_1, t)
                encoder_direction_loss.backward()
                encoder_direction_optimizer.step()

                loss_accum[1] += encoder_direction_loss.item() * x_0.shape[0]
                loss_count[1] += x_0.shape[0]

        epoch_stat["Score loss"] = loss_accum[0] / loss_count[0]
        epoch_stat["Encoder Direction loss"] = loss_accum[1] / loss_count[1]
        
        run.log(epoch_stat)

        model.eval()
        visualize(model, path=path, epoch=epoch, device=device)

        # Just rewrite weights
        th.save(model.state_dict(), os.path.join(path, f"model.pt"))