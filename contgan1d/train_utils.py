import torch as th
import torch.distributions as D
import matplotlib.pyplot as plt

import numpy as np
import os

def visualize(model, device, path, epoch, n=512):
    values = model.ode(n=n, device=device).detach().cpu().numpy()

    for i in range(n):
        plt.plot(np.linspace(1., 0., values.shape[0]), values[:, i], linewidth=1, color="green", alpha=0.2)

    plt.savefig(os.path.join(path, "image.png"))
    plt.close()

def train(model, device, run, path, num_epoch=500, num_steps=10, score_steps=5):
    model.to(device)
    run.watch(model)

    direction_optimizer = th.optim.AdamW(model.direction.parameters(), lr=1e-4, weight_decay=0.)
    score_optimizer = th.optim.AdamW(model.score.parameters(), lr=1e-4, weight_decay=0.)

    for epoch in range(1, num_epoch + 1):
        model.train()
        loss_accum = [0, 0]
        loss_count = [0, 0]

        epoch_stat = {"epoch": epoch}

        for step in range(1, num_steps + 1):
            mix = D.Categorical(th.tensor([1, 1]))
            comp = D.Independent(D.Normal(th.tensor([[-3.], [3.]]), th.tensor([[.5], [.5]])), 1)
            gmm = D.MixtureSameFamily(mix, comp)
            x_0 = gmm.sample((1024 * 1024,)).to(device)
            
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
                # Direction loss
                direction_optimizer.zero_grad()
                direction_loss = model.direction_loss(x_0, x_1, t)
                direction_loss.backward()
                direction_optimizer.step()

                loss_accum[1] += direction_loss.item() * x_0.shape[0]
                loss_count[1] += x_0.shape[0]

        epoch_stat["Score loss"] = loss_accum[0] / loss_count[0]
        epoch_stat["Direction loss"] = loss_accum[1] / loss_count[1]

        model.eval()
        visualize(model, device=device, path=path, epoch=epoch)
        
        run.log(epoch_stat)

    return model