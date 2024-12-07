# Continuos GAN

## Formulation

### Discrete case

Start definition with discrete time. First of all, define $`x_t = (1 - t) x_0 + t x_1`$, where $`x_0 \sim p(x_0)`$ - data to generate, $`t \in [0, 1]`$ - time and $`x_1 \sim \mathcal{N}(x_1, 0, I)`$. $`x_0`$ and $`x_1`$ are independent in simplest case. So, in $`t=0`$ and $`t=1`$ distributions are fixed. 

It can be viewed as diffusion model, but there are $`\alpha_t`$ and $`\sigma_t`$, here for simplicity $`\alpha_t = 1 - t`$ and $`\sigma_t = t`$. Using ideas from [Neural Flow Diffusion Models](https://arxiv.org/pdf/2404.12940) set $`x_t = (1 - t) x_0 + t x_1 + t (1 - t) F_\phi(x_0, x_1, t)`$, may be with other coefficients. Also there is opportunity to use other process to set current process: $`x_t = (1 - t) x_0 + t x_1 + t (1 - t) F_\phi(x_0, x_1, t, \epsilon_t)`$. Now focus on simplest case.

Our task - fit model $`E_\psi(x_t, t)`$ to reconstruct $`p(x)`$. By induction if we can reconstruct $`p(x_t)`$ with our procedure (with $`E_\psi(x_t, t)`$) than for training use sampling $`p(x_t)`$ and just learn to reconstruct $`p(x_{t-dt})`$ (by already knowing $`p(x_t)`$).

Start optimization with coefficients $`\lambda(t)`$. For every step use [Wasserstein GAN](https://arxiv.org/pdf/1701.07875). Our objective is $`\min\limits_{\psi} \max\limits_{\theta, \|D_\theta\|_L \le 1} \sum_{t} \lambda(t) \left[\mathbb{E}_{x_{t-dt} \sim p(x_{t-dt})} D_\theta (x_{t-dt}, t) - \mathbb{E}_{q_\psi(x_{t-dt} | x_t), p(x_t)} D_\theta (x_{t-dt}, t) \right]`$

For $`q_\psi(x_{t-dt} | x_t)`$ can be used different distributions. Also we can use $`F_\phi`$ or even $`\dfrac{\partial F_\phi}{\partial t}`$ here, but with our own $`\hat{x_0}`$ and $`\hat{x_1}`$. In simplest case use $`q_\psi(x_{t-dt} | x_t) = x_t -  dt E_\psi(x_t, t)`$. It allow us to go into continuos time and also use ODE sampler. If we use $`F_\phi`$, then $`q_\psi(x_{t-dt} | x_t) = x_t - dt \dfrac{\partial F_\phi}{\partial \hat{t}}(\hat{x_0}, \hat{x_1}, \hat{t}=t)`$, but it needs to do some proof about usage of WGAN in case when we do not have fixed distribution, may be need to add some limits on $`F_\phi`$ for that. 

In simplest case without using another process $`\epsilon_t`$ (there will also can do that, but not in so simple form) we can just simply rewrite our objective: $`\min\limits_{\psi} \max\limits_{\theta, \|D_\theta\|_L \le 1} T \mathbb{E}_{t, x_0, x_1} \lambda(t) \left[D_\theta (x_{t-dt}, t) - D_\theta (x_t - dt E_\psi(x_t, t), t) \right] = \min\limits_{\psi} \max\limits_{\theta, \|D_\theta\|_L \le 1} \mathbb{E}_{t, x_0, x_1} \lambda(t) \frac{D_\theta (x_{t-dt}, t) - D_\theta (x_t - dt E_\psi(x_t, t), t)}{dt}`$

### Continuos case

Thanks to WGAN, difference between discriminators is equal to $`0`$, so use $`\lim\limits_{dt \to 0}`$: $`\min\limits_{\psi} \max\limits_{\theta, \|D_\theta\|_L \le 1} \mathbb{E}_{t \sim [0, 1], x_0, x_1} \lambda(t) \left[ - \partial D_\theta (x_t, t) \partial x_t + \partial D_\theta (x_t, t) E_\psi(x_t, t)\right] = \min\limits_{\psi} \max\limits_{\theta, \|D_\theta\|_L \le 1} \mathbb{E}_{t \sim [0, 1], x_0, x_1} \lambda(t) \left[ \partial D_\theta (x_t, t) ( E_\psi(x_t, t) - (x_1 - x_0)) \right]`$


### Lipschitz constant

For defining $`\partial D_\theta (x_t, t), \|D_\theta\|_L \le 1`$ simply use other network. This network enough to have bounded norm. For that we use $`\text{out}_\text{new} = \text{out} \frac{ \text{tanh}(\|\text{out}\|_2)}{\|\text{out}\|_2}`$, where $`\text{out}= \text{score}_\theta (x_t, t)`$

Final objective: $`\min\limits_{\psi} \max\limits_{\theta} \mathbb{E}_{t \sim [0, 1], x_0, x_1} \lambda(t) \langle \text{score}_\theta (x_t, t) , E_\psi(x_t, t) - (x_1 - x_0)\rangle`$. In case with $`F_\phi`$ use jvp here for deriving, and using $`\epsilon_t`$ is somehow harder to derive. Good step here is to use not $`E_\psi(x_t, t)`$, but $`\hat{x_0}`$ or/and $`\hat{x_1}`$. 

### ODE

We can use $`E_\psi(x_t, t)`$ for ODE solving and sampling $`x_0`$. Simply use [torchdiffeq](https://github.com/rtqichen/torchdiffeq).


### Discussion

Intuitively, for every $`x_t`$ model should approximate weighted mean of $`x_1 - x_0`$. But it is hard problem, because need to sample a lot for accurate approximation. That is why using information about $`x_0`$ or $`x_1`$ can be useful (may be with ideas from Bridge models). It will something like using ELBO in VAE and diffusion models.

Also it can be useful to use $`F_\phi`$ and $`\epsilon t`$, because it simplifies work for $`E_\psi`$ and make our process wider. 

Another idea is to go into the latent space, just using prior loss and reconstruction loss, may we with GAN or WGAN ideas. May be it will fix some dimensionality problems.

If we have known distributions and want to create bridge, may be we can use similar formulas, just say that optimum is when we know weighted mean of $x_1 - x_0$ with condition on $x_t$. And calculate formulas for simple distributions. 

## Results:

In current repository you can see $`2`$ implementations - for simple 1D case and for CIFAR10 with using $F_\phi(x_t, t)$ (without additional limits and proving of correctness). In both cases model works not so good, it is unstable and generates bad samples.

In CIFAR10 we can see many image parts similar to real images, but the quality is bad. May be we should tune parameters (because of gans problems), may be implements some ideas from discussion, but may be I am wrong somethere. But generations that are sometimes good and formulas simplicity and beauty gives me hope :)

## Run:

Create logs folder before running 

1. CIFAR10 with $F_\phi$: `nohup python3 train_images.py --device "cuda:0" >> logs/contgan_images.txt &`
1. CIFAR10 linear: `nohup python3 train_images_simple.py --device "cuda:0" >> logs/contgan_images_simple.txt &`
3. 1D: `nohup python3 train1d.py --device "cuda:0" >> logs/contgan1d.txt &`

## Generation examples:

![CIFAR generation example](images/cifar_1.png)


![CIFAR generation example](images/cifar_2.png)


![CIFAR generation example](images/cifar_3.png)


![MNIST generation example](images/mnist_1.png)


![MNIST generation example](images/mnist_2.png)


![MNIST generation example](images/mnist_3.png)


![1D generation example](images/1d.png)