def count_parameters(model, top=None):
    total_params = 0
    params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        total_params += parameter.numel()
        params.append((parameter.numel(), name))
    if top is None:
        return total_params

    params = sorted(params, key=lambda x: x[0], reverse=True)
    top = params[:top]
    return total_params, top