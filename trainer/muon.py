"""
Muon optimizer from Keller Jordan et al.
Official implementation: https://github.com/KellerJordan/modded-nanogpt
Paper: https://kellerjordan.github.io/posts/muon/

Muon - MomentUm Orthogonalized by Newton-schulz
Applies SGD-momentum and then orthogonalizes 2D updates via Newton-Schulz iteration.
"""
import torch
from torch import Tensor
import torch.distributed as dist


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    We opt to use a quintic iteration whose coefficients are selected to maximize
    the slope at zero. For the purpose of minimizing steps, it turns out to be empirically
    effective to keep increasing the slope at zero even beyond the point where the iteration
    no longer converges all the way to one everywhere on the interval. This iteration therefore
    does not produce UV^T but rather something like US'V^T where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5),
    which turns out not to hurt model performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
      or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 0.02)
        momentum: momentum factor (default: 0.95)
        nesterov: whether to use Nesterov momentum (default: True)
        ns_steps: number of Newton-Schulz iteration steps (default: 5)
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        
        # Group params by size for efficiency
        param_groups = []
        for size in sorted({p.numel() for p in params}):
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            
            for p in params:
                g = p.grad
                if g is None:
                    continue
                
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                
                # Apply aspect-ratio scaled step
                p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
        
        return loss


class DistMuon(torch.optim.Optimizer):
    """
    Distributed version of Muon optimizer.
    
    Performs its own distributed synchronization:
    - reduce_scatter(AVG) for gradient averaging
    - all_gather to replicate updated weights

    Notes:
    * Designed for 2D parameters (e.g., linear/conv kernels reshaped to 2D).
      Do not use for 0D/1D params like embeddings or scalars.
    * Momentum buffers are maintained only on the 'owner' rank for each parameter.
      If you checkpoint optimizer state on a single rank, consolidate states beforehand.

    Args:
        params: iterable of Tensors
        lr: learning rate (default: 0.02)
        momentum: momentum coefficient in [0,1) (default: 0.95)
        nesterov: if True, Nesterov-style update (default: True)
        ns_steps: number of Newton-Schulz iterations for orthogonalization (default: 5)
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "DistMuon expects 2D parameters only"
        
        rank = dist.get_rank()
        
        # Group all parameters by their shape for efficiency
        shapes = sorted({p.shape for p in params})
        param_groups = []
        
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            
            if rank == 0:
                print(f"DistMuon: Grouping {len(group_params)} params of shape {shape}, device {device}, dtype {dtype}")
            
            param_groups.append(dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0])))
        
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with distributed synchronization."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Ensure all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), \
            "All params must have grads"
        
        # Kick off all the reduce scatter operations to average gradients
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            
            for base_i in range(0, len(params), world_size):
                owner_idx = base_i + rank
                
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)
        
        # Compute updates and gather
        future_idx = 0
        all_gather_futures = []
        
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            
            for base_i in range(0, len(params), world_size):
                owner_idx = base_i + rank
                
                # Wait for reduce scatter
                all_reduce_futures[future_idx].wait()
                future_idx += 1
                
                # Owner computes the Muon update
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad
                    
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    
                    scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                    p.add_(g, alpha=-group["lr"] * scale)
                
                # Replicate updated parameters to all ranks
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))])
                
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)
        
        # Wait for all gather to finish
        torch.futures.collect_all(all_gather_futures).wait()
        
        return loss

