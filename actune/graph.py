"""Shape- and precision-keyed CUDA Graph replay, from the optimized runtime."""
from contextlib import contextmanager
from unittest.mock import patch
import torch

class GraphReplay:
    def __init__(self, function, layers, *, shared_a8=False, max_graphs=12):
        self.function = function
        self.layers = layers
        self.shared_a8 = shared_a8
        self.max_graphs = max_graphs
        self.graphs = {}
        self.enabled = False
        self.replays = 0
        self.fallbacks = 0
        self.constants = {}

    @contextmanager
    def scope(self):
        from actune.kernels.w4a8_triton import shared_a8_cache_scope
        tensor = torch.tensor

        def constant(data, *args, **kwargs):
            # OpenPI builds small fixed attention-mask/time constants on the
            # host inside its forward. Pre-create their exact GPU values during
            # warmup; clone on every execution because time is updated in-place.
            device = kwargs.get('device')
            if device is None or torch.device(device).type != 'cuda':
                return tensor(data, *args, **kwargs)
            def literal(x):
                return isinstance(x, (bool, int, float)) or isinstance(x, (list, tuple)) and all(literal(y) for y in x)
            if not literal(data) or args or kwargs.get('requires_grad', False):
                return tensor(data, *args, **kwargs)
            key = (repr(data), repr(sorted(kwargs.items())))
            if key not in self.constants:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('Unwarmed CUDA constant in graph capture')
                self.constants[key] = tensor(data, **kwargs)
            return self.constants[key].clone()

        with shared_a8_cache_scope(enabled=self.shared_a8), patch.object(torch, 'tensor', constant):
            yield

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            return self.function(*args, **kwargs)
        try:
            from jax import tree_util
        except ImportError:
            from torch.utils import _pytree as tree_util
        from actune.precision import active_precision_profile
        leaves, tree = tree_util.tree_flatten((args, kwargs))
        signature = tuple((tuple(x.shape), tuple(x.stride()), str(x.dtype), str(x.device))
                          if isinstance(x, torch.Tensor) else (type(x).__name__, repr(x))
                          for x in leaves)
        profile = active_precision_profile()
        assignments = tuple(m.assignments[profile or m.default_profile] for m in self.layers.values())
        key = (str(tree), signature, assignments)
        if key not in self.graphs:
            if len(self.graphs) >= self.max_graphs:
                self.fallbacks += 1
                return self.function(*args, **kwargs)
            saved = {n: (m.calls, m.input_elements) for n, m in self.layers.items()}
            static = [x.clone() if isinstance(x, torch.Tensor) else x for x in leaves]
            # JAX takes (tree, leaves); PyTorch takes (leaves, tree).
            if tree_util.__name__.startswith('jax'):
                static_args, static_kwargs = tree_util.tree_unflatten(tree, static)
            else:
                static_args, static_kwargs = tree_util.tree_unflatten(static, tree)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(2):
                    with self.scope():
                        self.function(*static_args, **static_kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            for m in self.layers.values():
                m.calls = m.input_elements = 0
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode(), self.scope():
                output = self.function(*static_args, **static_kwargs)
            counts = {n: (m.calls, m.input_elements) for n, m in self.layers.items()}
            for n, m in self.layers.items():
                m.calls, m.input_elements = saved[n]
            self.graphs[key] = (graph, static, output, counts)
        graph, static, output, counts = self.graphs[key]
        for src, dst in zip(leaves, static, strict=True):
            if isinstance(src, torch.Tensor):
                dst.copy_(src)
        graph.replay()
        for n, (calls, elements) in counts.items():
            self.layers[n].calls += calls
            self.layers[n].input_elements += elements
        self.replays += 1
        return output
