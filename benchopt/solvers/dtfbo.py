from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.get_memory import get_memory
    from benchmark_utils.minibatch_sampler import init_sampler
    from benchmark_utils.learning_rate_scheduler import update_tfbo_lr
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.learning_rate_scheduler import init_tfbo_lr_scheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec
    from benchmark_utils.oracles import MultiLogRegOracle, DataCleaningOracle
    from benchmark_utils.learning_rate_scheduler import TFBOLearningRateScheduler

    import jax
    import jax.numpy as jnp
    import numpy as np
    from functools import partial


class Solver(BaseSolver):
    """Double-loop Tuning-free Bilevel Optimizer (D-TFBO)."""
    name = 'DTFBO'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'batch_size': [64],
        'eval_freq': [128],
        'random_state': [1],
        'framework': ["none"],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def skip(self, f_train, f_val, **kwargs):
        if self.framework == 'numba':
            if self.batch_size == 'full':
                return True, "Numba is not useful for full bach resolution."
            elif isinstance(f_train(),
                            (MultiLogRegOracle, DataCleaningOracle)):
                return True, "Numba implementation not available for " \
                      "this oracle."
            elif isinstance(f_val(), (MultiLogRegOracle, DataCleaningOracle)):
                return True, "Numba implementation not available for" \
                      "this oracle."
        elif self.framework not in ['jax', 'none', 'numba']:
            return True, f"Framework {self.framework} not supported."

        try:
            f_train(framework=self.framework)
        except NotImplementedError:
            return (
                True,
                f"Framework {self.framework} not compatible with "
                f"oracle {f_train()}"
            )
        return False, None

    def set_objective(self, f_train, f_val, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):

        self.f_inner = f_train(framework=self.framework)
        self.f_outer = f_val(framework=self.framework)
        self.n_inner_samples = n_inner_samples
        self.n_outer_samples = n_outer_samples

        if self.batch_size == "full":
            self.batch_size_inner = n_inner_samples
            self.batch_size_outer = n_outer_samples
        else:
            self.batch_size_inner = self.batch_size
            self.batch_size_outer = self.batch_size

        if self.framework == 'numba':
            # JIT necessary functions and classes
            self.dtfbo = njit(dtfbo)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                TFBOLearningRateScheduler, sched_spec
            )
        elif self.framework == "none":
            self.dtfbo = dtfbo
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = TFBOLearningRateScheduler
        elif self.framework == 'jax':
            self.f_inner = jax.jit(
                partial(self.f_inner, batch_size=self.batch_size_inner)
            )
            self.f_outer = jax.jit(
                partial(self.f_outer, batch_size=self.batch_size_outer)
            )
            inner_sampler, self.state_inner_sampler \
                = init_sampler(n_samples=n_inner_samples,
                               batch_size=self.batch_size_inner)
            outer_sampler, self.state_outer_sampler \
                = init_sampler(n_samples=n_outer_samples,
                               batch_size=self.batch_size_outer)
            self.dtfbo = partial(
                dtfbo_jax,
                inner_sampler=inner_sampler,
                outer_sampler=outer_sampler
            )
        else:
            raise ValueError(f"Framework {self.framework} not supported.")

        self.inner_var = inner_var0
        self.outer_var = outer_var0
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.memory = 0

    def warm_up(self):
        if self.framework in ['numba', 'jax']:
            self.run_once(2)
            self.inner_var = self.inner_var0
            self.outer_var = self.outer_var0

    def run(self, callback):
        eval_freq = self.eval_freq
        memory_start = get_memory()

        # Init variables
        inner_var = self.inner_var.copy()
        outer_var = self.outer_var.copy()
        if self.framework == "jax":
            v = jnp.zeros_like(inner_var)
            # Init lr scheduler
            state_lr = init_tfbo_lr_scheduler()
            carry = dict(
                state_lr=state_lr,
                state_inner_sampler=self.state_inner_sampler,
                state_outer_sampler=self.state_outer_sampler,
            )
        else:
            rng = np.random.RandomState(self.random_state)
            v = np.zeros_like(inner_var)

            # Init lr scheduler
            lr_scheduler = self.LearningRateScheduler()
            inner_sampler = self.MinibatchSampler(self.n_inner_samples,
                                                  self.batch_size_inner)
            outer_sampler = self.MinibatchSampler(self.n_outer_samples,
                                                  self.batch_size_outer)

        # Start algorithm
        while callback():
            if self.framework == 'jax':
                inner_var, outer_var, v, carry = self.dtfbo(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v, max_iter=eval_freq, **carry
                )
            else:
                inner_var, outer_var, v = self.dtfbo(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v,
                    inner_sampler=inner_sampler,
                    outer_sampler=outer_sampler,
                    lr_scheduler=lr_scheduler, max_iter=eval_freq,
                    seed=rng.randint(constants.MAX_SEED)
                )
            memory_end = get_memory()
            self.inner_var = inner_var
            self.outer_var = outer_var
            self.memory = memory_end - memory_start
            self.memory /= 1e6

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var,
                    memory=self.memory)


def dtfbo(inner_oracle, outer_oracle, inner_var, outer_var, v,
         inner_sampler=None, outer_sampler=None, lr_scheduler=None, max_iter=1,
         seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)
    
    eps = 1e-8
    threshold = 5.0

    for i in range(max_iter):

        # Step.1 - update the inner var
        slice_inner, _ = inner_sampler.get_batch()
        _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, slice_inner, inverse='id'
        )
        # while np.linalg.norm(grad_inner_var) ** 2 > threshold:
        for _ in range(10):
            beta = lr_scheduler.update('beta', grad_inner_var)
            inner_var -= (1 / (beta + eps)) * grad_inner_var

            slice_inner, _ = inner_sampler.get_batch()
            _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
                inner_var, outer_var, v, slice_inner, inverse='id'
            )

        # Step.2 - update the intermediate var
        slice_outer, _ = outer_sampler.get_batch()
        grad_in_outer, grad_out_outer = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )

        # while np.linalg.norm(hvp - grad_in_outer) ** 2 > threshold:
        for _ in range(10):
            gamma = lr_scheduler.update('gamma', hvp - grad_in_outer)
            v -= (1 / (gamma + eps)) * (hvp - grad_in_outer)

            slice_outer, _ = outer_sampler.get_batch()
            grad_in_outer, grad_out_outer = outer_oracle.grad(
                inner_var, outer_var, slice_outer
            )

        # Step.3 - update the outer var
        alpha = lr_scheduler.update('alpha', grad_out_outer - cross_v)
        outer_var -= (1 / (alpha + eps)) * (grad_out_outer - cross_v)

    return inner_var, outer_var, v


@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('inner_sampler', 'outer_sampler', 'max_iter'))
def dtfbo_jax(f_inner, f_outer, inner_var, outer_var, v,
             state_inner_sampler=None, state_outer_sampler=None, state_lr=None,
             inner_sampler=None, outer_sampler=None, max_iter=1):

    grad_inner = jax.grad(f_inner, argnums=0)
    grad_outer = jax.grad(f_outer, argnums=(0, 1))
    threshold = 1 / max_iter

    def dtfbo_one_iter(carry, _):
        eps = 1e-8

        # Step.1 - update the inner var
        start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
            carry['state_inner_sampler']
        )
        grad_inner_var, vjp_train = jax.vjp(
            lambda z, x: grad_inner(z, x, start_inner), carry['inner_var'],
            carry['outer_var']
        )
        # while jnp.greater(jnp.linalg.norm(grad_inner_var) ** 2, threshold):
        for _ in range(10):
            carry['state_lr']['beta'] = update_tfbo_lr(carry['state_lr']['beta'], jnp.linalg.norm(grad_inner_var))
            carry['inner_var'] -= (1 / (carry['state_lr']['beta'] + eps)) * grad_inner_var

            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            grad_inner_var, vjp_train = jax.vjp(
                lambda z, x: grad_inner(z, x, start_inner), carry['inner_var'],
                carry['outer_var']
            )

        # Step.2 - update the intermediate var
        hvp, cross_v = vjp_train(carry['v'])
        start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
            carry['state_outer_sampler']
        )
        grad_in_outer, grad_out_outer = grad_outer(
            carry['inner_var'], carry['outer_var'], start_outer
        )

        # while jnp.greater(jnp.linalg.norm(hvp - grad_in_outer) ** 2, threshold):
        for _ in range(10):
            carry['state_lr']['gamma'] = update_tfbo_lr(carry['state_lr']['gamma'], jnp.linalg.norm(hvp - grad_in_outer))
            carry['v'] -= (1 / (carry['state_lr']['gamma'] + eps)) * (hvp - grad_in_outer)

            hvp, cross_v = vjp_train(carry['v'])
            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            grad_in_outer, grad_out_outer = grad_outer(
                carry['inner_var'], carry['outer_var'], start_outer
            )

        # Step.3 - update the outer var
        carry['state_lr']['alpha'] = update_tfbo_lr(carry['state_lr']['alpha'], jnp.linalg.norm(grad_out_outer - cross_v))
        carry['outer_var'] -= (1 / (carry['state_lr']['alpha'] + eps)) * (grad_out_outer - cross_v)

        return carry, _

    init = dict(
        inner_var=inner_var, outer_var=outer_var, v=v, state_lr=state_lr,
        state_inner_sampler=state_inner_sampler,
        state_outer_sampler=state_outer_sampler
    )
    carry, _ = jax.lax.scan(
        dtfbo_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )

    return (
        carry['inner_var'], carry['outer_var'], carry['v'],
        {k: v for k, v in carry.items()
         if k not in ['inner_var', 'outer_var', 'v']}
    )
