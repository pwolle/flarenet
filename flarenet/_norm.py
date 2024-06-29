import flarejax as fj
import jax
from jaxtyping import Array, Float, jaxtyped

from ._activation import Standardize
from ._linear import Bias, Scale


import jax.numpy as jnp
import jax.lax as lax


class LayerNorm(fj.Module):
    __module_name = "flarenet.LayerNorm"

    b: Bias
    s: Scale

    epsilon: float = fj.field(static=True)
    axis: int = fj.field(static=True)

    @fj.typecheck
    @classmethod
    def init(cls, dim: int, epsilon: float = 1e-4, axis: int = -1):
        return cls(
            b=Bias.init(dim),
            s=Scale.init(dim),
            epsilon=epsilon,
            axis=axis,
        )

    @jaxtyped(typechecker=fj.typecheck)
    def __call__(
        self,
        x: Float[Array, "*b {self.dim}"],
    ) -> Float[Array, "*b {self.dim}"]:
        m = jnp.mean(x, axis=self.axis, keepdims=True)
        x = x - m

        v = jnp.mean(x**2, axis=self.axis, keepdims=True)
        x = x * lax.rsqrt(v + self.epsilon)
        return self.s(self.b(x))

    @fj.typecheck
    @property
    def dim(self) -> int:
        return self.b.dim


class RMSNorm(LayerNorm):
    __module_name = "flarenet.RMSNorm"

    @jaxtyped(typechecker=fj.typecheck)
    def __call__(
        self,
        x: Float[Array, "*b {self.dim}"],
    ) -> Float[Array, "*b {self.dim}"]:
        v = jnp.mean(x**2, axis=self.axis, keepdims=True)
        x = x * lax.rsqrt(v + self.epsilon)
        return self.s(self.b(x))
