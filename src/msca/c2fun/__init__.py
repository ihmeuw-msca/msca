from msca.c2fun.jax import c2fun_dict as c2fun_dict_jax
from msca.c2fun.main import C2Fun
from msca.c2fun.numpy import c2fun_dict as c2fun_dict_numpy

c2fun_dict: dict[str, dict[str, C2Fun]] = dict(
    numpy=c2fun_dict_numpy,
    jax=c2fun_dict_jax,
)
