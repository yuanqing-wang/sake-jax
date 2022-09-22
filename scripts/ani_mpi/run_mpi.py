import jax
import jax.numpy as jnp
import numpy as onp
import flax
from functools import partial

BATCH_SIZE = 128

from mpi4py import MPI
import mpi4jax
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N_DEVICES = size = comm.Get_size()

class Collater(object):
    def __init__(self, ds_tr, batch_size=128, n_devices=N_DEVICES):
        self.ds_tr = ds_tr # self._move_to_device(ds_tr)
        self.batch_size = batch_size
        self.pointers = []
        self.n_devices = n_devices

    @staticmethod
    def _move_to_device(ds_tr):
        for length in ds_tr:
            ds_tr[length]['i'] = jnp.array(ds_tr[length]['i'])
            ds_tr[length]['x'] = jnp.array(ds_tr[length]['x'])
            ds_tr[length]['y'] = jnp.array(ds_tr[length]['y'])
        return ds_tr

    def get_statistics(self):
        ys = jnp.concatenate([self.ds_tr[length]['y'].reshape(-1) for length in self.ds_tr])
        return ys.mean(), ys.std()

    
    def __len__(self):
        return len(self.get_pointers())

    def get_pointers(self):
        pointers = []
        for length in self.ds_tr:
            n_data = self.ds_tr[length]['x'].shape[0]
            n_batches = int(n_data / (self.batch_size * self.n_devices))
            idxs = onp.random.permutation(n_data)[:n_batches*self.batch_size*self.n_devices]
            idxs = idxs.reshape(n_batches, self.batch_size*self.n_devices)
            for idx in idxs:
                pointers.append((length, idx))
        import random
        random.shuffle(pointers)
        self.pointers = pointers
        return pointers

    def get_from_pointer(self, pointer):
        length, idxs = pointer
        i = jax.nn.one_hot(jnp.array(self.ds_tr[length]['i'][idxs]), 4)
        x = jnp.array(self.ds_tr[length]['x'][idxs])
        y = jnp.expand_dims(jnp.array(self.ds_tr[length]['y'][idxs]), -1)

        i = i.reshape(self.n_devices, self.batch_size, *i.shape[1:])
        x = x.reshape(self.n_devices, self.batch_size, *x.shape[1:])
        y = y.reshape(self.n_devices, self.batch_size, *y.shape[1:])

        return i, x, y

    def __iter__(self):
        self.get_pointers()
        return self

    def __next__(self):
        if len(self.pointers) == 0:
            raise StopIteration
        else:
            pointer = self.pointers.pop()
            return self.get_from_pointer(pointer)

def run():
    # data = ANIDataset()
    ds_tr = onp.load("ds_tr.npy", allow_pickle=True)[()]
    collater = Collater(ds_tr)

    mean, std = collater.get_statistics()
    import sake
    from functools import partial
    coloring = partial(sake.utils.coloring, mean=mean, std=std)

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=6,
    )

    @jax.jit
    def average_params(params):
        fn = lambda x: mpi4jax.allreduce(x, op=MPI.SUM, comm=comm)[0] / size
        params = jax.tree_util.tree_map(fn, params)
        return params

    def get_y_pred(params, i, x):
        y_pred, _, __ = model.apply(params, i, x)
        y_pred = y_pred.sum(axis=-2)
        y_pred = coloring(y_pred)
        return y_pred

    def get_loss(params, i, x, y):
        y_pred = get_y_pred(params,i, x)
        loss = jnp.abs(y_pred - y).mean()
        return loss

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint, restore_checkpoint
    params = restore_checkpoint("_checkpoint", target=None)["params"]

    import optax
    n_batches = len(collater)

    optimizer = optax.chain(
        optax.additive_weight_decay(1e-8),
        optax.clip(1.0),
        optax.zero_nans(),
        optax.adam(1e-5),
    )
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    @jax.jit
    def step(state, i, x, y):
        params = state.params
        grads = jax.grad(get_loss)(params, i, x, y)
        grads = average_params(params)
        state = state.apply_gradients(grads=grads)
        return state

    for idx_batch in range(1000):
        for i, x, y in collater:
            i, x, y = i[rank], x[rank], y[rank]
            state = step(state, i, x, y)
        save_checkpoint("_checkpoint", target=state, step=idx_batch)

if __name__ == "__main__":
    run()
