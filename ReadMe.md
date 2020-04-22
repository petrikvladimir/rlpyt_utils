# Introduction
Some utilities for managing experiments in rlpyt framework, e.g. to automatically increase run_id to not overwrite previous run under the same name.


# Utils

## RL default args and logger
```python
from rlpyt_utils import args
from rlpyt_utils.runners.minibatch_rl import MinibatchRlWithLog
from rlpyt_utils.agents_nn import AgentPgContinuous
from rlpyt.utils.logging.logger import record_tabular
import numpy as np

parser = args.get_default_rl_parser()
args.add_default_ppo_args(parser, ratio_clip=0.2)
options = parser.parse_args()

def log_diagnostics(itr, algo, agent, sampler):
    """ Log anything into tensorboard here. """
    std = np.random(3)
    for i in range(std.shape[0]):
        record_tabular('agent/std{}'.format(i), std[i])


sampler = ... # build your sampler as usual in rlpyt
agent = AgentPgContinuous(options.greedy_eval, initial_model_state_dict=args.load_initial_model_state(options),)
runner = MinibatchRlWithLog(algo=args.get_ppo_from_options(options),
                            agent=agent, sampler=sampler,
                            n_steps=int(100000 * options.horizon * options.num_parallel),
                            log_interval_steps=int(1 * options.horizon * options.num_parallel),
                            affinity=args.get_affinity(options), log_diagnostics_fun=log_diagnostics, seed=0)

if not args.is_evaluation(options):  # is training
    with args.get_default_context(options):
        runner.train()
else: # is either eval or greedy_eval
    runner.startup()
    print('Getting samples')
    samples_pyt, traj_infos = sampler.obtain_samples(0)
    print(np.sum(sampler.samples_np.env.reward, axis=0))
    runner.shutdown()
``` 
to run:
```
python script.py --help
python script.py
python script.py --eval
```

## Exponential reward

Computes unit reward for vector. Parameter _b_ is used to specify lengthscale of the reward according to the tables:

exp | d=0.01 | d=0.05 | d=0.1 | d=0.5 | d=1.0 | d=5.0 | d=10.0
--- | --- | --- | --- | --- | --- | --- | ---
**b=1e-02** | 1.0e+00 | 1.0e+00 | 1.0e+00 | 1.0e+00 | 1.0e+00 | 8.8e-01 | 6.1e-01
**b=1e-01** | 1.0e+00 | 1.0e+00 | 1.0e+00 | 9.9e-01 | 9.5e-01 | 2.9e-01 | 6.7e-03
**b=1e+01** | 1.0e+00 | 9.9e-01 | 9.5e-01 | 2.9e-01 | 6.7e-03 | 5.2e-55 | 7.1e-218
**b=1e+02** | 1.0e+00 | 8.8e-01 | 6.1e-01 | 3.7e-06 | 1.9e-22 | 0.0e+00 | 0.0e+00
**b=1e+03** | 9.5e-01 | 2.9e-01 | 6.7e-03 | 5.2e-55 | 7.1e-218 | 0.0e+00 | 0.0e+00
**b=1e+04** | 6.1e-01 | 3.7e-06 | 1.9e-22 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00

exp | d=0.01 | d=0.05 | d=0.1 | d=0.5 | d=1.0 | d=5.0 | d=10.0
--- | --- | --- | --- | --- | --- | --- | ---
**b=1e-02** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.88 | 0.61
**b=1e-01** | 1.0 | 1.0 | 1.0 | 0.99 | 0.95 | 0.29 | 0.01
**b=1e+01** | 1.0 | 0.99 | 0.95 | 0.29 | 0.01 | 0.0 | 0.0
**b=1e+02** | 1.0 | 0.88 | 0.61 | 0.0 | 0.0 | 0.0 | 0.0
**b=1e+03** | 0.95 | 0.29 | 0.01 | 0.0 | 0.0 | 0.0 | 0.0
**b=1e+04** | 0.61 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0