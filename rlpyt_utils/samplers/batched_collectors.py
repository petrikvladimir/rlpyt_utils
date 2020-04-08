import numpy as np

from rlpyt.samplers.collectors import (DecorrelatingStartCollector,
                                       BaseEvalCollector)
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import (torchify_buffer, numpify_buffer, buffer_from_example,
                                buffer_method)
from rlpyt.utils.logging import logger


class BatchedCpuResetCollector(DecorrelatingStartCollector):
    """Collector which executes ``agent.step()`` in the sampling loop (i.e.
    use in CPU or serial samplers.)

    It immediately resets any environment which finishes an episode.  This is
    typically indicated by the environment returning ``done=True``.  But this
    collector defers to the ``done`` signal only after looking for
    ``env_info["traj_done"]``, so that RL episodes can end without a call to
    ``env_reset()`` (e.g. used for episodic lives in the Atari env).  The
    agent gets reset based solely on ``done``.
    """

    mid_batch_reset = True

    def collect_batch(self, agent_inputs, traj_infos, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        self.agent.sample_mode(itr)
        for t in range(self.batch_T):
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            o_all, r_all, d_all, env_info_all = self.envs.step(action)
            # for b, env in enumerate(self.envs):
            for b in range(self.envs.batch_B):
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = o_all[b], r_all[b], d_all[b], env_info_all[b]
                traj_infos[b].step(observation[b], action[b], r, d, agent_info[b],
                                   env_info)
                if getattr(env_info, "traj_done", d):
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    if b == 0:  # reset only once for all
                        o_all = self.envs.reset()
                        o = o_all[b]
                if d:
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
                env_buf.done[t, b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return AgentInputs(observation, action, reward), traj_infos, completed_infos

    def start_envs(self, max_decorrelation_steps=0):
        """Calls ``reset()`` on every environment instance, then steps each
        one through a random number of random actions, and returns the
        resulting agent_inputs buffer (`observation`, `prev_action`,
        `prev_reward`)."""
        traj_infos = [self.TrajInfoCls() for _ in range(self.envs.batch_B)]
        observations = self.envs.reset()  # B x ObsSize
        observation = buffer_from_example(observations[0], self.envs.batch_B)
        for b, obs in enumerate(observations):
            observation[b] = obs  # numpy array or namedarraytuple
        prev_action = np.stack([self.envs.action_space.null_value() for _ in range(self.envs.batch_B)])
        prev_reward = np.zeros(self.envs.batch_B, dtype="float32")
        if self.rank == 0:
            logger.log("Sampler decorrelating envs, max steps: "
                       f"{max_decorrelation_steps}")
        if max_decorrelation_steps != 0:
            return NotImplementedError  # todo
            # for b, env in enumerate(self.envs):
            #     n_steps = 1 + int(np.random.rand() * max_decorrelation_steps)
            #     for _ in range(n_steps):
            #         a = env.action_space.sample()
            #         o, r, d, info = env.step(a)
            #         traj_infos[b].step(o, a, r, d, None, info)
            #         if getattr(info, "traj_done", d):
            #             o = env.reset()
            #             traj_infos[b] = self.TrajInfoCls()
            #         if d:
            #             a = env.action_space.null_value()
            #             r = 0
            #     observation[b] = o
            #     prev_action[b] = a
            #     prev_reward[b] = r
        # For action-server samplers.
        if hasattr(self, "step_buffer_np") and self.step_buffer_np is not None:
            self.step_buffer_np.observation[:] = observation
            self.step_buffer_np.action[:] = prev_action
            self.step_buffer_np.reward[:] = prev_reward
        return AgentInputs(observation, prev_action, prev_reward), traj_infos
