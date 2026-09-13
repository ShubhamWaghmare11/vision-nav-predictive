import numpy as np

class IDMExpert:
    """
    When env is created with agent_policy=IDMPolicy, the expert
    drives automatically. This class queries what action IDM took
    at each step for use as the behaviour cloning label.
    """

    def __init__(self, env):
        self.env = env

    def get_action(self) -> np.ndarray:
        """
        Returns the action IDM chose at the current step.
        Access via the policy object attached to the agent.
        """
        policy = self.env.engine.get_policy(self.env.agent.name)
        action = policy.act(self.env.agent.name)
        return np.array(action, dtype=np.float32)

    def reset(self):
        pass  # policy resets with the env