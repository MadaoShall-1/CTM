"""Multi-Pass Deep Q-Network baseline."""

from torch import Tensor, nn

from .networks import multipass_q_values
from .pdqn import PDQNAgent


class MPDQNAgent(PDQNAgent):
    """P-DQN whose Q values use one masked pass per discrete action."""

    algorithm_name = "mpdqn"

    def _q_values(self, network: nn.Module, states: Tensor, parameters: Tensor) -> Tensor:
        return multipass_q_values(network, states, parameters, self.spec)
