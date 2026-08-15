"""Agent wrapper around a trained sb3-contrib MaskablePPO checkpoint."""
import numpy as np

from agents.base_agent import Agent
from env.observation import build_observation, legal_action_mask
from poker.actions import Action


class RLAgent(Agent):
    """Loads a trained MaskablePPO model and plays legal-action-masked moves."""

    def __init__(self, model_path: str, deterministic: bool = True):
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(model_path)
        self.deterministic = deterministic
        self.name = f"rl({model_path})"

    def act(self, engine, seat: int) -> Action:
        action, _ = self._predict(engine, seat)
        return action

    def act_with_probabilities(self, engine, seat: int):
        """Return (chosen Action, {Action: probability}) for explainability."""
        obs = build_observation(engine, seat)
        mask = legal_action_mask(engine, seat)
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs[np.newaxis, :])
        dist = self.model.policy.get_distribution(obs_tensor, action_masks=mask[np.newaxis, :])
        probs = dist.distribution.probs.detach().cpu().numpy()[0]
        action, _ = self._predict(engine, seat, obs=obs, mask=mask)
        prob_by_action = {Action(i): float(probs[i]) for i in range(len(probs)) if mask[i]}
        return action, prob_by_action

    def _predict(self, engine, seat, obs=None, mask=None):
        obs = build_observation(engine, seat) if obs is None else obs
        mask = legal_action_mask(engine, seat) if mask is None else mask
        raw_action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return Action(int(raw_action)), mask
