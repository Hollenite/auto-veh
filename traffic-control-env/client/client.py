"""
client.py — TrafficEnv client for the Traffic Control Environment.

Provides an ``EnvClient`` subclass that communicates with the FastAPI
server over WebSocket, handling serialization/deserialization of
``TrafficAction`` and ``TrafficObservation`` models.

Supports both async and sync usage patterns:

    Async::

        async with TrafficEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            result = await env.step(TrafficAction(action=SignalAction.KEEP_CURRENT))

    Sync::

        env = TrafficEnv(base_url="http://localhost:8000").sync()
        with env:
            result = env.reset()
            result = env.step(TrafficAction(action=SignalAction.KEEP_CURRENT))

Classes:
    TrafficEnv: EnvClient subclass with typed action/observation/state handling.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

from models import TrafficAction, TrafficObservation, TrafficState


class TrafficEnv(EnvClient[TrafficAction, TrafficObservation, TrafficState]):
    """Client for the Traffic Control environment.

    Connects to a running TrafficEnvironment server over WebSocket
    and provides typed ``reset()``, ``step()``, and ``state()`` methods.

    Args:
        base_url: URL of the environment server
                  (e.g. ``"http://localhost:8000"``). Automatically
                  converted to ``ws://`` for the WebSocket connection.
        task_id: Difficulty level — ``"easy"``, ``"medium"``, or ``"hard"``.
                 Stored for reference; task selection is configured server-side.
        **kwargs: Additional arguments forwarded to ``EnvClient.__init__``.

    Example:
        >>> env = TrafficEnv(base_url="http://localhost:8000").sync()
        >>> with env:
        ...     result = env.reset()
        ...     while not result.done:
        ...         action = TrafficAction(action=SignalAction.KEEP_CURRENT)
        ...         result = env.step(action)
        ...     print(f"Episode done. Reward: {result.reward}")
    """

    def __init__(
        self,
        base_url: str,
        task_id: str = "easy",
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url=base_url, **kwargs)
        self.task_id: str = task_id

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _step_payload(self, action: TrafficAction) -> Dict[str, Any]:
        """Serialize a ``TrafficAction`` into the JSON payload for the server.

        Args:
            action: The agent's traffic signal control action.

        Returns:
            Dictionary with ``action`` (enum value string) and
            ``emergency_direction`` (string or None).
        """
        return {
            "action": action.action.value,
            "emergency_direction": action.emergency_direction,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TrafficObservation]:
        """Deserialize a server response into a typed ``StepResult``.

        The server sends back the observation fields either directly
        in the payload or nested under an ``"observation"`` key.

        Args:
            payload: Raw JSON dict from the server.

        Returns:
            ``StepResult`` containing the ``TrafficObservation``, reward, and done flag.
        """
        # The observation may be nested or flat depending on server response format
        obs_data = dict(payload.get("observation", payload))

        # OpenEnv may strip reward, done, and success from the observation
        # and promote them to the top level. We merge them back so our model is fully formed.
        if "reward" in payload and "reward" not in obs_data:
            obs_data["reward"] = payload["reward"]
        if "done" in payload and "done" not in obs_data:
            obs_data["done"] = payload["done"]
        if "success" in payload and "success" not in obs_data:
            obs_data["success"] = payload["success"]

        observation = TrafficObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TrafficState:
        """Deserialize a server state response into a typed ``TrafficState``.

        Args:
            payload: Raw JSON dict from the server's state endpoint.

        Returns:
            Populated ``TrafficState`` instance.
        """
        state_data = payload.get("state", payload)
        return TrafficState(**state_data)

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_hub(cls, repo_id: str, task_id: str = "easy", **kwargs: Any) -> "TrafficEnv":
        """Create a client pointing at a Hugging Face Space.

        This is a convenience method that constructs the HF Spaces URL
        from a repository ID. The Space must be running the
        TrafficEnvironment server.

        Args:
            repo_id: Hugging Face repository ID (e.g. ``"username/traffic-control-env"``).
            task_id: Difficulty level for the task.
            **kwargs: Additional arguments forwarded to the constructor.

        Returns:
            A ``TrafficEnv`` instance configured for the HF Space.

        Example:
            >>> env = TrafficEnv.from_hub("myuser/traffic-control-env", task_id="medium")
        """
        url = f"https://{repo_id.replace('/', '-')}.hf.space"
        return cls(base_url=url, task_id=task_id, **kwargs)

    def __repr__(self) -> str:
        return f"TrafficEnv(url='{self._ws_url}', task_id='{self.task_id}')"
