from abc import ABC, abstractmethod
from typing import Any


class AeroServeProcessor(ABC):
    """
    Base interface for all AeroServe processors.

    Every processor must implement execute(), which accepts
    a validated inference payload and returns a dictionary
    containing the inference result.
    """

    @abstractmethod
    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute inference and return a JSON-serializable dictionary."""
        raise NotImplementedError


class AeroServeManagedProcessor(AeroServeProcessor):
    """
    Framework-managed inference pipeline.

    AeroServe controls the execution sequence:
        preprocess -> predict -> postprocess

    Developers implement the three individual stages.
    """

    @abstractmethod
    def preprocess(self, payload: dict[str, Any]) -> Any:
        """Transform the validated payload into tensors, arrays, or model inputs."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, model_input: Any) -> Any:
        """Execute the model's forward pass."""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, inference_output: Any) -> dict[str, Any]:
        """Convert model outputs into a JSON-serializable dictionary."""
        raise NotImplementedError

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run the complete managed inference pipeline."""
        features = self.preprocess(payload)
        raw_output = self.predict(features)
        result = self.postprocess(raw_output)

        if not isinstance(result, dict):
            raise TypeError(
                "postprocess() must return a dictionary."
            )

        return result


class AeroServeCustomProcessor(AeroServeProcessor):
    """
    Developer-managed inference pipeline.

    Developers implement execute() themselves when their
    inference workflow requires custom orchestration.
    """

    @abstractmethod
    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the full inference cycle and return a result dictionary."""
        raise NotImplementedError

