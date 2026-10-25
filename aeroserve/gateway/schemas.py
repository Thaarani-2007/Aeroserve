import enum
import base64
from typing import Any
from pydantic import BaseModel, Field, field_validator,model_validator

class PriorityTier(str, enum.Enum):
    PREMIUM = "premium"
    STANDARD = "standard"
    BACKGROUND = "background"

class UserTier(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class InferencePayload(BaseModel):
    text: str | None = Field(default=None, description="Raw text string for NLP models")
    image_b64: str | None = Field(default=None, description="Base64 encoded image string for Computer Vision")
    features: list[float] | None = Field(default=None, description="1D array of numerical features for tabular models")
    tensors: list[Any] | None = Field(default=None, description="Multi-dimensional array structures for raw tensor input")

    @field_validator('image_b64')
    @classmethod
    def validate_image_b64(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                # Validate it is actually a decodable base64 string
                base64.b64decode(v, validate=True)
            except Exception:
                raise ValueError("image_b64 must be a valid base64 encoded string")
        return v

    @model_validator(mode='after')
    def check_at_least_one_input(self):
        if not any([self.text, self.image_b64 is not None, self.features, self.tensors]):
            raise ValueError("Payload must contain at least one input type (text, image_b64, features, or tensors).")
        return self

class PredictRequest(BaseModel):
    model_name: str = Field(..., description="Target model identifier registered in AeroServe")
    payload: InferencePayload = Field(..., description="Strictly typed input data payload")
    priority: PriorityTier = Field(
        default=PriorityTier.STANDARD, 
        description="Inference priority tier: premium, standard, or background"
    )
    user_tier: UserTier = Field(
        default=UserTier.GUEST,
        description="Access tier of the requesting user"
    )

class PredictResponse(BaseModel):
    request_id: str = Field(..., description="Correlation ID for end-to-end tracing")
    status: str = Field(default="enqueued", description="Lifecycle status of the request")
    queue: str = Field(..., description="Target Redis queue where the job is stored")
    timestamp: float = Field(..., description="UNIX timestamp when the request was ingested")
    inference_result: dict[str, Any] | None = Field(default=None, description="The final prediction output from the worker")

class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    models_loaded: list[str] = Field(default_factory=list, description="List of actively registered models")