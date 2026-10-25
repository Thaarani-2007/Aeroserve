import os
import joblib
import numpy as np
import time
from typing import Any, Dict, List

# Import the base class expected by AeroServe
from models.aeroserve_preprocessor import AeroServeProcessor

class IrisProcessor(AeroServeProcessor):
    # Enable AeroServe's dynamic batching engine
    supports_batching = True

    def __init__(self):
        # Resolve path dynamically so it works inside the Docker worker volume
        current_dir = os.path.dirname(os.path.abspath(__file__))
        artifact_path = os.path.join(current_dir, "iris_pipeline.joblib")
        
        print(f"Loading Iris Model and Scaler from {artifact_path}...")
        artifacts = joblib.load(artifact_path)
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a single inference request."""
        features = payload.get("features")
        
        if not features or len(features) != 4:
            return {"status": "error", "error": "Payload must contain a 'features' array of exactly 4 numeric values."}
        
        # Reshape for sklearn (1 sample, 4 features)
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        return {"status": "success", "prediction": str(prediction)}

    def execute_batch(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handles dynamically batched requests for high throughput."""
        time.sleep(10)
        X_list = []
        valid_indices = []
        
        # Initialize default error responses
        results = [{"status": "error", "error": "Invalid features array"} for _ in payloads]
        
        for idx, p in enumerate(payloads):
            feats = p.get("features")
            if feats and len(feats) == 4:
                X_list.append(feats)
                valid_indices.append(idx)
        
        # Perform vectorized inference on valid payloads
        if X_list:
            X_array = np.array(X_list)
            X_scaled = self.scaler.transform(X_array)
            predictions = self.model.predict(X_scaled)
            
            # Map predictions back to their original positions in the batch
            for i, pred_val in enumerate(predictions):
                original_idx = valid_indices[i]
                results[original_idx] = {"status": "success", "prediction": str(pred_val)}
                
        return results