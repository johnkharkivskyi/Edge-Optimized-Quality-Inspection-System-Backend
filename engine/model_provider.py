from .mock_engine import MockQualityModel

class ModelProvider:
    def __init__(self):
        self.models = {
            "INT8": MockQualityModel(mode="INT8"),
            "FP32": MockQualityModel(mode="FP32")
        }
        self.current_mode = "INT8"

    def get_active_model(self):
        return self.models[self.current_mode]

    def set_mode(self, mode):
        if mode in self.models:
            self.current_mode = mode

# Singleton instance
provider = ModelProvider()