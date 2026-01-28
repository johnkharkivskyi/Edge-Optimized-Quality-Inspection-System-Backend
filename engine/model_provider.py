from .model_engine import RealQualityModel

class ModelProvider:
    def __init__(self):
        # We use the same model class for both modes for now
        # until you create a quantized INT8 version
        self.models = {
            "INT8": RealQualityModel(mode="INT8"),
            "FP32": RealQualityModel(mode="FP32")
        }
        self.current_mode = "FP32"

    def get_active_model(self):
        return self.models[self.current_mode]

    def set_mode(self, mode):
        self.current_mode = mode

provider = ModelProvider()