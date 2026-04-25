import sys
import pandas as pd
from src.exception import MyException
from src.utils.main_utils import load_object

class PredictionPipeline:
    def __init__(self):
        try:
            self.model_path = "artifact/25_04_2026_20_35_12/Model_Trainer/Model/model.pkl"
            self.model = load_object(self.model_path)
        except Exception as e:
            raise MyException(e) from e

    def predict(self, data: dict):
        try:
            df = pd.DataFrame([data])
            prediction = self.model.predict(df)
            return int(prediction[0])
        except Exception as e:
            raise MyException(e) from e