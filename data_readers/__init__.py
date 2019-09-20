from .bair_predictions_data_reader import BairPredictionsDataReader
from .google_push_predictions_data_reader import GooglePushPredictionsDataReader


original_to_prediction_map = {'bair': BairPredictionsDataReader,
                              'google_push': GooglePushPredictionsDataReader}




