import os
import json
from action_inference.action_inference_gear import BaseActionInferenceGear
from video_prediction.savp.video_prediction.models import SAVPVideoPredictionModel


class SavpVaeGear(BaseActionInferenceGear):

    def __init__(self, *args, **kwargs):
        super(BaseActionInferenceGear, self).__init__(*args, **kwargs)

    def vp_forward_pass(self):
        """
        inputs
        ------

        outputs
        -------
        - generated frames
        - ground truth actions
        """
        pass

    def vp_restore_model(self):

        checkpoint_dir = os.path.join(self.ckpts_dir, self.dataset_name, 'savp_vae')

        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")

        VideoPredictionModel = SAVPVideoPredictionModel



    def create_predictions_dataset(self):
        pass

    def train_inference_model_online(self):
        pass