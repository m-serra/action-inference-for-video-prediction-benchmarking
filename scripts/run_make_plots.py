import os
from utils.make_plots import make_plots_vp_metrics
from utils.make_plots import make_plots_action_inference
from utils.make_plots import make_plots_regressions


def main():

    model_list = ['savp_vae', 'savp', 'sv2p', 'cdna', 'svg']
    make_plots_vp_metrics(model_list=model_list,
                          results_dir=os.path.join(os.path.expanduser('~/'),
                                                   'Tese/action-inference-for-video-prediction-benchmarking/results/bair'))

    results_dir = os.path.join(os.path.expanduser('~/'),
                               'Tese/action-inference-for-video-prediction-benchmarking/results/bair_predictions')
    inference_models_dir = os.path.join(os.path.expanduser('~/'),
                                        'Tese/action-inference-for-video-prediction-benchmarking/pretrained_inference_models/bair_predictions/')

    make_plots_action_inference(model_list=model_list, results_dir=results_dir)
    make_plots_regressions(model_list=model_list, results_dir=results_dir, inference_models_dir=inference_models_dir)


if __name__ == '__main__':
    main()
