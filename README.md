## Action-conditioned Benchmarking of Robotic Video Prediction Models

--------------------

[[Paper]](https://arxiv.org/abs/1910.02564)

### Overview
In the past few years research on video prediction models has been the subject of increasing interest. Applications include [weather nowcasting], future [semantic segmentation] or, when action information is available as input, [robotic planning]. While for the first tasks evaluating the quality of predictions using metrics designed to approximate human perception of quality, such as PSNR and SSIM, may be an obvious solution, from the standpoint of a robot planning its actions, image quality is not necessarily the most important aspect of the predictions.
With this in mind, we argue that in a planning context, video predictions should be evaluated by how well they encode action features. And if the action features were correctly encoded, then it should be possible to infer executed action from the predicted frames.

### Before starting
- We assume that the video prediction (VP) models being evaluated have been trained beforehand. Some authors have kindly made their own pre-trained models publicly available and we make use of some of these models in this repository. (see step 1.)

- All VP models mentioned in step 1. were trained on BAIR action-conditioned dataset, except for SVG which is action-free. Also, with exception of 'cdna' which was trained by us, all models were trained by the authors and are also available here [[1]] [[2]].

- We make some action inference models [available], pre-trained on the VP models from previous authors. If you prefer to use these please skip to step 4 of 'How to use'. 

### First steps
- Download the repository
```
git clone https://github.com/m-serra/action-inference-for-video-prediction-benchmarking.git
cd action-inference-for-video-prediction-benchmarking
```

- Download BAIR push dataset
```
bash utils/download_bair_dataset.sh path/to/save/directory
```

- If using python >= 3.6, add the root directory to the PYTHONPATH
```
export PYTHONPATH="${PYTHONPATH}:path/to/action-inference-for-video-prediction-benchmarking"
```

### How to use
**1. Download a pre-trained video prediction model**
```
bash utils/download_pretrained_model.sh model_name
```

The argument 'model_name' can be one of ('cdna', 'savp', 'savp_vae', 'sv2p', 'svg').

**2. Create a dataset of video predictions**
```
python scripts/create_prediction_datasets.py \
  --model_list cdna sv2p savp savp_vae svg  \
  --data_dir /path/to/bair/softmotion30_44k \
  --vp_ckpts_dir pretrained_vp_models \
  --save_dir prediction_datasets
```

**3. Train the action inference model on the predictions**
```
python scripts/train_inference_models.py \
  --model_list cdna sv2p savp savp_vae svg 
  --predictions_data_dir prediction_datasets  
  --model_save_dir pretrained_inference_models
```

**4. Evaluate the action inference model on the prediction test set and compute goodness of fit**
```
python scripts/evaluate_inference_models.py \
  --model_list sv2p savp_vae \
  --predictions_data_dir prediction_datasets\
  --model_ckpt_dir pretrained_inference_models \
  --results_save_dir results
```

### Adding new video prediction models
To evaluate video prediction models that we did not consider, a subclass of `BaseActionInferenceGear` should be created, implementing the methods `vp_restore_model()` and `vp_forward_pass()`. Then, having a pre-trained checkpoint of the model, steps 2-4 can be run.

### Next steps...
- Test on Datasets other than BAIR
- Train and infer without the need to save predictions datasets

### Citation
If somehow we've helped with your research consider citing using the following
```
@article{nunes2019action,
  title={Action-conditioned Benchmarking of Robotic Video Prediction Models: a Comparative Study},
  author={Nunes, Manuel Serra and Dehban, Atabak and Moreno, Plinio and Santos-Victor, Jos{\'e}},
  journal={arXiv preprint arXiv:1910.02564},
  year={2019}
}
```

## Last but not least
Part of the structure and some functions in this repository are inspired by [Alex Lee]'s code, under the MIT License 

<!-- Links -->
[weather nowcasting]: https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting
[semantic segmentation]: https://arxiv.org/abs/1703.07684
[robotic planning]: https://arxiv.org/abs/1610.00696
[1]: https://github.com/alexlee-gk/video_prediction
[2]: https://github.com/edenton/svg
[here]: https://github.com/m-serra/action-inference-for-video-prediction-benchmarking/tree/master/scripts
[available]: https://github.com/m-serra/action-inference-for-video-prediction-benchmarking/tree/master/pretrained_inference_models
[Alex Lee]: https://github.com/alexlee-gk
