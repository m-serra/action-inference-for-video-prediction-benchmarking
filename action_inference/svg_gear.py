import os
import sys
import random
import torch
import numpy as np
import video_prediction.svg.utils as utils
from torch.utils.data import DataLoader
from action_inference.base_action_inference_gear import BaseActionInferenceGear


class SVGGear(BaseActionInferenceGear):

    def __init__(self, random_seed=7, **kwargs):
        super(SVGGear, self).__init__(**kwargs)
        self.random_seed = random_seed
        self.num_stochastic_samples = 1
        self.last_frame_skip = None

    def vp_forward_pass(self, model, inputs, sess, mode=None, feeds=None):

        inpts = next(inputs)
        gt_actions = inpts['actions']
        gt_states = inpts['states']
        gt_frames = inpts['images']

        # get approx posterior sample
        model['frame_predictor'].hidden = model['frame_predictor'].init_hidden()
        model['posterior'].hidden = model['posterior'].init_hidden()

        posterior_gen = [inpts['images'][0]]
        x_in = inpts['images'][0]
        for i in range(1, self.context_frames+self.n_future):
            h = model['encoder'](x_in)
            h_target = model['encoder'](inpts['images'][i])[0].detach()
            if self.last_frame_skip or i < self.context_frames:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            _, z_t, _ = model['posterior'](h_target)  # take the mean
            if i < self.context_frames:
                model['frame_predictor'](torch.cat([h, z_t], 1))
                posterior_gen.append(inpts['images'][i])
                x_in = inpts['images'][i]
            else:
                h_pred = model['frame_predictor'](torch.cat([h, z_t], 1)).detach()
                x_in = model['decoder']([h_pred, skip]).detach()
                posterior_gen.append(x_in)

        all_gen = []
        for s in range(self.num_stochastic_samples):
            gen_seq = []
            gt_seq = []
            model['frame_predictor'].hidden = model['frame_predictor'].init_hidden()
            model['posterior'].hidden = model['posterior'].init_hidden()
            model['prior'].hidden = model['prior'].init_hidden()
            x_in = inpts['images'][0]
            all_gen.append([])
            all_gen[s].append(x_in)
            for i in range(1, self.context_frames+self.n_future):
                h = model['encoder'](x_in)
                if self.last_frame_skip or i < self.context_frames:
                    h, skip = h
                else:
                    h, _ = h
                h = h.detach()
                if i < self.context_frames:
                    h_target = model['encoder'](inpts['images'][i])[0].detach()
                    z_t, _, _ = model['posterior'](h_target)
                    model['prior'](h)
                    model['frame_predictor'](torch.cat([h, z_t], 1))
                    x_in = inpts['images'][i]
                    all_gen[s].append(x_in)
                else:
                    z_t, _, _ = model['prior'](h)
                    h = model['frame_predictor'](torch.cat([h, z_t], 1)).detach()
                    x_in = model['decoder']([h, skip]).detach()
                    gen_seq.append(x_in.data.cpu().numpy())
                    gt_seq.append(inpts['images'][i].data.cpu().numpy())
                    all_gen[s].append(x_in)

        # Convert to numpy
        gt_states = gt_states.cpu().numpy()
        gen_np = [batch.transpose(1, 2).transpose(2, 3).cpu().numpy() for i, batch in enumerate(all_gen[0])]
        gen_np = np.stack(gen_np, axis=0)
        gen_np = np.transpose(gen_np, [1, 0, 2, 3, 4])

        gt_np = [batch.transpose(1, 2).transpose(2, 3).cpu().numpy() for i, batch in enumerate(gt_frames)]
        gt_np = np.stack(gt_np, axis=0)
        gt_np = np.transpose(gt_np, [1, 0, 2, 3, 4])

        # --> In the future I have to replace the states with actions, for generality
        # return np.array(all_gen)[:, -self.n_future:], gt_actions, np.array(gt_states)[:, -self.n_future:]
        return gen_np[:, -self.n_future:], gt_np[:, -self.n_future:], gt_states[:, -self.n_future:]

    def vp_restore_model(self, dataset, mode, ckpts_dir):

        # to restore the model the original svg directory should be in the path
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        'video_prediction/svg'))

        ckpts_dir = os.path.join(ckpts_dir, 'model.pth')

        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        dtype = torch.cuda.FloatTensor

        # ---------------- load the models  ----------------
        tmp = torch.load(ckpts_dir)
        frame_predictor = tmp['frame_predictor']
        posterior = tmp['posterior']
        prior = tmp['prior']

        for i, (name, module) in enumerate(frame_predictor._modules.items()):
            module = self.recursion_change_bn(frame_predictor)
        for i, (name, module) in enumerate(prior._modules.items()):
            module = self.recursion_change_bn(prior)

        frame_predictor.eval()
        prior.eval()
        posterior.eval()
        encoder = tmp['encoder']
        decoder = tmp['decoder']

        for i, (name, module) in enumerate(encoder._modules.items()):
            module = self.recursion_change_bn(encoder)
        for i, (name, module) in enumerate(decoder._modules.items()):
            module = self.recursion_change_bn(decoder)
        encoder.eval()
        decoder.eval()

        encoder.train()
        decoder.train()
        frame_predictor.batch_size = dataset.batch_size
        posterior.batch_size = dataset.batch_size
        prior.batch_size = dataset.batch_size

        # --------- transfer to gpu --------------------
        frame_predictor.cuda()
        posterior.cuda()
        prior.cuda()
        encoder.cuda()
        decoder.cuda()

        # --------- options --------------------
        self.last_frame_skip = tmp['opt'].last_frame_skip

        # ---------------- get the data ----------------
        dataset.set_mode(mode)
        # data = dataset.get_seq_from_png(mode=mode)

        data_loader = DataLoader(dataset,
                                 num_workers=0,  # dataset.n_threads,
                                 batch_size=dataset.batch_size,
                                 shuffle=dataset.shuffle,
                                 drop_last=True,
                                 pin_memory=True)

        batch_generator = self.get_batch(data_loader, dataset.dataset_name, dtype)

        model = {'frame_predictor': frame_predictor, 'encoder': encoder, 'decoder': decoder,
                 'prior': prior, 'posterior': posterior}

        return model, batch_generator, None, None

    def evaluate_inference_model_online(self):
        raise NotImplementedError

    def recursion_change_bn(self, module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = 1
            module.num_batches_tracked = None
        if isinstance(module, torch.nn.UpsamplingNearest2d):
            module.align_corners = None
        else:
            for i, (name, module1) in enumerate(module._modules.items()):
                module1 = self.recursion_change_bn(module1)
        return module

    @staticmethod
    def get_batch(data_loader, dataset, dtype):
        while True:
            for sequence in data_loader:
                img_batch = utils.normalize_data(dataset, dtype, sequence['images'])
                yield {'images': img_batch, 'actions': sequence['actions'], 'states': sequence['states']}
