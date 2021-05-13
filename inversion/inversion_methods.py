from tqdm import tqdm
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.image_precossing import _sigmoid_to_tanh, _tanh_to_sigmoid, _add_batch_one
import pickle as pkl

from utils.file_utils import image_files, load_as_tensor, Tensor2PIL, split_to_batches


def get_inversion(inversion_type, args):
    if inversion_type == 'GD':
        return GradientDescent(args.lr, optimizer=optim.SGD, args=args)
    elif inversion_type == 'Adam':
        return GradientDescent(args.lr, optimizer=optim.Adam, args=args)


class GradientDescent(object):
    def __init__(self, lr, optimizer, args):
        self.lr = lr
        self.optimizer = optimizer
        self.init_type = args.init_type  # ['Zero', 'Normal']

    def invert(self, generator, gt_image, loss_function, batch_size=1, video=True,
               init=(), z_iterations=3000, args=None, out_dir=None, image_name=None):
        input_size_list = generator.input_size()
        if len(init) == 0:
            if generator.init is False:
                latent_estimate = []
                for input_size in input_size_list:
                    if self.init_type == 'Zero':
                        print("Zero")
                        latent_estimate.append(torch.zeros((batch_size,) + input_size).cuda())
                    elif self.init_type == 'Normal':
                        latent_estimate.append(torch.randn((batch_size,) + input_size).cuda())
            else:
                latent_estimate = list(generator.init_value(batch_size))
        else:
            assert len(init) == len(input_size_list), 'Please check the number of init value'
            latent_estimate = init

        for latent in latent_estimate:
            latent.requires_grad = True  # 19 PGGAN - 0: 1   4 - 18： 1e-8 x 10 = 1e-6
        print(latent_estimate[0].requires_grad)
        optimizer = self.optimizer(latent_estimate, lr=self.lr)

        history = []
        # Opt
        for i in tqdm(range(z_iterations)):
            y_estimate = generator(latent_estimate)
            if i % args.report_image == 0:
                if out_dir is not None:
                    y_estimate_pil = Tensor2PIL(torch.clamp(_tanh_to_sigmoid(y_estimate.detach().cpu()), min=0.0, max=1.0))
                    y_estimate_pil.save(os.path.join(out_dir, image_name.split('.')[0] + "_" + str(i) + "_layer_" +
                                        str(args.composing_layer) + ".png"))
                    zs = latent_estimate[0].detach().cpu().numpy()
                    ps = latent_estimate[1].detach().cpu().numpy()
                    save_dict = {'zs': zs, 'ps': ps}
                    with open(os.path.join(out_dir, image_name.split('.')[0] + "_" + str(i) + "_layer_" +
                                           str(args.composing_layer) + ".pkl"), 'wb') as file_out:
                        pkl.dump(save_dict, file_out)

            optimizer.zero_grad()
            loss = loss_function(y_estimate, gt_image)
            loss.backward()
            optimizer.step()
            if video:
                history.append(copy.deepcopy(latent_estimate))

        return latent_estimate, history

