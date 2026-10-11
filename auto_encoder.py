import os
import random
import time
from typing import Union, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.autograd import Variable

import utilize

import numpy as np


class AEBackbone(pl.LightningModule):
    def __init__(
            self,
            encoder,
            decoder,
            dim_num: int
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dim_num = dim_num
        self.dim_num_with_channel = dim_num if dim_num == 1 else dim_num + 1

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, x, x_hat, rec_type: str = 'mse', rec_name: str = ''):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        # loss = self.value_loss_func(x, x_hat)
        if rec_type.lower() == 'mse':
            x_hat = nn.Tanh()(x_hat)
            loss = F.mse_loss(x_hat, x, reduction="mean")
        else:
            x_hat = nn.Sigmoid()(x_hat)
            loss = nn.BCELoss(reduction="mean")(x_hat, x.float())
        # loss = loss.sum(dim=list(range(1, len(x.shape)))).mean(dim=[0])
        if rec_name == '':
            self.log("train_rec_loss", loss)
        else:
            self.log(f"train_rec_loss_{rec_name}", loss)
        # loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def on_train_epoch_end(self) -> None:
        cur_version_dir = os.path.join(self.trainer.log_dir, 'checkpoints')
        if not os.path.exists(cur_version_dir):
            return
        cur_epoch = [int(epoch.split('=')[1].split('.')[0]) for epoch in os.listdir(cur_version_dir) if
                     epoch[:9] == 'net-epoch']
        if len(cur_epoch) > 20:
            del_epochs = sorted(cur_epoch)[:10]
            del_epochs_dir = [
                os.path.join(self.trainer.log_dir, 'checkpoints',
                             'net-epoch=' + '{:0>2d}'.format(epoch) + '.ckpt') if epoch < 100 else
                os.path.join(self.trainer.log_dir, 'checkpoints', 'net-epoch=' + '{}'.format(epoch) + '.ckpt')
                for epoch in del_epochs
            ]
            [os.remove(del_dir) for del_dir in del_epochs_dir]
        return

    def test_step(self, batch, batch_idx):
        mini_batch_data, mini_batch_label, mini_batch_idx, mini_batch_treatment_ID = batch

        z_batch = self.encoder(mini_batch_data)
        x_rec_batch = self.decoder(z_batch)
        x_rec_batch = nn.Tanh()(x_rec_batch)

        z_enc_rec_dis_batch = nn.MSELoss(reduction='none')(input=x_rec_batch, target=mini_batch_data)
        z_enc_rec_dis_batch = z_enc_rec_dis_batch.mean(list(range(1, self.dim_num_with_channel + 1)))

        if batch_idx == 0:
            self.z_enc_rec_all = z_enc_rec_dis_batch
            self.label_all = mini_batch_label
            self.idx_all = mini_batch_idx
            self.latent_all = z_batch
            self.treatment_ID_all = mini_batch_treatment_ID
        else:
            self.z_enc_rec_all = torch.cat((self.z_enc_rec_all, z_enc_rec_dis_batch))
            self.label_all = torch.cat((self.label_all, mini_batch_label), dim=0)
            self.idx_all = torch.cat((self.idx_all, mini_batch_idx), dim=0)
            self.latent_all = torch.cat((self.latent_all, z_batch), dim=0)
            self.treatment_ID_all = torch.cat((self.treatment_ID_all, mini_batch_treatment_ID), dim=0)

    def on_test_epoch_end(self) -> None:
        self.z_enc_rec_all = self.z_enc_rec_all.cpu().detach().numpy()
        self.label_all = self.label_all.cpu().detach().numpy()
        self.idx_all = self.idx_all.cpu().detach().numpy()
        self.latent_all = self.latent_all.cpu().detach().numpy()
        self.treatment_ID_all = self.treatment_ID_all.cpu().detach().numpy()


class VanillaAEMultDim(AEBackbone):
    """
    mult dim version
    """

    def __init__(
            self,
            encoder,
            decoder,
            learning_rate=1e-5,
            dim_num=2,
            **kwargs
    ):
        super().__init__(encoder, decoder, dim_num)
        # networks
        self.value_loss_func = nn.MSELoss(reduction='mean')
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        x, _, _, _ = batch
        z_hidden = self.encoder(x)
        x_hat = self.decoder(z_hidden)
        loss = self._get_reconstruction_loss(x, x_hat)
        self.log("train_rec_loss", loss)
        return loss

    def configure_optimizers(self):
        opt_aut = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return [opt_aut], []


class VanillaAEMultDimMultTarget(AEBackbone):
    """
    mult dim and mult target version
    """

    def __init__(
            self,
            encoder,
            decoder_shape,
            decoder_docs,
            learning_rate=1e-5,
            dim_num=2,
            **kwargs
    ):
        super().__init__(encoder, decoder_shape, dim_num)
        # networks
        self.value_loss_func = nn.MSELoss(reduction='mean')
        self.learning_rate = learning_rate
        self.decoder_shape = decoder_shape
        self.decoder_docs = decoder_docs

    def training_step(self, batch, batch_idx):
        x, _, _, _ = batch
        x_shape = ((x > -1) * 1).type_as(x)
        x_docs = torch.from_numpy(x.cpu().numpy().max(-1).max(-1)).type_as(x)
        z_hidden = self.encoder(x)
        x_hat_shape = self.decoder_shape(z_hidden)
        x_hat_docs = self.decoder_docs(z_hidden)
        shape_loss = self._get_reconstruction_loss(x_shape, x_hat_shape, 'cross_entropy', 'shape')
        docs_loss = self._get_reconstruction_loss(x_docs, x_hat_docs, 'mse', 'docs')
        dice_loss = self._get_dice_loss(x_shape, x_hat_shape)
        return shape_loss + docs_loss

    def test_step(self, batch, batch_idx):
        mini_batch_data, mini_batch_label, mini_batch_idx, mini_batch_treatment_ID = batch
        x_shape = (mini_batch_data > -1) * 1
        x_docs = torch.from_numpy(mini_batch_data.cpu().numpy().max(-1).max(-1))
        z_batch = self.encoder(mini_batch_data)
        x_hat_shape = self.decoder_shape(z_batch)
        x_hat_docs = self.decoder_docs(z_batch)
        x_hat_shape = nn.Sigmoid()(x_hat_shape)
        x_hat_docs = nn.Tanh()(x_hat_docs)

        # 分别获得剂量和孔径图
        x_hat_docs_for_result = x_hat_docs.reshape([*x_hat_docs.shape, 1, 1]).repeat(1, 1, x_hat_shape.shape[-2], x_hat_shape.shape[-1])
        x_hat_shape_for_result = (x_hat_shape > 0.5) * 1
        x_hat_for_result = x_hat_docs_for_result * x_hat_shape_for_result
        x_hat_for_result[x_hat_shape <= 0.5] = -1

        z_enc_rec_docs_dis_batch = nn.MSELoss(reduction='none')(input=x_hat_docs, target=x_docs)
        # z_enc_rec_docs_dis_batch = F.mse_loss(x_hat_docs, x_docs, reduction="none")
        z_enc_rec_docs_dis_batch = z_enc_rec_docs_dis_batch.mean(list(range(1, len(z_enc_rec_docs_dis_batch.shape))))
        z_enc_rec_shape_dis_batch = nn.BCELoss(reduction='none')(input=x_hat_shape, target=x_shape.float())
        z_enc_rec_shape_dis_batch = z_enc_rec_shape_dis_batch.mean(list(range(1, len(mini_batch_data.shape))))
        z_enc_rec_dis_batch = z_enc_rec_shape_dis_batch + z_enc_rec_docs_dis_batch
        # z_enc_rec_dis_batch = z_enc_rec_docs_dis_batch

        # 直接计算通量图差异
        z_enc_rec_dis_batch = nn.MSELoss(reduction='none')(input=x_hat_for_result, target=mini_batch_data)
        z_enc_rec_dis_batch = z_enc_rec_dis_batch.mean(list(range(1, self.dim_num_with_channel + 1)))

        if batch_idx == 0:
            self.z_enc_rec_all = z_enc_rec_dis_batch
            self.label_all = mini_batch_label
            self.idx_all = mini_batch_idx
            self.latent_all = z_batch
            self.treatment_ID_all = mini_batch_treatment_ID
        else:
            self.z_enc_rec_all = torch.cat((self.z_enc_rec_all, z_enc_rec_dis_batch))
            self.label_all = torch.cat((self.label_all, mini_batch_label), dim=0)
            self.idx_all = torch.cat((self.idx_all, mini_batch_idx), dim=0)
            self.latent_all = torch.cat((self.latent_all, z_batch), dim=0)
            self.treatment_ID_all = torch.cat((self.treatment_ID_all, mini_batch_treatment_ID), dim=0)

    def configure_optimizers(self):
        opt_aut = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return [opt_aut], []

    def _get_dice_loss(self, x, x_hat):
        eps = 1e-5
        x_hat = nn.Sigmoid()(x_hat)
        N = x.size(0)
        pred_flat = x_hat.view(N, -1)
        gt_flat = x.view(N, -1)
        inter = (pred_flat * gt_flat).sum(1)
        union = pred_flat.sum(1) + gt_flat.sum(1)
        dice = 2 * (inter + eps) / (union + eps)
        dice = dice.mean(dim=[0])
        self.log("train_dice", dice)
        return 1 - dice


class AdversarialAEMultDim(AEBackbone):
    """
    mult dim version
    """

    def __init__(
            self,
            encoder,
            decoder,
            discriminator,
            learning_rate=1e-5,
            learning_rate_dis_z=1e-5,
            dim_num=2,
            tau=4,
            **kwargs
    ):
        super().__init__(encoder, decoder, dim_num)
        # networks
        self.discriminator = discriminator
        self.learning_rate = learning_rate
        self.learning_rate_dis_z = learning_rate_dis_z
        self.tau = tau

        self.mu_gauss, self.z_continous_samples_all = utilize.sample_gaussian_mixture(self.tau)
        self.z_continous_samples_all = torch.from_numpy(self.z_continous_samples_all)

    def _get_discriminate_loss(self, d_z, is_real):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        d_z = nn.Sigmoid()(d_z)
        if is_real:
            target = torch.ones(d_z.shape).type_as(d_z)
        else:
            target = torch.zeros(d_z.shape).type_as(d_z)
        discriminate_loss = nn.BCELoss(reduction='mean')(target=target, input=d_z)
        # self.log("train_rec_loss", loss)
        return discriminate_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _, _, _ = batch
        if optimizer_idx == 0:
            # train encoder and decoder by reconstruction loss
            z_hidden = self.encoder(x)
            x_hat = self.decoder(z_hidden)
            reconstruction_loss = self._get_reconstruction_loss(x, x_hat)
            self.log("train_rec_loss", reconstruction_loss)
            return reconstruction_loss
        if optimizer_idx == 1:
            # train discriminator by discriminate the z generate by encoder or sample from Gaussian
            # The z sampled from the Gaussian needs to be judged as a positive example by the discriminator
            z_target_batch = self.z_continous_samples_all[
                             random.sample(range(0, self.z_continous_samples_all.shape[0]), x.shape[0]), :
                             ]
            z_target_batch = z_target_batch.type_as(x)
            # The z generated by the encoder needs to be judged as a negative example by the discriminator
            z_hidden_de = self.encoder(x).detach()

            d_gauss = self.discriminator(z_target_batch)
            d_encoder = self.discriminator(z_hidden_de)
            # determine discriminator classification target variables
            discriminator_loss_real = self._get_discriminate_loss(d_gauss, True)
            discriminator_loss_fake = self._get_discriminate_loss(d_encoder, False)
            discriminator_loss = discriminator_loss_fake + discriminator_loss_real
            self.log("train_dis_loss", discriminator_loss, prog_bar=True)
            return discriminator_loss
        if optimizer_idx == 2:
            # Train the encoder by using the z generated by the encoder deceiving discriminator
            z_hidden = self.encoder(x)
            d_encoder = self.discriminator(z_hidden)
            generator_loss = self._get_discriminate_loss(d_encoder, True)
            self.log("train_gen_loss", generator_loss, prog_bar=True)
            return generator_loss

    def configure_optimizers(self):
        opt_aut = torch.optim.Adam(
            [{"params": self.encoder.parameters()}, {"params": self.decoder.parameters()}],
            lr=self.learning_rate)
        # opt_dec = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate_dec)
        opt_dis = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate_dis_z)
        opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        return [opt_aut, opt_dis, opt_enc], []


class ContractiveAEOneDim(AEBackbone):
    """
    Contractive AE, 仅能支持线性核的一层编解码器
    """

    def __init__(
            self,
            encoder,
            decoder,
            learning_rate=1e-5,
            dim_num=1,
            contractive_lam: float = None,
            **kwargs
    ):
        super().__init__(encoder, decoder, dim_num)
        self.learning_rate = learning_rate
        self.contractive_lam = 10 if contractive_lam is None else contractive_lam
        assert dim_num == 1, 'contractive AE is only supported for which dim is 1'

    def _get_contractive_loss(self, W, z_hidden):
        dh = z_hidden * (1 - z_hidden)  # Hadamard product produces size N_batch x N_hidden
        w_sum = torch.sum(Variable(W) ** 2, dim=1)
        w_sum = w_sum.unsqueeze(1)  # shape N_hidden x 1
        contractive_loss = torch.mean(torch.mm(dh ** 2, w_sum), 0)
        self.log("train_contractive_loss", contractive_loss, prog_bar=True)
        return contractive_loss

    def training_step(self, batch, batch_idx):
        x, _, _, _ = batch  # We do not need the labels
        z_hidden = self.encoder(x)
        x_hat = self.decoder(z_hidden)
        W = self.state_dict()['encoder.layers.0.block.0.weight']
        rec_loss = self._get_reconstruction_loss(x, x_hat)
        con_loss = self._get_contractive_loss(W, z_hidden)
        return rec_loss + self.contractive_lam * con_loss

    def configure_optimizers(self):
        opt_aut = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return [opt_aut], []


class AnoDetectVanillaAEEntirety(VanillaAEMultDim):
    """
    计算重建误差来识别异常计划
    """
    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().test_epoch_end(outputs)
        test_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        utilize.output_entire_distance(
            self.idx_all, self.treatment_ID_all, self.label_all, self.z_enc_rec_all,
            self.trainer.default_root_dir, f'{self.dim_num}d_{test_time_str}'
        )

class AnoDetectVanillaAEEntiretyInterp(VanillaAEMultDim):
    """
    计算重建误差来识别异常计划
    """
    def on_test_epoch_start(self):
        train_data = np.array(self.train_data)

        average_regular_continuous_feature = np.mean(train_data, axis=0)
        self.average_regular_feature = average_regular_continuous_feature

    def test_step(self, batch, batch_idx):
        mini_batch_data, mini_batch_label, mini_batch_idx, mini_batch_treatment_ID = batch
        average_regular_minibatch = torch.from_numpy(self.average_regular_feature).unsqueeze(0).repeat(mini_batch_data.size(0), *([1] * len(self.average_regular_feature.shape)))

        z_batch = self.encoder(mini_batch_data)
        x_rec_batch = self.decoder(z_batch)
        x_rec_batch = nn.Tanh()(x_rec_batch)

        z_enc_rec_dis_batch = nn.MSELoss(reduction='none')(input=x_rec_batch, target=mini_batch_data).mean(dim=tuple(range(1, mini_batch_data.ndim)))
        # z_enc_rec_dis_batch = z_enc_rec_dis_batch.mean(list(range(1, self.dim_num_with_channel + 1)))
        replaced_rec_dis_list = []

        # 计算更换为正常值之后的重建值
        for var in range(180):
            var_p = var
            replaced_mini_batch_data = mini_batch_data.clone()
            replaced_mini_batch_data[:, var_p: var_p + 1] = average_regular_minibatch[:, var_p: var_p + 1]

            z_batch = self.encoder(replaced_mini_batch_data)
            x_rec_batch = self.decoder(z_batch)
            x_rec_batch = nn.Tanh()(x_rec_batch)
            z_enc_rec_dis_batch_replace = nn.MSELoss(reduction='none')(input=x_rec_batch, target=replaced_mini_batch_data).mean(dim=tuple(range(1, mini_batch_data.ndim)))
            # z_enc_rec_dis_batch = z_enc_rec_dis_batch.mean(list(range(1, self.dim_num_with_channel + 1)))

            replaced_rec_dis_list.append(z_enc_rec_dis_batch_replace)

        if batch_idx == 0:
            self.z_enc_rec_all = z_enc_rec_dis_batch
            self.label_all = mini_batch_label
            self.idx_all = mini_batch_idx
            self.latent_all = z_batch
            self.treatment_ID_all = mini_batch_treatment_ID
            self.replaced_rec_dis_list = replaced_rec_dis_list
        else:
            self.z_enc_rec_all = torch.cat((self.z_enc_rec_all, z_enc_rec_dis_batch))
            self.label_all = torch.cat((self.label_all, mini_batch_label), dim=0)
            self.idx_all = torch.cat((self.idx_all, mini_batch_idx), dim=0)
            self.latent_all = torch.cat((self.latent_all, z_batch), dim=0)
            self.treatment_ID_all = torch.cat((self.treatment_ID_all, mini_batch_treatment_ID), dim=0)
            self.replaced_rec_dis_list = [
                torch.cat((l_dis, dis)) for l_dis, dis in zip(self.replaced_rec_dis_list, replaced_rec_dis_list)
            ]

    def on_test_epoch_end(self) -> None:
        # super().test_epoch_end(outputs)
        # test_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        # utilize.output_entire_distance(
        #     self.idx_all, self.treatment_ID_all, self.label_all, self.z_enc_rec_all,
        #     self.trainer.default_root_dir, f'{self.dim_num}d_{test_time_str}'
        # )
        self.z_enc_rec_all = self.z_enc_rec_all.cpu().detach().numpy()
        replaced_rec_dis_np = np.array([dis.cpu().detach().numpy() for dis in self.replaced_rec_dis_list])
        self.replaced_rec_dis_delta_np = np.tile(self.z_enc_rec_all, (replaced_rec_dis_np.shape[0], 1)) - replaced_rec_dis_np
        self.label_all = self.label_all.cpu().detach().numpy()
        self.idx_all = self.idx_all.cpu().detach().numpy()
        self.treatment_ID_all = self.treatment_ID_all.cpu().detach().numpy()

        self.test_result = {
            'idx': self.idx_all,
            'z_replaced_rec_delta': self.replaced_rec_dis_delta_np,
            'z_enc_rec': self.z_enc_rec_all,
            'label': self.label_all,
            'treatment_ID': self.treatment_ID_all,
        }


class AnoDetectVanillaAECluster(VanillaAEMultDim):
    """
    使用隐藏层的特征进行聚类进而来识别异常计划
    """

    def __init__(
            self,
            encoder,
            decoder,
            learning_rate=1e-5,
            dim_num=2,
            n_cluster=10,
            **kwargs
    ):
        super().__init__(encoder, decoder, learning_rate, dim_num)
        self.n_cluster = n_cluster

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().test_epoch_end(outputs)
        test_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        utilize.output_cluster_distance(
            self.idx_all, self.label_all, self.latent_all,
            self.trainer.default_root_dir, f'{self.dim_num}d_cluster_{test_time_str}',
            None, self.n_cluster
        )


class AnoDetectContractiveAEEntirety(ContractiveAEOneDim):
    """
    使用CAE网络计算重建误差来识别异常计划
    """

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().test_epoch_end(outputs)
        test_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        utilize.output_entire_distance(
            self.idx_all, self.treatment_ID_all, self.label_all, self.z_enc_rec_all,
            self.trainer.default_root_dir, f'contractive_{self.dim_num}d_{test_time_str}'
        )


class AnoDetectAdversarialAECluster(AdversarialAEMultDim):
    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().test_epoch_end(outputs)
        test_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        utilize.output_scatter(
            self.latent_all,
            self.label_all,
            self.mu_gauss,
            self.trainer.default_root_dir, f'{self.dim_num}d_cluster_{test_time_str}',
        )
        utilize.output_cluster_distance(
            self.idx_all, self.label_all, self.latent_all,
            self.trainer.default_root_dir, f'{self.dim_num}d_cluster_{test_time_str}',
            self.mu_gauss
        )


class AnoDetectVanillaAEMultTargetEntirety(VanillaAEMultDimMultTarget):
    """
    计算重建误差来识别异常计划
    """

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().test_epoch_end(outputs)
        test_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        utilize.output_entire_distance(
            self.idx_all, self.treatment_ID_all, self.label_all, self.z_enc_rec_all,
            self.trainer.default_root_dir, f'{self.dim_num}d_{test_time_str}'
        )


class AdversarialAEMultDimMultTarget(AEBackbone):
    """
    mult dim and mult target version
    """

    def __init__(
            self,
            encoder,
            decoder_shape,
            decoder_docs,
            discriminator,
            learning_rate=1e-5,
            learning_rate_dis_z=1e-5,
            dim_num=2,
            tau=1,
            **kwargs
    ):
        super().__init__(encoder, decoder_shape, dim_num)
        # networks
        self.value_loss_func = nn.MSELoss(reduction='mean')
        self.learning_rate = learning_rate
        self.learning_rate_dis_z = learning_rate_dis_z
        self.decoder_shape = decoder_shape
        self.decoder_docs = decoder_docs
        self.discriminator = discriminator

        self.tau = tau

        self.mu_gauss, self.z_continous_samples_all = utilize.sample_gaussian_mixture(self.tau, sigma=0.1)
        self.z_continous_samples_all = torch.from_numpy(self.z_continous_samples_all)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _, _, _ = batch
        if optimizer_idx == 0:
            x_shape = (x > -1) * 1
            x_docs = torch.from_numpy(x.cpu().numpy().max(-1).max(-1)).to(x.device)
            z_hidden = self.encoder(x)
            x_hat_shape = self.decoder_shape(z_hidden)
            x_hat_docs = self.decoder_docs(z_hidden)
            shape_loss = self._get_reconstruction_loss(x_shape, x_hat_shape, 'cross_entropy', 'shape')
            docs_loss = self._get_reconstruction_loss(x_docs, x_hat_docs, 'mse', 'docs')
            dice_loss = self._get_dice_loss(x_shape, x_hat_shape)
            return shape_loss + docs_loss
        if optimizer_idx == 1:
            # train discriminator by discriminate the z generate by encoder or sample from Gaussian
            # The z sampled from the Gaussian needs to be judged as a positive example by the discriminator
            z_target_batch = self.z_continous_samples_all[
                             random.sample(range(0, self.z_continous_samples_all.shape[0]), x.shape[0]), :
                             ]
            z_target_batch = z_target_batch.type_as(x)
            # The z generated by the encoder needs to be judged as a negative example by the discriminator
            z_hidden_de = self.encoder(x).detach()

            d_gauss = self.discriminator(z_target_batch)
            d_encoder = self.discriminator(z_hidden_de)
            # determine discriminator classification target variables
            discriminator_loss_real = self._get_discriminate_loss(d_gauss, True)
            discriminator_loss_fake = self._get_discriminate_loss(d_encoder, False)
            discriminator_loss = discriminator_loss_fake + discriminator_loss_real
            self.log("train_dis_loss", discriminator_loss, prog_bar=True)
            return discriminator_loss
        if optimizer_idx == 2:
            # Train the encoder by using the z generated by the encoder deceiving discriminator
            z_hidden = self.encoder(x)
            d_encoder = self.discriminator(z_hidden)
            generator_loss = self._get_discriminate_loss(d_encoder, True)
            self.log("train_gen_loss", generator_loss, prog_bar=True)
            return generator_loss

    def test_step(self, batch, batch_idx):
        mini_batch_data, mini_batch_label, mini_batch_idx, mini_batch_treatment_ID = batch
        x_shape = (mini_batch_data > -1) * 1
        x_docs = torch.from_numpy(mini_batch_data.cpu().numpy().max(-1).max(-1)).to(mini_batch_data.device)
        z_batch = self.encoder(mini_batch_data)
        x_hat_shape = self.decoder_shape(z_batch)
        x_hat_docs = self.decoder_docs(z_batch)
        x_hat_shape = nn.Sigmoid()(x_hat_shape)
        x_hat_docs = nn.Tanh()(x_hat_docs)

        z_enc_rec_docs_dis_batch = nn.MSELoss(reduction='none')(input=x_hat_docs, target=x_docs)
        # z_enc_rec_docs_dis_batch = F.mse_loss(x_hat_docs, x_docs, reduction="none")
        z_enc_rec_docs_dis_batch = z_enc_rec_docs_dis_batch.mean(list(range(1, len(z_enc_rec_docs_dis_batch.shape))))
        z_enc_rec_shape_dis_batch = nn.BCELoss(reduction='none')(input=x_hat_shape, target=x_shape.float())
        z_enc_rec_shape_dis_batch = z_enc_rec_shape_dis_batch.mean(list(range(1, len(mini_batch_data.shape))))
        z_enc_rec_dis_batch = z_enc_rec_shape_dis_batch + z_enc_rec_docs_dis_batch
        # z_enc_rec_dis_batch = z_enc_rec_docs_dis_batch

        if batch_idx == 0:
            self.z_enc_rec_all = z_enc_rec_dis_batch
            self.label_all = mini_batch_label
            self.idx_all = mini_batch_idx
            self.latent_all = z_batch
            self.treatment_ID_all = mini_batch_treatment_ID
        else:
            self.z_enc_rec_all = torch.cat((self.z_enc_rec_all, z_enc_rec_dis_batch))
            self.label_all = torch.cat((self.label_all, mini_batch_label), dim=0)
            self.idx_all = torch.cat((self.idx_all, mini_batch_idx), dim=0)
            self.latent_all = torch.cat((self.latent_all, z_batch), dim=0)
            self.treatment_ID_all = torch.cat((self.treatment_ID_all, mini_batch_treatment_ID), dim=0)

    def configure_optimizers(self):
        opt_aut = torch.optim.Adam(
            [{"params": self.encoder.parameters()}, {"params": self.decoder_shape.parameters()},
             {"params": self.decoder_docs.parameters()}],
            lr=self.learning_rate)
        # opt_dec = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate_dec)
        opt_dis = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate_dis_z)
        opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        return [opt_aut, opt_dis, opt_enc], []

    def _get_dice_loss(self, x, x_hat):
        eps = 1e-5
        x_hat = nn.Sigmoid()(x_hat)
        N = x.size(0)
        pred_flat = x_hat.view(N, -1)
        gt_flat = x.view(N, -1)
        inter = (pred_flat * gt_flat).sum(1)
        union = pred_flat.sum(1) + gt_flat.sum(1)
        dice = 2 * (inter + eps) / (union + eps)
        dice = dice.mean(dim=[0])
        self.log("train_dice", dice)
        return 1 - dice

    def _get_discriminate_loss(self, d_z, is_real):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        d_z = nn.Sigmoid()(d_z)
        if is_real:
            target = torch.ones(d_z.shape).type_as(d_z)
        else:
            target = torch.zeros(d_z.shape).type_as(d_z)
        discriminate_loss = nn.BCELoss(reduction='mean')(target=target, input=d_z)
        # self.log("train_rec_loss", loss)
        return discriminate_loss

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().test_epoch_end(outputs)
        test_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        utilize.output_scatter(
            self.latent_all,
            self.label_all,
            self.mu_gauss,
            self.trainer.default_root_dir, f'{self.dim_num}d_cluster_{test_time_str}',
        )
        utilize.output_cluster_distance(
            self.idx_all, self.label_all, self.latent_all,
            self.trainer.default_root_dir, f'{self.dim_num}d_cluster_{test_time_str}',
            self.mu_gauss
        )
        utilize.output_entire_distance(
            self.idx_all, self.treatment_ID_all, self.label_all, self.z_enc_rec_all,
            self.trainer.default_root_dir, f'{self.dim_num}d_{test_time_str}'
        )