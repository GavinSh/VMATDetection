import pickle

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import DataReader
import net
from DataSet import AEDataSetWithLabel, AEDataSetWithLabelRadiomics
from net import EncoderMultConv, DecoderMultConv, EncoderMult1d, DecoderMult1d

import utilize

if __name__ == '__main__':
    kernel_dim = 2
    treatment_convert_func = DataReader.convert_concat_3d_heat_map
    ano_detection_net = net.AnoDetectVanillaAEEntirety
    concat_shape = None
    contractive_lambda = 1
    data_shape = [90, 80, 80]
    is_data_composed_3d_to_2d = True
    is_data_flatten_to_1d = False
    is_data_augment = False
    USE_CUDA = False
    num_epochs = 20000
    mini_batch_size = 32
    latent_dim = 128
    hidden_dim = [128, 64, 64, 16]

    iter_num = 1
    logger_path = r'./tf_logs'

    excel_list = []
    treatment_list = []
    for excel_path in excel_list:
        treatment_list.extend(DataReader.get_VMAT_data_from_execl(excel_path))
    treatment_list, label_list = utilize.fold_treatment(treatment_list, 'data/version_1/label.xlsx')

    with open('backup_train_test_idx_fold0.txt', 'rb') as f:
        backup_train_test_idx_dict = pickle.load(f)
    train_treatment_list = [treatment for treatment_idx, treatment in enumerate(treatment_list) if
                           treatment_idx in backup_train_test_idx_dict['train_treatment_idx']]
    train_treatment_label_list = [treatment_label for treatment_idx, treatment_label in enumerate(label_list) if
                                 treatment_idx in backup_train_test_idx_dict['train_treatment_idx']]
    test_treatment_list = [treatment for treatment_idx, treatment in enumerate(treatment_list) if
                           treatment_idx in backup_train_test_idx_dict['test_treatment_idx']]
    test_treatment_label_list = [treatment_label for treatment_idx, treatment_label in enumerate(label_list) if
                                 treatment_idx in backup_train_test_idx_dict['test_treatment_idx']]

    for _ in range(iter_num):
        if kernel_dim > 1 or is_data_flatten_to_1d:
            train_dataset = AEDataSetWithLabel(
                train_treatment_list, train_treatment_label_list,
                data_shape,
                treatment_convert_func,
                -1, -1,
                is_data_composed_3d_to_2d,
                is_data_flatten_to_1d,
                is_data_augment,
                concat_shape=concat_shape
            )
        else:
            train_dataset = AEDataSetWithLabelRadiomics(
                train_treatment_list, train_treatment_label_list,
                DataReader.convert_sum_heat_map,
                -1, -1,
                is_data_augment
            )
        train_dataloader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, num_workers=0)
        if kernel_dim == 1:
            encoder = EncoderMult1d(train_dataset.data_features, latent_dim, hidden_dim)
            decoder = DecoderMult1d(train_dataset.data_features, latent_dim, hidden_dim[::-1])
        else:
            encoder = EncoderMultConv(train_dataset.data_channels, latent_dim, hidden_dim, data_shape[-1], kernel_dim)
            decoder = DecoderMultConv(train_dataset.data_channels, latent_dim, list(reversed(hidden_dim)),
                                      data_shape[-1], kernel_dim)

        raw_ae = ano_detection_net(
            encoder, decoder,
            dim_num=kernel_dim,
            contractive_lam=contractive_lambda
        )
        logger = TensorBoardLogger(logger_path, name=f'med_ae_{kernel_dim}d')

        checkpoint_callback = ModelCheckpoint(
            monitor='train_rec_loss',
            filename='net-{epoch:02d}',
            save_last=True,
            save_top_k=-1,
            mode='min',
        )
        if torch.backends.cudnn.version() and USE_CUDA:
            trainer = pl.Trainer(logger=logger, gpus=-1, max_epochs=num_epochs,
                                 callbacks=[checkpoint_callback])
        else:
            trainer = pl.Trainer(logger=logger, accelerator="cpu", max_epochs=num_epochs,
                                 callbacks=[checkpoint_callback])

        trainer.fit(raw_ae, train_dataloader)
