import pickle

import pytorch_lightning as pl
from sklearn.metrics import roc_curve, precision_score, accuracy_score, f1_score, auc
from torch.utils.data import DataLoader

import DataReader
import net
import utilize
from DataSet import AEDataSetWithLabel, AEDataSetWithLabelRadiomics
from net import EncoderMultConv, DecoderMultConv, EncoderMult1d, DecoderMult1d
import matplotlib.pyplot as plt
import os

import numpy as np

if __name__ == '__main__':
    kernel_dim = 2
    treatment_convert_func = DataReader.convert_concat_3d_heat_map
    ano_detection_net = net.AnoDetectVanillaAEEntiretyInterp
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

    checkpoint_path = 'tf_logs/med_ae_2d/version_1/checkpoints/net-epoch=19999.ckpt'

    iter_num = 1
    logger_path = r'./tf_logs'
    result_path = r'./result'
    # seed_value = 5000

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
            test_dataset = AEDataSetWithLabel(
                test_treatment_list, test_treatment_label_list,
                data_shape,
                treatment_convert_func,
                train_dataset.map_max, train_dataset.map_min,
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
            test_dataset = AEDataSetWithLabelRadiomics(
                test_treatment_list, test_treatment_label_list,
                DataReader.convert_sum_heat_map,
                -1, -1,
                is_data_augment
            )
        test_dataloader = DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=True, num_workers=0)
        if kernel_dim == 1:
            encoder = EncoderMult1d(test_dataset.data_features, latent_dim, hidden_dim)
            decoder = DecoderMult1d(test_dataset.data_features, latent_dim, hidden_dim[::-1])
        else:
            encoder = EncoderMultConv(test_dataset.data_channels, latent_dim, hidden_dim, data_shape[-1], kernel_dim)
            decoder = DecoderMultConv(test_dataset.data_channels, latent_dim, list(reversed(hidden_dim)),
                                      data_shape[-1], kernel_dim)

        # raw_ae = ano_detection_net(
        #     encoder, decoder,
        #     dim_num=kernel_dim,
        #     contractive_lam=contractive_lambda
        # )
        raw_ae = ano_detection_net.load_from_checkpoint(
            encoder=encoder, decoder=decoder, dim_num=kernel_dim, contractive_lam=contractive_lambda,
            checkpoint_path=checkpoint_path
        )
        raw_ae.train_data = [(d - train_dataset.map_min)/ (train_dataset.map_max - train_dataset.map_min) * 2 - 1 for d in train_dataset.treatment_map_list]

        trainer = pl.Trainer(default_root_dir=result_path)

        trainer.test(raw_ae, test_dataloader)

        test_result = raw_ae.test_result

        data_idx = test_result['idx']
        # data_z_enc_transactions = test_result['z_enc_transactions']
        data_label = test_result['label']
        data_z_rec = test_result['z_enc_rec']
        data_z_replaced_rec_dis_delta = test_result['z_replaced_rec_delta']
        data_treatment_id = test_result['treatment_ID']

        as_fig, as_axes = plt.subplots(1, 1, sharey=True, figsize=(60, 12))
        auc_fig, auc_axes = plt.subplots(1, 1, sharey=True, figsize=(12, 12))
        kfold_idx=0

        regular_z_data = data_z_rec[data_label == 0]
        global_z_outliers = data_z_rec[data_label == 1]
        # local_outliers = z_mode[z_mode['label'] == 'local']
        z_outliers_replaced_rec_dis_delta = data_z_replaced_rec_dis_delta[:, data_label == 1]
        average_outliers_replaced_rec_dis_delta = np.mean(z_outliers_replaced_rec_dis_delta, axis=1)
        # utils.circle_table_plot(z_outliers_replaced_rec_dis_delta, title='AE')
        pass
        utilize.circle_table_plot(z_outliers_replaced_rec_dis_delta, title='AE')

        # temp = val_regular_idx[data_label == 0]
        # temp = ((temp - np.min(temp)) / (np.max(temp) - np.min(temp)) * label.shape[0]).astype(int)
        as_axes.scatter(data_idx[data_label == 0], regular_z_data,
                                   c='C0', marker='o', s=30, linewidth=0.3, label='Regular', edgecolors='w')

        # create train scatter plot of global anomalies
        as_axes.scatter(data_idx[data_label == 1], global_z_outliers,
                                   c='C1', marker='x', s=120, linewidth=3, label='Anomalous', edgecolors='w')
        # as_axes[kfold_idx].scatter(val_regular_idx[data_label == 1], global_z_outliers,
        #                         c='C1', marker='x', s=120, linewidth=3, label='Anomalous', edgecolors='w')

        xlabel = 'Mode Number ($\\tau$)'

        # as_fig.text(0.35, 0.05, xlabel, fontsize=24)

        # plt.suptitle('Anomaly Score | $\\alpha = {:.1f}$'.format(alpha), fontsize=24)
        as_fig.suptitle('Anomaly Score', fontsize=24)

        data_anomalous = data_z_rec

        # fig, axes = plt.subplots(1, 1, sharey=True, figsize=(12, 12))
        raw_kmeans_fpr, raw_kmeans_tpr, raw_kmeans_ther = roc_curve(data_label, data_anomalous, pos_label=1)
        raw_kmeans_auc = auc(raw_kmeans_fpr, raw_kmeans_tpr)
        auc_axes.plot(
            raw_kmeans_fpr, raw_kmeans_tpr, 'k--', lw=2,
            c=plt.cm.Set3(kfold_idx),
            label='{0}(AUC={1:.2f})'.format('AE KFold = {}'.format(kfold_idx), raw_kmeans_auc)
        )

        th = [th for f, t, th in zip(raw_kmeans_fpr, raw_kmeans_tpr, raw_kmeans_ther) if t == 1][0]
        fpr = [f for f, t, th in zip(raw_kmeans_fpr, raw_kmeans_tpr, raw_kmeans_ther) if t == 1][0]
        preb = (data_anomalous > th) * 1
        prec = precision_score(data_label, preb)
        f1 = f1_score(data_label, preb)
        acc = accuracy_score(data_label, preb)
        ann = len(np.where(preb == 1)[0])

        # with open(os.path.join('recon_violin', 'AE_' + str(kfold_idx)), 'wb') as file:
        #     pickle.dump(
        #         {
        #             'label': data_label,
        #             'recon': data_z_rec
        #         },
        #         file
        #     )

        utilize.draw_replaced_feature_bar_fig(average_outliers_replaced_rec_dis_delta, model_name=f'AE')
        auc_fig.legend(loc='lower right')
        auc_fig.savefig('2d_dim_auc.png', dpi=400)
        as_fig.savefig('2d_dim_bar.png', dpi=400)

        default_root_dir = r'result_interp'
        utilize.output_entire_distance(data_idx, data_treatment_id, data_label, data_z_rec, default_root_dir, f'AE')


