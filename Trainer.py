import torch
from Models import SegErrConvClassifier, SegErrTransformerClassifier
from SegmentDataset import SegmentErrorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import ConcatDataset
import torch.nn.functional as t_n_f
import os


def compute_tp_fp_fn(predict: Tensor, target: Tensor):
    b, n = predict.shape
    tp = torch.zeros(size=(n,))
    fp = torch.zeros(size=(n,))
    fn = torch.zeros(size=(n,))
    predict_index = torch.argmax(predict, dim=1)
    for idx in range(b):
        for c in range(n):
            if c == predict_index[idx] and 1 == target[idx][c]:
                tp[c] += 1
            if c == predict_index[idx] and 0 == target[idx][c]:
                fp[c] += 1
            if c != predict_index[idx] and 1 == target[idx][c]:
                fn[c] += 1
    return tp, fp, fn


def train(
        model: any,
        datasets: (),
        epochs=100,
        beg_epoch=0,  # 0 means training starts from scratch
        model_dir='',
        n_class=3,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size=8,
        learning_rate=1e-4,
        weight_decay_param=1e-3,
        beta_param=(0.9, 0.99),
):
    assert beg_epoch < epochs, 'the number of epochs should be greater than the starting epoch'
    if beg_epoch > 0:
        checkpoint = torch.load(model_dir, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = model.to(device)
    model = model.to(device)

    train_set, test_set, model_save_dir = datasets[0], datasets[1], datasets[2]
    model_save_dir = model_save_dir + '_' + model.name + '\\'
    os.makedirs(model_save_dir, exist_ok=True)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=beta_param,
        weight_decay=weight_decay_param)
    loss_func = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [epochs // 2, ], 0.1)

    if beg_epoch > 0:
        train_info = open(model_save_dir + 'info.txt', mode='a')
    else:
        train_info = open(model_save_dir + 'info.txt', mode='w+')
        train_info.write('epoch, train_loss, train_oa, test_loss, test_oa\n')

    for epoch in range(beg_epoch, epochs):
        # train
        model.train()
        train_loss = 0.0
        tp, fp, fn = torch.zeros(size=(n_class,)), torch.zeros(size=(n_class,)), torch.zeros(size=(n_class,))
        train_bar = tqdm(train_loader)
        for label, data in train_bar:
            label = label.to(device, dtype=torch.float)
            image = data[0].to(device, dtype=torch.float)
            mask = data[1].to(device, dtype=torch.float)
            optimizer.zero_grad()
            result = model(image, mask)
            loss = loss_func(result, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tp_t, fp_t, fn_t = compute_tp_fp_fn(result, label)
            tp += tp_t
            fp += fp_t
            fn += fn_t
            oa_batch = tp_t.sum() / len(label)
            train_bar.set_description(desc='[%d/%d], train loss: %.4f, batch oa: %.4f\n' %
                                           (epoch + 1, epochs, loss.item(), oa_batch))
        train_loss /= len(train_set)
        oa = tp.sum() / len(train_set)
        print('[%d/%d] train_loss: %.4f, batch oa: %.4f' % (epoch + 1, epochs, train_loss, oa))
        train_info.write(f'%d, %.4f, %.4f, ' % (epoch + 1, train_loss, oa))
        scheduler.step()

        # eval (test set)
        model.eval()
        test_loss = 0.0
        tp, fp, fn = torch.zeros(size=(n_class,)), torch.zeros(size=(n_class,)), torch.zeros(size=(n_class,))
        test_bar = tqdm(test_loader)
        with torch.no_grad():
            for label, data in test_bar:
                label = label.to(device, dtype=torch.float)
                image = data[0].to(device, dtype=torch.float)
                mask = data[1].to(device, dtype=torch.float)
                result = model(image, mask)
                loss = loss_func(result, label)
                test_loss += loss.item()
                tp_t, fp_t, fn_t = compute_tp_fp_fn(result, label)
                tp += tp_t
                fp += fp_t
                fn += fn_t
                oa_batch = tp_t.sum() / len(label)
                test_bar.set_description(desc='[%d/%d], test loss: %.4f, batch miou: %.4f\n' %
                                              (epoch + 1, epochs, loss.item(), oa_batch))
            test_loss /= len(test_set)
            oa = tp.sum() / len(test_set)
            print('[%d/%d] test_loss: %.4f, oa: %.4f' % (epoch + 1, epochs, test_loss, oa))
            train_info.write(f'%.4f, %.4f\n' % (test_loss, oa))
        # save model and record info
        torch.save(model.state_dict(), model_save_dir + '\\net_epoch_%d.pth' % (epoch + 1))
        train_info.flush()
    train_info.close()


def get_combined_datasets(
        image_dir_list,
        mask_dir_list,
        image_list_info_list,
        data_aug: bool = True,
):
    assert (len(image_dir_list) == len(mask_dir_list) and
            len(image_dir_list) == len(image_list_info_list)), 'inconsistent length for dataset list'
    data_sets = []
    for idx in range(len(image_dir_list)):
        data_sets.append(SegmentErrorDataset(
            sample_list_csv=image_list_info_list[idx],
            segment_image_dir=image_dir_list[idx],
            segment_mask_dir=mask_dir_list[idx],
            data_aug=data_aug))
    combined_ds = ConcatDataset(data_sets)
    return combined_ds


def configure_datasets_wuyuan_jilin1(border_len: int = 0):
    train_ds = get_combined_datasets(
        image_dir_list=[
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_50\\segment_imagettes_%d\\' %
            border_len,
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_20\\segment_imagettes_%d\\' %
            border_len,
        ],
        mask_dir_list=[
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_50\\segment_masks_%d\\' %
            border_len,
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_20\\segment_masks_%d\\' %
            border_len,
        ],
        image_list_info_list=[
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_50\\'
            'sample_division_init_sr_60_step_10_end_sr_80\\train_set_rep_1_sampling_ratio_index_3.csv',
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_20\\'
            'sample_division_init_sr_60_step_10_end_sr_80\\train_set_rep_1_sampling_ratio_index_3.csv',
        ],
        data_aug=True,
    )
    test_ds = get_combined_datasets(
        image_dir_list=[
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_50\\segment_imagettes_%d\\' %
            border_len,
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_20\\segment_imagettes_%d\\' %
            border_len,
        ],
        mask_dir_list=[
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_50\\segment_masks_%d\\' %
            border_len,
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_20\\segment_masks_%d\\' %
            border_len,
        ],
        image_list_info_list=[
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_50\\'
            'sample_division_init_sr_60_step_10_end_sr_80\\test_set_rep_1_sampling_ratio_index_3.csv',
            'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_20\\'
            'sample_division_init_sr_60_step_10_end_sr_80\\test_set_rep_1_sampling_ratio_index_3.csv',
        ],
        data_aug=False,
    )
    model_save_dir = 'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\trained_model_seg_err_cls_%d' % border_len
    return train_ds, test_ds, model_save_dir


def configure_datasets_wuhan_gf2(border_len: int = 0):
    train_ds = get_combined_datasets(
        image_dir_list=[
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_50\\segment_imagettes_%d\\' %
            border_len,
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_10\\segment_imagettes_%d\\' %
            border_len,
        ],
        mask_dir_list=[
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_50\\segment_masks_%d\\' % border_len,
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_10\\segment_masks_%d\\' % border_len,
        ],
        image_list_info_list=[
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_50\\'
            'sample_division_init_sr_60_step_10_end_sr_80\\train_set_rep_1_sampling_ratio_index_3.csv',
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_10\\'
            'sample_division_init_sr_60_step_10_end_sr_80\\train_set_rep_1_sampling_ratio_index_3.csv',
        ],
        data_aug=True,
    )
    test_ds = get_combined_datasets(
        image_dir_list=[
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_50\\segment_imagettes_%d\\' %
            border_len,
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_10\\segment_imagettes_%d\\' %
            border_len,
        ],
        mask_dir_list=[
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_50\\segment_masks_%d\\' % border_len,
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_10\\segment_masks_%d\\' % border_len,
        ],
        image_list_info_list=[
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_50\\'
            'sample_division_init_sr_60_step_10_end_sr_80\\test_set_rep_1_sampling_ratio_index_3.csv',
            'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segment_sample_scale_10\\'
            'sample_division_init_sr_60_step_10_end_sr_80\\test_set_rep_1_sampling_ratio_index_3.csv',
        ],
        data_aug=False,
    )
    model_save_dir = 'D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\trained_model_seg_err_cls_%d' % border_len
    return train_ds, test_ds, model_save_dir


def train_experiment_with_different_depth_models():
    for layer in range(4, 7):
        train(
            model=SegErrConvClassifier(3, 1, layer,),
            datasets=configure_datasets_wuyuan_jilin1(5),
            epochs=500,
            beg_epoch=0,
            model_dir='',
            n_class=3,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            batch_size=120,
            learning_rate=1e-4,
            weight_decay_param=1e-3,
            beta_param=(0.9, 0.99),
        )
        train(
            model=SegErrConvClassifier(3, 1, layer, ),
            datasets=configure_datasets_wuhan_gf2(5),
            epochs=500,
            beg_epoch=0,
            model_dir='',
            n_class=3,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            batch_size=120,
            learning_rate=1e-4,
            weight_decay_param=1e-3,
            beta_param=(0.9, 0.99),
        )


if '__main__' == __name__:
    # train(
    #     model=SegErrConvClassifier(),
    #     datasets=configure_datasets_wuyuan_jilin1(0),
    #     epochs=500,
    #     beg_epoch=0,
    #     model_dir='',
    #     n_class=3,
    #     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    #     batch_size=120,
    #     learning_rate=1e-5,
    #     weight_decay_param=1e-3,
    #     beta_param=(0.9, 0.99),
    # )
    train_experiment_with_different_depth_models()
