import torch
from SegmentDataset import data_extraction
from Models import SegErrConvClassifier
import ctypes
import os


def use_trained_model_to_do_evaluation(
        # model specific parameters
        model: any,
        model_file: str,
        # image specific parameters
        src_tif_image: str,
        ref_seg_file: str,
        ref_seg_pts_file: str,
        # input seg specific parameters (set according to the input segmentation result)
        test_seg_file: str,  # input
        matched_ref_eval_seg_csv_file: str,  # output
        imagette_dir: str,  # input and output
        imagette_mask_dir: str,  # input and output
        result_record_file: str,  # output
        ose_map_file: str,  # output
        use_map_file: str,  # output
        # overall default parameters
        dll_file: str = '',
        segment_image_name_template: str = 'segment_imagette_head_pixel_x_%d_y_%d.tif',
        buffer_size: int = 5,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> ():
    lib = ctypes.CDLL(dll_file)

    # 1. match reference segments and the segments in the segmentation result
    lib.MATCH_REF_SEG_AND_TEST_SEG(
        ctypes.c_wchar_p(ref_seg_file),
        ctypes.c_wchar_p(ref_seg_pts_file),
        ctypes.c_wchar_p(test_seg_file),
        ctypes.c_wchar_p(matched_ref_eval_seg_csv_file),
    )

    matched_seg_info = []
    with open(matched_ref_eval_seg_csv_file, mode='r') as f_pts:
        for idx, line in enumerate(f_pts):
            if 0 == idx:  # the first line is skipped
                continue
            line = line.strip('\n')
            elements = line.split(',')
            n_matched_seg = int(elements[0])
            info_list = []
            for idx_tmp in range(n_matched_seg):
                info_list.append(
                    (
                        int(elements[idx_tmp * 3 + 1]),
                        int(elements[idx_tmp * 3 + 2]),
                        int(elements[idx_tmp * 3 + 3])
                    )
                )
            matched_seg_info.append(info_list)

    # 2. # use the matching information to produce imagette and mask imagette for model evaluation
    lib.USE_MATCH_INFO_TO_PRODUCE_IMAGETTES_AND_MASK(
        ctypes.c_wchar_p(matched_ref_eval_seg_csv_file),
        ctypes.c_wchar_p(src_tif_image),
        ctypes.c_wchar_p(test_seg_file),
        ctypes.c_wchar_p(imagette_dir),
        ctypes.c_wchar_p(imagette_mask_dir),
        ctypes.c_wchar_p(segment_image_name_template),
        ctypes.c_int(buffer_size),
    )

    # 3. do evaluation
    checkpoint = torch.load(model_file, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    error_record = []
    area_ref_seg = []
    for info_list in matched_seg_info:
        rec_cur = []
        area_cur = 0
        for info in info_list:
            area, xv, yv = info[0], info[1], info[2]
            imagette_file = imagette_dir + segment_image_name_template % (xv, yv)
            imagette_mask_file = imagette_mask_dir + segment_image_name_template % (xv, yv)
            image, mask = data_extraction(imagette_file, imagette_mask_file,)
            prediction = model(image.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
            ose, use = prediction[0][1], prediction[0][2]
            rec_cur.append((area, float(ose), float(use)))
            area_cur += area
        error_record.append(rec_cur)
        area_ref_seg.append(area_cur)

    # record errors
    weighted_ose_list, weighted_use_list = [], []
    with open(result_record_file, mode='w+') as f_ret:
        f_ret.write('ose, use\n')
        for idx, error_record in enumerate(error_record):
            total_area = 0
            for item in error_record:
                total_area += item[0]
            weighted_ose = 0
            weighted_use = 0
            for item in error_record:
                weighted_ose += (float(item[0]) / total_area) * item[1]
                weighted_use += (float(item[0]) / total_area) * item[2]
            f_ret.write('%f, %f\n' % (weighted_ose, weighted_use))
            weighted_ose_list.append(weighted_ose)
            weighted_use_list.append(weighted_use)

    # produce error map (value)
    ose_list_values = (ctypes.c_float * len(weighted_ose_list))(*weighted_ose_list)
    lib.PRODUCE_SEGMENTATION_ERROR_DISTRIBUTION_MAP_VALUE(
        ctypes.c_wchar_p(ref_seg_file),
        ctypes.c_wchar_p(ref_seg_pts_file),
        ctypes.c_wchar_p(ose_map_file),
        ose_list_values,
    )
    use_list_values = (ctypes.c_float * (len(weighted_use_list)))(*weighted_use_list)
    lib.PRODUCE_SEGMENTATION_ERROR_DISTRIBUTION_MAP_VALUE(
        ctypes.c_wchar_p(ref_seg_file),
        ctypes.c_wchar_p(ref_seg_pts_file),
        ctypes.c_wchar_p(use_map_file),
        use_list_values,
    )

    # produce error map (tif)
    ose_list_values = (ctypes.c_float * (len(weighted_ose_list)))(*weighted_ose_list)
    lib.PRODUCE_SEGMENTATION_ERROR_DISTRIBUTION_MAP_TIF(
        ctypes.c_wchar_p(ref_seg_file),
        ctypes.c_wchar_p(ref_seg_pts_file),
        ctypes.c_wchar_p(ose_map_file + '.tif'),
        ose_list_values,
    )
    use_list_values = (ctypes.c_float * (len(weighted_use_list)))(*weighted_use_list)
    lib.PRODUCE_SEGMENTATION_ERROR_DISTRIBUTION_MAP_TIF(
        ctypes.c_wchar_p(ref_seg_file),
        ctypes.c_wchar_p(ref_seg_pts_file),
        ctypes.c_wchar_p(use_map_file + '.tif'),
        use_list_values,
    )

    # compute fused over-segmentation error (fose) and fused under-segmentation error (fuse)
    whole_ref_area = 0
    for item in area_ref_seg:
        whole_ref_area += item
    fose, fuse = 0., 0.
    for idx in range(len(weighted_ose_list)):
        fose += area_ref_seg[idx] * weighted_ose_list[idx]
        fuse += area_ref_seg[idx] * weighted_use_list[idx]
    fose /= whole_ref_area
    fuse /= whole_ref_area
    print('fused over-segmentation error (FOSE) = %f' % fose)
    print('fused under-segmentation error (FUSE) = %f' % fuse)
    return fose, fuse


def do_evaluation_experiments_for_a_series_of_segmentations(
        src_tif_image: str = 'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\image_bip_display.tif',
        ref_seg_file: str = 'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\val_data\\segmentation_result.seg',
        ref_seg_pts_file: str = 'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\val_data\\segment_location_information.csv',
        model_file: str = 'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\trained_model_seg_err_cls\\net_epoch_464.pth',
        work_dir: str = 'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segmentation results mrs_gmbf\\',
        work_sub_dir: str = 'deep_seg_error_eval',
        report_txt_file: str = 'deep_seg_error_eval_report.txt',
        buffer_size: int = 5,
):
    test_seg_files = ['seg_scale_%d' % s for s in range(10, 170, 20)]
    error_records = []
    result_dir = work_dir + work_sub_dir + '\\'
    os.makedirs(result_dir, exist_ok=True)
    for seg_file in test_seg_files:
        scale_specific_eval_data_dir = result_dir + 'eval_data\\' + seg_file + '\\'
        os.makedirs(scale_specific_eval_data_dir, exist_ok=True)
        os.makedirs(scale_specific_eval_data_dir + '\\imagette\\', exist_ok=True)
        os.makedirs(scale_specific_eval_data_dir + '\\imagette mask\\', exist_ok=True)
        fose, fuse = use_trained_model_to_do_evaluation(
            # model param
            model=SegErrConvClassifier(3, 1, 7, 96, 3),
            model_file=model_file,
            # image param
            src_tif_image=src_tif_image,
            ref_seg_file=ref_seg_file,
            ref_seg_pts_file=ref_seg_pts_file,
            # seg param
            test_seg_file=work_dir + seg_file + '.seg',
            matched_ref_eval_seg_csv_file=work_dir + 'deep_seg_error_eval\\' + seg_file + '_matched_ref_seg_loc_info.csv',
            imagette_dir=scale_specific_eval_data_dir + '\\imagette\\',
            imagette_mask_dir=scale_specific_eval_data_dir + '\\imagette mask\\',
            result_record_file=result_dir + seg_file + '_deep_seg_error_val_report.txt',
            ose_map_file=result_dir + seg_file + '_deep_seg_error_ose_map',
            use_map_file=result_dir + seg_file + '_deep_seg_error_use_map',
            # default param
            dll_file='D:\\docs\\code\\C_CPP_notcopy\\ListSegmentEngine\\x64\\Release\\ListSegmentEngine.dll',
            segment_image_name_template='segment_imagette_head_pixel_x_%d_y_%d.tif',
            buffer_size=buffer_size,
        )
        error_records.append((fose, fuse))

    with open(work_dir + report_txt_file, mode='w+') as f:
        for idx, s in enumerate(range(10, 170, 20)):
            f.write('scale=%d, fose=%f, fuse=%f\n' % (s, error_records[idx][0], error_records[idx][1]))


if '__main__' == __name__:
    # experiment for jilin1
    do_evaluation_experiments_for_a_series_of_segmentations(
        src_tif_image='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\image_bip_display.tif',
        ref_seg_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\val_data\\segmentation_result.seg',
        ref_seg_pts_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\val_data\\segment_location_information.csv',
        model_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\trained_model_seg_err_cls_10\\net_epoch_491.pth',
        work_dir='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segmentation results mrs_hac\\',
        buffer_size=10,
    )
    do_evaluation_experiments_for_a_series_of_segmentations(
        src_tif_image='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\image_bip_display.tif',
        ref_seg_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\val_data\\segmentation_result.seg',
        ref_seg_pts_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\val_data\\segment_location_information.csv',
        model_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\trained_model_seg_err_cls_10\\net_epoch_491.pth',
        work_dir='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segmentation results mrs_gmbf\\',
        buffer_size=10,
    )
    # experiment for gf2
    do_evaluation_experiments_for_a_series_of_segmentations(
        src_tif_image='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\image_bip_display.tif',
        ref_seg_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\val_data\\segmentation_result.seg',
        ref_seg_pts_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\val_data\\segment_location_information.csv',
        model_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\trained_model_seg_err_cls_10\\net_epoch_246.pth',
        work_dir='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segmentation results mrs_hac\\',
        buffer_size=10,
    )
    do_evaluation_experiments_for_a_series_of_segmentations(
        src_tif_image='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\image_bip_display.tif',
        ref_seg_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\val_data\\segmentation_result.seg',
        ref_seg_pts_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\val_data\\segment_location_information.csv',
        model_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\trained_model_seg_err_cls_10\\net_epoch_246.pth',
        work_dir='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segmentation results mrs_gmbf\\',
        buffer_size=10,
    )
    # transfer from jilin1 (model) to gf2 (data)
    do_evaluation_experiments_for_a_series_of_segmentations(
        src_tif_image='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\image_bip_display.tif',
        ref_seg_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\val_data\\segmentation_result.seg',
        ref_seg_pts_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\val_data\\segment_location_information.csv',
        model_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\trained_model_seg_err_cls_10\\net_epoch_491.pth',
        work_dir='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segmentation results mrs_hac\\',
        work_sub_dir='deep_seg_error_eval_transfer_from_jilin1',
        report_txt_file='deep_seg_error_eval_report_transfer_from_jilin1.txt',
        buffer_size=10,
    )
    do_evaluation_experiments_for_a_series_of_segmentations(
        src_tif_image='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\image_bip_display.tif',
        ref_seg_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\val_data\\segmentation_result.seg',
        ref_seg_pts_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\val_data\\segment_location_information.csv',
        model_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\trained_model_seg_err_cls_10\\net_epoch_491.pth',
        work_dir='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\segmentation results mrs_gmbf\\',
        work_sub_dir='deep_seg_error_eval_transfer_from_jilin1',
        report_txt_file='deep_seg_error_eval_report_transfer_from_jilin1.txt',
        buffer_size=10,
    )
    # transfer from gf2 (model) to jilin1 (data)
    do_evaluation_experiments_for_a_series_of_segmentations(
        src_tif_image='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\image_bip_display.tif',
        ref_seg_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\val_data\\segmentation_result.seg',
        ref_seg_pts_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\val_data\\segment_location_information.csv',
        model_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\trained_model_seg_err_cls_10\\net_epoch_246.pth',
        work_dir='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segmentation results mrs_hac\\',
        work_sub_dir='deep_seg_error_eval_transfer_from_gf2',
        report_txt_file='deep_seg_error_eval_report_transfer_from_gf2.txt',
        buffer_size=10,
    )
    do_evaluation_experiments_for_a_series_of_segmentations(
        src_tif_image='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\image_bip_display.tif',
        ref_seg_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\val_data\\segmentation_result.seg',
        ref_seg_pts_file='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\val_data\\segment_location_information.csv',
        model_file='D:\\data\\data_deep_seg_error\\201502_wuhan_gf2\\trained_model_seg_err_cls_10\\net_epoch_246.pth',
        work_dir='D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segmentation results mrs_gmbf\\',
        work_sub_dir='deep_seg_error_eval_transfer_from_gf2',
        report_txt_file='deep_seg_error_eval_report_transfer_from_gf2.txt',
        buffer_size=10,
    )
