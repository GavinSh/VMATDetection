import random
from typing import List, Tuple

import cv2
import numpy as np
import openpyxl

import DataTemplate

LEAF_MOVE_MAX_POSITION = 2000
THIN_LEAF_THICKNESS_PIXEL = 1
SUB_FIELD_SIZE_PIXEL = 80


def get_leaf_thickness_pixel_list(treatment_room: DataTemplate.TreatmentRoom):
    if treatment_room in [DataTemplate.TreatmentRoom.R4_Novalis, DataTemplate.TreatmentRoom.R5_Unique]:
        return (
                [THIN_LEAF_THICKNESS_PIXEL * 2] * 10
                + [THIN_LEAF_THICKNESS_PIXEL] * 40
                + [THIN_LEAF_THICKNESS_PIXEL * 2] * 10
        )
    else:
        return [THIN_LEAF_THICKNESS_PIXEL] * 80


def get_VMAT_data_from_execl(xlsx_path: str):
    work_book = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    data_generator = work_book['Sheet1'].values
    # last_patient_id: str = ''
    last_treatment_id: str = ''
    treatment_data_list: List[DataTemplate.TreatmentData] = []
    treatment_data: DataTemplate.TreatmentData = DataTemplate.TreatmentData()
    treatment_arc: DataTemplate.Arc = DataTemplate.Arc()

    for row_idx, row_data in enumerate(data_generator):
        if row_idx == 0:
            continue
        patient_id = row_data[0]
        treatment_id = row_data[9]
        arc_name = row_data[4]
        if treatment_id != last_treatment_id or row_data[3] is None:
            if treatment_data.num_leaf != 0:
                treatment_data.arcs.append(treatment_arc)
                treatment_data.num_arc += 1
                treatment_data_list.append(treatment_data)

            if row_data[3] is None:
                break

            treatment_data = DataTemplate.TreatmentData()
            treatment_data.patient_ID = patient_id
            treatment_data.patient_name = row_data[1]
            treatment_data.treatment_ID = treatment_id
            treatment_data.treatment_room = DataTemplate.TreatmentRoom[row_data[2]]
            treatment_data.treatment_type = DataTemplate.RadiotherapyType[row_data[3]]

            treatment_data.organ_code = row_data[8]
            treatment_data.organ_name = row_data[11]
            treatment_data.treatment_name = row_data[10]

            treatment_data.num_leaf = (len(row_data) - 15) // 2

            treatment_arc = DataTemplate.Arc()
            treatment_arc.arc_name = arc_name
            treatment_arc.prescription_dose = row_data[6]

            last_treatment_id = treatment_id
        if arc_name != treatment_arc.arc_name:
            # treatment_arc.num_field = len(treatment_arc.fields)
            treatment_data.arcs.append(treatment_arc)
            treatment_data.num_arc += 1
            treatment_arc = DataTemplate.Arc()
            treatment_arc.arc_name = arc_name
            treatment_arc.prescription_dose = row_data[6]
        treatment_field: DataTemplate.Field = DataTemplate.Field()
        treatment_field.num_sub_field = 1
        treatment_field.gantry_idx = row_data[5]
        treatment_field.gantry_angle = row_data[12]
        treatment_field.gantry_rotation_direction = DataTemplate.FieldRotationDirection(row_data[13])

        treatment_sub_field: DataTemplate.SubField = DataTemplate.SubField()
        treatment_sub_field.dose = row_data[7]
        treatment_sub_field.coll_angle = row_data[14]
        treatment_sub_field.equivalent_dose = treatment_sub_field.dose / treatment_arc.prescription_dose * DataTemplate.FIXED_PRESCRIPTION_DOSE
        leaf_position_list = list(row_data[15:])
        assert len(leaf_position_list) == treatment_data.num_leaf * 2
        treatment_sub_field.leaf_position = np.zeros([treatment_data.num_leaf, 2])
        treatment_sub_field.leaf_position[:, 0] = leaf_position_list[: treatment_data.num_leaf]
        treatment_sub_field.leaf_position[:, 1] = leaf_position_list[treatment_data.num_leaf:]
        treatment_field.sub_fields.append(treatment_sub_field)
        treatment_arc.fields.append(treatment_field)
        treatment_arc.num_field += 1
    return treatment_data_list


def convert_aperture_map_list(
        treatment_room: DataTemplate.TreatmentRoom,
        arc_data: DataTemplate.Arc,
        random_move: bool = False
):
    """
    单个Arc中各个控制点的孔径图和跳数组合为List
    :param treatment_room: 加速器室，用于判断叶片厚度规格
    :param arc_data:
    :param random_move: 数据增强用的孔径图随机偏移
    :return:
    """
    concat_aperture_map: List[np.ndarray] = []
    concat_dose = []
    last_dose = 0
    if random_move:
        top_bottom_move_vector = random.randint(-5, 5)
    for field in arc_data.fields:
        for sub_field in field.sub_fields:
            aperture_map = np.ones([SUB_FIELD_SIZE_PIXEL, LEAF_MOVE_MAX_POSITION * 2])
            sum_dose = sub_field.equivalent_dose
            cur_dose = sum_dose - last_dose
            last_dose = sum_dose
            leaf_thick_list = get_leaf_thickness_pixel_list(treatment_room)

            leaf_top_position_pixel = 0
            for leaf_idx, leaf_thick in enumerate(leaf_thick_list):
                leaf_bottom_position_pixel = leaf_top_position_pixel + leaf_thick
                if sub_field.leaf_position[leaf_idx, 0] == -25 and sub_field.leaf_position[leaf_idx, 1] == 25:
                    aperture_map[leaf_top_position_pixel: leaf_bottom_position_pixel, :] = 0
                else:
                    left_leaf_position = int(sub_field.leaf_position[leaf_idx, 0]) + LEAF_MOVE_MAX_POSITION
                    right_leaf_position = int(sub_field.leaf_position[leaf_idx, 1]) + LEAF_MOVE_MAX_POSITION
                    if random_move:
                        left_right_move_vector = random.randint(-1, 1) * 50
                        left_leaf_position += left_right_move_vector
                        right_leaf_position += left_right_move_vector
                    aperture_map[leaf_top_position_pixel: leaf_bottom_position_pixel, 0: left_leaf_position] = 0
                    aperture_map[leaf_top_position_pixel: leaf_bottom_position_pixel, right_leaf_position:] = 0
                leaf_top_position_pixel += leaf_thick

            if random_move:
                ex_aperture_map = np.zeros([SUB_FIELD_SIZE_PIXEL + 10, LEAF_MOVE_MAX_POSITION * 2])
                ex_aperture_map[5: -5, :] = aperture_map
                top_position = 5 + top_bottom_move_vector
                bottom_position = -5 + top_bottom_move_vector
                if bottom_position == 0:
                    aperture_map = ex_aperture_map[top_position:, :]
                else:
                    aperture_map = ex_aperture_map[top_position: bottom_position, :]

            aperture_map = cv2.resize(aperture_map, [SUB_FIELD_SIZE_PIXEL, SUB_FIELD_SIZE_PIXEL])
            # 根据小机头角度进行旋转
            rotated_matrix = cv2.getRotationMatrix2D(center=[SUB_FIELD_SIZE_PIXEL/2, SUB_FIELD_SIZE_PIXEL/2], angle=sub_field.coll_angle, scale=1)
            rotated_aperture_map = cv2.warpAffine(src=aperture_map, M=rotated_matrix, dsize=[SUB_FIELD_SIZE_PIXEL, SUB_FIELD_SIZE_PIXEL], flags=cv2.INTER_NEAREST)
            concat_aperture_map.append(rotated_aperture_map)
            concat_dose.append(cur_dose)
    return concat_aperture_map, concat_dose


def convert_sum_heat_map(treatment_data: DataTemplate.TreatmentData, random_move: bool = False, **kwargs):
    """
    生成2维热图的方法
    各个控制点的剂量图直接相加计算累积图，直接使用每个孔径图
    孔径图1 * 剂量1 + 孔径图2 * 剂量2 + ... + 孔径图n * 剂量n
    :param treatment_data:
    :param random_move: 数据增强用的孔径图随机偏移
    :return:
    """
    sum_heat_map_list: List[np.ndarray] = []
    for arc in treatment_data.arcs:
        aperture_map_list, dose_list = convert_aperture_map_list(treatment_data.treatment_room, arc, random_move)
        sum_heat_map = np.sum([aperture_map * dose for aperture_map, dose in zip(aperture_map_list, dose_list)], axis=0)
        sum_heat_map_list.append(sum_heat_map)
    return sum_heat_map_list


def convert_concat_heat_map(treatment_data: DataTemplate.TreatmentData, concat_shape: Tuple[int, int] = None,
                            random_move: bool = False, **kwargs):
    """
    生成2维热图的方法
    各个控制点的剂量图concat连结为二维大图，直接使用每个孔径图
    [[孔径图1 * 剂量1, 孔径图2 * 剂量2, ..., 孔径图m * 剂量m],
    [孔径图m+1 * 剂量m+1, 孔径图m+2 * 剂量m+2, ..., 孔径图2m * 剂量2m],
    ...
    [孔径图mn-m+1 * 剂量mn-m+1, 孔径图mn-m+2 * 剂量mn-m+2, ..., 孔径图mn * 剂量mn]]
    :param concat_shape: 转换后每行和每列heatmap的个数
    :param treatment_data:
    :param random_move: 数据增强用的孔径图随机偏移
    :return:
    """
    if concat_shape is None:
        concat_shape = [9, 10]
    concat_heat_map_list = []
    for arc in treatment_data.arcs:
        aperture_map_list, dose_list = convert_aperture_map_list(treatment_data.treatment_room, arc, random_move)
        concat_row_list = []
        for row_idx in range(concat_shape[0]):
            concat_row_list.append(
                np.concatenate(
                    [
                        aperture_map_list[heat_map_idx + 1] * dose_list[heat_map_idx + 1]
                        for heat_map_idx in range(concat_shape[1] * row_idx, concat_shape[1] * (row_idx + 1))
                    ],
                    axis=1
                )
            )
        concat_row = np.concatenate(concat_row_list, axis=0)
        concat_heat_map_list.append(concat_row)
    return concat_heat_map_list


def convert_sum_heat_map_diff(treatment_data: DataTemplate.TreatmentData, random_move: bool = False, **kwargs):
    """
    生成2维热图的方法
    各个控制点的剂量图直接相加计算累积图，使用孔径图变化值
    (孔径图1 - 孔径图0) * 剂量1 + (孔径图2 - 孔径图1) * 剂量2 + ... + (孔径图n - 孔径图n-1) * 剂量n
    :param treatment_data:
    :param random_move: 数据增强用的孔径图随机偏移
    :return:
    """
    sum_heat_map_list: List[np.ndarray] = []
    for arc in treatment_data.arcs:
        aperture_map_list, dose_list = convert_aperture_map_list(treatment_data.treatment_room, arc, random_move)
        sum_heat_map = np.sum([
            (aperture_map_list[map_idx] - aperture_map_list[map_idx - 1]) * dose_list[map_idx]
            for map_idx in range(1, len(dose_list))
        ], axis=0)
        sum_heat_map_list.append(sum_heat_map)
    return sum_heat_map_list


def convert_concat_heat_map_diff(treatment_data: DataTemplate.TreatmentData, concat_shape: Tuple[int, int] = None,
                                 random_move: bool = False,
                                 **kwargs):
    """
    生成2维热图的方法
    各个控制点的剂量图concat连结为二维大图，使用孔径图变化值
    [[(孔径图1 - 孔径图0) * 剂量1, (孔径图2 - 孔径图1) * 剂量2, ..., (孔径图m - 孔径图m-1) * 剂量m],
    [(孔径图m+1 - 孔径图m) * 剂量m+1, (孔径图m+2 - 孔径图m+1) * 剂量m+2, ..., (孔径图2m - 孔径图2m-1) * 剂量2m],
    ...
    [(孔径图mn-m+1 - 孔径图mn-m) * 剂量mn-m+1, (孔径图mn-m+2 - 孔径图mn-m+1) * 剂量mn-m+2, ..., (孔径图mn - 孔径图mn-1) * 剂量mn]]
    :param concat_shape: 转换后每行和每列heatmap的个数
    :param treatment_data:
    :param random_move: 数据增强用的孔径图随机偏移
    :return:
    """
    if concat_shape is None:
        concat_shape = [9, 10]
    concat_heat_map_list = []
    for arc in treatment_data.arcs:
        aperture_map_list, dose_list = convert_aperture_map_list(treatment_data.treatment_room, arc, random_move)
        concat_row_list = []
        for row_idx in range(concat_shape[0]):
            concat_row_list.append(
                np.concatenate(
                    [
                        (aperture_map_list[heat_map_idx + 1] - aperture_map_list[heat_map_idx]) * dose_list[
                            heat_map_idx + 1]
                        for heat_map_idx in range(concat_shape[1] * row_idx, concat_shape[1] * (row_idx + 1))
                    ],
                    axis=1
                )
            )
        concat_row = np.concatenate(concat_row_list, axis=0)
        concat_heat_map_list.append(concat_row)
    return concat_heat_map_list


def convert_concat_3d_heat_map(treatment_data: DataTemplate.TreatmentData, random_move: bool = False, **kwargs):
    """
    生成3维热图的方法
    各个控制点的剂量图concat连结为三维图，直接使用每个孔径图
    [孔径图1 * 剂量1, 孔径图2 * 剂量2, ..., 孔径图n * 剂量n]
    :param treatment_data:
    :param random_move: 数据增强用的孔径图随机偏移
    :return:
    """
    sum_heat_map_list: List[np.ndarray] = []
    for arc in treatment_data.arcs:
        aperture_map_list, dose_list = convert_aperture_map_list(treatment_data.treatment_room, arc, random_move)
        sum_heat_map = np.concatenate([
            np.expand_dims(aperture_map * dose, axis=0)
            for aperture_map, dose in zip(aperture_map_list[1:], dose_list[1:])
        ], axis=0)
        sum_heat_map_list.append(sum_heat_map)
    return sum_heat_map_list


if __name__ == '__main__':
    treatment_list = get_VMAT_data_from_execl('data/version_2/rectum_data.xlsx')
    for treatment_idx, treatment in enumerate(treatment_list):
        heat_map_list = convert_sum_heat_map(treatment)
        # heat_map_list = convert_concat_3d_heat_map(treatment)
        for map_idx, heat_map in enumerate(heat_map_list):
            heat_map = cv2.resize(heat_map, (500, 500), cv2.INTER_CUBIC)
            cv2.imshow('treatment_{0}_arc_{1}.png'.format(treatment.patient_ID, treatment.arcs[map_idx].arc_name),
                       ((heat_map - np.min(heat_map)) / (
                               np.max(heat_map) - np.min(heat_map)) * 255).astype(np.uint8))
            cv2.waitKey(0)
