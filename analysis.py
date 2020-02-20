import math
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# cap_result = pyshark.FileCapture("./data/raw/test1.pcap")
plt.rcParams['font.sans-serif'] = ['STXihei']  # （替换sans-serif字体）显示中文


# 画一下长度-时间散点图
def plt_scatter(original_data: pd.DataFrame, background_traffic: pd.DataFrame):
    plt.figure(figsize=[9, 6])
    plt.subplot(2, 1, 1)
    plt.scatter(original_data["Time"].values, original_data["Length"].values, marker=".", label='packets')
    plt.plot([0, original_data["Time"].values[-1]], [50, 50], c='k', label='len = 50')
    plt.plot([0, original_data["Time"].values[-1]], [70, 70], c='k', label='len = 70')
    plt.legend(loc='best')
    plt.title("An example of busy traffic flow")
    plt.xlabel("Time(s)")
    plt.ylabel("Packet Length")
    plt.subplot(2, 1, 2)
    plt.scatter(background_traffic["Time"].values, background_traffic["Length"].values, marker=".", c='red')
    plt.plot([0, original_data["Time"].values[-1]], [50, 50], c='k', label='len = 50')
    plt.plot([0, original_data["Time"].values[-1]], [70, 70], c='k', label='len = 70')
    plt.legend(loc='best')
    plt.title("An example of background traffic flow")
    plt.xlabel("Time(s)")
    plt.ylabel("Packet Length")
    plt.show()
    plt.scatter(original_data["Time"].values, original_data["Length"].values, marker=".")
    plt.scatter(background_traffic["Time"].values, background_traffic["Length"].values, marker=".", c='red')
    plt.plot([0, original_data["Time"].values[-1]], [50, 50], c='k', label='len = 50')
    plt.plot([0, original_data["Time"].values[-1]], [70, 70], c='k', label='len = 70')
    plt.legend(loc='best')
    plt.title("An example of traffic flow")
    plt.xlabel("Time(s)")
    plt.ylabel("Packet Length")
    plt.show()


def cal_flow_density(flow_time: np.ndarray, win_size: np.float64):
    """
    计算流量密度
    :param flow_time:
    :param win_size:
    :return: 以Hz为单位
    """
    return flow_time.shape[0] / win_size


# the first order difference of F TD
def cal_1st_order_diff(flow: np.ndarray):
    """
    从原始时间序列计算序列的时间延迟
    :param flow: 原始输入的流
    :return: 结果向量 delta_flow_packet_time
    """
    delta_flow_packet_time = [0.0, ]
    for index in range(1, flow.shape[0]):
        delta = flow[index] - flow[index - 1]
        delta_flow_packet_time.append(delta)
    delta_flow_packet_time = np.array(delta_flow_packet_time)
    return delta_flow_packet_time


# calculate global features -- descriptive statistics
# including standard deviation,
# median, minimum, skewness, kurtosis, and standard error
def cal_descriptive_features(flow: np.ndarray):
    """
    计算descriptive statistics
    结果应该是一个向量，按照标准差、中位数、最小值、偏度、丰度、标准误差
    :param flow: 输入的流
    :return: 结果向量
    """
    # TODO: 标准误差（均方根误差）可以使用
    #  rmse = sqrt(sklearn.metrics.mean_squared_error(y_actual, y_predicted))来计算
    #  但是这地方只有一个向量，没办法计算

    """
    最小值没有意义，每个窗口的最小值必然是54.0 Bytes
    """
    return [np.std(flow, ddof=1), np.median(flow),
            pd.Series(flow).skew(), pd.Series(flow).kurt()]


# calculate local features
def cal_variance(flow: np.ndarray):
    """
    计算流前后方向的方差
    :param flow:
    :return: Variances in backward and forward directions
    """
    variances = []
    quartile = flow.shape[0] / 4
    ob_pos = [math.floor(quartile), math.floor(quartile * 2), math.floor(quartile * 3)]
    for each in ob_pos:
        var_backward = np.var(flow[0: each])
        var_forward = np.var(flow[each:])
        variances.append(var_backward)
        variances.append(var_forward)
    return variances


def cal_hopping_counts(flow: np.ndarray, threshold: int):
    """
    计算数据包大小变化跳数
    :param flow: 数据长度的列表
    :param threshold: 阈值，超过阈值跳数++
    :return: 跳数 as a feature
    """
    hopping_count = 0
    for idx in range(0, flow.shape[0] - 1):
        if flow[idx] - flow[idx + 1] > threshold:
            hopping_count += 1
    return hopping_count


def feature_extraction(window_len: np.ndarray, window_time: np.ndarray):
    f_vector = [cal_descriptive_features(window_len), cal_descriptive_features(window_time),
                cal_variance(window_len), cal_variance(window_time), [cal_hopping_counts(window_len, 800)]]
    f_vector = np.concatenate(f_vector, axis=0)
    return f_vector


"""
时间窗口Wn
窗口长度固定--窗口内最后一个包的到达时间 - 第0个包的到达时间 <= 窗口长度
时间间隔--下一个窗口的第0个包的达到时间 - 本窗口的最后一个包的到达时间 <= delta

每一个时间窗口都会被表示为一个向量 Vn = FeatureExtraction(Wn)
"""


def distance_metric(v: np.ndarray, u: np.ndarray, std_dev: np.ndarray):
    """
    这个函数衡量任意两个特征项向量u, v之间的距离
    其中sigma是v向量的标准差
    :param v: 第一个特征向量
    :param u: 第二个特征向量
    :param std_dev: 所有特征值的标准差
    :return: 距离值
    """
    dist = 0
    for idx in range(0, v.shape[0]):
        dist += np.exp(np.square(v[idx] - u[idx]) / std_dev[idx])
    return dist


def drop_bg_traffic(windows_flow: np.ndarray):
    """
    把背景流量从窗口中分离出来
    :param windows_flow:
    :return:
    """
    to_be_deleted = []
    for idx in range(0, windows_flow.shape[0]):
        if windows_flow[idx].shape[0] <= 16 and \
                packets_length_counter(windows_flow[idx], np.float64(100.0)) <= 5:
            to_be_deleted.append(idx)
    windows_flow = np.delete(windows_flow, to_be_deleted)
    return windows_flow


def packets_length_counter(win: np.ndarray, threshold: np.float64):
    """
    计算一个窗口中，包长度大于threshold的包的个数
    :param win: 一个窗口--ndarray
    :param threshold: 包大小阈值
    :return: 包长度大于threshold的包的个数
    """
    counter = 0
    for each in win:
        if each[0] > threshold:
            counter += 1
    return counter


# TODO: Part 3 中的流量分段——聚类
WINDOW_SIZE = 0.05
TIME_GAP = 1


def segment_by_window(original_data: np.ndarray, win_size: np.float64):
    """
    对原始数据根据窗口大小分段
    :param original_data: 原始数据，应该是一个二维ndarray
    :param win_size: 窗口大小，单位是秒
    :return: 分窗后的数据
    """
    segments = []
    target_time = win_size
    last_index = 0
    for idx in range(0, original_data.shape[0]):
        if original_data[idx][2] > target_time:
            seg = copy.deepcopy(original_data[last_index: idx])
            segments.append(seg)
            last_index = idx
            target_time += win_size
    segments.append(copy.deepcopy(original_data[last_index:]))
    segments = np.array(segments)
    return segments


def cal_distances(segmented_flow: np.ndarray):
    feature_list = []
    for idx in range(0, segmented_flow.shape[0]):
        feature = feature_extraction(segmented_flow[idx].T[0], segmented_flow[idx].T[1])
        feature_list.append(feature)
    feature_list = np.array(feature_list)

    std_scale = StandardScaler()
    feature_list = std_scale.fit_transform(feature_list)

    std_dev_list = []
    for each in feature_list.T:
        std_dev_list.append(np.std(each, ddof=1))

    dist_list = []
    for idx in range(0, len(feature_list) - 1):
        dist_list.append(distance_metric(feature_list[idx], feature_list[idx + 1], np.array(std_dev_list)))
    return feature_list, dist_list, std_dev_list


# def connectivity_clustering(segmented_flow: np.ndarray, threshold: np.float64):
#


if __name__ == "__main__":
    cap_result = pd.read_csv("./data/raw/video-picture.csv")
    bg_traffic = pd.read_csv("./data/raw/background-traffic.csv")

    # 获取所有TCP包的信息，按照到达时间排序
    tcp_result = copy.deepcopy(cap_result[cap_result.Protocol == 'TCP'].reset_index(drop=True))
    tcp_result.sort_values(axis=0, by="Time")

    bg_traffic = copy.deepcopy(bg_traffic[bg_traffic.Protocol == 'TCP'].reset_index(drop=True))
    bg_traffic.sort_values(axis=0, by="Time")

    # 决定取所有数据中比较密集的一部分来测试一下
    # 64.8(Index: 2564) ~  65.46614(Index: 3892)
    # View Construction
    # test_data = tcp_result[2564: 3892]
    # flow_packet_len = test_data["Length"].values  # F PL
    # flow_packet_time = test_data["Time"].values  # F TD

    feature_time = cal_1st_order_diff(tcp_result["Time"].values)
    data = [[tcp_result["Length"].values], [feature_time], [tcp_result["Time"]]]
    data = np.concatenate(data, axis=0).T
    windows = segment_by_window(data, np.float64(WINDOW_SIZE))
    windows = drop_bg_traffic(windows)

    features, distances, std_deviation = cal_distances(windows)
