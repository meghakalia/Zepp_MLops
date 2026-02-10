import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

weights = {
    'sleep_duration': 0.3,
    'sleep_start_time': 0.1,
    'deep_sleep_ratio': 0.2,
    'wakeup_count': 0.1,
    'awake_duration': 0.1
}


def compute_sleep_duration_score(age, sleep_duration):
    """
    计算睡眠时长得分。

    参数:
        age (int): 用户年龄。
        sleep_duration (int): 睡眠时长，单位为分钟。

    返回:
        float: 睡眠时长得分 (0-100)。
    """
    # 根据年龄确定mu
    if age <= 5:
        mu = 12
    elif age <= 13:
        mu = 10
    elif age <= 17:
        mu = 9
    elif age <= 64:
        mu = 8
    else:
        mu = 7.5

    sigma = 3
    sleep_hours = sleep_duration / 60.0
    tmp = -((sleep_hours - mu) ** 2) / (2 * sigma ** 2)
    score_initial = math.exp(tmp)
    score = score_initial * weights['sleep_duration']
    score = max(0, min(score * 100, 100))  # 将分数限制在0到100之间
    return score, max(0, min(score_initial * 100, 100))


def compute_sleep_start_time_score(sleep_start_time):
    """
    计算入睡时间得分。

    参数:
        sleep_start_time (float): 入睡时间，单位为小时（24小时制）。

    返回:
        float: 入睡时间得分 (0-100)。
    """
    sleep_hour = sleep_start_time / 60

    if 26 < sleep_hour:
        score_initial = 2 / 15
        score = score_initial * weights['sleep_start_time']
    else:
        mu = 22.0  # 22:00
        sigma = 2.0
        tmp = -((sleep_hour - mu) ** 2) / (2 * sigma ** 2)
        score_initial = math.exp(tmp)
        score = score_initial * weights['sleep_start_time']
        score = max(0, min(score * 100, 100))  # 将分数限制在0到100之间
    return score, max(0, min(score_initial * 100, 100))


def compute_deep_sleep_ratio_score(deep_sleep_ratio):
    """
    计算深睡比例得分。

    参数:
        deep_sleep_ratio (float): 深睡比例，范围0到1（即0%到100%）。

    返回:
        float: 深睡比例得分 (0-100)。
    """
    mu = 0  # 100%
    sigma = 0.0833
    tmp = -((deep_sleep_ratio - mu) ** 2) / (2 * sigma ** 2)
    score_initial = (1 - math.exp(tmp)) * 100
    score = score_initial * weights['deep_sleep_ratio']
    score = max(0, min(score, 100))  # 将分数限制在0到100之间
    return score, max(0, min(score_initial, 100))


def compute_wakeup_count_score(wakeup_count):
    """
    计算醒来次数得分。

    参数:
        wakeup_count (int): 醒来次数。

    返回:
        float: 醒来次数得分 (0-100)。
    """
    mu = 0
    sigma = 1.3333
    tmp = -((wakeup_count - mu) ** 2) / (2 * sigma ** 2)
    score_initial = math.exp(tmp)
    score = score_initial * weights['wakeup_count']
    score = max(0, min(score * 100, 100))
    return score, max(0, min(score_initial * 100, 100))


def compute_awake_duration_score(awake_duration):
    """
    计算醒来累积时长得分。

    参数:
        awake_duration (int): 醒来累积时长，单位为分钟。

    返回:
        float: 醒来累积时长得分 (0-100)。
    """
    mu = 0
    sigma = 40
    tmp = -((awake_duration - mu) ** 2) / (2 * sigma ** 2)
    score_initial = math.exp(tmp)
    score = score_initial * weights['awake_duration']
    score = max(0, min(score * 100, 100))
    return score, max(0, min(score_initial * 100, 100))


def calculate_sleep_score(age, sleep_duration, sleep_start_time, deep_sleep_ratio, wakeup_count, awake_duration):
    """
    计算总睡眠评分。

    参数:
        age (int): 用户年龄。
        sleep_duration (int): 睡眠时长，单位为分钟。
        sleep_start_time (float): 入睡时间，单位为小时（24小时制）。
        deep_sleep_ratio (float): 深睡比例，范围0到1。
        wakeup_count (int): 醒来次数。
        awake_duration (int): 醒来累积时长，单位为分钟。

    返回:
        float: 总睡眠评分，范围0到100。
    """
    sleep_start_time = convert_time_to_minutes(sleep_start_time)
    sleep_duration_score, sleep_duration_score_initial = compute_sleep_duration_score(age, sleep_duration)
    sleep_start_time_score, sleep_start_time_score_initial = compute_sleep_start_time_score(sleep_start_time)
    deep_sleep_ratio_score, deep_sleep_ratio_score_initial = compute_deep_sleep_ratio_score(deep_sleep_ratio)
    wakeup_count_score, wakeup_count_score_initial = compute_wakeup_count_score(wakeup_count)
    awake_duration_score, awake_duration_score_initial = compute_awake_duration_score(awake_duration)

    scalingFactor = 100.0 / (sum(weights.values())*100)


    score_component_names = ["sleep duration", "deep sleep ratio", "sleep start time",
                             "wakefulness after sleep onset (WASO) frequency",
                             "wakefulness after sleep onset (WASO) duration"]
    sleep_component_score = [sleep_duration_score, deep_sleep_ratio_score, sleep_start_time_score,
                             wakeup_count_score,
                             awake_duration_score]


    sleep_component_score = [max(0, min(int(score * scalingFactor), 100)) for score in sleep_component_score]
    total_score = sum(sleep_component_score)
    total_score = max(0, min(int(total_score), 100))


    return total_score, {'sleep_duration_score': sleep_duration_score_initial, 'sleep_start_time_score': sleep_start_time_score_initial,
                         'deep_sleep_ratio_score': deep_sleep_ratio_score_initial, 'WASO_frequency_score': wakeup_count_score_initial,
                         'WASO_duration_score': awake_duration_score_initial}


def convert_time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(":"))
    total_minutes = hours * 60 + minutes
    # 次日0-12点入睡+1440
    if total_minutes < 720:
        total_minutes += 1440
    return total_minutes


if __name__ == "__main__":
    # age = 34
    # sleep_duration = 347
    # sleep_start_time = 1098
    # deep_sleep_ratio = 72/380
    # wakeup_count = 1
    # awake_duration = 1
    folder_path = '../data/readiness_data_1111/user_score_data/'
    userid = '1069129974'
    df_user = pd.read_csv(f"{folder_path}{userid}.csv")
    age = df_user["age"][0]

    # 初始化要存储的分数列（如果它们尚未存在的话）
    score_columns = [
        'sleep_duration_score',
        'deep_sleep_ratio_score',
        'start_time_score',
        'wakefulness_after_sleep_onset_frequency_score',
        'wakefulness_after_sleep_onset_duration_score'
    ]
    for col in score_columns:
        if col not in df_user.columns:
            df_user[col] = None  # 或者使用 NaN 进行初始化

    # 遍历 df_user 来计算并赋值
    for idx, row in df_user.iterrows():
        sleep_duration = row["sleep_duration"]
        sleep_start_time_str = row["sleep_start_time"]
        deep_sleep_ratio = row["deep_sleep_ratio"]
        wakeup_count = row["wakefulness_after_sleep_onset_frequency"]
        awake_duration = row["wakefulness_after_sleep_onset_duration"]

        # 转换时间和计算分数
        # sleep_start_time = convert_time_to_minutes(sleep_start_time_str)
        total_score, scores = calculate_sleep_score(age, sleep_duration, sleep_start_time_str, deep_sleep_ratio,
                                                    wakeup_count, awake_duration)
        # 输入数据按行打印
        print("age:", age)
        print("sleep_duration:", sleep_duration)
        print("sleep_start_time:", sleep_start_time_str)
        print("deep_sleep_ratio:", deep_sleep_ratio)
        print("wakeup_count:", wakeup_count)
        print("awake_duration:", awake_duration)

        print("total score:", total_score)
        for key, value in scores.items():
            print(f"{key}: {value}")
            df_user.at[idx, key] = value
        df_user.at[idx, "total_score"] = total_score
    df_user['score_diff'] = df_user['total_score'] - df_user['sleep_score']
    df_user['score_diff'] = df_user['score_diff'].apply(lambda x: abs(x))
    print(df_user['score_diff'].describe())
    print('pass')