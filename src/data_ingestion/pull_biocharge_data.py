import requests
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import numpy as np
import os
import openpyxl
import warnings
import struct
import concurrent.futures
from tqdm import tqdm
import ast # Used for converting string-lists back to lists in offline mode
from zoneinfo import ZoneInfo # Python 3.9+
from concurrent.futures import ThreadPoolExecutor
import re

warnings.filterwarnings("ignore")

# Global tokens loaded from environment variables.
# Set XIAOMI_SUPERTOKEN and XIAOMI_TOKEN in your .env file.
supertoken = os.environ.get('XIAOMI_SUPERTOKEN', '')
token = os.environ.get('XIAOMI_TOKEN', supertoken)

# ==============================================================================
# SECTION 1: DATA DECODING & UTILITY FUNCTIONS
# This section contains all data decoding, time conversion, and helper functions.
# ==============================================================================

def decodeHrData(text):
    """Decodes the heartRateData field."""
    if not isinstance(text, str) or not text:
        return [255] * 1440
    import base64
    try:
        decodeBytes = base64.b64decode(text)
        if len(decodeBytes) != 1440:
            return []
    except Exception:
        return []
    return [b & 0xff for b in decodeBytes]


def decodeRawDataActiveAndStep(b):
    return b & 0xff if ((b & 0x80) == 0x80) else b


def decodeRawData(text):
    """Decodes the mergedRawData field."""
    if not isinstance(text, str) or not text:
        return [[], [], []]
    import base64
    try:
        decodeBytes = base64.b64decode(text)
        if len(decodeBytes) < 4320 or len(decodeBytes) % 1440 != 0:
            return [[], [], []]
    except Exception:
        return [[], [], []]
    times = len(decodeBytes) // 1440
    mode_out_list = [decodeBytes[i * times] for i in range(1440)]
    active_list = [decodeRawDataActiveAndStep(decodeBytes[i * times + 1]) for i in range(1440)]
    steps_list = [decodeRawDataActiveAndStep(decodeBytes[i * times + 2]) for i in range(1440)]
    return [mode_out_list, active_list, steps_list]

def merge_nap_segments(odd_stage):
    if not odd_stage:
        return []
    naps = []
    current_nap = None

    for entry in odd_stage:
        start, stop = entry['start'], entry['stop']
        if current_nap is None:
            current_nap = {'start': start, 'stop': stop}
        else:
            if start - current_nap['stop'] <= 15:
                current_nap['stop'] = stop
            else:
                naps.append({
                    'start': current_nap['start'] % 1440,
                    'stop': current_nap['stop'] % 1440,
                    'duration': current_nap['stop'] - current_nap['start']
                })
                current_nap = {'start': start, 'stop': stop}

    if current_nap is not None:
        naps.append({
            'start': current_nap['start'] % 1440,
            'stop': current_nap['stop'] % 1440,
            'duration': current_nap['stop'] - current_nap['start']
        })
    return naps

def format_sleep_minute_offset(minute_offset):
    """
    Formats a minute offset (which can be negative) into HH:MM format.
    Negative values are shifted to represent times from the previous day.
    Returns string in 24-hour format, e.g. '22:47', never negative.
    """
    minute_offset = int(minute_offset)
    # If negative, shift to previous day by adding 1440 minutes (24 hours)
    if minute_offset < 0:
        minute_offset = (minute_offset + 1440) % 1440
    # If greater than 1440, wrap to next day
    elif minute_offset >= 1440:
        minute_offset = minute_offset % 1440
    
    hour = minute_offset // 60
    minute = minute_offset % 60
    return f"{hour:02}:{minute:02}"


def get_analysis_summary(summary):
    EPISON = 1e-10
    default_keys = [
        "sleep_score", "sleep_start_time", "sleep_end_time", "sleep_duration",
        "deep_sleep_duration", "light_sleep_duration", "rem_sleep_duration",
        "deep_sleep_ratio", "light_sleep_ratio", "rem_sleep_ratio",
        "wakefulness_after_sleep_onset_frequency", "wakefulness_after_sleep_onset_duration",
        "sleep_latency_minutes", "sleep_inertia_minutes", "nap_start_time",
        "nap_end_time", "nap_duration"
    ]
    if not isinstance(summary, dict) or not summary:
        return {k: np.nan for k in default_keys}
        
    slp_summary = summary.get('slp', {})
    if not slp_summary or 'stage' not in slp_summary or not slp_summary['stage']:
        return {k: np.nan for k in default_keys}
    
    start = slp_summary['stage'][0]['start']
    stop = slp_summary['stage'][-1]['stop']

    # sleep_start_timestamp = slp_summary['st']
    # sleep_end_timestamp = slp_summary['ed']
    # datetime_object_st = datetime.fromtimestamp(sleep_start_timestamp)
    # formatted_datetime_st = datetime_object_st.strftime("%Y-%m-%d %H:%M:%S")

    # datetime_object_ed = datetime.fromtimestamp(sleep_end_timestamp)
    # formatted_datetime_ed = datetime_object_ed.strftime("%Y-%m-%d %H:%M:%S")

    # sleep_duration = (datetime_object_ed - datetime_object_st) / 60

    sleep_duration = int(stop) - int(start)

    
    deep_sleep_minutes = slp_summary.get('dp', 0)
    light_sleep_minutes = slp_summary.get('lt', 0)
    rem_sleep_duration = sum(item['stop'] - item['start'] + 1 for item in slp_summary['stage'] if item.get('mode') == 8)
    
    deep_sleep_ratio = deep_sleep_minutes / (sleep_duration + EPISON) if sleep_duration > 0 else 0
    light_sleep_ratio = light_sleep_minutes / (sleep_duration + EPISON) if sleep_duration > 0 else 0
    rem_sleep_ratio = rem_sleep_duration / (sleep_duration + EPISON) if sleep_duration > 0 else 0

    nap_st, nap_ed, nap_duration = [], [], []
    odd_stage = slp_summary.get('odd_stage', [])
    if odd_stage:
        naps = merge_nap_segments(odd_stage)
        for item in naps:
            nap_st.append(f"{item['start'] // 60:02}:{item['start'] % 60:02}")
            nap_ed.append(f"{item['stop'] // 60:02}:{item['stop'] % 60:02}")
            nap_duration.append(item['duration'])
    
    return {
        "sleep_score": slp_summary.get('ss', np.nan),
        "sleep_start_time": format_sleep_minute_offset(start),
        "sleep_end_time": format_sleep_minute_offset(stop),
        "sleep_duration": sleep_duration,
        "deep_sleep_duration": deep_sleep_minutes,
        "light_sleep_duration": light_sleep_minutes,
        "rem_sleep_duration": rem_sleep_duration,
        "deep_sleep_ratio": deep_sleep_ratio,
        "light_sleep_ratio": light_sleep_ratio,
        "rem_sleep_ratio": rem_sleep_ratio,
        "wakefulness_after_sleep_onset_frequency": slp_summary.get('wc', 0),
        "wakefulness_after_sleep_onset_duration": slp_summary.get('wk', 0),
        "sleep_latency_minutes": slp_summary.get('is', 0),
        "sleep_inertia_minutes": slp_summary.get('lb', 0),
        "nap_start_time": nap_st or np.nan,
        "nap_end_time": nap_ed or -np.nan,
        "nap_duration": nap_duration or np.nan,
    }

def get_sleep_data(stage, nap_stage):
    """
    Generates a 2880-minute array for sleep stages, combining main sleep and naps.
    This logic is from the original script to handle overnight sleep correctly.
    """
    sleep_data = [-1] * 2880
    # mode_values = {4: 2, 8: 1, 5: 0} # Deep, REM, Light # REVERTED: No mapping

    # Process main sleep stages
    for time_range in stage:
        start, stop, mode = time_range['start'], time_range['stop'], time_range['mode']
        if start < 1000: start += 1440
        if stop < 1000: stop += 1440
        # value = mode_values.get(mode, -1) # REVERTED:
        value = mode
        for i in range(start, stop + 1):
            if i < 2880: sleep_data[i] = value

    # Process nap stages (just keep original process, no location change for naps)
    for time_range in nap_stage:
        start, stop, mode = time_range['start'], time_range['stop'], time_range['mode']
        if start > 1440: start -= 1440
        if stop > 1440: stop -= 1440
        value = mode
        for i in range(start, stop + 1):
            if i < 2880:
                sleep_data[i] = value
    return sleep_data

def calculate_sleep_and_nap_states(times_list, nap_start_times, nap_end_times, sleep_stage_list, day_date_str):
    """
    Calculates minute-by-minute nap_state and sleep_state based on the original script's logic.
    Adapted to check for mapped sleep stages [0, 1, 2].
    """
    nap_intervals = []
    # Ensure nap_start_times is a list, not np.nan
    if nap_start_times and isinstance(nap_start_times, list):
        for start_str, end_str in zip(nap_start_times, nap_end_times):
            try:
                # Naps are always on the *previous* day's date context
                start_dt = pd.to_datetime(f"{day_date_str} {start_str}")
                end_dt = pd.to_datetime(f"{day_date_str} {end_str}")
                nap_intervals.append((start_dt, end_dt))
            except ValueError:
                continue

    nap_states = []
    sleep_states = []
    times_list_dt = pd.to_datetime(times_list)

    for i, time_point in enumerate(times_list_dt):
        # Check if the current time point falls within any nap interval
        is_nap = any(start <= time_point <= end for start, end in nap_intervals)
        nap_states.append(1 if is_nap else 0)
        
        # Check for main sleep using the raw mapped values
        # 4: Deep, 5: Light, 8: REM
        is_main_sleep = sleep_stage_list[i] in [4, 5, 8] # REVERTED: Check for raw modes
        
        if is_nap or is_main_sleep:
            sleep_states.append(1)
        else:
            sleep_states.append(0)

    return sleep_states, nap_states

def calculate_age(birthdate_str):
    try:
        birthdate = datetime.strptime(birthdate_str, "%Y-%m")
        current_date = datetime.now()
        age = current_date.year - birthdate.year - ((current_date.month, current_date.day) < (birthdate.month, birthdate.day))
        return age
    except:
        return np.nan

def convert_date_to_datetime_ms_str(date_string: str) -> str:
    shanghai_tz = ZoneInfo("Asia/Shanghai")
    dt_naive = datetime.strptime(date_string, "%Y-%m-%d")
    dt_aware = dt_naive.astimezone(shanghai_tz)
    timestamp_ms = int(dt_aware.timestamp() * 1000)
    return str(timestamp_ms)

def generate_time_range(start_time_str, end_time_str):
    time_format = "%Y-%m-%d %H:%M"
    try:
        start_time = datetime.strptime(start_time_str, time_format)
        end_time = datetime.strptime(end_time_str, time_format)
    except ValueError:
         return []
    time_list = []
    current_time = start_time
    while current_time <= end_time:
        time_list.append(current_time.strftime(time_format))
        current_time += timedelta(minutes=1)
    return time_list

# ==============================================================================
# SECTION 2: ATOMIC DATA FETCHING FUNCTIONS
# ==============================================================================

def resolve_timezone(tz_value, user_id):
    """
    Convert the API-returned timeZone value into a tzinfo for timestamp conversion.
    Supports IANA names (e.g., 'Asia/Shanghai'), offsets like '+08:00' / '-0500',
    or numeric offsets (seconds / milliseconds / hours). Falls back to a
    user-prefix-based default when unresolvable.
    """
    default_map = {
        '1': 'Asia/Shanghai',
        '3': 'America/New_York',
        '7': 'Europe/Berlin',
    }
    fallback_name = default_map.get(str(user_id)[0], 'UTC')

    def _offset_timezone_from_string(val):
        match = re.match(r'^([+-]?)(\d{1,2}):?(\d{2})?$', val)
        if not match:
            return None
        sign, hours, minutes = match.groups()
        hours = int(hours)
        minutes = int(minutes or 0)
        total_minutes = hours * 60 + minutes
        if sign == '-':
            total_minutes = -total_minutes
        return timezone(timedelta(minutes=total_minutes))

    # IANA name or offset string
    if isinstance(tz_value, str) and tz_value:
        try:
            return ZoneInfo(tz_value)
        except Exception:
            tz_offset = _offset_timezone_from_string(tz_value)
            if tz_offset:
                return tz_offset

    # Numeric offset (interpret heuristically)
    if isinstance(tz_value, (int, float)) and not pd.isna(tz_value):
        offset_seconds = int(tz_value)
        # If value looks like hours (e.g., 8), convert to seconds.
        if abs(offset_seconds) < 24:
            offset_seconds *= 3600
        # If value is very large, it might be milliseconds.
        elif abs(offset_seconds) > 24 * 3600 * 5:
            offset_seconds = offset_seconds // 1000
        try:
            return timezone(timedelta(seconds=offset_seconds))
        except Exception:
            pass

    # Fallback to prefix-based default or UTC
    try:
        return ZoneInfo(fallback_name)
    except Exception:
        return timezone.utc


def get_dynamic_data(userId, fromDate, toDate, token):
    headers = {'content-type': 'application/json', 'apptoken': token, 'appname': 'com.xiaomi.hm.health'}
    params = {'userId': userId, 'fromDate': fromDate, 'toDate': toDate, 'appName': 'com.xiaomi.hm.health'}
    urls = {
        'cn2': ['https://mifit-health-data-private-cn3.zepp.com/admin/user/originalData/'],
        'de':  ['https://mifit-health-data-private-de2.huami.com/admin/user/originalData/'],
        'us2': ['https://mifit-health-data-private-us2.huami.com/admin/user/originalData/', 'https://mifit-health-data-private-us3.zepp.com/admin/user/originalData/']
    }
    if str(userId).startswith('3'):
        user_region = 'us2'
    elif str(userId).startswith('7'):
        user_region = 'de'
        print('de')
    else:
        user_region = 'cn2'

    for url in urls[user_region]:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            res_json = response.json()
            if isinstance(res_json, list) and res_json:
                df = pd.DataFrame(res_json)
                df['dateTime'] = pd.to_datetime(df['dateTime'])
                df['date_only'] = df['dateTime'].dt.date
                df['userid_raw'] = df['uid']
                df['heartratedata_raw'] = df['heartRateData'].apply(decodeHrData)
                df['mode_raw'], df['active_raw'], df['step_raw'] = zip(*df['data'].apply(decodeRawData))
                df['summary_raw'] = df['summary'].apply(json.loads)

                full_date_range = pd.to_datetime(pd.date_range(start=fromDate, end=toDate, freq='D')).date
                scaffold = pd.DataFrame({'date_only': full_date_range})
                df['date_only'] = pd.to_datetime(df['date_only']).dt.date
                merged_df = pd.merge(scaffold, df, on='date_only', how='left')

                default_list = [255] * 1440
                merged_df['hr'] = merged_df['heartratedata_raw'].apply(lambda x: x if isinstance(x, list) and len(x) == 1440 else list(default_list))
                merged_df['mode'] = merged_df['mode_raw'].apply(lambda x: x if isinstance(x, list) and len(x) == 1440 else list(default_list))
                merged_df['active'] = merged_df['active_raw'].apply(lambda x: x if isinstance(x, list) and len(x) == 1440 else list(default_list))
                merged_df['step'] = merged_df['step_raw'].apply(lambda x: x if isinstance(x, list) and len(x) == 1440 else list(default_list))
                merged_df['summary'] = merged_df['summary_raw'].apply(lambda x: x if isinstance(x, dict) else {})
                merged_df['userid'] = merged_df['userid_raw'].fillna(method='ffill').fillna(method='bfill')
                merged_df['dateTime'] = pd.to_datetime(merged_df['date_only'])

                return merged_df[['userid', 'dateTime', 'hr', 'mode', 'active', 'step', 'summary']].reset_index(drop=True)

        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Error in get_dynamic_data from {url}: {e}")
            continue
            
    return pd.DataFrame()

def get_basic_info(userid, supertoken):
    # 根据 userid 前缀选择对应的数据中心 API
    # 1开头: 中国, 3开头: 美国, 7开头: 德国
    user_prefix = str(userid)[0]
    if user_prefix == '1':
        urls = [
            'https://mifit-user-profile-private-cn3.zepp.com/internal/users/batchProfiles'
        ]
    elif user_prefix == '3':
        urls = [
            'https://mifit-user-profile-private-us2.huami.com/internal/users/batchProfiles',
            'https://mifit-user-profile-private-us3.zepp.com/internal/users/batchProfiles'
        ]
    elif user_prefix == '7':
        urls = [
            'https://mifit-user-profile-private-de2.huami.com/internal/users/batchProfiles'
        ]
    else:
        # 兜底，仍然保留中美两个常见API
        urls = [
            'https://mifit-user-profile-private-us2.huami.com/internal/users/batchProfiles',
            'https://mifit-user-profile-private-cn3.zepp.com/internal/users/batchProfiles'
        ]
    url = None
    for try_url in urls:
        try:
            headers = {'apptoken': supertoken, 'appname': 'com.xiaomi.hm.health', 'Content-Type': 'application/json'}
            response = requests.post(try_url, headers=headers, data=f'[{userid}]', timeout=10)
            response.raise_for_status()
            res_json = response.json()
            if str(userid) in res_json and 'profile' in res_json[str(userid)]:
                url = try_url
                break
        except Exception:
            continue
    if url is None:
        return {"age": np.nan, "gender": np.nan, "weight": np.nan, "height": np.nan}
    headers = {'apptoken': supertoken, 'appname': 'com.xiaomi.hm.health', 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=f'[{userid}]', timeout=10)
        response.raise_for_status()
        profile = response.json()[str(userid)]['profile']
        return {
            "age": calculate_age(profile.get('birthday', '')),
            "gender": profile.get('gender', np.nan),
            "weight": profile.get('weight', np.nan),
            "height": profile.get('height', np.nan)
        }
        
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        print(f"Error fetching basic info for {userid}: {e}")
        return {"age": np.nan, "gender": np.nan, "weight": np.nan, "height": np.nan}
        

def get_stress_data(user_id, from_date_str, to_date_str):
    all_stress_dfs = []
    start_date = datetime.strptime(from_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(to_date_str, "%Y-%m-%d")
    
    headers_map = {
        '3': {'apptoken': 'UQVBQFZCPmpHSm56LntKMmo2Jlpya0p6PjwEAAQAAAABDeJbD30jm0zw-ijj4genZd2bW-KIGbzO11cH1rgEsg2vZhhKHYJ9xyUQgAnQq_sY9ozag71uqepWoz82_Uj9GfjN_vgIgbLFgALuzrYXHFdY5ge2JtgpkTmPDfFPnxMD5V9iAujxUCfR3QRE2WXS5ufSA1qhQtIfxcRmo5w2sQRkUlyCpDvs1iW-TJ_AFD3M', 'appName': 'com.huami.midong'},
        '5': {'apptoken': 'PQVBQFZCPmpHSm56LntKMmo2Jlpya0p6PjwEAAQAAAABKHXMC23pOvgnFvu9lQ1T0b_BXSV3fE7gDZMyVhsfYWfPAhZ4yCzpoheIQDdd_yVpPNBXpdux10BeKvjNENwaGDns4qV7904O5fe3rqrfUh9bRVHoR0I1hbq6Rc6Hx6vA5TvvjB5lBMrWfIyFYfB3HsyHpzCtMMSNUOwM_W6nqbWcBBHfsq7NgEXDpJTcChH0', 'appName': 'com.huami.midong'},
        '7': {'apptoken': 'DQVBQFZCPmpHSm56LntKMmo2Jlpya0p6PjwEAAQAAAABEYcoCvLVaEO5Eq4YD8397uXe2h6uIKuSizT7XzlaQCwywKZDJ3k3CwW74_e28vR_JHkf5IFJFF2gcWq5eu6-0AHGWM3HxQUfhuK98aehhRBXQNaWa-SENctvGcyRtdXlx3rRmVL-jTSfzOw7wjZeKo92k_Fqkmr6HZEdPwReH7CH98sfoLk8t9ub-WIqgqpc', 'appName': 'com.huami.midong'},
        'default': {'apptoken': 'NQVBQFZCPmpHSm56LntKMmo2Jlpya0p6PjwEAAQAAAAC3eZXQG7EM2HDUbzwb0ViXYFBhSk6Go88YBt9uI6f6rATc_ztC39I4LmXUnV8sGJlpNWdUnrApsHYuKwEc4IPpk0Nb2EQq6FvL_h_67P7FwXr9OjGIO2yF2Hqb_itPrBcw1RoBLMj0pucISPm7NyMIJA5YJK6keMQpv3L1VMK5ehShZyCHj_3crOdSFzdNDKI', 'appName': 'com.huami.midong'}
    }
    headers = headers_map.get(str(user_id)[0], headers_map['default'])
    
    current_start = start_date
    while current_start <= end_date:
        current_end = current_start + timedelta(days=13)
        if current_end > end_date: current_end = end_date
        
        time_start_ms = convert_date_to_datetime_ms_str(current_start.strftime('%Y-%m-%d'))
        time_end_ms = convert_date_to_datetime_ms_str(current_end.strftime('%Y-%m-%d'))
        url = f"http://api-mifit-us2.huami.com/v2/users/{user_id}/events?eventType=all_day_stress&from={time_start_ms}&limit=20&subType=all_day_stress&to={time_end_ms}"
        
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            items = response.json().get('items', [])
            if items:
                flat_data = [item for day in items if 'value' in day and 'data' in day['value'] for item in json.loads(day['value']['data'])]
                if flat_data:
                    all_stress_dfs.append(pd.DataFrame(flat_data))
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching stress data for chunk {current_start.date()} to {current_end.date()}: {e}")
        current_start = current_end + timedelta(days=1)
        
    if all_stress_dfs:
        final_df = pd.concat(all_stress_dfs, ignore_index=True)
        if 'time' in final_df.columns:
            final_df = final_df.drop_duplicates(subset=['time'])
            final_df['datetime'] = pd.to_datetime(final_df['time'], unit='ms')
            tz = 'Asia/Shanghai' if str(user_id).startswith('1') else 'America/New_York'
            final_df['datetime'] = final_df['datetime'].dt.tz_localize('UTC').dt.tz_convert(tz)
            return final_df.sort_values('datetime')
    return pd.DataFrame()

def get_readiness_data(userid, fromDate, toDate, supertoken=None):
    if supertoken is None:
        supertoken = 'UQVBQFZCPmpHSm56LntKMmo2Jlpya0p6PjwEAAQAAAADt8B0qIacr3oIc9YMTwAoP2UNHxFccyyahlWEjaB3hV1sIMl5Rsl_sAPZDitbrLswJSQky0Roet_8noVUGYha9-NnNjKBWoJjWupX0J9DyTJ2LLtOvO87k6KyjXdA4nspirUXOLxNQdYLS71zJ2lDIUAfre1nK6z8YKbKnXU6JYwPf84P5VJL0KKuk4aeC-BM'
    user_prefix = str(userid)[0]
    urls = []
    if user_prefix == '1':
        urls = [
            f'https://mifit-health-data-private-cn3.zepp.com/v2/users/{userid}/events'
        ]
    elif user_prefix == '3':
        urls = [
            f'https://mifit-health-data-private-us2.huami.com/v2/users/{userid}/events',
            f'https://mifit-health-data-private-us3.zepp.com/v2/users/{userid}/events'
        ]
    elif user_prefix == '7':
        urls = [
            f'https://mifit-health-data-private-de2.huami.com/v2/users/{userid}/events'
        ]
    else:
        urls = [
            f'https://mifit-health-data-private-us2.huami.com/v2/users/{userid}/events',
            f'https://mifit-health-data-private-cn3.zepp.com/v2/users/{userid}/events'
        ]
    headers = {'appname': 'com.huami.midong', 'apptoken': supertoken}
    params = {
        'eventType': 'readiness',
        'subType': 'watch_score',
        'from': convert_date_to_datetime_ms_str(fromDate),
        'to': convert_date_to_datetime_ms_str(toDate),
        'limit': '100'
    }
    response_ok = None
    for url in urls:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            items = response.json().get('items', [])
            if items:
                response_ok = response
                break
        except Exception:
            continue
    if response_ok is None:
        print(f"Error fetching readiness data for {userid}: 所有区域API均失败")
        return pd.DataFrame()
    try:
        items = response_ok.json().get('items', [])
        if not items:
            return pd.DataFrame()
        records = []
        for item in items:
            if 'value' in item and 'timestamp' in item:
                value = item['value']
                records.append({
                    'time': item['timestamp'],
                    "rdnsScore": value.get('rdnsScore', np.nan),
                    "mentScore": value.get('mentScore', np.nan),
                    "phyScore": value.get('phyScore', np.nan),
                    "skinTempScore": value.get('skinTempScore', np.nan),
                    "hrvScore": value.get('hrvScore', np.nan),
                    "rhrScore": value.get('rhrScore', np.nan),
                    "afibScore": value.get('afibScore', np.nan),
                    "ahiScore": value.get('ahiScore', np.nan),
                    "sleep_temperature_fluctuation": value.get('skinTempCalibrated', np.nan),
                    "sleep_rhr": value.get('sleepRHR', np.nan),
                    "sleep_hrv": value.get('sleepHRV', np.nan),
                    "hrvBaseline": value.get('hrvBaseline', np.nan),
                    "skinTempBaseLine": value.get('skinTempBaseLine', np.nan),
                    "rhrBaseline": value.get('rhrBaseline', np.nan),
                    "ahiBaseline": value.get('ahiBaseline', np.nan),
                    "skinTempCalibrated": value.get('skinTempCalibrated', np.nan)
                })
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
        return df.drop_duplicates(subset=['date'], keep='last').set_index('date')
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing readiness data for {userid}: {e}")
        return pd.DataFrame()

def get_exertion_data(userid, fromDate, toDate, supertoken=None):
    if supertoken is None:
        supertoken = 'UQVBQFZCPmpHSm56LntKMmo2Jlpya0p6PjwEAAQAAAADt8B0qIacr3oIc9YMTwAoP2UNHxFccyyahlWEjaB3hV1sIMl5Rsl_sAPZDitbrLswJSQky0Roet_8noVUGYha9-NnNjKBWoJjWupX0J9DyTJ2LLtOvO87k6KyjXdA4nspirUXOLxNQdYLS71zJ2lDIUAfre1nK6z8YKbKnXU6JYwPf84P5VJL0KKuk4aeC-BM'
    user_prefix = str(userid)[0]
    # 多区域 API fallback
    if user_prefix == '1':
        urls = [
            f'https://mifit-health-data-private-cn3.zepp.com/v2/users/{userid}/events'
        ]
    elif user_prefix == '3':
        urls = [
            f'https://mifit-health-data-private-us2.huami.com/v2/users/{userid}/events',
            f'https://mifit-health-data-private-us3.zepp.com/v2/users/{userid}/events'
        ]
    elif user_prefix == '7':
        urls = [
            f'https://mifit-health-data-private-de2.huami.com/v2/users/{userid}/events'
        ]
    else:
        urls = [
            f'https://mifit-health-data-private-us2.huami.com/v2/users/{userid}/events',
            f'https://mifit-health-data-private-cn3.zepp.com/v2/users/{userid}/events'
        ]

    headers = {'appname': 'com.huami.midong', 'apptoken': supertoken}
    params = {
        'eventType': 'exertion',
        'subType': 'algo_result',
        'from': convert_date_to_datetime_ms_str(fromDate),
        'to': convert_date_to_datetime_ms_str(toDate),
        'limit': '100'
    }
    response_ok = None
    for url in urls:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            items = response.json().get('items', [])
            if items:
                response_ok = response
                break
        except Exception:
            continue
    if response_ok is None:
        print(f"Error fetching exertion data for {userid}: 所有区域API均失败")
        return pd.DataFrame()
    try:
        items = response_ok.json().get('items', [])
        if not items:
            return pd.DataFrame()
        records = [
            {'time': item['timestamp'], 'totalScore': item['value'].get('totalScore', np.nan)}
            for item in items if 'value' in item and 'timestamp' in item
        ]
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
        return df.drop_duplicates(subset=['date'], keep='first').set_index('date')
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"Error fetching exertion data: {e}")
        return pd.DataFrame()


def get_charge_wake_data(userid, fromDate, toDate, supertoken=None):
    # 和get_readiness_data一样，自动适配API和token
    from datetime import datetime, timedelta
    if supertoken is None:
        supertoken = 'UQVBQFZCPmpHSm56LntKMmo2Jlpya0p6PjwEAAQAAAADt8B0qIacr3oIc9YMTwAoP2UNHxFccyyahlWEjaB3hV1sIMl5Rsl_sAPZDitbrLswJSQky0Roet_8noVUGYha9-NnNjKBWoJjWupX0J9DyTJ2LLtOvO87k6KyjXdA4nspirUXOLxNQdYLS71zJ2lDIUAfre1nK6z8YKbKnXU6JYwPf84P5VJL0KKuk4aeC-BM'
    user_prefix = str(userid)[0]
    urls = []
    if user_prefix == '1':
        urls = [
            f'https://mifit-health-data-private-cn3.zepp.com/v2/users/{userid}/events'
        ]
    elif user_prefix == '3':
        urls = [
            f'https://mifit-health-data-private-us2.huami.com/v2/users/{userid}/events',
            f'https://mifit-health-data-private-us3.zepp.com/v2/users/{userid}/events'
        ]
    elif user_prefix == '7':
        urls = [
            f'https://mifit-health-data-private-de2.huami.com/v2/users/{userid}/events'
        ]
    else:
        urls = [
            f'https://mifit-health-data-private-us2.huami.com/v2/users/{userid}/events',
            f'https://mifit-health-data-private-cn3.zepp.com/v2/users/{userid}/events'
        ]

    headers = {'appname': 'com.huami.midong', 'apptoken': supertoken}
    # 开始时间提前一天
    fromDate_dt = datetime.strptime(fromDate, "%Y-%m-%d") - timedelta(days=1)
    fromDate_early = fromDate_dt.strftime("%Y-%m-%d")
    params = {
        'eventType': 'Charge',
        'subType': 'wake_data',
        'from': convert_date_to_datetime_ms_str(fromDate_early),  # 提前一天
        'to': convert_date_to_datetime_ms_str(toDate),
        'limit': '100'
    }
    response_ok = None
    for url in urls:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            items = response.json().get('items', [])
            if items:
                response_ok = response
                break
        except Exception:
            continue
    if response_ok is None:
        print(f"Error fetching Charge wake data for {userid}: 所有区域API均失败")
        return pd.DataFrame()
    try:
        items = response_ok.json().get('items', [])
        if not items:
            return pd.DataFrame()
        records = []
        for item in items:
            if 'value' in item and 'timestamp' in item and item['value'].get('samples'):
                sample = item['value']['samples'][0]
                records.append({
                    'time': item['timestamp'],
                    'mentalWake': sample.get('mentalWake', np.nan),
                    'physicalWake': sample.get('physicalWake', np.nan),
                    'wakeCharge': sample.get('wakeCharge', np.nan),
                    'chronicWeightDaily': sample.get('chronicWeightDaily', np.nan),
                    'dailyFitnessScore': sample.get('dailyFitnessScore', np.nan),
                    'exertionScore': sample.get('exertionScore', np.nan),
                    'timeZone': item.get('timeZone', np.nan),
                })
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        # Use the per-record timeZone (when provided) to derive the local date.
        df['date'] = df.apply(
            lambda r: datetime.fromtimestamp(r['time'] / 1000, tz=timezone.utc)
            .astimezone(resolve_timezone(r.get('timeZone'), userid))
            .strftime('%Y-%m-%d'),
            axis=1
        )
        df = df.drop_duplicates(subset=['date'], keep='first').set_index('date')
        return df
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"Error fetching Charge wake data: {e}")
        return pd.DataFrame()

def get_exercise_data(userid, from_date, to_date, supertoken=None):
    # 映射不同userid前缀到不同的supertoken
    supertoken_map = {
        '1': 'PQVBQFZCPmpHSm56LntKMmo2Jlpya0p6PjwEAAQAAAABKHXMC23pOvgnFvu9lQ1T0b_BXSV3fE7gDZMyVhsfYWfPAhZ4yCzpoheIQDdd_yVpPNBXpdux10BeKvjNENwaGDns4qV7904O5fe3rqrfUh9bRVHoR0I1hbq6Rc6Hx6vA5TvvjB5lBMrWfIyFYfB3HsyHpzCtMMSNUOwM_W6nqbWcBBHfsq7NgEXDpJTcChH0',
        '3': 'UQVBQFZCPmpHSm56LntKMmo2Jlpya0p6PjwEAAQAAAABDeJbD30jm0zw-ijj4genZd2bW-KIGbzO11cH1rgEsg2vZhhKHYJ9xyUQgAnQq_sY9ozag71uqepWoz82_Uj9GfjN_vgIgbLFgALuzrYXHFdY5ge2JtgpkTmPDfFPnxMD5V9iAujxUCfR3QRE2WXS5ufSA1qhQtIfxcRmo5w2sQRkUlyCpDvs1iW-TJ_AFD3M',
        '7': 'DQVBQFZCPmpHSm56LntKMmo2Jlpya0p6PjwEAAQAAAABEYcoCvLVaEO5Eq4YD8397uXe2h6uIKuSizT7XzlaQCwywKZDJ3k3CwW74_e28vR_JHkf5IFJFF2gcWq5eu6-0AHGWM3HxQUfhuK98aehhRBXQNaWa-SENctvGcyRtdXlx3rRmVL-jTSfzOw7wjZeKo92k_Fqkmr6HZEdPwReH7CH98sfoLk8t9ub-WIqgqpc',
        'default': 'NQVBQFZCPmpHSm56LntKMmo2Jlpya0p6PjwEAAQAAAAC3eZXQG7EM2HDUbzwb0ViXYFBhSk6Go88YBt9uI6f6rATc_ztC39I4LmXUnV8sGJlpNWdUnrApsHYuKwEc4IPpk0Nb2EQq6FvL_h_67P7FwXr9OjGIO2yF2Hqb_itPrBcw1RoBLMj0pucISPm7NyMIJA5YJK6keMQpv3L1VMK5ehShZyCHj_3crOdSFzdNDKI'
    }
    exercise_columns = [
        'date', 'start_time', 'end_time', 'run_time', 'dis', 
        'avg_hr', 'trackid', 'source', 'exercise_type'
    ]
    
    user_str = str(userid)
    # 选择ap域名和supertoken
    if user_str.startswith('1'):
        api_domains = [
            "api-mifit-cn3.zepp.com",
        ]
        use_supertoken = supertoken_map['1']
    elif user_str.startswith('3'):
        api_domains = [
            "api-mifit-us2.zepp.com",
            "api-mifit-us3.zepp.com",
        ]
        use_supertoken = supertoken_map['3']
    elif user_str.startswith('7'):
        api_domains = [
            "api-mifit-de2.huami.com",
        ]
        use_supertoken = supertoken_map['7']
    else:
        api_domains = [
            "api-mifit-us2.zepp.com",
            "api-mifit-us3.zepp.com",
        ]
        use_supertoken = supertoken_map['default']

    # 如果外部传入supertoken优先生效（手动指定时）
    if supertoken:
        use_supertoken = supertoken

    from_ts, to_ts = int(datetime.strptime(from_date, "%Y-%m-%d").timestamp()), int(datetime.strptime(to_date, "%Y-%m-%d").timestamp())
    headers = {'appname': 'com.xiaomi.hm.health', 'apptoken': use_supertoken}
    last_exception = None
    for domain in api_domains:
        url = f"https://{domain}/v2/sport/run/history.json?huamiId={userid}&startTrackId={from_ts}&stopTrackId={to_ts}"
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            summaries = response.json().get('data', {}).get('summary', [])
            if not summaries:
                continue

            records = []
            for item in summaries:
                dt_object = datetime.fromtimestamp(int(item['end_time']))
                date_str = dt_object.strftime('%Y-%m-%d')
                run_time = round(float(item.get('run_time', 0)) / 60)
                end_minutes = dt_object.hour * 60 + dt_object.minute
                start_minutes = end_minutes - run_time
                if start_minutes < 0:
                    start_minutes = 0
                record = {
                    'date': date_str,
                    'start_time': f"{start_minutes // 60:02}:{start_minutes % 60:02}",
                    'end_time': f"{end_minutes // 60:02}:{end_minutes % 60:02}",
                    'run_time': run_time,
                    'dis': round(float(item.get('dis', 0))/1000),
                    'avg_hr': item.get('avg_heart_rate', np.nan),
                    'trackid': item.get('trackid', np.nan),
                    'source': item.get('source', np.nan),
                    'exercise_type': item.get('type', np.nan)
                }
                records.append(record)
            if records:
                return pd.DataFrame(records)
        except (requests.RequestException, json.JSONDecodeError) as e:
            last_exception = e
            continue

    print(f"Error fetching exercise data for {userid}: {last_exception or 'No data found'}")
    return pd.DataFrame([], columns=exercise_columns)
        
# ==============================================================================
# SECTION 3: MASTER DATA FETCHER
# ==============================================================================

def fetch_all_data_for_period(userid, fromDate, toDate, supertoken, token, raw_data_folder_path):
    print(f"[{userid}] Starting data fetching for period: {fromDate} to {toDate}")
    
    toDate_obj = datetime.strptime(toDate, "%Y-%m-%d")
    toDate_extended = (toDate_obj + timedelta(days=1)).strftime("%Y-%m-%d")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_exercise = executor.submit(get_exercise_data, userid, fromDate, toDate, supertoken)
        future_dynamic = executor.submit(get_dynamic_data, userid, fromDate, toDate, supertoken)
        future_basic = executor.submit(get_basic_info, userid, supertoken)
        future_stress = executor.submit(get_stress_data, userid, fromDate, toDate)
        future_readiness = executor.submit(get_readiness_data, userid, fromDate, toDate_extended)
        future_exertion = executor.submit(get_exertion_data, userid, fromDate, toDate_extended)
        future_charge_wake = executor.submit(get_charge_wake_data, userid, fromDate, toDate_extended)
        dynamic_df, basic_info, stress_df, exercise_df, readiness_df, exertion_df, charge_wake_df = (f.result() for f in [future_dynamic, future_basic, future_stress, future_exercise, future_readiness, future_exertion, future_charge_wake])
    if dynamic_df.empty:
        print(f"[{userid}] No dynamic data found. Aborting processing.")
        return None
        
    user_raw_data_path = os.path.join(raw_data_folder_path, str(userid))
    os.makedirs(user_raw_data_path, exist_ok=True)
    
    basic_df = pd.DataFrame([basic_info]) if basic_info else pd.DataFrame()
    dfs_to_save = {
        "dynamic_df": dynamic_df, "stress_df": stress_df, "exercise_df": exercise_df,
        "readiness_df": readiness_df, "exertion_df": exertion_df, "charge_wake_df": charge_wake_df,
        "basic_df": basic_df,
    }

    for name, df in dfs_to_save.items():
        if not df.empty:
            df_to_save = df.reset_index() if 'date' in df.index.names else df.copy()
            file_path = os.path.join(user_raw_data_path, f"{name}.csv")
            df_to_save.to_csv(file_path, index=False)
            
    print(f"[{userid}] Raw data saved.")
    
    return {"dynamic_df": dynamic_df, "basic_info": basic_info, "stress_df": stress_df, "exercise_df": exercise_df, "readiness_df": readiness_df, "exertion_df": exertion_df, "charge_wake_df": charge_wake_df}

# ==============================================================================
# SECTION 4: DATA PROCESSING
# ==============================================================================

def process_fetched_data(userid, fetched_data, user_score_data_path, user_sleep_data_path, processing_mode='ONLINE'):
    print(f"[{userid}] Starting data processing.")
    dynamic_df, basic_info, exercise_df, readiness_df, exertion_df, stress_df, charge_wake_df = (fetched_data[k] for k in ["dynamic_df", "basic_info", "exercise_df", "readiness_df", "exertion_df", "stress_df", "charge_wake_df"])
    
    # Part 1: Generate Daily Score CSV
    dynamic_df['date_str'] = dynamic_df['dateTime'].dt.strftime('%Y-%m-%d')
    all_scores = []

    for _, row in dynamic_df.iterrows():
        date_str = row['date_str']
        summary_analysis = get_analysis_summary(row['summary'])
        readiness_score = readiness_df.loc[date_str] if date_str in readiness_df.index else pd.Series(dtype='float64')
        exertion_score = exertion_df.loc[date_str] if date_str in exertion_df.index else pd.Series(dtype='float64')
        charge_wake_score = charge_wake_df.loc[date_str] if date_str in charge_wake_df.index else pd.Series(dtype='float64')
        
        daily_exercise = exercise_df[exercise_df['date'] == date_str]
        if len(row['summary']) == 0:
            continue
        all_scores.append({
            "userid": userid, "date": date_str, **basic_info, **summary_analysis,
            'sleep_rhr': readiness_score.get('sleep_rhr', np.nan),
            'sleep_hrv': readiness_score.get('sleep_hrv', np.nan),
            'rdnsScore': readiness_score.get('rdnsScore', np.nan),
            'mentScore': readiness_score.get('mentScore', np.nan),
            'phyScore': readiness_score.get('phyScore', np.nan),
            'skinTempScore': readiness_score.get('skinTempScore', np.nan),
            'hrvScore': readiness_score.get('hrvScore', np.nan),
            'rhrScore': readiness_score.get('rhrScore', np.nan),
            'afibScore': readiness_score.get('afibScore', np.nan),
            'ahiScore': readiness_score.get('ahiScore', np.nan),
            'sleep_temperature_fluctuation': readiness_score.get('sleep_temperature_fluctuation', np.nan),
            'hrvBaseline': readiness_score.get('hrvBaseline', np.nan),
            'exertion_score': exertion_score.get('totalScore', np.nan),
            'chronicWeightDaily': charge_wake_score.get('chronicWeightDaily', np.nan),
            'mentalWake': charge_wake_score.get('mentalWake', np.nan),
            'physicalWake': charge_wake_score.get('physicalWake', np.nan),
            'wakeCharge': charge_wake_score.get('wakeCharge', np.nan),
            'dailyFitnessScore': charge_wake_score.get('dailyFitnessScore', np.nan),
            'exertionScore': charge_wake_score.get('exertionScore', np.nan),
            'exercise_st': daily_exercise['start_time'].tolist() or np.nan,
            'exercise_ed': daily_exercise['end_time'].tolist() or np.nan,
            'exercise_duration': daily_exercise['run_time'].tolist() or np.nan,
            'exercise_type': daily_exercise['exercise_type'].tolist() or np.nan,
            'exercise_distance': daily_exercise['dis'].tolist() or np.nan,
            'exercise_avg_hr': daily_exercise['avg_hr'].tolist() or np.nan,
        })
        
    scores_df = pd.DataFrame(all_scores).sort_values('date').reset_index(drop=True)

    # Part 2: Generate Minute-by-Minute Time-Series CSVs
    print(f"[{userid}] Starting time-series data generation.")
    dynamic_df['date_str'] = dynamic_df['dateTime'].dt.strftime('%Y-%m-%d')
    scores_df_copy = scores_df.copy()
    scores_df_copy['date_str'] = pd.to_datetime(scores_df_copy['date']).dt.strftime('%Y-%m-%d')
    
    dynamic_df = dynamic_df[dynamic_df['summary'] != {}]
    if processing_mode == 'OFFLINE':
        import ast
        for col in ['hr', 'mode', 'active', 'step']:
            dynamic_df[col] = dynamic_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        dynamic_df['summary'] = dynamic_df['summary'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        dynamic_df['summary'] = dynamic_df['summary'].apply(lambda x: x if isinstance(x, dict) else {})

    dynamic_df = dynamic_df[dynamic_df['hr'].apply(lambda hr_list: not all(hr >= 250 for hr in hr_list))]


    available_dates = dynamic_df['date_str'].unique()
    scores_df_copy = scores_df_copy[scores_df_copy['date_str'].isin(available_dates)].reset_index(drop=True)
    
    merged_df = pd.merge(dynamic_df, scores_df_copy, on='date_str', how='left', suffixes=('', '_score'))

    stress_lists = {}
    if not stress_df.empty and 'datetime' in stress_df.columns:
        tz = stress_df['datetime'].dt.tz
        stress_df_indexed = stress_df.set_index('datetime')
        for day_str in merged_df['date_str'].unique():
            day_start = pd.to_datetime(day_str).tz_localize(tz)
            day_end = day_start + timedelta(days=1, minutes=-1)
            minute_index = pd.date_range(start=day_start, end=day_end, freq='T')
            resampled = stress_df_indexed['value'].reindex(minute_index, fill_value=255)
            stress_lists[day_str] = resampled.tolist()
            
    merged_df['stressdata'] = merged_df['date_str'].map(stress_lists).apply(lambda x: x if isinstance(x, list) else [255] * 1440)
    
    # +++ NEW: Pre-computation step before the loop +++
    # Helper to convert HH:MM string to minutes from midnight
    def time_to_minutes(t):
        if pd.isna(t) or t == -1:
            return np.nan
        try:
            h, m = map(int, str(t).split(':'))
            return h * 60 + m
        except (ValueError, AttributeError):
            return np.nan
    # Create the numeric 'sleep_end_minutes' column in the reference dataframe
    scores_df_copy['sleep_end_minutes'] = scores_df_copy['sleep_end_time'].apply(time_to_minutes)
    scores_df_copy['date_dt'] = pd.to_datetime(scores_df_copy['date_str']).dt.date


    # MODIFIED FUNCTION: Implements the robust, data-centric lookback logic.
    def get_smart_sleep_time(current_row, original_scores_df, current_date_str, time_type):
        """
        智能判断睡眠起止时间，并返回场景分类信息。
        返回: (时间字符串, 是否默认, init_flag, 场景描述, 历史平均时间或NaN)
        """
        time_col = 'sleep_end_time'
        time_col_minutes = 'sleep_end_minutes'
        
        current_day_score = original_scores_df[original_scores_df['date_str'] == current_date_str]
        
        original_time_minutes = np.nan
        if not current_day_score.empty:
            original_time_minutes = current_day_score.iloc[0][time_col_minutes]

        # 场景1: 存在主睡眠记录
        if pd.notna(original_time_minutes):
            original_time_str = current_day_score.iloc[0][time_col]
            return original_time_str, False, -1, 'Actual', np.nan

        if time_type == 'end':
            # 场景 2 & 3: 当日晨起时间不存在
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d").date()
            lookback_start_date = current_date - timedelta(days=3)
            
            lookback_df = original_scores_df[
                (original_scores_df['date_dt'] >= lookback_start_date) &
                (original_scores_df['date_dt'] < current_date)
            ]

            historical_wake_up_minutes = lookback_df[time_col_minutes].dropna()
            
            scenario_base = ''
            historical_avg_str = np.nan
            if not historical_wake_up_minutes.empty:
                reference_minute = int(historical_wake_up_minutes.mean())
                scenario_base = 'Default (Historical Avg)'
                # Store the historical average for output
                ref_hour_hist, ref_min_hist = divmod(reference_minute, 60)
                historical_avg_str = f"{ref_hour_hist:02d}:{ref_min_hist:02d}"
            else:
                reference_minute = 8 * 60
                scenario_base = 'Default (08:00)'
            
            ref_hour, ref_minute_of_hour = divmod(reference_minute, 60)
            reference_time_str = f"{ref_hour:02d}:{ref_minute_of_hour:02d}"

            four_hours_before = reference_minute - (4 * 60)
            hr_data = current_row['hr']
            start_window = max(0, four_hours_before)
            end_window = min(len(hr_data), reference_minute)
            is_worn_in_window = any(hr != 255 for hr in hr_data[start_window:end_window])

            if is_worn_in_window:
                return reference_time_str, True, 0, f'{scenario_base} - Worn', historical_avg_str
            else:
                return reference_time_str, True, 1, f'{scenario_base} - No Wear', historical_avg_str
                
        else:  # time_type == 'start'
            last_hr_index = next((i for i, x in reversed(list(enumerate(current_row['hr']))) if x != 255), None)
            if last_hr_index is not None and last_hr_index > 1080:
                return f"{last_hr_index // 60:02}:{last_hr_index % 60:02}", True, -1, 'Default Start', np.nan
            return "22:00", True, -1, 'Default Start', np.nan
    
    # +++ NEW: Initialize columns in scores_df_copy for scenario logging +++
    scores_df_copy['wake_up_scenario'] = ''
    scores_df_copy['historical_wake_up_time'] = np.nan


    for i in range(1, len(merged_df)):
        pre_date_str = merged_df.iloc[i-1]['date_str']
        cur_date_str = merged_df.iloc[i]['date_str']

        pre_date_start_str, _, _, _, _ = get_smart_sleep_time(merged_df.iloc[i-1], scores_df_copy, pre_date_str, 'start')
        pre_date_ed_str, prev_end_defaulted, _, _, _ = get_smart_sleep_time(merged_df.iloc[i-1], scores_df_copy, pre_date_str, 'end')
        cur_date_ed_str, curr_end_defaulted, cur_init_flag, cur_scenario, cur_hist_wake_time = get_smart_sleep_time(merged_df.iloc[i], scores_df_copy, cur_date_str, 'end')

        merged_df.loc[i-1, 'sleep_start_time'] = pre_date_start_str
        merged_df.loc[i-1, 'sleep_end_time'] = pre_date_ed_str
        merged_df.loc[i, 'sleep_end_time'] = cur_date_ed_str
        
        # +++ MODIFIED: Store scenario results in the scores_df_copy +++
        scores_df_copy.loc[i, 'wake_up_scenario'] = cur_scenario
        scores_df_copy.loc[i, 'historical_wake_up_time'] = cur_hist_wake_time


        # if curr_end_defaulted and prev_end_defaulted:
        #      start_dt = pd.to_datetime(f"{pre_date_str} 23:00")
        #      end_dt = pd.to_datetime(f"{cur_date_str} {cur_date_ed_str}")
        #      duration = (end_dt - start_dt).total_seconds() / 60
        #      merged_df.loc[i, 'sleep_duration'] = duration
        #      scores_df_copy.loc[i, 'sleep_duration'] = duration # Also update score file
        #      merged_df.loc[i, 'sleep_score'] = 100
        #      scores_df_copy.loc[i, 'sleep_score'] = 100 # Also update score file


        if prev_end_defaulted:
            scores_df_copy.loc[scores_df_copy['date_str'] == pre_date_str, 'sleep_end_time'] = pre_date_ed_str
        if curr_end_defaulted:
            scores_df_copy.loc[scores_df_copy['date_str'] == cur_date_str, 'sleep_end_time'] = cur_date_ed_str

        final_pre_date_ed_str = merged_df.loc[i-1, 'sleep_end_time']
        final_cur_date_ed_str = merged_df.loc[i, 'sleep_end_time']
        
        current_row, previous_row = merged_df.iloc[i], merged_df.iloc[i-1]
        
        start_slice_index_1440 = int(final_pre_date_ed_str.split(':')[0]) * 60 + int(final_pre_date_ed_str.split(':')[1])
        end_slice_index_1440 = int(final_cur_date_ed_str.split(':')[0]) * 60 + int(final_cur_date_ed_str.split(':')[1])

        final_hr = previous_row['hr'][start_slice_index_1440:] + current_row['hr'][:end_slice_index_1440]
        final_active = previous_row['active'][start_slice_index_1440:] + current_row['active'][:end_slice_index_1440]
        final_mode = previous_row['mode'][start_slice_index_1440:] + current_row['mode'][:end_slice_index_1440]
        final_step = previous_row['step'][start_slice_index_1440:] + current_row['step'][:end_slice_index_1440]
        final_stress = previous_row['stressdata'][start_slice_index_1440:] + current_row['stressdata'][:end_slice_index_1440]
        final_stress = [255 if j < len(final_hr) and final_hr[j] == 255 else final_stress[j] for j in range(len(final_stress))]
        
        times = generate_time_range(f"{pre_date_str} {final_pre_date_ed_str}", f"{cur_date_str} {final_cur_date_ed_str}")
        if not times or len(times) <= 1: continue
        times = times[:-1]
        temp_length = len(final_hr)

        if curr_end_defaulted:
            # 情况一：当天没有主睡眠记录。
            # 我们直接创建一个长度与其它数据流一致的、由 -1 填充的占位符列表。
            final_sleep_stage = [-1] * temp_length
        else:
            current_summary = current_row['summary']
            sleep_stage_input = current_summary.get('slp', {}).get('stage', [])
            # nap_stage_input要从前一天取，同时要判断前一天值是否存在
            previous_summary = previous_row['summary'] if 'summary' in previous_row else {}
            nap_stage_input = []
            if previous_summary:
                nap_stage_input = previous_summary.get('slp', {}).get('odd_stage', [])
            combined_sleep_data = get_sleep_data(sleep_stage_input, nap_stage_input)
            start_slice_2880 = int(final_pre_date_ed_str.split(':')[0]) * 60 + int(final_pre_date_ed_str.split(':')[1])
            end_slice_2880 = int(final_cur_date_ed_str.split(':')[0]) * 60 + int(final_cur_date_ed_str.split(':')[1]) + 1440
            final_sleep_stage = combined_sleep_data[start_slice_2880:end_slice_2880]

        nap_info = get_analysis_summary(previous_row['summary'])
        nap_st_list = nap_info.get('nap_start_time', [])
        nap_ed_list = nap_info.get('nap_end_time', [])

        lists_to_sync = [times, final_hr, final_active, final_mode, final_step, final_stress, final_sleep_stage]
        try:
            min_len = min(len(lst) for lst in lists_to_sync)
        except ValueError:
            print(f"[{userid}] An essential data list for {cur_date_str} is empty. Skipping this entry.")
            continue

        if any(len(lst) != min_len for lst in lists_to_sync):
            print(f"[{userid}] Mismatch found for {cur_date_str}. Truncating all lists to safe length: {min_len}.")
            times, final_hr, final_active, final_mode, final_step, final_stress, final_sleep_stage = (lst[:min_len] for lst in lists_to_sync)

        final_sleep_state, final_nap_state = calculate_sleep_and_nap_states(
            times, nap_st_list, nap_ed_list, final_sleep_stage, pre_date_str
        )
        
        # previous_odd_stage = previous_row['summary'].get('slp', {}).get('odd_stage', [])
        # current_odd_stage = current_row['summary'].get('slp', {}).get('odd_stage', [])

        # final_sleep_state, final_nap_state = create_sleep_and_nap_states_from_odd_stage(
        #     previous_odd_stage,
        #     current_odd_stage,
        #     start_slice_index_1440,
        #     end_slice_index_1440,
        #     final_sleep_stage # Pass the generated sleep stage list
        # )


        output_df = pd.DataFrame({
            'time': times, 'hr': final_hr, 'active': final_active, 'step': final_step,
            'mode': final_mode, 'stress': final_stress, 'sleep_stage': final_sleep_stage,
            'nap_state': final_nap_state, 'sleep_state': final_sleep_state
        })
        
        output_df['exercise'] = 0
        exercise_to_check_pre = exercise_df[exercise_df['date'] == pre_date_str]
        exercise_to_check_cur = exercise_df[exercise_df['date'] == cur_date_str]
        daily_exercise = pd.concat([exercise_to_check_pre, exercise_to_check_cur]).drop_duplicates()

        if not daily_exercise.empty:
            output_df_time_index = pd.to_datetime(output_df['time'])
            for _, ex_row in daily_exercise.iterrows():
                ex_start = pd.to_datetime(f"{ex_row['date']} {ex_row['start_time']}")
                ex_end = pd.to_datetime(f"{ex_row['date']} {ex_row['end_time']}")
                output_df.loc[(output_df_time_index >= ex_start) & (output_df_time_index <= ex_end), 'exercise'] = 1
        
        cur_date_dur = current_row.get('sleep_duration', 0)
        if pd.isna(cur_date_dur): cur_date_dur = 0
        cur_date_dur = int(cur_date_dur)
        
        final_length = len(output_df)
        sleep_markers = [0] * final_length
        sleep_start_idx = max(0, final_length - cur_date_dur)
        if sleep_start_idx < final_length:
            sleep_markers[sleep_start_idx:final_length] = [1] * (final_length - sleep_start_idx)
        output_df['sleep_markers'] = sleep_markers

        output_df['init_flag'] = cur_init_flag
        output_df['userid'] = userid
        
        # --- REMOVED scenario columns from daily sleep file ---
        final_columns = [
            'userid', 'time', 'hr', 'mode', 'active', 'step',
            'stress', 'sleep_stage', 'nap_state', 'sleep_state', 'exercise', 'sleep_markers', 
            'init_flag'
        ]
        output_df = output_df[final_columns]
        
        file_path = os.path.join(user_sleep_data_path, str(userid))
        os.makedirs(file_path, exist_ok=True)
        output_df.to_csv(os.path.join(file_path, f"{cur_date_str}.csv"), index=False)

    # Clean up temporary columns before saving the final score file
    scores_df_copy.drop(columns=['date_str', 'sleep_end_minutes', 'date_dt'], inplace=True, errors='ignore')
    
    # Add new columns to the final list to ensure they are saved
    final_score_columns = list(scores_df.columns) + ['wake_up_scenario', 'historical_wake_up_time']
    # Reorder to make it logical, if desired
    try:
        end_time_index = final_score_columns.index('sleep_end_time')
        final_score_columns.insert(end_time_index + 1, 'wake_up_scenario')
        final_score_columns.insert(end_time_index + 2, 'historical_wake_up_time')
        # Remove duplicates if they exist from the simple `+` operation
        final_score_columns = list(dict.fromkeys(final_score_columns))
    except ValueError:
        pass # sleep_end_time not in list, just append at the end
    
    scores_df_copy = scores_df_copy.reindex(columns=final_score_columns)

    scores_df_copy.to_csv(os.path.join(user_score_data_path, f"{userid}.csv"), index=False)
    print(f"[{userid}] User score data saved.")
    print(f"[{userid}] Time-series data processing finished.")


# ==============================================================================
# SECTION 5: MAIN EXECUTION PIPELINE
# ==============================================================================

def run_user_pipeline(userid, fromDate, toDate, supertoken, token, user_score_data_path, user_sleep_data_path, raw_data_folder_path, mode):
    start_time = time.time()
    print(f"--- Starting pipeline for user {userid} in {mode} mode ---")

    if mode == 'ONLINE':
        fetched_data = fetch_all_data_for_period(userid, fromDate, toDate, supertoken, token, raw_data_folder_path)
    elif mode == 'OFFLINE':
        fetched_data = fetch_all_data_from_local(userid, raw_data_folder_path)
    else:
        print(f"Invalid mode: {mode}")
        return

    if fetched_data:
        process_fetched_data(userid, fetched_data, user_score_data_path, user_sleep_data_path, processing_mode=mode)
    print(f"--- Pipeline for user {userid} finished in {time.time() - start_time:.2f} seconds ---")


def fetch_all_data_from_local(userid, raw_data_folder_path):
    print(f"[{userid}] Starting data fetching from local files.")
    user_raw_data_path = os.path.join(raw_data_folder_path, str(userid))
    if not os.path.exists(user_raw_data_path):
        print(f"No local data found for user {userid} at {user_raw_data_path}")
        return None
        
    dfs = {}
    
    list_converters = {
        'heartratedata': ast.literal_eval, 
        'mode': ast.literal_eval, 
        'active': ast.literal_eval,
        'step': ast.literal_eval, 
        'summary': ast.literal_eval,
    }
    date_columns = ['dateTime', 'datetime', 'date']

    for df_name in ["dynamic_df", "stress_df", "exercise_df", "readiness_df", "exertion_df", "charge_wake_df", "basic_df"]:
        file_path = os.path.join(user_raw_data_path, f"{df_name}.csv")
        if os.path.exists(file_path):
            converters = list_converters if df_name == 'dynamic_df' else None
            try:
                temp_df = pd.read_csv(file_path, converters=converters)
            except Exception as e:
                print(f"Could not read {file_path} with converters, trying without. Error: {e}")
                temp_df = pd.read_csv(file_path)

            if df_name == 'dynamic_df' and 'heartratedata' in temp_df.columns:
                temp_df.rename(columns={'heartratedata': 'hr'}, inplace=True)

            for col in date_columns:
                if col in temp_df.columns:
                    temp_df[col] = pd.to_datetime(temp_df[col], errors='coerce')
                    
                    if pd.api.types.is_datetime64_any_dtype(temp_df[col]) and temp_df[col].dt.tz is not None:
                        temp_df[col] = temp_df[col].dt.tz_convert('UTC').dt.tz_localize(None)
            
            if 'date' in temp_df.columns and df_name in ['readiness_df', 'exertion_df', 'charge_wake_df']:
                temp_df['date'] = pd.to_datetime(temp_df['date']).dt.strftime('%Y-%m-%d')
                temp_df = temp_df.set_index('date')
                
            dfs[df_name] = temp_df
        else:
            print(f"Warning: Raw data file not found: {file_path}")
            # --- MODIFICATION START ---
            # If a file is missing, create an empty DataFrame with the correct columns
            # to prevent KeyErrors downstream. This is especially important for exercise_df.
            if df_name == 'exercise_df':
                exercise_columns = [
                    'date', 'start_time', 'end_time', 'run_time', 'dis', 
                    'avg_hr', 'trackid', 'source', 'exercise_type'
                ]
                dfs[df_name] = pd.DataFrame(columns=exercise_columns)
            else:
                dfs[df_name] = pd.DataFrame()
            # --- MODIFICATION END ---


    basic_info = dfs.get("basic_df", pd.DataFrame()).iloc[0].to_dict()
    
    print(f"[{userid}] Data fetching from local files complete.")
    
    return {
        "dynamic_df": dfs.get("dynamic_df", pd.DataFrame()), 
        "basic_info": basic_info, 
        "stress_df": dfs.get("stress_df", pd.DataFrame()), 
        "exercise_df": dfs.get("exercise_df", pd.DataFrame()),
        "readiness_df": dfs.get("readiness_df", pd.DataFrame()), 
        "exertion_df": dfs.get("exertion_df", pd.DataFrame()),
        "charge_wake_df": dfs.get("charge_wake_df", pd.DataFrame()),
    }



if __name__ == "__main__":
    PROCESSING_MODE = 'ONLINE'
    # PROCESSING_MODE = 'OFFLINE'
    # RUNNING_MODE = "SERIES"
    RUNNING_MODE = "PARALLE"

    # Load tokens from environment variables
    supertoken = os.environ.get('XIAOMI_SUPERTOKEN')
    token = os.environ.get('XIAOMI_TOKEN', supertoken)
    if not supertoken:
        raise ValueError("Set XIAOMI_SUPERTOKEN environment variable (see .env.example)")

    current_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_name = 'test_data_new'

    data_folder = os.path.join(current_folder, f'../data/{dataset_name}/')

    user_score_data_path = os.path.join(data_folder, 'user_score_data')
    user_sleep_data_path = os.path.join(data_folder, 'user_sleep_data')
    raw_data_folder_path = os.path.join(data_folder, 'user_raw_data')
    
    os.makedirs(user_score_data_path, exist_ok=True)
    os.makedirs(user_sleep_data_path, exist_ok=True)
    os.makedirs(raw_data_folder_path, exist_ok=True)

    print(f"User score data will be saved to: {user_score_data_path}")
    print(f"User sleep data will be saved to: {user_sleep_data_path}")
    print(f"Raw data will be saved to: {raw_data_folder_path}")

     
    # 允许为每个用户分别指定开始和结束日期（或从配置表读取），例如：
    # 支持：手动填写/从csv导入
    # 示例1：直接在代码里写
    # 增加一个开关：控制是否只处理未存在用户（True：只处理未存在用户，False：强制重新生成并覆盖已有数据）
    CHECK_EXIST_USER = False    # True: 只处理未存在的用户（已存在的不处理）；False: 总是覆盖处理（并删除本地已存在的数据）

    user_date_ranges = [
        # {'userid': '1113363497', 'from_date': '2025-10-01', 'to_date': '2025-11-03'},
        # {'userid': '3089520729', 'from_date': '2025-10-01', 'to_date': '2025-11-03'},
        # {'userid': '7083547893', 'from_date': '2025-10-01', 'to_date': '2025-11-03'},
        # {'userid': '3095596688', 'from_date': '2025-10-01', 'to_date': '2025-11-03'},
        # {'userid': '7083052373', 'from_date': '2025-10-01', 'to_date': '2025-11-03'},
        # {'userid': '3095351182', 'from_date': '2025-10-01', 'to_date': '2025-11-03'},
        # {'userid': '1196709240', 'from_date': '2025-10-01', 'to_date': '2025-11-03'},
        {'userid': '7083662941', 'from_date': '2025-11-10', 'to_date': '2025-11-25'},
    ]

    # 示例2：也可以从csv批量读入（例如特殊用户csv），如下反注释即可：
    # df_special = pd.read_csv(f"{current_folder}/userid/biocharge_{dataset_name}_userids.csv")
    # user_date_ranges = []
    # for userid in df_special['userid']:
    #     user_date_ranges.append({
    #         'userid': str(userid),
    #         'from_date': '2024-10-15',
    #         'to_date': '2024-12-01',
    #     })


    exist_userids = os.listdir(user_score_data_path)
    exist_userids = [userid.split('.')[0] for userid in exist_userids]

    import shutil

    # 根据开关决定最终需要处理的用户列表
    if CHECK_EXIST_USER:
        # 只处理未存在的用户
        ID_list_complete = [user for user in user_date_ranges if user['userid'] not in exist_userids]
    else:
        # 不管已经存在的，全部重处理，且先删除本地已有数据
        # 删除score/sleep/raw三个文件夹下该用户的数据（如有）
        for item in user_date_ranges:
            userid = item['userid']
            # 删除user_score_data_path 中该userid文件
            score_file = os.path.join(user_score_data_path, f"{userid}.csv")
            if os.path.exists(score_file):
                try:
                    os.remove(score_file)
                except Exception as e:
                    print(f"Error deleting {score_file}: {e}")
            # 删除user_sleep_data_path 中该userid文件
            sleep_file = os.path.join(user_sleep_data_path, f"{userid}.csv")
            if os.path.exists(sleep_file):
                try:
                    os.remove(sleep_file)
                except Exception as e:
                    print(f"Error deleting {sleep_file}: {e}")
            # 删除raw_data_folder_path 下面该userid文件夹
            raw_folder = os.path.join(raw_data_folder_path, f"{userid}")
            if os.path.exists(raw_folder):
                try:
                    shutil.rmtree(raw_folder)
                except Exception as e:
                    print(f"Error deleting {raw_folder}: {e}")
        ID_list_complete = user_date_ranges

    if RUNNING_MODE == "SERIES":
        # 串行处理
        for item in ID_list_complete:
            userid = item['userid']
            fromDate = item['from_date']
            toDate = item['to_date']
            run_user_pipeline(
                userid, fromDate, toDate,
                supertoken, token,
                user_score_data_path, user_sleep_data_path, raw_data_folder_path,
                PROCESSING_MODE
            )
    elif RUNNING_MODE == "PARALLE":
        # 并行处理
        MAX_WORKERS = 10
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for item in ID_list_complete:
                userid = item['userid']
                fromDate = item['from_date']
                toDate = item['to_date']
                future = executor.submit(
                    run_user_pipeline,
                    userid, fromDate, toDate, 
                    supertoken, token, 
                    user_score_data_path, user_sleep_data_path, raw_data_folder_path, 
                    PROCESSING_MODE
                )
                futures.append(future)
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing users", position=0, leave=False, colour='green'):
                try:
                    future.result()
                except Exception as e:
                    print(f"A task in the pipeline generated an exception: {e}")
    else:
        print("Not correct running mode.")

    print("All user processing tasks have been submitted and completed.")



