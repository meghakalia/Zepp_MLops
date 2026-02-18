import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1.  计算动态 SWC (14d)、Baseline & CV (5d)、几天未佩戴
# ------------------------------------------------------------------
def compute_hrv_metrics(hrv_series: pd.Series,
                        swc_win: int = 14,
                        swc_k: float = 0.5,
                  
                        base_win: int = 5,
                        base_min_valid: int = 3,
                        stale_days: int = 28           # NEW: 过期阈值
                        ) -> pd.DataFrame:
    idx = hrv_series.index
    n = len(hrv_series)
    has_hrv = ~hrv_series.isna()

    swc_low, swc_up = np.full(n, np.nan), np.full(n, np.nan)
    baseline = np.full(n, np.nan)
    cv       = np.full(n, np.nan)
    conf     = np.zeros(n)

    # NEW 计算 day_gap
    _last_seen = np.where(has_hrv, np.arange(n), np.nan)
    _last_seen = pd.Series(_last_seen).ffill().fillna(-1).values
    gap_arr = np.arange(n) - _last_seen  # 每天距上次有效夜的天数

    last_base, last_cv = np.nan, np.nan

    for i in range(n):
        # 统一判断：近5天 >=3 个有效
        vals_b = hrv_series[max(0, i - base_win + 1): i + 1].dropna()
        base_ready = (len(vals_b) >= base_min_valid)

        if base_ready and has_hrv[i]:
            # baseline / cv（5天窗口）
            mean_b = vals_b.mean()
            std_b  = vals_b.std(ddof=0)
            last_base = mean_b
            last_cv   = (std_b / mean_b) * 100

            # SWC（14天窗口）
            win   = min(swc_win, i + 1)
            vals_s = hrv_series[max(0, i - win + 1): i + 1].dropna()
            mu = vals_s.mean()
            sigma = vals_s.std(ddof=0)
            swc_low[i] = mu - swc_k * sigma
            swc_up[i]  = mu + swc_k * sigma

        else:
            # NEW: 若基线过期（> stale_days），不再沿用旧值，直接置空并重置状态
            if gap_arr[i] > stale_days:
                swc_low[i] = np.nan
                swc_up[i]  = np.nan
                last_base  = np.nan   
                last_cv    = np.nan
            else:
                # 未达标但仍在保质期内，沿用昨日
                swc_low[i] = swc_low[i-1] if i else np.nan
                swc_up[i]  = swc_up[i-1]  if i else np.nan

        # 写回当日 baseline/cv/conf
        baseline[i] = last_base
        cv[i]       = last_cv
        conf[i]     = len(vals_b) / base_win

    day_gap = gap_arr

    return pd.DataFrame({
        "daily_hrv": hrv_series.values,
        "swc_low": swc_low, "swc_up": swc_up,
        "baseline": baseline, "cv": cv,
        "confidence": conf, "day_gap": day_gap
    }, index=idx)

# ------------------------------------------------------------------
# 2.  按文档里的趋势分类
# ------------------------------------------------------------------
def classify_hrv_trend(row: pd.Series) -> str:
    if pd.isna(row.daily_hrv):
        return "No Data"
    # 基线或 SWC 尚无效，或无数据
    if pd.isna(row.baseline) or pd.isna(row.swc_low):
        return "No Data"

    # 1. No relevant trends
    if (row.swc_low <= row.daily_hrv <= row.swc_up and
        row.swc_low <= row.baseline <= row.swc_up):
        return "No relevant trends"

    # 2. Baseline > upper SWC
    if row.baseline > row.swc_up:
        return "Positive Adaptation to Stressors"

    # calculate delta cv and delta baseline
    delta_cv   = row.cv - row.cv_prev
    delta_base = row.baseline - row.baseline_prev

    # 3. CV increased & Baseline increased
    if delta_cv > 0 and delta_base > 0:
        return "Most likely Coping well with Stressors"
    # 4. CV decreased & Baseline increased 
    if delta_cv < 0 and delta_base > 0:
        return "Positive Adaptation to Stressors"
    # 5. CV decreased & Baseline decreased
    if delta_cv < 0 and delta_base < 0:
        return "Risk of maladaptation"
    # 6. CV increased & Baseline decreased
    if delta_cv > 0 and delta_base < 0:
        return "Risk of maladaptation"

    return "Others"

# ------------------------------------------------------------------
# 3.  将分类映射到HRV Factor
# ------------------------------------------------------------------
_BASE_FACTOR = {
    "Positive Adaptation to Stressors"      : 1.10,
    "Most likely Coping well with Stressors": 0.90,
    "Risk of maladaptation"                 : 0.40,
    "No relevant trends"                    : 0.70,
    "No Data"                               : 1.00,
    "Others"                                : 1.00,
}

def _factor_from_trend(trend: str, confidence: float,
                       day_gap: int) -> float:
    if trend == "No Data":
        
        return 0.70 * max(0.3, 1 - 0.1 * day_gap)

    base = _BASE_FACTOR[trend]
    return 0.70 + confidence * (base - 0.70)

# ------------------------------------------------------------------
# 4.  从HRV原始值输出 hrv_factor 
# ------------------------------------------------------------------
def hrv_factor_series(hrv_series: pd.Series,missing_codes=(255, -1, 0, 999, ''),decay_missing: bool = False) -> pd.Series:
    hrv_series = (pd.to_numeric(hrv_series, errors='coerce')
                    .replace(list(missing_codes), np.nan))

    m = compute_hrv_metrics(hrv_series)

    # 前一天 baseline / cv，用于计算差值
    m["baseline_prev"] = m["baseline"].shift(1)
    m["cv_prev"]       = m["cv"].shift(1)
    # 趋势分类
    m["trend"] = m.apply(classify_hrv_trend, axis=1)
    
    #print(m[["daily_hrv","baseline","cv","swc_low","swc_up"]])
    
    # factor
    if decay_missing:
        factors = m.apply(lambda r: _factor_from_trend(
            r.trend, r.confidence, r.day_gap), axis=1)
    else:
        factors = (
    m['trend'].map(_BASE_FACTOR)        
      .astype(float, errors='ignore')   
      .fillna(1.0)                     
)

    return factors

# Constants for HRV_score
ALPHA = 0.20 

def hrv_score_modified(series_raw, dates=None):
    """
    series_raw : pandas Series of raw HRV  
                 may contain 255 / -1 / NaN, treated as invalid & skipped
    dates      : 假设已包含每个自然天
    returns    : scores
    """
    # Use same preprocessing as hrv_factor
    series_clean = (pd.to_numeric(series_raw, errors='coerce')
                      .replace((255, -1, 0, 999, ''), np.nan))

    # Create time series if dates provided
    if dates is not None:
        hrv_series = pd.Series(
            series_clean.values,
            index=pd.to_datetime(dates)
        ).sort_index()
    else:
        hrv_series = series_clean

    # Use same metrics calculation as hrv_factor (5 days for baseline)
    m = compute_hrv_metrics(hrv_series)
    
    scores = []
    prev_score = np.nan
    cv_history = []

    for i, row in m.iterrows():
        # Skip invalid nights
        if (pd.isna(row.daily_hrv) or 
            pd.isna(row.baseline) or 
            pd.isna(row.cv) or 
            pd.isna(row.swc_low) or 
            pd.isna(row.swc_up)):
            scores.append(np.nan)
            cv_history.append(np.nan)
            prev_score = np.nan
            continue
        
        # S1: Baseline position relative to SWC (use hrv_factor's baseline and SWC)
        if row.baseline > row.swc_up:
            S1 = +22
        elif row.baseline < row.swc_low:
            S1 = -22
        else:
            S1 = 0
        
        # S2: Daily change from baseline (use hrv_factor's baseline)
        delta = row.daily_hrv - row.baseline
        if delta >= 0.04:
            S2 = +12
        elif delta <= -0.05:
            S2 = -12
        elif delta <= -0.02:
            S2 = -6
        else:
            S2 = 0
        
        # S3: CV trend (use hrv_factor's CV)
        cv_t = min(row.cv, 10)  # Cap CV at 10
        if len(cv_history) > 0 and not pd.isna(cv_history[-1]):
            delta_cv = cv_t - cv_history[-1]
            if delta_cv < 0:
                S3 = +8
            elif delta_cv > 0:
                S3 = -8
            else:
                S3 = 0
        else:
            S3 = 0
        
        cv_history.append(cv_t)
        
        # Calculate raw score
        raw_score = 77 + 0.5*S1 + 0.3*S2 + 0.20*S3
        raw_score = np.clip(raw_score, 0, 100)
        
        # Apply smoothing
        if np.isnan(prev_score):
            final = raw_score
        else:
            final = (1 - ALPHA) * raw_score + ALPHA * prev_score
        
        scores.append(round(final, 1))
        prev_score = final
        

    return np.array(scores)

# ------------------------------------------------------------------
# 5.  获取单日的HRV factor
# ------------------------------------------------------------------
def get_hrv_factor_for_day(daily_data: pd.DataFrame, 
                          current_index: int,
                          decay_missing: bool = False) -> float:
    
    hrv_part = daily_data.loc[:current_index, ["date", "sleep_hrv"]]

    hrv_series = pd.Series(
        hrv_part["sleep_hrv"].values,
        index=pd.to_datetime(hrv_part["date"])
    ).sort_index()

    # Check for empty or all-NaT index
    if hrv_series.empty or hrv_series.index.isna().all():
        return np.nan

    min_idx = hrv_series.index.min()
    max_idx = hrv_series.index.max()
    if pd.isna(min_idx) or pd.isna(max_idx):
        return np.nan

    # 按自然日补齐到“今天”为止，缺失为NaN
    hrv_series = hrv_series.reindex(
        pd.date_range(min_idx, max_idx, freq="D")
    )

    factors = hrv_factor_series(hrv_series, decay_missing=decay_missing)

    return factors.iloc[-1]