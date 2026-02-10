import numpy as np
import math
import pandas as pd

# --- Helper Functions (Maintained from original) ---

def get_hrv_factor(hrv_score):
    """Calculates a recovery factor based on HRV score."""
    if 69 < hrv_score <= 85:
        return 0.7
    elif 0 <= hrv_score <= 69:
        return 0.5
    return 1.0 # Optimal or missing score

def get_rhr_factor(rhr_score):
    """Calculates a recovery factor based on RHR score."""
    if 69 < rhr_score <= 85:
        return 0.7
    elif 0 <= rhr_score <= 69:
        return 0.5
    return 1.0 # Optimal or missing score

def get_circadian_output(time_in_hour, major_peak=18, minor_peak_shift=3, beta=0.5):
    """Calculates the circadian rhythm output."""
    return np.cos(2 * math.pi * (time_in_hour - major_peak) / 24) + \
           beta * np.cos(2 * math.pi * (time_in_hour - major_peak - minor_peak_shift) / 12)

def get_sleep_prospensity(circadian_value, a_val=0.55):
    """Calculates sleep propensity from circadian output."""
    return -a_val * circadian_value

def get_sleep_debt(curr_battery, f_val=0.0026564, total_battery=2880):
    """Calculates sleep debt from the current battery level."""
    return f_val * (total_battery - curr_battery)

def get_sleep_intensity(sp, sd, curr_press, press_thresh, expand_ratio=1):
    """Calculates sleep intensity."""
    return min((sp + sd) * np.exp((press_thresh - curr_press) * expand_ratio), 4.4)

def main_synthetic_process(Rt, Rc):
    """Converts absolute cognitive reserve (Rt) to a percentage score (Et)."""
    return 100 * (Rt / Rc)

def Et2Rt(Et, Rc):
    """Converts a percentage score (Et) back to absolute cognitive reserve (Rt)."""
    return (Et * Rc) / 100

def smoothstep_curve(score, parameter, margin, direction):
    if direction == "decrease":
        k_modified = parameter * (1 - (margin * (3 * score**2 - 2 * score**3)))
    elif direction == "increase":
        k_modified = parameter * (1 + (margin * (3 * score**2 - 2 * score**3)))
    return k_modified

# --- Optimized Mental Battery Class ---

class mental_batteryFT(object):
    """
    Calculates the Mental Battery score by modeling cognitive resource dynamics.
    It simulates depletion from stress and exertion, and replenishment from rest and sleep.
    """
    def __init__(self, ts, allks, press_thresh=0.117, expand1=5, expand2=1, Rt=2880, Rc=2880, 
                 usual_awake_ts=8, usual_slp_ts=0, mental_no_wear_fitting=[], param_margin=0.1):
        """Initializes the mental battery model."""
        self.Rt = Rt  # Current cognitive reserve
        self.Rc = Rc  # Max cognitive reserve
        self.Et = []  # History of battery scores (%)
        self.time = ts  # Initial time of day in hours
        self.Rts = [Rt] # History of cognitive reserves

        self.expand1 = expand1
        self.expand2 = expand2
        self.press_thresh = press_thresh
        self.no_wear_fitting = mental_no_wear_fitting
        self.not_wear_last_Rt = self.Rt
        self.previous_Rt = self.Rt
        self.allks = allks
        
        self.param_margin = param_margin

        # Unpack parameters with descriptive names for clarity
        # self.params = {
        #     'recovery_si_multiplier': allks[3],    # 1.3
        #     'recovery_exp_high_press': allks[4],   # -2.0
        #     'recovery_exp_low_press': allks[5],    # -2.4
        #     'stage_deep_multiplier': allks[6],     # 3
        #     'stage_rem_multiplier': allks[7],      # 9
        #     'stage_light_multiplier': allks[8],    # 12
        #     'health_modulation': allks[9]          # 0.001
        # }

        self.params = {
            'k_score_low_battery': allks[0],      # Dynamic, initialized here for completeness
            'k_score_high_battery': allks[1],     # Dynamic, initialized here for completeness
            'depletion_min_rate': allks[2],       # Dynamic, initialized here for completeness
            'recovery_si_multiplier': allks[3],    # 1.3
            'recovery_exp_high_press': allks[4],   # -2.0
            'recovery_exp_low_press': allks[5],    # -2.4
            'stage_deep_multiplier': allks[6],     # 3
            'stage_rem_multiplier': allks[7],      # 9
            'stage_light_multiplier': allks[8],    # 12
            'health_modulation': allks[9]          # 0.001
        }

    def update_time(self, step=1):
        """Updates the internal time by one minute."""
        self.time = (self.time + step / 60) % 24

    # Method for hypothetical expenditure calculation
    def calculate_hypothetical_expenditure(self, curr_press, exertion_score):
        """
        Calculates the hypothetical mental expenditure for one minute without updating the state.
        This is intended for use when the battery is critically low (<5) and normal depletion is paused.
        """
        # 1. Apply exertion-based modifier to stress (same as in _update_Rt)
        modified_press = curr_press * np.exp(0.01 * (exertion_score - 50))

        # 2. Check if in a depletion state
        if modified_press < self.press_thresh:
            return 0.0 # No depletion if below threshold

        # 3. Calculate depletion amount (logic copied from _update_Rt depletion branch)
        if (100 * self.Rt / self.Rc) < 50:
            battery_pct = max(100 * (self.Rt / self.Rc), 0.1)
            k_score = self.params['k_score_low_battery'] / math.exp(battery_pct / 50)
        else:
            k_score = self.params['k_score_high_battery']
        
        depletion_amount = max(self.params['depletion_min_rate'], (modified_press - self.press_thresh) * 10) * k_score
        
        return depletion_amount

    def _update_Rt(self, sleep_state, curr_press, sleep_duration_score, sleep_start_time_score, 
                   WASO_score, hrv_factor, rhr_score, sleep_stage, exertion_score,
                   nap_pointer, fitness_fatigue_score,
                   original):
        """
        Updates the cognitive reserve (Rt) for one minute based on physiological state.
        This function contains the core logic for depletion and recovery.
        """
        # Update fitness_fatigue_score every 2 minutes (i.e., every other call)

        # 优化后的代码：合并分支，减少重复计算，提升可读性
        new_score = max(min(fitness_fatigue_score, 0.99), 0)
        if not hasattr(self, 'fitness_fatigue_score') or self.fitness_fatigue_score != new_score:
            self.fitness_fatigue_score = new_score
            # self.params['k_score_low_battery'] = smoothstep_curve(score=self.fitness_fatigue_score, parameter=self.allks[0], margin=self.param_margin, direction="decrease")
            # self.params['k_score_high_battery'] = smoothstep_curve(score=self.fitness_fatigue_score, parameter=self.allks[1], margin=self.param_margin, direction="decrease")
            # self.params['depletion_min_rate'] = smoothstep_curve(score=self.fitness_fatigue_score, parameter=self.allks[2], margin=self.param_margin, direction="decrease")
            
            if original:
                self.fitness_fatigue_score_sleep = 1
            else:
                self.fitness_fatigue_score_sleep = 1 - self.fitness_fatigue_score**3

        # 1. Calculate foundational sleep metrics
        circadian_value = get_circadian_output(self.time)
        sp = get_sleep_prospensity(circadian_value)
        sd = get_sleep_debt(self.Rt, total_battery=self.Rc)
        si = get_sleep_intensity(sp, sd, curr_press, self.press_thresh, expand_ratio=self.expand2)
        
        # 2. Apply exertion-based modifier to stress
        modified_press = curr_press * np.exp(0.01 * (exertion_score - 50))
        
        # 3. Determine new Rt based on state (Awake vs. Sleep)
        if sleep_state != 1:  # --- AWAKE STATE ---
            if modified_press >= self.press_thresh: # Depletion
                if (100 * self.Rt / self.Rc) < 50:
                    battery_pct = max(100 * (self.Rt / self.Rc), 0.1)
                    k_score = self.params['k_score_low_battery'] / math.exp(battery_pct / 50)
                else:
                    k_score = self.params['k_score_high_battery']
                self.Rt -= max(self.params['depletion_min_rate'], (modified_press - self.press_thresh) * 10) * k_score
            else:  # Replenishment
                self.Rt += si * (self.press_thresh - modified_press)
        
        else:  # --- SLEEP STATE ---
            # Determine sleep stage multiplier
            stage_map = {0: 'stage_deep_multiplier', 1: 'stage_rem_multiplier', 2: 'stage_light_multiplier'}
            stage_multiplier = self.params.get(stage_map.get(sleep_stage))
            
            # Calculate combined scores for recovery
            if nap_pointer:
                sleep_quality = 1 / stage_multiplier
            else:
                sleep_quality = (0.50 * sleep_duration_score + 0.25 * sleep_start_time_score + 0.25 * WASO_score) / stage_multiplier
            physio_factor = 0.7 * hrv_factor + 0.3 * get_rhr_factor(rhr_score)
            
            # Determine recovery exponent based on pressure
            recovery_exp = self.params['recovery_exp_high_press'] if modified_press >= self.press_thresh else self.params['recovery_exp_low_press']
            
            # Update Rt with sleep recovery formula
            recovery_amount = self.params['recovery_si_multiplier'] * (si / np.exp(recovery_exp * sleep_quality)) * physio_factor * self.fitness_fatigue_score_sleep
            self.Rt += recovery_amount

        self.Rt = min(self.Rt, self.Rc)

    def run_battery(self, sleep_state, shutdown_count, curr_press, min_mode_state, health_score_list, 
                    sleep_duration_score, sleep_start_time_score, WASO_score, hrv_score, rhr_score,
                    sleep_stage, exertion_score, time, 
                    nap_pointer, fitness_fatigue_score,
                    original):
        """
        Runs the battery simulation for a single time step (1 minute).
        """
        tmpE = 0 # Initialize battery score for the current step

        # Case 1: Device not worn (long period) -> Use pre-calculated fitting curve
        if min_mode_state[time] == 2:
            tmpE = self.no_wear_fitting[time]
            self.not_wear_last_Rt = Et2Rt(tmpE, self.Rc)
        
        # Case 2: Device not worn (short period) -> Adjust last known value by fitting curve's delta
        elif min_mode_state[time] == 3:
            current_Et = main_synthetic_process(self.Rt, self.Rc)
            delta_E = self.no_wear_fitting[time] - self.no_wear_fitting[time - shutdown_count[time]]
            tmpE = current_Et + delta_E
            self.not_wear_last_Rt = Et2Rt(tmpE, self.Rc)

        # Case 3: Normal operation -> Run the full model
        else:
            # If returning from a non-wear period, reset Rt
            if time > 0 and min_mode_state[time - 1] in [2, 3]:
                self.Rt = self.not_wear_last_Rt
            
            # Update cognitive reserve using the core model
            self._update_Rt(
                sleep_state[time], curr_press[time], sleep_duration_score, sleep_start_time_score, 
                WASO_score, hrv_score, rhr_score, sleep_stage[time], exertion_score[time],
                nap_pointer, fitness_fatigue_score,
                original
            )
            
            # Convert updated Rt to a percentage score
            tmpE = main_synthetic_process(self.Rt, self.Rc)
            self.Rts.append(self.Rt)

        # Apply health score modulation to dampen large changes when health is poor
        prior_battery = self.Et[-1] if self.Et else main_synthetic_process(self.Rts[0], self.Rc)
        out_battery_diff = tmpE - prior_battery
        modulation = self.params['health_modulation'] * (1 - health_score_list[time]) * abs(out_battery_diff)
        modulated_diff = out_battery_diff * (1 - modulation)
        battery_final = prior_battery + modulated_diff
        
        # Apply external caps to prevent charge from moving in a disallowed direction
        # if self.Et:
        #     previous_Et = self.Et[-1]
        #     if (previous_Et < battery_final and not increase_flag) or \
        #        (previous_Et > battery_final and not decrease_flag):
        #         battery_final = previous_Et
        #         self.Rt = self.previous_Rt # Revert Rt if change was capped
        
        self.previous_Rt = self.Rt
        self.Et.append(battery_final)
        self.update_time()

        return battery_final