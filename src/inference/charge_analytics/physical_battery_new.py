import numpy as np

def smoothstep_curve(score, parameter, margin, direction):
    if direction == "decrease":
        k_modified = parameter * (1 - (margin * (3 * score**2 - 2 * score**3)))
    elif direction == "increase":
        k_modified = parameter * (1 + (margin * (3 * score**2 - 2 * score**3)))
    return k_modified

class physical_batteryFT:
    """
    Calculates the Physical Battery by modeling energy expenditure and recovery.
    
    This class tracks 'expenditure' internally (0-100, where higher means more tired)
    and outputs 'battery' (100 - expenditure) based on physiological data.
    """
    def __init__(self, age, CP, acc_threshold, hr_rest, allks,
                 init_expenditure=100, physical_no_wear_fitting=[], fitness_fatigue_score=0, exertion_growth_rate=0.05, **kwargs):
        """
        Initializes the physical battery model.

        Args:
            age (int): User's age.
            CP (float): Critical Power threshold for HRR to determine expenditure.
            acc_threshold (float): Acceleration threshold to determine expenditure.
            hr_rest (float): User's resting heart rate.
            allks (list): A list of tuned model parameters (hyperparameters).
            init_expenditure (float): The starting battery level (0-100). 
                                      Note: Despite the name, this represents battery level.
            physical_no_wear_fitting (list): A pre-calculated curve for non-wear periods.
            **kwargs: Catches any other arguments passed from the main script.
        """
        # Convert initial battery (100=full) to internal expenditure (0=full)
        internal_init_expenditure = 100 - init_expenditure
        self.expenditure_history = [internal_init_expenditure]
        self.last_known_expenditure = internal_init_expenditure
        self.output_battery = []
        
        self.age = age
        self.hr_rest = hr_rest
        self.no_wear_fitting = physical_no_wear_fitting

        self.allks = allks
        self.acc_threshold_org = acc_threshold
        self.CP = CP
                
        # Unpack model hyperparameters into a dictionary for clarity and readability
        self.params = {
            'recovery_k1': allks[1],                 # 22000.0
            'recovery_k2': allks[2],                 # 40.0
            'recovery_k3': allks[3],                 # 200.0
            'recovery_multiplier': allks[4],         # 25.6
            'stage_deep_multiplier': allks[5],       # 1
            'stage_light_multiplier': allks[6],      # 1.5
            'stage_rem_multiplier': allks[7],        # 2
            'health_modulation_k': allks[8]          # 0.001
        }

        self.growth_param = exertion_growth_rate
        self.params['expenditure_k_val'] = allks[0]  # 0.1, will be modified dynamically


        self.fitness_fatigue_score = fitness_fatigue_score
        self.fitness_fatigue_score_sleep = 1 - self.fitness_fatigue_score**3
        
        self.CP1 = smoothstep_curve(score = self.fitness_fatigue_score, parameter = self.CP, margin = min(0.25, self.fitness_fatigue_score), direction = "increase")
        self.CP2 = smoothstep_curve(score = self.fitness_fatigue_score, parameter = 10, margin = min(0.25, self.fitness_fatigue_score), direction = "increase")
        self.acc_threshold = smoothstep_curve(score = self.fitness_fatigue_score, parameter = self.acc_threshold_org, margin = min(0.25, self.fitness_fatigue_score), direction = "increase")

        self.params['expenditure_k_val'] = smoothstep_curve(score = self.fitness_fatigue_score, parameter = self.allks[0], margin = 0.25, direction = "decrease")
        self.exertion_growth_rate = smoothstep_curve(score = self.fitness_fatigue_score, parameter = self.growth_param, margin = 0.5, direction = "decrease")
        # self.exertion_growth_rate = exertion_growth_rate
        # --- Debugging Lists ---
        self.logic_branch_flags = []
        self.original_hrr_series = []
        self.modified_hrr_series = []
        
    # Method for hypothetical expenditure calculation
    def calculate_hypothetical_expenditure(self, current_hrr, current_acc, exertion_score):
        """
        Calculates the hypothetical physical expenditure increase for one minute without updating state.
        This focuses only on the expenditure logic branch.
        """
        # 1. Prepare inputs (logic copied from _update_expenditure)

        modified_hrr = current_hrr * np.exp(self.exertion_growth_rate * (exertion_score - 50))
        current_acc = self.acc_threshold if current_acc == -1 else current_acc
        
        # The CP threshold is based on being awake, which is the only state where this matters.
        # We assume awake state for hypothetical expenditure.
        cp_threshold = 15 

        # 2. Check for expenditure state (logic copied from _update_expenditure)
        is_in_expenditure_state = (modified_hrr - cp_threshold >= 0) and (current_acc > self.acc_threshold)

        if is_in_expenditure_state:
            # 3. Calculate expenditure increase (logic copied from the expenditure branch)
            k_val = self.params['expenditure_k_val'] * (0.8 if modified_hrr <= 15 else 1.0)
            expenditure_increase = (modified_hrr - cp_threshold) * k_val
            return expenditure_increase
        
        return 0.0 # No expenditure increase if conditions aren't met
    
    def _update_expenditure(self, current_hrr, current_acc, current_mode, sleep_state,
                           sleep_stage, sleep_duration_score, sleep_start_time_score,
                           waso_score, exertion_score,
                           nap_pointer, fitness_fatigue_score):
        """
        Updates the energy expenditure for one minute. 
        
        This method contains the core calculation logic, which has been preserved 
        exactly from the original script to ensure identical output.
        """
        new_score = max(min(fitness_fatigue_score, 0.99), 0)
        if not hasattr(self, 'fitness_fatigue_score') or self.fitness_fatigue_score != new_score:
            self.fitness_fatigue_score = new_score
            self.fitness_fatigue_score_sleep = 1 - self.fitness_fatigue_score**3

            self.CP1 = smoothstep_curve(score = self.fitness_fatigue_score, parameter = self.CP, margin = min(0.25, self.fitness_fatigue_score), direction = "increase")
            self.CP2 = smoothstep_curve(score = self.fitness_fatigue_score, parameter = 10, margin = min(0.25, self.fitness_fatigue_score), direction = "increase")
            self.acc_threshold = smoothstep_curve(score = self.fitness_fatigue_score, parameter = self.acc_threshold_org, margin = min(0.25, self.fitness_fatigue_score), direction = "increase")

            self.params['expenditure_k_val'] = smoothstep_curve(score = self.fitness_fatigue_score, parameter = self.allks[0], margin = 0.25, direction = "decrease")
            self.exertion_growth_rate = smoothstep_curve(score = self.fitness_fatigue_score, parameter = 0.05, margin = 0.5, direction = "decrease")

        # --- 1. Data Preparation ---
        self.original_hrr_series.append(current_hrr)
        
        # Modify HRR based on the daily exertion score to account for strain
        modified_hrr = current_hrr * np.exp(self.exertion_growth_rate * (exertion_score - 50))
        self.modified_hrr_series.append(modified_hrr)
        
        # Fill missing acceleration values with the threshold
        current_acc = self.acc_threshold if current_acc == -1 else current_acc
        
        last_expenditure = self.expenditure_history[-1]
        
        # The Critical Power (CP) threshold changes based on awake/sleep state
        self.CP = self.CP2 if current_mode == 0 else self.CP1
        
        # Combine sleep scores into a single weighted value
        if nap_pointer:
            weighted_sleep_score = 1
        else:
            weighted_sleep_score = (sleep_duration_score * 0.5) + (sleep_start_time_score * 0.25) + (waso_score * 0.25)
        
        # Map sleep stage code (0,1,2) to its corresponding parameter multiplier
        stage_multipliers = {
            0: self.params['stage_deep_multiplier'],
            1: self.params['stage_rem_multiplier'],
            2: self.params['stage_light_multiplier']
        }
        stage_multiplier = stage_multipliers.get(sleep_stage, self.params['stage_light_multiplier'])

        # --- 2. Core Expenditure & Recovery Logic ---
        is_in_expenditure_state = (modified_hrr - self.CP >= 0) and (current_acc > self.acc_threshold)

        if is_in_expenditure_state:
            # BRANCH 1: EXPENDITURE (High HRR & High Acceleration)
            if sleep_state == 0:
                # BRANCH 1.1: Expenditure while Awake
                k_val = self.params['expenditure_k_val'] * (0.8 if modified_hrr <= self.CP else 1.0)
                new_expenditure = last_expenditure + ((modified_hrr - self.CP) * k_val)
                self.logic_branch_flags.append(0)
            else: # sleep_state == 1
                # BRANCH 1.2: Expenditure while Asleep
                k_val_recovery = self.params['recovery_multiplier'] / self.params['expenditure_k_val']
                denominator = stage_multiplier * k_val_recovery * (201 - last_expenditure) * self.fitness_fatigue_score_sleep
                exponent = (modified_hrr - 100) * weighted_sleep_score / denominator
                new_expenditure = last_expenditure * np.exp(exponent)
                self.logic_branch_flags.append(1)
        else:
            # BRANCH 2: RECOVERY (Low HRR or Low Acceleration)
            if modified_hrr < self.CP:
                # BRANCH 2.1: Recovery triggered by Low Heart Rate Reserve
                current_battery_level = 100 - last_expenditure
                if 40 <= current_battery_level <= 80:
                    # BRANCH 2.1.1: Recovery at mid-range battery levels
                    if sleep_state == 0: # Awake
                        k_rm11 = self.params['recovery_k1'] * self.params['recovery_multiplier']
                        new_expenditure = last_expenditure * np.exp((modified_hrr - 100) / k_rm11)
                    else: # Asleep
                        rm11 = self.params['recovery_k1']
                        denominator = stage_multiplier * rm11 * self.fitness_fatigue_score_sleep
                        new_expenditure = last_expenditure * np.exp((modified_hrr - 100) * weighted_sleep_score / denominator)
                    self.logic_branch_flags.append(2)
                else:
                    # BRANCH 2.1.2: Recovery at low/high battery levels
                    if sleep_state == 0: # Awake
                        k_rm12 = self.params['recovery_k2'] * self.params['recovery_multiplier']
                        new_expenditure = last_expenditure * np.exp(-(self.CP - modified_hrr) / (k_rm12 * (201 - last_expenditure)))
                    else: # Asleep
                        rm12 = self.params['recovery_k2']
                        denominator = stage_multiplier * (rm12 * (201 - last_expenditure)) * self.fitness_fatigue_score_sleep
                        new_expenditure = last_expenditure * np.exp(-(self.CP - modified_hrr) * weighted_sleep_score / denominator)
                    self.logic_branch_flags.append(3)
            else:
                # BRANCH 2.2: Recovery triggered by Low Acceleration (even with high HRR)
                if sleep_state == 0: # Awake
                    k_rm2 = self.params['recovery_k3'] * self.params['recovery_multiplier']
                    new_expenditure = last_expenditure * np.exp((modified_hrr - 100) / (k_rm2 * (201 - last_expenditure)))
                else: # Asleep
                    rm2 = self.params['recovery_k3']
                    denominator = stage_multiplier * (rm2 * (201 - last_expenditure)) * self.fitness_fatigue_score_sleep
                    new_expenditure = last_expenditure * np.exp((modified_hrr - 100) * weighted_sleep_score / denominator)
                self.logic_branch_flags.append(4)
                
        return max(0, new_expenditure)
        # return new_expenditure

    def run_battery(self, hrr_series, acc_series, min_mode_state_series, shutdown_count_series, 
                         sleep_state_series, sleep_stage_series, health_score_list, 
                         sleep_duration_score, sleep_start_time_score, WASO_score, 
                         exertion_score_series, idx,
                         nap_pointer=False, fitness_fatigue_score=0, 
                         increase_flag=True, decrease_flag=True):
        """
        Runs the battery simulation for a single minute-long step.
        This method is called from the main script for each minute of the day.
        """
        current_mode = min_mode_state_series[idx]
        new_expenditure = 0

        if current_mode == 2:
            # Case 1: Long non-wear period -> Use pre-calculated fitting curve
            new_expenditure = self.no_wear_fitting[idx]
            self.logic_branch_flags.append(-1)
            self.original_hrr_series.append(0)
            self.modified_hrr_series.append(0)

        elif current_mode == 3:
            # Case 2: Short non-wear period -> Adjust based on fitting curve delta
            # WARNING: The original script's logic is preserved here. It was flagged
            # as problematic ("！有问题") because it uses a future data point (idx + 1),
            # but is kept to ensure identical output to the original script.
            delta_expenditure = self.no_wear_fitting[idx] - self.no_wear_fitting[idx + 1]
            new_expenditure = self.last_known_expenditure + delta_expenditure
            self.logic_branch_flags.append(-2)
            self.original_hrr_series.append(0)
            self.modified_hrr_series.append(0)

        elif current_mode in [0, 1]:
            # Case 3: Normal operation (Awake/Sleep) -> Run the full model
            new_expenditure = self._update_expenditure(
                hrr_series[idx], acc_series[idx], current_mode,
                sleep_state_series[idx], sleep_stage_series[idx],
                sleep_duration_score, sleep_start_time_score, WASO_score,
                exertion_score_series[idx], nap_pointer, fitness_fatigue_score
            )
        else:
            # Case 4: Invalid data -> Mark as invalid and skip
            new_expenditure = -999 # Sentinel value for invalid
            self.logic_branch_flags.append(-999)
            self.modified_hrr_series.append(-999)
            
        # Convert internal expenditure (0-100, tired) to final battery (100-0, charged)
        output_battery = 100 - new_expenditure
        
        # Determine previous battery level to calculate the minute-over-minute change
        previous_battery = self.output_battery[-1] if idx > 0 else (100 - self.last_known_expenditure)
        battery_difference = output_battery - previous_battery
        
        # Apply a non-linear modulation based on health score. This creates a
        # parabolic scaling effect that dampens large changes based on health.
        health_factor = 1.0 - health_score_list[idx]
        dampening_k = self.params['health_modulation_k'] * health_factor * battery_difference
        modulated_difference = battery_difference * (1 - dampening_k)
        final_battery = previous_battery + modulated_difference

        # Apply external flags to cap battery changes if they are disallowed
        if idx > 0:
            previous_output_battery = self.output_battery[-1]
            if (previous_output_battery < final_battery and not increase_flag) or \
               (previous_output_battery > final_battery and not decrease_flag):
                final_battery = previous_output_battery
                # Revert expenditure if the battery change was capped
                new_expenditure = 100 - final_battery
                
        # Update state for the next iteration
        self.last_known_expenditure = new_expenditure
        self.expenditure_history.append(new_expenditure)
        self.output_battery.append(final_battery)