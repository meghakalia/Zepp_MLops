import torch
import torch.nn as nn


class MLP_delta(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, layers: int = 3, dropout: float = 0.1, norm: str = 'batch'):
        super().__init__()
        dims = [input_dim] + [hidden] * layers + [1]
        blocks = []
        for i in range(len(dims) - 2):
            blocks.append(nn.Linear(dims[i], dims[i+1]))
            if norm == 'batch':
                blocks.append(nn.BatchNorm1d(dims[i+1])) # batch norm 
            elif norm == 'layer':
                blocks.append(nn.LayerNorm(dims[i+1])) # layer norm
            
            blocks.append(nn.ReLU(inplace=True))
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        y = self.net(x)
        return y


class ChargePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        # Learnable stage multipliers - initialized with domain knowledge
        # Using log scale to ensure positivity: multiplier = exp(param)
        self.log_stage_multipliers = nn.Parameter(
            torch.tensor([torch.log(torch.tensor(1.50)), torch.log(torch.tensor(2.57)), torch.log(torch.tensor(3.43))])  # deep, REM, light
        )
        
        # Rest of network
        self.mlp = MLP_delta(input_dim + 1, hidden=hidden_dim, layers=2, dropout=0.1
        )
    
    def get_stage_recovery_rate(self, sleep_stage, sleep_state):
        """
        sleep_stage: (batch,) tensor with values 0, 1, 2
        sleep_state: (batch,) tensor with values 0 (awake), 1 (asleep)
        """
        # Get multipliers (ensure positive via exp)
        multipliers = torch.exp(self.log_stage_multipliers)  # [1.50, 2.57, 3.43] initially
        
        # Index into multipliers based on sleep_stage
        # Clamp stage to valid range [0, 2]
        stage_clamped = torch.clamp(sleep_stage, 0, 2).long()
        stage_multiplier = multipliers[stage_clamped]  # (batch,)
        
        # Recovery rate = 1 / multiplier (0 when awake)
        recovery_rate = (1.0 / stage_multiplier) * sleep_state.float()
        
        return recovery_rate
    
    def forward(self, features, sleep_stage, sleep_state):
        # Compute learnable recovery rate
        recovery_rate = self.get_stage_recovery_rate(sleep_stage, sleep_state)
        
        # Concatenate recovery_rate to features
        recovery_rate = recovery_rate.unsqueeze(-1)  # (batch, 1)
        x = torch.cat([recovery_rate, features], dim=-1)
        
        # Predict delta
        delta = self.mlp(x)
        return delta


class ChargePredictorDirectConcat(nn.Module):
    """
    Approach 1: Direct concatenation of learnable parameters as input features.
    All mental and physical parameters are concatenated with input features.
    """
    # Number of learnable parameters: 10 mental + 9 physical = 19
    NUM_PARAMS = 19

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        # ==================== Mental Battery Parameters (10) ====================
        # Depletion parameters
        self.mental_log_k_score_low_battery = nn.Parameter(torch.log(torch.tensor(1.2)))
        self.mental_log_k_score_high_battery = nn.Parameter(torch.log(torch.tensor(1.4)))
        self.mental_log_depletion_min_rate = nn.Parameter(torch.log(torch.tensor(2.2921)))

        # Recovery parameters
        self.mental_log_recovery_si_multiplier = nn.Parameter(torch.log(torch.tensor(1.3)))
        self.mental_recovery_exp_high_press = nn.Parameter(torch.tensor(-2.0))
        self.mental_recovery_exp_low_press = nn.Parameter(torch.tensor(-2.4))

        # Sleep stage multipliers [deep, REM, light]
        self.mental_log_stage_multipliers = nn.Parameter(
            torch.tensor([torch.log(torch.tensor(3.0)),
                          torch.log(torch.tensor(9.0)),
                          torch.log(torch.tensor(12.0))])
        )

        # Health modulation
        self.mental_log_health_modulation = nn.Parameter(torch.log(torch.tensor(0.001)))

        # ==================== Physical Battery Parameters (9) ====================
        # Expenditure parameter
        self.physical_log_expenditure_k_val = nn.Parameter(torch.log(torch.tensor(0.002)))

        # Recovery parameters
        self.physical_log_recovery_k1 = nn.Parameter(torch.log(torch.tensor(22000.0)))
        self.physical_log_recovery_k2 = nn.Parameter(torch.log(torch.tensor(40.0)))
        self.physical_log_recovery_k3 = nn.Parameter(torch.log(torch.tensor(200.0)))
        self.physical_log_recovery_multiplier = nn.Parameter(torch.log(torch.tensor(25.6)))

        # Sleep stage multipliers [deep, light, REM]
        self.physical_log_stage_multipliers = nn.Parameter(
            torch.tensor([torch.log(torch.tensor(1.0)),
                          torch.log(torch.tensor(1.5)),
                          torch.log(torch.tensor(2.0))])
        )

        # Health modulation
        self.physical_log_health_modulation_k = nn.Parameter(torch.log(torch.tensor(0.001)))

        # MLP with input_dim + 19 parameters
        self.mlp = MLP_delta(input_dim + self.NUM_PARAMS, hidden=hidden_dim, layers=2, dropout=0.1)

    def get_all_params(self, batch_size):
        """
        Stack all learnable parameters into a single vector, apply scaling (sigmoid/tanh), and expand for batch.
        Returns: (batch_size, 19) tensor
        """
        # Helper functions
        def scale_sigmoid(x):
            return torch.sigmoid(x)
        def scale_tanh(x):
            return torch.tanh(x)

        # Exponentiate log parameters, then scale
        mental_k_score_low_battery = scale_sigmoid(torch.exp(self.mental_log_k_score_low_battery)).unsqueeze(0)
        mental_k_score_high_battery = scale_sigmoid(torch.exp(self.mental_log_k_score_high_battery)).unsqueeze(0)
        mental_depletion_min_rate = scale_sigmoid(torch.exp(self.mental_log_depletion_min_rate)).unsqueeze(0)
        mental_recovery_si_multiplier = scale_sigmoid(torch.exp(self.mental_log_recovery_si_multiplier)).unsqueeze(0)
        # These are not log, use tanh to keep in [-1,1]
        mental_recovery_exp_high_press = scale_tanh(self.mental_recovery_exp_high_press).unsqueeze(0)
        mental_recovery_exp_low_press = scale_tanh(self.mental_recovery_exp_low_press).unsqueeze(0)
        # Stage multipliers: exp then sigmoid
        mental_stage_multipliers = scale_sigmoid(torch.exp(self.mental_log_stage_multipliers))
        mental_health_modulation = scale_sigmoid(torch.exp(self.mental_log_health_modulation)).unsqueeze(0)

        # Physical parameters
        physical_expenditure_k_val = scale_sigmoid(torch.exp(self.physical_log_expenditure_k_val)).unsqueeze(0)
        physical_recovery_k1 = scale_sigmoid(torch.exp(self.physical_log_recovery_k1)).unsqueeze(0)
        physical_recovery_k2 = scale_sigmoid(torch.exp(self.physical_log_recovery_k2)).unsqueeze(0)
        physical_recovery_k3 = scale_sigmoid(torch.exp(self.physical_log_recovery_k3)).unsqueeze(0)
        physical_recovery_multiplier = scale_sigmoid(torch.exp(self.physical_log_recovery_multiplier)).unsqueeze(0)
        physical_stage_multipliers = scale_sigmoid(torch.exp(self.physical_log_stage_multipliers))
        physical_health_modulation_k = scale_sigmoid(torch.exp(self.physical_log_health_modulation_k)).unsqueeze(0)

        params = torch.cat([
            # Mental parameters (10)
            mental_k_score_low_battery,
            mental_k_score_high_battery,
            mental_depletion_min_rate,
            mental_recovery_si_multiplier,
            mental_recovery_exp_high_press,
            mental_recovery_exp_low_press,
            mental_stage_multipliers,  # 3 values
            mental_health_modulation,
            # Physical parameters (9)
            physical_expenditure_k_val,
            physical_recovery_k1,
            physical_recovery_k2,
            physical_recovery_k3,
            physical_recovery_multiplier,
            physical_stage_multipliers,  # 3 values
            physical_health_modulation_k,
        ])  # Shape: (19,)

        # Expand to batch size
        return params.unsqueeze(0).expand(batch_size, -1)  # (batch_size, 19)

    def forward(self, features, sleep_stage=None, sleep_state=None):
        """
        Args:
            features: (batch, input_dim) input features
            sleep_stage: unused, kept for API compatibility
            sleep_state: unused, kept for API compatibility
        Returns:
            delta: (batch, 1) predicted change
        """
        batch_size = features.shape[0]

        # Get all parameters expanded to batch size
        params = self.get_all_params(batch_size)  # (batch, 19)

        # Concatenate parameters with features
        x = torch.cat([params, features], dim=-1)  # (batch, input_dim + 19)

        # Predict delta
        delta = self.mlp(x)
        return delta


class ChargePredictor3(nn.Module):
    """
    Approach 2: Physics-informed model using actual battery equations.
    Parameters are used in the actual mental/physical battery formulas.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        # ==================== Mental Battery Parameters ====================
        self.mental_log_k_score_low_battery = nn.Parameter(torch.log(torch.tensor(1.2)))
        self.mental_log_k_score_high_battery = nn.Parameter(torch.log(torch.tensor(1.4)))
        self.mental_log_depletion_min_rate = nn.Parameter(torch.log(torch.tensor(2.2921)))
        self.mental_log_recovery_si_multiplier = nn.Parameter(torch.log(torch.tensor(1.3)))
        self.mental_recovery_exp_high_press = nn.Parameter(torch.tensor(-2.0))
        self.mental_recovery_exp_low_press = nn.Parameter(torch.tensor(-2.4))
        self.mental_log_stage_multipliers = nn.Parameter(
            torch.tensor([torch.log(torch.tensor(3.0)),
                          torch.log(torch.tensor(9.0)),
                          torch.log(torch.tensor(12.0))])  # deep, REM, light
        )
        self.mental_log_health_modulation = nn.Parameter(torch.log(torch.tensor(0.001)))

        # ==================== Physical Battery Parameters ====================
        self.physical_log_expenditure_k_val = nn.Parameter(torch.log(torch.tensor(0.002)))
        self.physical_log_recovery_k1 = nn.Parameter(torch.log(torch.tensor(22000.0)))
        self.physical_log_recovery_k2 = nn.Parameter(torch.log(torch.tensor(40.0)))
        self.physical_log_recovery_k3 = nn.Parameter(torch.log(torch.tensor(200.0)))
        self.physical_log_recovery_multiplier = nn.Parameter(torch.log(torch.tensor(25.6)))
        self.physical_log_stage_multipliers = nn.Parameter(
            torch.tensor([torch.log(torch.tensor(1.0)),
                          torch.log(torch.tensor(1.5)),
                          torch.log(torch.tensor(2.0))])  # deep, light, REM
        )
        self.physical_log_health_modulation_k = nn.Parameter(torch.log(torch.tensor(0.001)))

        # Constants
        self.press_thresh = 0.117
        self.Rc = 2880.0  # Max cognitive reserve

        # Residual MLP to learn corrections to physics model
        # Input: features + physics_delta_mental + physics_delta_physical
        self.mlp = MLP_delta(input_dim + 2, hidden=hidden_dim, layers=2, dropout=0.1)

    def compute_mental_delta(self, curr_battery, curr_press, sleep_state, sleep_stage,
                              sleep_quality, hrv_factor, exertion_score, time_of_day):
        """
        Compute mental battery delta using physics equations.

        Args:
            curr_battery: (batch,) current battery percentage [0-100]
            curr_press: (batch,) current pressure/stress level
            sleep_state: (batch,) 0=awake, 1=asleep
            sleep_stage: (batch,) 0=deep, 1=REM, 2=light
            sleep_quality: (batch,) weighted sleep quality score
            hrv_factor: (batch,) HRV-based recovery factor
            exertion_score: (batch,) daily exertion score [0-100]
            time_of_day: (batch,) hour of day [0-24]
        """
        batch_size = curr_battery.shape[0]

        # Get parameters
        k_low = torch.exp(self.mental_log_k_score_low_battery)
        k_high = torch.exp(self.mental_log_k_score_high_battery)
        depletion_min = torch.exp(self.mental_log_depletion_min_rate)
        recovery_si_mult = torch.exp(self.mental_log_recovery_si_multiplier)
        stage_mults = torch.exp(self.mental_log_stage_multipliers)  # [deep, REM, light]

        # Compute circadian rhythm
        circadian = torch.cos(2 * 3.14159 * (time_of_day - 18) / 24) + \
                    0.5 * torch.cos(2 * 3.14159 * (time_of_day - 18 - 3) / 12)

        # Sleep propensity and debt
        sp = -0.55 * circadian
        sd = 0.0026564 * (100 - curr_battery)  # Normalized to percentage

        # Sleep intensity
        si = torch.clamp((sp + sd) * torch.exp((self.press_thresh - curr_press)), max=4.4)

        # Modify pressure by exertion
        modified_press = curr_press * torch.exp(0.01 * (exertion_score - 50))

        # Initialize delta
        delta = torch.zeros(batch_size, device=curr_battery.device)

        # Awake state
        awake_mask = (sleep_state == 0)

        # Awake + Depletion (high pressure)
        depletion_mask = awake_mask & (modified_press >= self.press_thresh)
        if depletion_mask.any():
            # k_score based on battery level
            low_battery_mask = depletion_mask & (curr_battery < 50)
            high_battery_mask = depletion_mask & (curr_battery >= 50)

            k_score = torch.zeros(batch_size, device=curr_battery.device)
            if low_battery_mask.any():
                battery_pct = torch.clamp(curr_battery[low_battery_mask], min=0.1)
                k_score[low_battery_mask] = k_low / torch.exp(battery_pct / 50)
            if high_battery_mask.any():
                k_score[high_battery_mask] = k_high

            depletion_amount = torch.maximum(
                depletion_min.expand(batch_size),
                (modified_press - self.press_thresh) * 10
            ) * k_score
            delta[depletion_mask] = -depletion_amount[depletion_mask] / self.Rc * 100

        # Awake + Replenishment (low pressure)
        replenish_mask = awake_mask & (modified_press < self.press_thresh)
        if replenish_mask.any():
            replenish_amount = si * (self.press_thresh - modified_press)
            delta[replenish_mask] = replenish_amount[replenish_mask] / self.Rc * 100

        # Sleep state - Recovery
        sleep_mask = (sleep_state == 1)
        if sleep_mask.any():
            # Get stage multiplier
            stage_clamped = torch.clamp(sleep_stage, 0, 2).long()
            stage_mult = stage_mults[stage_clamped]

            # Recovery exponent
            recovery_exp = torch.where(
                modified_press >= self.press_thresh,
                self.mental_recovery_exp_high_press,
                self.mental_recovery_exp_low_press
            )

            # Recovery formula
            recovery_amount = recovery_si_mult * (si / torch.exp(recovery_exp * sleep_quality / stage_mult)) * hrv_factor
            delta[sleep_mask] = recovery_amount[sleep_mask] / self.Rc * 100

        return delta

    def compute_physical_delta(self, curr_battery, hrr, acc, sleep_state, sleep_stage,
                                sleep_quality, exertion_score, CP=15.0, acc_threshold=5.0):
        """
        Compute physical battery delta using physics equations.

        Args:
            curr_battery: (batch,) current battery percentage [0-100]
            hrr: (batch,) heart rate reserve
            acc: (batch,) acceleration
            sleep_state: (batch,) 0=awake, 1=asleep
            sleep_stage: (batch,) 0=deep, 1=light, 2=REM
            sleep_quality: (batch,) weighted sleep quality score
            exertion_score: (batch,) daily exertion score
            CP: critical power threshold
            acc_threshold: acceleration threshold
        """
        batch_size = curr_battery.shape[0]

        # Get parameters
        expenditure_k = torch.exp(self.physical_log_expenditure_k_val)
        recovery_k1 = torch.exp(self.physical_log_recovery_k1)
        recovery_k2 = torch.exp(self.physical_log_recovery_k2)
        recovery_k3 = torch.exp(self.physical_log_recovery_k3)
        recovery_mult = torch.exp(self.physical_log_recovery_multiplier)
        stage_mults = torch.exp(self.physical_log_stage_multipliers)  # [deep, light, REM]

        # Convert battery to expenditure (internal representation)
        expenditure = 100 - curr_battery

        # Modify HRR by exertion
        modified_hrr = hrr * torch.exp(0.05 * (exertion_score - 50))

        # Initialize new expenditure
        new_expenditure = expenditure.clone()

        # Expenditure state: high HRR and high acceleration
        expenditure_mask = (modified_hrr >= CP) & (acc > acc_threshold)

        # Expenditure while awake
        exp_awake = expenditure_mask & (sleep_state == 0)
        if exp_awake.any():
            k_val = expenditure_k * torch.where(modified_hrr <= CP, 0.8, 1.0)
            new_exp = expenditure + (modified_hrr - CP) * k_val
            new_expenditure[exp_awake] = new_exp[exp_awake]

        # Expenditure while asleep
        exp_asleep = expenditure_mask & (sleep_state == 1)
        if exp_asleep.any():
            stage_clamped = torch.clamp(sleep_stage, 0, 2).long()
            stage_mult = stage_mults[stage_clamped]
            k_val_recovery = recovery_mult / expenditure_k
            denominator = stage_mult * k_val_recovery * (201 - expenditure)
            exponent = (modified_hrr - 100) * sleep_quality / denominator
            new_exp = expenditure * torch.exp(exponent)
            new_expenditure[exp_asleep] = new_exp[exp_asleep]

        # Recovery state: low HRR or low acceleration
        recovery_mask = ~expenditure_mask

        # Recovery from low HRR
        low_hrr_mask = recovery_mask & (modified_hrr < CP)

        # Mid-range battery (40-80)
        mid_range = (curr_battery >= 40) & (curr_battery <= 80)
        recovery_mid = low_hrr_mask & mid_range

        if recovery_mid.any():
            stage_clamped = torch.clamp(sleep_stage, 0, 2).long()
            stage_mult = stage_mults[stage_clamped]

            # Awake
            awake_mid = recovery_mid & (sleep_state == 0)
            if awake_mid.any():
                k_rm = recovery_k1 * recovery_mult
                new_exp = expenditure * torch.exp((modified_hrr - 100) / k_rm)
                new_expenditure[awake_mid] = new_exp[awake_mid]

            # Asleep
            asleep_mid = recovery_mid & (sleep_state == 1)
            if asleep_mid.any():
                denominator = stage_mult * recovery_k1
                new_exp = expenditure * torch.exp((modified_hrr - 100) * sleep_quality / denominator)
                new_expenditure[asleep_mid] = new_exp[asleep_mid]

        # Low/high battery range
        recovery_extreme = low_hrr_mask & ~mid_range
        if recovery_extreme.any():
            stage_clamped = torch.clamp(sleep_stage, 0, 2).long()
            stage_mult = stage_mults[stage_clamped]

            awake_ext = recovery_extreme & (sleep_state == 0)
            if awake_ext.any():
                k_rm = recovery_k2 * recovery_mult
                new_exp = expenditure * torch.exp(-(CP - modified_hrr) / (k_rm * (201 - expenditure)))
                new_expenditure[awake_ext] = new_exp[awake_ext]

            asleep_ext = recovery_extreme & (sleep_state == 1)
            if asleep_ext.any():
                denominator = stage_mult * (recovery_k2 * (201 - expenditure))
                new_exp = expenditure * torch.exp(-(CP - modified_hrr) * sleep_quality / denominator)
                new_expenditure[asleep_ext] = new_exp[asleep_ext]

        # Recovery from low acceleration (high HRR but low acc)
        low_acc_mask = recovery_mask & (modified_hrr >= CP)
        if low_acc_mask.any():
            stage_clamped = torch.clamp(sleep_stage, 0, 2).long()
            stage_mult = stage_mults[stage_clamped]

            awake_low_acc = low_acc_mask & (sleep_state == 0)
            if awake_low_acc.any():
                k_rm = recovery_k3 * recovery_mult
                new_exp = expenditure * torch.exp((modified_hrr - 100) / (k_rm * (201 - expenditure)))
                new_expenditure[awake_low_acc] = new_exp[awake_low_acc]

            asleep_low_acc = low_acc_mask & (sleep_state == 1)
            if asleep_low_acc.any():
                denominator = stage_mult * (recovery_k3 * (201 - expenditure))
                new_exp = expenditure * torch.exp((modified_hrr - 100) * sleep_quality / denominator)
                new_expenditure[asleep_low_acc] = new_exp[asleep_low_acc]

        # Clamp expenditure
        new_expenditure = torch.clamp(new_expenditure, min=0)

        # Convert back to battery and compute delta
        new_battery = 100 - new_expenditure
        delta = new_battery - curr_battery

        return delta

    def forward(self, features, curr_mental_battery, curr_physical_battery,
                curr_press, hrr, acc, sleep_state, sleep_stage,
                sleep_quality, hrv_factor, exertion_score, time_of_day):
        """
        Forward pass using physics equations + residual MLP.

        Args:
            features: (batch, input_dim) other input features
            curr_mental_battery: (batch,) current mental battery [0-100]
            curr_physical_battery: (batch,) current physical battery [0-100]
            curr_press: (batch,) current pressure/stress
            hrr: (batch,) heart rate reserve
            acc: (batch,) acceleration
            sleep_state: (batch,) 0=awake, 1=asleep
            sleep_stage: (batch,) sleep stage code
            sleep_quality: (batch,) weighted sleep quality
            hrv_factor: (batch,) HRV recovery factor
            exertion_score: (batch,) daily exertion score
            time_of_day: (batch,) hour of day

        Returns:
            mental_delta: (batch, 1) mental battery change
            physical_delta: (batch, 1) physical battery change
        """
        # Compute physics-based deltas
        mental_physics_delta = self.compute_mental_delta(
            curr_mental_battery, curr_press, sleep_state, sleep_stage,
            sleep_quality, hrv_factor, exertion_score, time_of_day
        )

        physical_physics_delta = self.compute_physical_delta(
            curr_physical_battery, hrr, acc, sleep_state, sleep_stage,
            sleep_quality, exertion_score
        )

        # Concatenate physics deltas with features for residual learning
        physics_deltas = torch.stack([mental_physics_delta, physical_physics_delta], dim=-1)  # (batch, 2)
        x = torch.cat([features, physics_deltas], dim=-1)  # (batch, input_dim + 2)

        # Residual correction from MLP
        residual = self.mlp(x)  # (batch, 1)

        # Final deltas = physics + residual (applied equally to both for simplicity)
        # Alternatively, you could have separate residuals for mental and physical
        mental_delta = mental_physics_delta.unsqueeze(-1) + residual * 0.5
        physical_delta = physical_physics_delta.unsqueeze(-1) + residual * 0.5

        return mental_delta, physical_delta


class GatedDualHeadMLP(nn.Module):
    """
    Gated dual-head architecture for biocharge delta prediction.

    Mirrors the analytical model's hard if/else branching on sleep_state:
    - Awake head: learns stress-driven depletion, micro-recovery, HRR/ACC expenditure
    - Sleep head: learns stage/quality/HRV-modulated exponential recovery
    - Hard gate on sleep_markers selects which head's output to use
    - Shared encoder projects all features into a common representation

    Uses SiLU (Swish) activation to match the analytical model's exponential formulas.
    ~10k parameters with default dims (shared=48, awake=64/32, sleep=48/24).
    """
    def __init__(self, input_dim=21, shared_dim=48, awake_hidden=64,
                 awake_hidden2=32, sleep_hidden=48, sleep_hidden2=24,
                 dropout=0.1, gate_idx=8):
        super().__init__()
        self.gate_idx = gate_idx  # sleep_markers index in feature vector

        # Shared encoder
        self.shared = nn.Linear(input_dim, shared_dim)
        self.shared_ln = nn.LayerNorm(shared_dim)

        # Awake head (larger — more complex dynamics)
        self.awake_h1 = nn.Linear(shared_dim, awake_hidden)
        self.awake_ln1 = nn.LayerNorm(awake_hidden)
        self.awake_h2 = nn.Linear(awake_hidden, awake_hidden2)
        self.awake_ln2 = nn.LayerNorm(awake_hidden2)
        self.awake_out = nn.Linear(awake_hidden2, 1)

        # Sleep head (smaller — regular exponential recovery)
        self.sleep_h1 = nn.Linear(shared_dim, sleep_hidden)
        self.sleep_ln1 = nn.LayerNorm(sleep_hidden)
        self.sleep_h2 = nn.Linear(sleep_hidden, sleep_hidden2)
        self.sleep_ln2 = nn.LayerNorm(sleep_hidden2)
        self.sleep_out = nn.Linear(sleep_hidden2, 1)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        gate = x[:, self.gate_idx].unsqueeze(-1)  # 0=awake, 1=sleep

        h = self.act(self.shared_ln(self.shared(x)))
        h = self.dropout(h)

        a = self.act(self.awake_ln1(self.awake_h1(h)))
        a = self.dropout(a)
        a = self.act(self.awake_ln2(self.awake_h2(a)))
        awake_delta = self.awake_out(a)

        s = self.act(self.sleep_ln1(self.sleep_h1(h)))
        s = self.dropout(s)
        s = self.act(self.sleep_ln2(self.sleep_h2(s)))
        sleep_delta = self.sleep_out(s)

        delta = gate * sleep_delta + (1 - gate) * awake_delta
        return delta
