import numpy as np
import json
import math

EPISON = 1e-10

def domain_readiness_hrv(x,age):
    """
        初始化
        当用户历史数据不足时，按照高医生提供的先验知识打分，分段线性函数  
        f(0)=0,f(30)=80, f(100)=100, f(200)=60 age<=50
        f(15)=60,f(20)=80, f(60)=100, f(90)=60, age>50
    """
    if age<=50:
        if x<=30:
            a=8/3
            b=0
        elif x<=100:
            a=20/70
            b=80-30*a
        else:
            a=-0.4
            b=60-a*200
    else:
        if x<=20:
            a=4
            b=60-15*a
        elif x<=60:
            a=0.5
            b=70
        else:
            a=-4/3
            b=100-a*60
    y=max(min(a*x+b,100),0)
    return y

def domain_readiness_hrv_update(hrv_input_list,age,baseline_flag):
    """
    hrv_input_list: [前一天的子项打分, 当天是运行的第几天,当天的hrv值,hrv的历史均值,hrv的标准差]
    """
    S_pre = hrv_input_list[0]
    t     = hrv_input_list[1]
    curr  = hrv_input_list[2]
    mean  = hrv_input_list[3]
    std   = hrv_input_list[4]
    Xt = domain_readiness_hrv(curr,age)

    if t <= 30 :
        a = 0.8
    else:
        a = 1

    if t <= 7:
        b = 0
    else:
        b = min(1,t/30)   
    
    Zt = (curr - mean) / (std + EPISON)
    if baseline_flag == True:
        if Zt >= 0 :
            Zt = Zt * np.exp(-Zt/20)
        else:
            Zt = Zt * np.exp(Zt/10)
    
    if age < 50:
        if curr <= 100:
            Yt = 10*Zt + 80
        else:
            Yt = -0.4*curr +140
    else:
        if curr <= 60:
            Yt = 10*Zt + 80
        else:
            Yt = (-4/3)*curr + 180
    # 如果是第一天的话默认使用Xt
    S_pre = Xt if S_pre == 0 else S_pre
    Score = a*((1-b)*Xt + b*Yt)+(1-a)*S_pre

    return Score

def domain_readiness_rhr(x_in):
    """
       当用户历史数据不足时，按照高医生提供的先验知识打分，分段线性函数
       f(30)=59,f(40)=100, f(80)=79, f(90)=60
    """
    if x_in>=80:
        a=-1.9
        b=79+80*1.9
    elif x_in>=40:
        a=-21/40
        b=200-79
    else:
        a=41/10
        b=59-30*a
    y=max(min(a*x_in+b,100),0)
    return y

def domain_readiness_rhr_update(rhr_input_list,baseline_flag):
    """
    rhr_input_list: [前一天的子项打分, 当天是运行的第几天,当天的rhr值,rhr的历史均值,rhr的标准差]
    """
    S_pre = rhr_input_list[0]
    t     = rhr_input_list[1]
    curr  = rhr_input_list[2]
    mean  = rhr_input_list[3]
    std   = rhr_input_list[4]
    Xt = domain_readiness_rhr(curr)

    if t <= 30 :
        a = 0.8
    else:
        a = 1

    if t <= 7:
        b = 0
    else:
        b = min(1,t/30)   
    
    Zt = (curr - mean) / (std + EPISON)
    if baseline_flag == True:
        if Zt >= 0 :
            Zt = Zt * np.exp(-Zt/20)
        else:
            Zt = Zt * np.exp(Zt/10)

    if curr >= 40:
        Yt = -12.5*Zt + 85
    else:
        Yt = 4.1*curr - 64

    S_pre = Xt if S_pre == 0 else S_pre
    Score = a*((1-b)*Xt + b*Yt)+(1-a)*S_pre
    return Score

def domain_readiness_temperature(x):
    """
       当用户历史数据不足时，按照高医生提供的先验知识打分，分段线性函数  
       temperature为正时，pay attention上界是temperature为1，因此f(1)=59, f(0)=100,得到相应的a,b取值
       temperature为负时，按照f(-1)=80, f(-2)=55得到线性函数a,b值
    """
    if x>0:
        a=-41
        b=100
    else:
        a=25
        b=105
    y=max(min(a*x+b,100),0)
    return y

def domain_readiness_temperature_update(temp_input_list,baseline_flag):
    """
    temp_input_list: [前一天的子项打分, 当天是运行的第几天,当天的temp值,temp的历史均值,temp的标准差]
    """
    S_pre = temp_input_list[0]
    t     = temp_input_list[1]
    curr  = temp_input_list[2]
    mean  = temp_input_list[3]
    std   = temp_input_list[4]
    Xt = domain_readiness_temperature(curr)

    a = 0.9

    if t <= 7:
        b = 0
    else:
        b = min(1,t/30)   
    
    # 均值方差用的是校准后的温度的均值和方差
    Zt = (curr - mean) / (std + EPISON)
    if baseline_flag == True:
        if Zt >= 0 :
            Zt = Zt * np.exp(-Zt/30)
        else:
            Zt = Zt * np.exp(Zt/30)
    if Zt > 0.15:
        Yt = 105 - 25*Zt
    else:
        Yt = 105 + 10*Zt
    S_pre = Xt if S_pre == 0 else S_pre
    Score = a*((1-b)*Xt + b*Yt)+(1-a)*S_pre
    Score = min(max(Score,0),100) 
    return Score

def domain_knowledge_function_monotonic_increase(x):
    """
        基于某项指标的单调递增逻辑的打分
    """
    y= min(max(80+10*x,0),100)
    return y

# def domain_knowledge_function_monotonic_decrease(x):
#     """
#         基于某项指标的单调递减逻辑的打分
#     """
#     y= min(max(85-12.5*x,0),100)
#     return y

def domain_knowlege_function_temperature(x):
    """
        基于温度数据的分段函数打分逻辑
    """
    if x>0.15:
        y=105-25*x
    else:
        y=105+10*x
    y=min(max(y,0),100)
    return y

def get_rhr_readiness_score(inputs_RHR):
    """
        基于rhr数值的readiness的计算逻辑
    """
    rhr=inputs_RHR['curr']
        # 判断是不是第一天
    if inputs_RHR['t_days'] == 0:
        score = domain_readiness_rhr(rhr)
    else:
        rhr_input_list = [inputs_RHR['pre'],inputs_RHR['t_days'],inputs_RHR['curr'],inputs_RHR['lt_mean'],inputs_RHR['lt_std']]
        baseline_flag = True
        score = domain_readiness_rhr_update(rhr_input_list,baseline_flag)
    return score

def get_hrv_readiness_score(inputs_hrv,age):
    """
        基于hrv数值的readiness的计算逻辑
    """
    hrv=inputs_hrv['curr']
    if inputs_hrv['t_days'] == 0:
        score=domain_readiness_hrv(hrv,age)
    else:
        hrv_input_list = [inputs_hrv['pre'],inputs_hrv['t_days'],inputs_hrv['curr'],inputs_hrv['lt_mean'],inputs_hrv['lt_std']]
        baseline_flag = True
        score = domain_readiness_hrv_update(hrv_input_list,age,baseline_flag)
    return score 

def get_temperature_readiness_score(inputs_temperature): 
    """
        基于体温数据的readiness的计算逻辑
    """
    temperature = inputs_temperature['curr']
    if inputs_temperature['t_days'] == 0:
        score = domain_readiness_temperature(temperature)
    else:
        temp_input_list = [inputs_temperature['pre'],inputs_temperature['t_days'],inputs_temperature['curr'],inputs_temperature['lt_mean'],inputs_temperature['lt_std']]
        baseline_flag = True
        # temp_input_list = [100,3,0.39,0.3374,0.003677861697599132] 
        # temp_input_list = [100,3,1,0.5,0.5]
        score = domain_readiness_temperature_update(temp_input_list,baseline_flag)
    return score

def get_osa_score(ahi,n_abnormal):
    if n_abnormal>=3:
        judge_range=[0,5,10,25]
    else:
        judge_range=[0,5,15,30]
    if ahi<judge_range[1]:
        score=100
    elif judge_range[1] <ahi<=judge_range[2]:
        score=79
    elif judge_range[2] <ahi<=judge_range[3]:
        score=59
    else:
        score=0
    return score 

def weights_for_items(item_scores):
    """
        Calculate weights for readiness items using a non-linear weighting scheme.
        
        The weighting is done using an exponential decay function:
        - For each item score, calculate weight = exp(-(score/100)^2)
        - This gives lower weights to higher scores (exponential decay)
        - The weights are then normalized by dividing by their sum
        
        For example:
        - A score of 100 gets weight = exp(-1) = 0.368
        - A score of 50 gets weight = exp(-0.25) = 0.779
        - A score of 0 gets weight = exp(0) = 1.0
        
        This weighting scheme:
        - Gives more importance to lower scores that indicate issues
        - De-emphasizes high scores that indicate good readiness
        - Helps identify potential problems by weighting concerning scores more heavily
        
        Args:
            item_scores: List of individual readiness component scores (0-100)
            
        Returns:
            Tuple of:
            - Weighted average score (rounded down to integer)
            - List of calculated weights for each item
    """
    item_scores =  np.array(item_scores)
    # weights     =  item_scores/100
    # weights     =  np.exp(-weights)
    weights = [math.exp(-(i/100)**2) for i in item_scores]
    # weights = [math.exp(-(i/60)**2) for i in item_scores]
    weights=weights/(np.sum(weights))
    weighted_score=item_scores*weights
    if np.isnan(np.sum(weighted_score)):
        return (0,weights)
    return (math.floor(np.sum(weighted_score)),weights)

def calc_readiness_main(readiness_inputs):
    """
        根据各项指标的入参数据得到总readiness的主函数
    """
    inputs_physical=readiness_inputs['morning_physical']
    age = readiness_inputs['age']
    physical_valid = inputs_physical['flag']

    if physical_valid==0:
        physical_readiness=-1
    else:
        # inputs_physical_list = [inputs_physical['pre'],inputs_physical['t_days'],inputs_physical['curr'],inputs_physical['lt_mean'],inputs_physical['lt_std']]
        physical_readiness = inputs_physical['curr']

    inputs_mental=readiness_inputs['morning_mental']
    mental_valid = inputs_mental['flag']
    if mental_valid==0:
        mental_readiness=-1
    else:
        # inputs_mental_list = [inputs_mental['pre'],inputs_mental['t_days'],inputs_mental['curr'],inputs_mental['lt_mean'],inputs_mental['lt_std']]
        mental_readiness = inputs_mental['curr']

    inputs_rhr=readiness_inputs['rhr']
    rhr_valid = inputs_rhr['flag']
    if rhr_valid==0:
        rhr_readiness=-1
    else:
        rhr_readiness = get_rhr_readiness_score(inputs_rhr)

    inputs_hrv=readiness_inputs['hrv']
    hrv_valid = inputs_hrv['flag']
    if hrv_valid==0:
        hrv_readiness=-1
    else:
        hrv_readiness  = get_hrv_readiness_score(inputs_hrv,age)
    
    inputs_temperature=readiness_inputs['temperature']
    temperature_valid = inputs_temperature['flag']
    if temperature_valid==0:
        temperature_readiness=-1
    else:
        # mean_temp = np.mean(inputs_temperature['hist'])
        # inputs_temperature['hist'] = (np.array(inputs_temperature['hist'])-mean_temp).tolist()
        # inputs_temperature['curr'] = inputs_temperature['curr']-mean_temp
        temperature_readiness  = get_temperature_readiness_score(inputs_temperature)

    inputs_af=readiness_inputs['af']
    af_readiness=-1
    # if inputs_af==-1:
    #     af_readiness = -1
    # else:
    #     af_readiness  = get_illness_readiness_score(inputs_af)
    
    ahi=readiness_inputs['osa']['ahi']
    osa_count=readiness_inputs['osa']['count']

    if ahi==-1:
        osa_readiness=-1
    else:
        osa_readiness  = get_osa_score(ahi,osa_count)
    
    all_items = [physical_readiness,mental_readiness,rhr_readiness,hrv_readiness,temperature_readiness,af_readiness,osa_readiness]
    valid_items = [tmpitem for tmpitem in all_items if tmpitem!=-1]
    
    item_tags = ['morning_physical','morning_mental','rhr','hrv','temperature','af','osa']
    valid_item_tag = np.array([item_tags[idx] for idx in range(len(all_items)) if all_items[idx]!=-1])
    
    total_readiness,item_weights = weights_for_items(valid_items)

    indexes = np.argsort(item_weights)[::-1]
    sorted_tag = valid_item_tag[indexes]

    temperature_sign = 1 if readiness_inputs['temperature']['curr']>0 else 0
    rhr_sign  = 1 if readiness_inputs['rhr']['curr'] >= readiness_inputs['rhr']['valid_range'][0] else 0
    hrv_sign  = 1 if readiness_inputs['hrv']['curr'] > readiness_inputs['hrv']['valid_range'][1] else 0


    outputs={}
    outputs['morning_physical']=physical_readiness
    outputs['morning_mental']=mental_readiness
    outputs['rhr']=rhr_readiness
    outputs['hrv']=hrv_readiness
    outputs['temperature']=temperature_readiness
    outputs['osa']=osa_readiness
    outputs['af']=af_readiness
    outputs['readiness_total']=total_readiness
    outputs['tag_weight']=sorted_tag.tolist()
    outputs['temperature_sign']=temperature_sign
    outputs['rhr_sign']=rhr_sign
    outputs['hrv_sign']=hrv_sign
    outputs['slp']=readiness_inputs['slp']
    return outputs


## New Readiness Score Functions
## ______________________________________________________________________________________________

def calculate_personalized_weight(current_score, mean_score):
    """
    Calculates a personalized weight based on the difference between a score and a mean.

    This function uses a modified sigmoid function. It is fortified to handle
    non-numeric inputs and potential floating-point overflow errors during the
    exponential calculation.

    Args:
        current_score: The current score (should be a numeric type).
        mean_score: The mean score to compare against (should be a numeric type).

    Returns:
        float: A weight value between 0.0 and 0.5. 
               Returns 0.0 if inputs are invalid.
               Returns 0.5 if a numeric overflow occurs.
    """
    try:
        # Calculate the argument for the exponential function.
        z = -0.05 * (float(current_score) - float(mean_score))

        # The np.exp() function can overflow if z is a large positive number.
        sigmoid_val = 1 / (1 + np.exp(z))
        
        # Calculate the final weight
        w = abs(sigmoid_val - 0.5)
        
        return w

    except (ValueError, TypeError):
        # If inputs are not numeric, return a default weight of 0.0
        return 0.0
        
    except OverflowError:
        # If np.exp(z) overflows, the sigmoid value is effectively 0.
        # The resulting weight is abs(0 - 0.5) = 0.5.
        return 0.5

def calculate_bias(score_values):
    """
    Calculates bias for a list of scores, where bias = (100 - score) / 100.

    This function is fortified to handle non-iterable inputs and lists that
    contain non-numeric values by skipping them.

    Args:
        score_values: An iterable (e.g., a list) of scores.

    Returns:
        list: A list of calculated bias values.
              Returns an empty list if the input is invalid or contains no
              valid numeric scores.
    """
    bias_list = []
    try:
        for score in score_values:
            try:
                # Attempt to convert score to float and calculate bias
                bias = (100 - float(score)) / 100
                bias_list.append(bias)
            except (ValueError, TypeError):
                # If a specific item is not a number, skip it and continue.
                continue
        return bias_list
    except TypeError:
        # If score_values is not an iterable (e.g., an integer was passed),
        # return an empty list.
        return []

def calculate_personalized_softmax_weights(scores, temperature=1.0, alpha=None, bias=None):
    """
    Calculates weights using a personalized softmax function.

    Args:
        scores (list or np.array): The list of raw input scores.
        temperature (float): Controls the sharpness of the distribution.
        alpha (list or np.array, optional): Input scaling factors for each score.
        bias (list or np.array, optional): Bias values to add to each score.

    Returns:
        np.array: The resulting weights, which sum to 1.
    """
    scores = np.array(scores)

    # Set defaults if hyperparameters are not provided
    # An alpha of 1 and a bias of 0 have no effect on the calculation.
    if alpha is None:
        alpha = np.ones_like(scores)
    if bias is None:
        bias = np.zeros_like(scores)

    # Apply all hyperparameters to the scores
    transformed_scores = (alpha * scores + bias) / temperature
    
    # Calculate the softmax using the transformed scores
    # The subtraction of max(transformed_scores) is for numerical stability
    exps = np.exp(transformed_scores - np.max(transformed_scores))
    
    return exps / np.sum(exps)

def multiply_sum_weights_scores(score_values, score_weights):
    """
    Calculates the weighted sum of scores and clamps the result between 0 and 100.

    This function is fortified to handle non-iterable inputs, lists of
    mismatched lengths, and lists containing non-numeric values.

    Args:
        score_values (list): A list of numeric scores.
        score_weights (list): A list of corresponding numeric weights.

    Returns:
        float: The weighted score, clamped between 0 and 100.
               Returns 0.0 if inputs are invalid or cannot be processed.
    """
    try:
        # Check if the lengths of the lists are mismatched.
        if len(score_values) != len(score_weights):
            return 0.0

        total_sum = 0
        for score, weight in zip(score_values, score_weights):
            try:
                # Attempt to multiply, converting to float to handle ints/floats.
                total_sum += float(score) * float(weight)
            except (ValueError, TypeError, AttributeError):
                # If any value is not numeric (e.g., None, string), skip this pair.
                continue
        
        # Clamp the final rounded value between 0 and 100.
        return min(100, max(0, round(total_sum, 2)))

    except TypeError:
        # If either input is not a list/iterable (can't call len()), return 0.0.
        return np.nan