#%% 

import torch
from torch.distributions import Normal
from torchvision.transforms import Resize
from math import log, sqrt
from torch import nn 

from utils import duration, print, print_duration



def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

def episodes_steps(this):
    return(this.shape[0], this.shape[1])



def pad_zeros(value, length):
    rows_to_add = length - value.size(-2)
    if(rows_to_add == 0):
        return(value)
    padding_shape = list(value.shape)
    padding_shape[-2] = rows_to_add
    if(value.get_device() == -1):
        device = "cpu"
    else:
        device = value.get_device()
    padding = torch.zeros(padding_shape).to(device)
    padding[..., 0] = 1
    value = torch.cat([value, padding], dim=-2)
    return value



def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

def sample(mu, std, device):
    e = Normal(0, 1).sample(std.shape).to(device)
    return(mu + e * std)

def rnn_cnn(do_this, to_this):
    episodes, steps = episodes_steps(to_this)
    this = to_this.view((episodes * steps, to_this.shape[2], to_this.shape[3], to_this.shape[4]))
    this = do_this(this)
    this = this.view((episodes, steps, this.shape[1], this.shape[2], this.shape[3]))
    return(this)



def model_start(model_input_list, device = "cpu", half = False, recurrent = False):
    start_time = duration()
    new_model_inputs = []
    for model_input, layer_type in model_input_list:
        model_input = model_input.to(device)
        if(half):
            model_input = model_input.to(dtype=torch.float16)
        if(layer_type == "lin"):
            if(len(model_input.shape) == 2):   model_input = model_input.unsqueeze(1)
            episodes, steps = episodes_steps(model_input)
            if(not recurrent):
                model_input = model_input.reshape(episodes * steps, model_input.shape[2])
        if(layer_type == "cnn"):
            if(len(model_input.shape) == 4): model_input = model_input.unsqueeze(1)
            episodes, steps = episodes_steps(model_input)
            model_input = model_input.reshape(episodes * steps, model_input.shape[2], model_input.shape[3], model_input.shape[4]).permute(0, -1, 1, 2)
        if(layer_type == "voice"):
            if(len(model_input.shape) == 2):  model_input = model_input.unsqueeze(0)
            if(len(model_input.shape) == 3):  model_input = model_input.unsqueeze(1)
            episodes, steps = episodes_steps(model_input)
            model_input = model_input.reshape(episodes * steps, model_input.shape[2], model_input.shape[3])
        new_model_inputs.append(model_input)
    return start_time, episodes, steps, new_model_inputs



def model_end(start_time, episodes, steps, model_output_list, duration_text = None):
    new_model_outputs = []
    for model_output, layer_type in model_output_list:
        if(layer_type == "lin"):
            model_output = model_output.reshape(episodes, steps, model_output.shape[-1])
        if(layer_type == "cnn"):
            model_output = model_output.permute(0, 2, 3, 1)
            model_output = model_output.reshape(episodes, steps, model_output.shape[1], model_output.shape[2], model_output.shape[3])
        if(layer_type == "voice"):
            model_output = model_output.reshape(episodes, steps, model_output.shape[1], model_output.shape[2])
        new_model_outputs.append(model_output)
    print_duration(start_time, duration(), duration_text)
    return new_model_outputs