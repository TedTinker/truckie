#%%
#------------------
# truck_models.py provides the encoders and decoders the robot needs, written for
# general_FEP_RL. They follow the original communication project's submodules.py
# (Vision_IN, Vision_OUT, Touch_IN, Touch_OUT, Voice_IN, Voice_OUT, Wheels_Joints_IN)
# layer for layer, with two changes:
#
#   1. Each is a Shape_to_Shape_Model, so it takes (batch, episode_length, *input_shape)
#      and returns (batch, episode_length, *output_shape), and the buffer reads its
#      shapes off the built world model.
#   2. Touch and proprioception are separate. In the original, joint angles and speeds
#      rode along inside touch; here prop has its own inner state, scaler, beta and eta.
#
# Decoders get input_size from the world model layer (the width of all inner states
# together), so that argument stays open in the partials in agent.py.
#------------------

import torch
from torch import nn
import torch.nn.functional as F

from general_FEP_RL.shape_to_shape_models import Shape_to_Shape_Model
from general_FEP_RL.actor_critic import Action_Decoder



# Same initialisation as the original utils_submodule.init_weights.
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)



def flatten_steps(value, shape):
    batch_size, episode_length = value.shape[:2]
    return value.reshape(batch_size * episode_length, *shape), batch_size, episode_length



#------------------
# Vision: (image_size, image_size, 4) channels last, rgb + depth in [0, 1].
#------------------

class Vision_Encoder(Shape_to_Shape_Model):
    """Original Vision_IN: no convolution, just rescale to [-1, 1], flatten, linear."""

    def __init__(self, name, image_size, output_size, dropout = .001, verbose = False):
        super().__init__(
            name = name,
            input_shape = (image_size, image_size, 4),
            output_shape = (output_size,),
            arg_dict = {'dropout' : dropout},
            verbose = verbose)

    def build_model(self, arg_dict):
        self.model = nn.Sequential(
            nn.Linear(
                in_features = self.input_shape[0] * self.input_shape[1] * self.input_shape[2],
                out_features = self.output_shape[0]),
            nn.PReLU(),
            nn.Dropout(arg_dict['dropout']))
        self.apply(init_weights)

    def forward(self, value):
        value, batch_size, episode_length = flatten_steps(value, self.input_shape)
        value = (value * 2) - 1
        encoding = self.model(value.flatten(1))
        return encoding.reshape(batch_size, episode_length, *self.output_shape)



class Vision_Decoder(Shape_to_Shape_Model):
    """Original Vision_OUT: linear to a small feature map, conv, then PixelShuffle up."""

    def __init__(
            self, name, input_size, image_size,
            hidden_channels = 64, divisions = 2, dropout = .001, verbose = False):
        if image_size % divisions != 0:
            raise ValueError(f"image_size {image_size} must be divisible by divisions {divisions}.")
        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (image_size, image_size, 4),
            arg_dict = {'hidden_channels' : hidden_channels, 'divisions' : divisions, 'dropout' : dropout},
            verbose = verbose)

    def build_model(self, arg_dict):
        self.hidden_channels = arg_dict['hidden_channels']
        self.divisions = arg_dict['divisions']
        self.small_size = self.output_shape[0] // self.divisions

        self.a = nn.Linear(
            in_features = self.input_shape[0],
            out_features = self.hidden_channels * self.small_size * self.small_size)
        self.b = nn.Sequential(
            nn.BatchNorm2d(self.hidden_channels),           # Original: "Tested, use this"
            nn.PReLU(),
            nn.Dropout(arg_dict['dropout']),
            nn.Conv2d(
                in_channels = self.hidden_channels,
                out_channels = 4 * self.divisions ** 2,     # PixelShuffle needs 4 * r^2 channels.
                kernel_size = 3,
                padding = 1,
                padding_mode = 'reflect'),
            nn.PixelShuffle(self.divisions),
            nn.Tanh())
        self.apply(init_weights)

    def forward(self, value):
        value, batch_size, episode_length = flatten_steps(value, self.input_shape)
        value = self.a(value).reshape(-1, self.hidden_channels, self.small_size, self.small_size)
        vision = (self.b(value) + 1) / 2                    # [0, 1]
        vision = vision.permute(0, 2, 3, 1)                 # channels last, like the observation
        return vision.reshape(batch_size, episode_length, *self.output_shape)

    @staticmethod
    def loss_func(predicted_values, target_values):
        # Original used binary cross-entropy for vision. reduction='none': the epoch masks.
        return F.binary_cross_entropy(predicted_values, target_values, reduction = 'none')



#------------------
# Touch and proprioception: vectors in [0, 1].
#------------------

class Sensor_Encoder(Shape_to_Shape_Model):
    """Original Touch_IN: linear, batch norm, PReLU."""

    def __init__(self, name, input_size, output_size, verbose = False):
        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (output_size,),
            verbose = verbose)

    def build_model(self, arg_dict):
        self.model = nn.Sequential(
            nn.Linear(self.input_shape[0], self.output_shape[0]),
            nn.BatchNorm1d(self.output_shape[0]),           # Original: "Tested, use this"
            nn.PReLU())
        self.apply(init_weights)

    def forward(self, value):
        value, batch_size, episode_length = flatten_steps(value, self.input_shape)
        return self.model(value).reshape(batch_size, episode_length, *self.output_shape)



class Sensor_Decoder(Shape_to_Shape_Model):
    """Original Touch_OUT: linear, tanh, rescaled to [0, 1]."""

    def __init__(self, name, input_size, output_size, verbose = False):
        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (output_size,),
            verbose = verbose)

    def build_model(self, arg_dict):
        self.model = nn.Sequential(
            nn.Linear(self.input_shape[0], self.output_shape[0]),
            nn.Tanh())
        self.apply(init_weights)

    def forward(self, value):
        value, batch_size, episode_length = flatten_steps(value, self.input_shape)
        value = (self.model(value) + 1) / 2
        return value.reshape(batch_size, episode_length, *self.output_shape)

    @staticmethod
    def loss_func(predicted_values, target_values):
        return F.mse_loss(predicted_values, target_values, reduction = 'none')



#------------------
# Voice: (max_voice_len, voice_shape) one-hots, one row per character.
#------------------

class Voice_Encoder(Shape_to_Shape_Model):
    """Original Voice_IN: embed each character, GRU over the characters, linear."""

    def __init__(
            self, name, max_voice_len, voice_shape, output_size,
            char_encode_size = 8, hidden_size = 64, dropout = .001, verbose = False):
        super().__init__(
            name = name,
            input_shape = (max_voice_len, voice_shape),
            output_shape = (output_size,),
            arg_dict = {'char_encode_size' : char_encode_size, 'hidden_size' : hidden_size, 'dropout' : dropout},
            verbose = verbose)

    def build_model(self, arg_dict):
        hidden_size = arg_dict['hidden_size']
        self.embed = nn.Sequential(
            nn.Embedding(
                num_embeddings = self.input_shape[1],
                embedding_dim = arg_dict['char_encode_size']),
            nn.PReLU(),
            nn.Dropout(arg_dict['dropout']),
            nn.Linear(arg_dict['char_encode_size'], hidden_size),
            nn.PReLU())
        self.gru = nn.GRU(input_size = hidden_size, hidden_size = hidden_size, batch_first = True)
        self.out = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_size, self.output_shape[0]))
        self.apply(init_weights)

    def forward(self, value):
        value, batch_size, episode_length = flatten_steps(value, self.input_shape)
        characters = torch.argmax(value, dim = -1)          # Also works on predicted logits, for dreams.
        _, last_hidden = self.gru(self.embed(characters))
        encoding = self.out(last_hidden[-1])
        return encoding.reshape(batch_size, episode_length, *self.output_shape)



class Voice_Decoder(Shape_to_Shape_Model):
    """Original Voice_OUT (actor = False): linear to one row per character, GRU, logits."""

    def __init__(
            self, name, input_size, max_voice_len, voice_shape,
            hidden_size = 64, dropout = .001, verbose = False):
        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (max_voice_len, voice_shape),
            arg_dict = {'hidden_size' : hidden_size, 'dropout' : dropout},
            verbose = verbose)

    def build_model(self, arg_dict):
        self.hidden_size = arg_dict['hidden_size']
        max_voice_len = self.output_shape[0]
        self.a = nn.Sequential(
            nn.Linear(self.input_shape[0], self.hidden_size * max_voice_len),
            nn.BatchNorm1d(self.hidden_size * max_voice_len),   # Original: "Tested, use this"
            nn.PReLU(),
            nn.Dropout(arg_dict['dropout']))
        self.gru = nn.GRU(input_size = self.hidden_size, hidden_size = self.hidden_size, batch_first = True)
        self.logits = nn.Linear(self.hidden_size, self.output_shape[1])
        self.apply(init_weights)

    def forward(self, value):
        value, batch_size, episode_length = flatten_steps(value, self.input_shape)
        value = self.a(value).reshape(-1, self.output_shape[0], self.hidden_size)
        value, _ = self.gru(value)
        logits = self.logits(value)
        return logits.reshape(batch_size, episode_length, *self.output_shape)

    @staticmethod
    def loss_func(predicted_values, target_values):
        # Cross-entropy per character, against the index of each one-hot row.
        voice_shape = predicted_values.shape[-1]
        loss = F.cross_entropy(
            predicted_values.reshape(-1, voice_shape),
            torch.argmax(target_values, dim = -1).reshape(-1),
            reduction = 'none')
        return loss.reshape(target_values.shape[:-1])



#------------------
# Actions: wheels and joints, (wheels_joints_shape,) in [-1, 1].
#------------------

class Wheels_Joints_Encoder(Shape_to_Shape_Model):
    """Original Wheels_Joints_IN: linear, PReLU. Built once inside the world model's
    prior and once inside each critic, from the same partial, with separate weights."""

    def __init__(self, name, input_size, output_size, verbose = False):
        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (output_size,),
            verbose = verbose)

    def build_model(self, arg_dict):
        self.model = nn.Sequential(
            nn.Linear(self.input_shape[0], self.output_shape[0]),
            nn.PReLU())
        self.apply(init_weights)

    def forward(self, value):
        value, batch_size, episode_length = flatten_steps(value, self.input_shape)
        return self.model(value).reshape(batch_size, episode_length, *self.output_shape)



class Wheels_Joints_Decoder(Action_Decoder):
    """The package's tanh-squashed Gaussian, with the original actor's two PReLU layers
    in front of mu and std instead of one."""

    def build_model(self, arg_dict):
        hidden_size = arg_dict['hidden_size']
        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_shape[0], hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU())
        self.mu = nn.Linear(hidden_size, self.output_shape[0])
        self.std = nn.Linear(hidden_size, self.output_shape[0])
        self.apply(init_weights)



def make_value_decoder(input_size, hidden_size):
    """Original Critic head after concatenating hidden state and encoded action."""
    def make():
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, 1))
        model.apply(init_weights)
        return model
    return make



######################



if __name__ == '__main__':

    batch_size, episode_length = 4, 3
    image_size, touch_size, prop_size, voice_len, voice_shape, inner = 16, 16, 4, 3, 18, 280

    vision_in = Vision_Encoder('vision', image_size, 128)
    vision_out = Vision_Decoder('vision', inner, image_size)
    touch_in = Sensor_Encoder('touch', touch_size, 20)
    touch_out = Sensor_Decoder('touch', inner, touch_size)
    voice_in = Voice_Encoder('command_voice', voice_len, voice_shape, 64)
    voice_out = Voice_Decoder('command_voice', inner, voice_len, voice_shape)
    action_in = Wheels_Joints_Encoder('wheels_joints', 4, 8)
    action_out = Wheels_Joints_Decoder('wheels_joints', input_size = 128, output_size = 4, hidden_size = 64)

    image = torch.rand(batch_size, episode_length, image_size, image_size, 4)
    voice = F.one_hot(torch.randint(0, voice_shape, (batch_size, episode_length, voice_len)), voice_shape).float()
    z = torch.randn(batch_size, episode_length, inner)

    print('vision  ', list(vision_in(image).shape), list(vision_out(z).shape),
          list(Vision_Decoder.loss_func(vision_out(z), image).shape))
    print('touch   ', list(touch_in(torch.rand(batch_size, episode_length, touch_size)).shape), list(touch_out(z).shape))
    print('voice   ', list(voice_in(voice).shape), list(voice_out(z).shape),
          list(Voice_Decoder.loss_func(voice_out(z), voice).shape))
    print('action  ', list(action_in(torch.rand(batch_size, episode_length, 4)).shape),
          {k : list(v.shape) for k, v in action_out(torch.randn(batch_size, episode_length, 128)).items()})
# %%
