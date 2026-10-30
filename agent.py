#%%
#------------------
# truckie, on the new architecture.
#
# Contract with the cluster framework:
#   main.py builds   Agent(args = args, i = i)   and calls   agent.training(q).
#   training() puts (i, fraction_done) on q, which main.py prints as a progress bar.
#   save_dicts() writes value_dict_###.pickle and min_max_dict_###.pickle into
#   saved_{comp}/{arg_name}/, which finish_dicts.py merges across agents.
#   GUI.py's ping files (save_{arg_name}_{i}_plot_dict, save_{arg_name}_{i}_agent)
#   are checked once per epoch.
#
# The saved dictionary keeps the original communication project's plot_dict keys
# (wins_*, gen_wins_*, rolled_*, accumulated_*, vision_loss, critics_loss, ...).
# The package's own training logs are saved beside them.
#
# Following the original agent.py:
#   - One processor, 'all': every allowed task, colour and shape, two objects, parenting.
#   - One epoch per training episode. Every epochs_per_gen_test epochs, one test episode
#     on held-out goals (test = True) which is not pushed to the buffer.
#   - Per-epoch losses are logged every keep_data epochs (and the first and last).
#   - reward_inflation_type and hidden_state_eta_feedback_voice_reduction_type schedules.
#
# Parenting only: the actor makes wheels_joints. In the original, voice_out was zeroed
# whenever parenting was True, so it is left out here.
#------------------

import glob
import gzip
import os
import pickle
import sys
from functools import partial
from itertools import accumulate
from math import exp, log

# Before anything imports general_FEP_RL, so a sibling copy works without pip install.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from utils import (
    folder, print, task_name_list, rolling_average,
    Obs, get_goal_from_one_hots, adjust_action, wait_for_button_press)
from arena import Arena
from processor import Processor
from truck_models import (
    Vision_Encoder, Vision_Decoder, Sensor_Encoder, Sensor_Decoder,
    Voice_Encoder, Voice_Decoder, Wheels_Joints_Encoder, Wheels_Joints_Decoder,
    make_value_decoder)

from general_FEP_RL.agent import Agent as FEP_Agent        # renamed: our class is also Agent
from general_FEP_RL.buffer import Recurrent_Replay_Buffer, shapes_from_world_model



OBSERVATION_NAMES = ['vision', 'touch', 'prop', 'command_voice', 'feedback_voice']
ACTION_NAME = 'wheels_joints'

# The original only trained on predictions of the next observation. The package can
# also train posterior reconstructions of the current one; this scales those relative
# to the prediction (0 = original behaviour).
POSTERIOR_ACCURACY = 0.

# Keys in plot_dict that are not per-episode or per-epoch series.
NOT_SERIES = ['args', 'arg_title', 'arg_name', 'behavior', 'composition_data',
              'division_epochs', 'training_log', 'training_log_actor']



def to_plain(obj):
    """A copy of a training log with one-number tensors turned into floats."""
    if isinstance(obj, dict):
        return {k : to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if hasattr(obj, 'item') and getattr(obj, 'numel', lambda: 1)() == 1:
        return obj.item()
    return obj


def plain_log(log):
    """The package logs whole (batch, steps, 1) reward/done/mask tensors each epoch.
    Leave those out, so the pickle stays small and loads without torch."""
    log = {k : v for k, v in log.items() if k not in ('reward', 'done', 'mask')}
    return to_plain(log)


def min_max_of(values):
    numbers = [v for v in values if v is not None and not isinstance(v, str)]
    return (min(numbers), max(numbers)) if numbers else (None, None)


def schedule(kind, fraction_done, epochs, total_epochs):
    """Rising schedules from 0 to 1, as in the original's reward inflation:
    'None' (always 1), 'linear', 'exp_k' (fraction ** k), 'sigmoid_k'."""
    if kind == 'None':
        return 1.
    if kind == 'linear':
        return fraction_done
    if kind.startswith('exp'):
        return fraction_done ** float(kind.split('_')[-1])
    if kind.startswith('sigmoid'):
        k = float(kind.split('_')[-1])
        return 1 / (1 + exp(-k * (epochs - total_epochs / 2)))
    raise ValueError(f"Unknown schedule '{kind}'. Use None, linear, exp_k, or sigmoid_k.")



class Agent:

    def __init__(self, args, i, GUI = False, verbose = False):
        self.args = args
        self.i = i
        self.agent_num = i
        self.GUI = GUI
        self.arena_1 = None
        self.processor = None

        self.check_args()

        self.epochs = 0
        self.episodes = 0
        self.steps = 0
        self.reward_inflation = schedule(args.reward_inflation_type, 0., 0, args.epochs)
        self.feedback_voice_eta_reduction = 1.

        self.build_agent(verbose)
        self.make_plot_dict()



    def check_args(self):
        args = self.args
        unsupported = []
        if args.curiosity == 'prediction_error':
            unsupported.append("curiosity 'prediction_error' (general_FEP_RL only has hidden_state curiosity)")
        if args.processor != 'all':
            unsupported.append(f"processor '{args.processor}' (only 'all')")
        if args.save_compositions:
            unsupported.append('save_compositions')
        if args.pb_vector:
            unsupported.append('pb_vector')
        if args.lstm:
            unsupported.append('lstm')
        if unsupported:
            raise NotImplementedError('This Agent does not implement: ' + '; '.join(unsupported) + '.')



    #------------------
    # The general_FEP_RL agent, with the original's sizes and scalars.
    #------------------

    def build_agent(self, verbose = False):
        args = self.args
        n_actions = args.wheels_joints_shape

        vision_encoder = partial(
            Vision_Encoder, name = 'vision', image_size = args.image_size,
            output_size = args.vision_encode_size, dropout = args.dropout)
        vision_decoder = partial(
            Vision_Decoder, name = 'vision', image_size = args.image_size,
            hidden_channels = args.hidden_size, divisions = args.divisions, dropout = args.dropout)
        touch_encoder = partial(
            Sensor_Encoder, name = 'touch', input_size = args.touch_shape, output_size = args.touch_encode_size)
        touch_decoder = partial(
            Sensor_Decoder, name = 'touch', output_size = args.touch_shape)
        prop_encoder = partial(
            Sensor_Encoder, name = 'prop', input_size = args.joint_aspects, output_size = args.prop_encode_size)
        prop_decoder = partial(
            Sensor_Decoder, name = 'prop', output_size = args.joint_aspects)

        def voice_encoder(name):
            return partial(
                Voice_Encoder, name = name, max_voice_len = args.max_voice_len, voice_shape = args.voice_shape,
                output_size = args.voice_encode_size, char_encode_size = args.char_encode_size,
                hidden_size = args.hidden_size, dropout = args.dropout)

        def voice_decoder(name):
            return partial(
                Voice_Decoder, name = name, max_voice_len = args.max_voice_len, voice_shape = args.voice_shape,
                hidden_size = args.hidden_size, dropout = args.dropout)

        wheels_joints_encoder = partial(
            Wheels_Joints_Encoder, name = ACTION_NAME,
            input_size = n_actions, output_size = args.wheels_joints_encode_size)
        wheels_joints_decoder = partial(
            Wheels_Joints_Decoder, name = ACTION_NAME,
            output_size = n_actions, hidden_size = args.hidden_size)

        # Per inner state: (encoder, decoder, state size, accuracy scaler, beta, eta).
        voice_scaler = args.voice_scaler
        self.modalities = {
            'vision' :         (vision_encoder, vision_decoder, args.vision_state_size,
                                args.vision_scaler, args.beta_vision, args.hidden_state_eta_vision),
            'touch' :          (touch_encoder, touch_decoder, args.touch_state_size,
                                args.touch_scaler, args.beta_touch, args.hidden_state_eta_touch),
            'prop' :           (prop_encoder, prop_decoder, args.prop_state_size,
                                args.prop_scaler, args.beta_prop, args.hidden_state_eta_prop),
            'command_voice' :  (voice_encoder('command_voice'), voice_decoder('command_voice'), args.voice_state_size,
                                voice_scaler, args.beta_command_voice, args.hidden_state_eta_command_voice),
            'feedback_voice' : (voice_encoder('feedback_voice'), voice_decoder('feedback_voice'), args.voice_state_size,
                                voice_scaler, args.beta_feedback_voice, args.hidden_state_eta_feedback_voice)}

        # The original clamped DKL to [0, dkl_max], then multiplied by eta. The package
        # computes eta * clamp(DKL * eta_before_clamp, 0, 1); these two settings make that
        # exactly eta_original * min(DKL, dkl_max).
        curious = args.curiosity == 'hidden_state'
        self.base_etas = {}
        inner_state_scalars = {}
        for name, (_, _, _, scaler, beta, eta) in self.modalities.items():
            self.base_etas[name] = (eta if curious else 0.) * args.dkl_max
            inner_state_scalars[name] = {
                'upsilon_prior' : scaler,
                'upsilon_posterior' : scaler,
                'beta' : beta,
                'eta_before_clamp' : 1. / args.dkl_max,
                'eta' : self.base_etas[name]}

        # The original's log-probability was the MEAN over the wheels_joints dimensions;
        # the package's is the SUM. To keep alpha and target_entropy meaning what they
        # meant in the original, divide alpha by the number of dimensions and multiply
        # target_entropy by it.
        if args.alpha is None:
            initial_alpha, lr_alpha = 1. / n_actions, args.lr     # The original started at alpha = 1.
        else:
            initial_alpha, lr_alpha = max(float(args.alpha) / n_actions, 1e-6), 0.

        self.agent = FEP_Agent(
            hidden_state_sizes = [args.pvrnn_mtrnn_size],
            time_constants = [1],

            list_of_dict_of_prior_input_encoder_class_dicts = [
                {ACTION_NAME : {'class' : wheels_joints_encoder}}],

            list_of_dict_of_posterior_input_encoder_class_dicts = [
                {name : {'class' : encoder, 'decoding_output_size' : state_size}
                 for name, (encoder, _, state_size, _, _, _) in self.modalities.items()}],

            list_of_dict_of_prediction_decoder_class_dicts = [
                {name : {'class' : decoder}
                 for name, (_, decoder, _, _, _, _) in self.modalities.items()}],

            lower_layer_posterior_sample_decoding_output_sizes = [0],

            dict_of_action_decoder_class_dicts = {ACTION_NAME : {'class' : wheels_joints_decoder}},
            dict_of_action_encoder_class_dicts = {ACTION_NAME : {'class' : wheels_joints_encoder}},

            list_of_dict_of_inner_state_scalar_dicts = [inner_state_scalars],

            dict_of_action_scalar_dicts = {
                ACTION_NAME : {
                    'target_entropy' : args.target_entropy * n_actions,
                    'action_cost' : args.action_cost,     
                    'initial_alpha' : initial_alpha,
                    'lr_alpha' : lr_alpha,
                    'delta' : 0.}},

            number_of_critics = args.critics,
            tau = args.tau,
            gamma = args.GAMMA,
            d = args.d,
            make_value_decoder = make_value_decoder(
                args.pvrnn_mtrnn_size + args.wheels_joints_encode_size, args.hidden_size),
            lr = args.lr,
            weight_decay = args.weight_decay,
            capacity = args.capacity,
            max_steps = args.max_steps,
            max_epochs_in_log = args.epochs // args.keep_data + 4,     # Logged every keep_data epochs; never pruned.
            verbose = verbose)



    def make_plot_dict(self):
        args = self.args
        self.plot_dict = {
            'args' : args,
            'arg_title' : args.arg_title,
            'arg_name' : args.arg_name,

            'division_epochs' : [],
            'steps' : [],
            'behavior' : {},
            'composition_data' : {},

            'accuracy_loss' : [],
            'complexity_loss' : [],
            **{f'{name}_loss' : [] for name in OBSERVATION_NAMES},

            'actor_loss' : [],
            'critics_loss' : [[] for _ in range(args.critics)],
            'alpha_loss' : [],
            'alpha_text_loss' : [],

            'reward' : [],
            'gen_reward' : [],
            'q' : [],
            'extrinsic' : [],
            'intrinsic_curiosity' : [],
            'intrinsic_entropy' : [],

            **{f'{name}_prediction_error_curiosity' : [] for name in OBSERVATION_NAMES},
            'prediction_error_curiosity' : [],
            **{f'{name}_hidden_state_curiosity' : [] for name in OBSERVATION_NAMES},
            'hidden_state_curiosity' : [],

            'wins_all' : [],
            'gen_wins_all' : []}
        for task_name in task_name_list:
            self.plot_dict[f'wins_{task_name}'] = []
            self.plot_dict[f'gen_wins_{task_name}'] = []
        self.plot_dict['wins_exception'] = []      # Last: plotting.py drops it when exceptions == 0.



    #------------------
    # Physics.
    #------------------

    def start_physics(self, GUI = None):
        args = self.args
        self.arena_1 = Arena(GUI = self.GUI if GUI == None else GUI, args = args)
        self.processor = Processor(
            args, self.arena_1, None,
            tasks_and_weights = [(task, 1) for task in args.allowed_tasks],
            objects = 2, colors = args.allowed_colors, shapes = args.allowed_shapes,
            parenting = True, full_name = 'All Tasks')

    def stop_physics(self):
        if self.arena_1 is not None:
            self.arena_1.stop()
        self.arena_1 = None
        self.processor = None



    #------------------
    # Observations: Obs from the processor -> dict of (1, 1, ...) tensors.
    #------------------

    def to_observation(self, obs):
        return {
            'vision' : obs.vision.unsqueeze(1),
            'touch' : obs.touch.unsqueeze(1),
            'prop' : obs.prop.unsqueeze(1),
            'command_voice' : obs.command_voice.one_hots.unsqueeze(0).unsqueeze(0),
            'feedback_voice' : obs.feedback_voice.one_hots.unsqueeze(0).unsqueeze(0)}

    @staticmethod
    def for_buffer(value_dict):
        return {k : v.squeeze(0).squeeze(0) for k, v in value_dict.items()}



    #------------------
    # One episode. Training episodes go into the buffer; test episodes do not.
    #------------------

    def episode(self, test = False, record_behavior = False):
        args = self.args
        processor = self.processor
        buffer = self.agent.buffer

        self.agent.eval()               # Dropout off and batch norm on running stats while acting.
        self.agent.begin(batch_size = 1)
        processor.begin(test = test)

        observation = self.to_observation(processor.obs())
        total_reward, win, behavior = 0., False, []

        for step in range(args.max_steps):
            step_dict = self.agent.step_in_episode(observation)
            wheels_joints = step_dict['action'][ACTION_NAME]
            reward, done, win = processor.step(wheels_joints[0, 0].clone())
            reward *= self.reward_inflation
            next_observation = self.to_observation(processor.obs())

            if not test:
                self.steps += 1
                buffer.push(
                    observation_dict = self.for_buffer(observation),
                    action_dict = self.for_buffer(step_dict['action']),
                    reward = reward,
                    next_observation_dict = self.for_buffer(next_observation),
                    done = done,
                    best_action_dict = None)
            if record_behavior:
                behavior.append(processor.feedback_voice_1.char_text)

            total_reward += reward
            if done:
                break
            observation = next_observation

        processor.done()
        return total_reward, win, step + 1, behavior



    def record_wins(self, prefix, win):
        """One entry per episode in every wins list: the result for this goal's task,
        None for the others."""
        goal_task = self.processor.goal.task.name
        self.plot_dict[f'{prefix}wins_all'].append(win)
        for task_name in task_name_list:
            self.plot_dict[f'{prefix}wins_{task_name}'].append(win if task_name == goal_task else None)
        if prefix == '':
            exception = getattr(self.processor.arena_1, 'supposed_to_be_exception', False)
            self.plot_dict['wins_exception'].append(win if exception else None)



    def training_episode(self):
        args = self.args
        self.episodes += 1
        record = (args.save_behaviors
                  and (args.agents_per_behavior_analysis == -1 or self.i <= args.agents_per_behavior_analysis)
                  and self.episodes % args.episodes_per_behavior_analysis == 0)

        total_reward, win, steps, behavior = self.episode(test = False, record_behavior = record)

        self.plot_dict['steps'].append(steps)
        self.plot_dict['reward'].append(total_reward)
        self.record_wins('', win)
        if record:
            self.plot_dict['behavior'][self.episodes] = behavior



    def gen_test(self):
        """One episode on held-out goals."""
        try:
            total_reward, win, _, _ = self.episode(test = True)
        except IndexError:
            # make_objects_and_task found no held-out combination for the drawn task.
            self.processor.done()
            self.plot_dict['gen_reward'].append(0.)
            self.plot_dict['gen_wins_all'].append(None)
            for task_name in task_name_list:
                self.plot_dict[f'gen_wins_{task_name}'].append(None)
            return None
        self.plot_dict['gen_reward'].append(total_reward)
        self.record_wins('gen_', win)
        return win



    #------------------
    # Watching one episode (test_agent.py). Nothing is pushed to the buffer, and no
    # counters, schedules or plot_dict entries change.
    #
    # Alignment in general_FEP_RL: at step t the agent is given o_t (and, internally,
    # a_{t-1}). Its PRIOR prediction at step t is what it expected o_t to be BEFORE
    # seeing it (from h_{t-1} and a_{t-1}); its POSTERIOR prediction is its
    # reconstruction of o_t AFTER seeing it. Then it chooses a_t.
    # So frame t of the episode shows o_t, both predictions of o_t, the DKL/curiosity
    # of o_t, and the action a_t taken from there.
    #
    # Dreaming: from step 1 on, the world model runs on its prior alone
    # (use_posterior = False) and never sees o_t; the arena still moves with its actions,
    # so the real o_t is shown next to what it imagined.
    #------------------

    @staticmethod
    def to_display_observation(value_dict):
        """A dict of (1, 1, ...) tensors (real or predicted) -> an Obs of plain values,
        with voices turned into Goals. Voice predictions are logits; argmax picks the
        characters."""
        if value_dict is None:
            return None
        def goal(name):
            return get_goal_from_one_hots(value_dict[name].detach().cpu().reshape(value_dict[name].shape[-2:]))
        return Obs(
            vision = value_dict['vision'].detach().cpu().reshape(value_dict['vision'].shape[-3:]),
            touch = value_dict['touch'].detach().cpu().flatten(),
            prop = value_dict['prop'].detach().cpu().flatten(),
            command_voice = goal('command_voice'),
            feedback_voice = goal('feedback_voice'))



    def save_episodes(
            self,
            test = False,               # Held-out (test) goals instead of training goals?
            verbose = False,            # Print the goal, and the arena's per-step details.
            display = True,             # Large per-step figure of real vs. predicted observations.
            video_display = True,       # Compact per-step frame, saved as a .png.
            sleep_time = None,          # Delay per arena step.
            waiting = False,            # Wait for a button press after each frame.
            user_action = False,        # Adjust each action with sliders before it is taken.
            dreaming = False,           # Run the world model on its prior alone after step 0.
            deterministic = False,      # tanh(mu) for actions and inner states instead of samples.
            set_positions = None,       # Object positions, passed to the processor.
            set_goal = None,            # (task, objects_1, objects_2) from make_objects_and_task.
            video_folder = 'saved_deigo/thesis_pics/video_pics',
            live_view = False,          # general_FEP_RL's LiveView window, updated every step.
            live_view_kwargs = None):   # Extra LiveView settings, e.g. {'error_limit' : 1.}.
        """Run one episode for watching. Returns win; the recorded episode is kept in
        self.episode_dict.

        With live_view, one window is kept open across episodes (self.live_view; see
        close_live_view). Its 'Dreaming' button switches dreaming on and off mid-episode,
        on top of the dreaming argument."""

        if display or video_display:
            from plotting_episodes import plot_step, plot_video_step     # matplotlib only when watching.
        view = self.open_live_view(**(live_view_kwargs or {})) if live_view else None

        args = self.args
        if self.processor is None:
            self.start_physics()
        processor = self.processor
        scalars = self.agent.list_of_dict_of_inner_state_scalar_dicts[0]

        episode_dict = {
            'goal' : None,
            'objects' : None,
            'dreaming' : dreaming,          # As asked for; 'dreamed' says what each step did.
            'dreamed' : [],
            'obs' : [],                     # Real o_t, as Obs.
            'birds_eye' : [],
            'prior_predictions' : [],       # Expected o_t before seeing it (None at step 0).
            'posterior_predictions' : [],   # Reconstructed o_t after seeing it (None while dreaming).
            'action' : [],                  # a_t, (wheels_joints_shape,).
            'critic_predictions' : [],      # Q(h_t, a_t) from each critic.
            'reward' : [],                  # Reward after a_t (not inflated).
            **{f'{name}_dkl' : [] for name in OBSERVATION_NAMES},
            **{f'{name}_curiosity' : [] for name in OBSERVATION_NAMES}}
        self.episode_dict = episode_dict

        def show(step, step_dict, real_observation, last_step = False):
            if view is not None:
                self.update_live_view(view, step, step_dict, real_observation, last_step = last_step, win = win)
            if display:
                plot_step(step, episode_dict, args = args, last_step = last_step)
            if video_display:
                plot_video_step(step, episode_dict, args = args, last_step = last_step, save_folder = video_folder)
            if (display or video_display or view is not None) and waiting:
                wait_for_button_press()

        def record(step_dict, real_observation, dreamed, step):
            episode_dict['dreamed'].append(dreamed)
            episode_dict['obs'].append(self.to_display_observation(real_observation))
            episode_dict['birds_eye'].append(processor.arena_1.photo_from_above()[:, :, :3])
            prior = {k : v for k, v in step_dict['prior_predictions'][0].items()}
            episode_dict['prior_predictions'].append(None if step == 0 else self.to_display_observation(prior))
            episode_dict['posterior_predictions'].append(
                None if dreamed else self.to_display_observation(step_dict['posterior_predictions'][0]))
            inner_states = step_dict['list_of_inner_states'][0]
            for name in OBSERVATION_NAMES:
                if dreamed or step == 0:        # No posterior, or a prior from nothing.
                    episode_dict[f'{name}_dkl'].append(None)
                    episode_dict[f'{name}_curiosity'].append(None)
                    continue
                dkl = inner_states[name]['dkl'].sum().item()
                episode_dict[f'{name}_dkl'].append(dkl)
                episode_dict[f'{name}_curiosity'].append(
                    scalars[name]['eta'] * min(max(dkl * scalars[name]['eta_before_clamp'], 0.), 1.))

        win = False
        with torch.no_grad():
            self.agent.eval()
            self.agent.begin(batch_size = 1)
            if view is not None:
                view.begin()
            processor.begin(test = test, verbose = verbose, set_positions = set_positions, set_goal = set_goal)
            episode_dict['goal'] = processor.goal
            episode_dict['objects'] = processor.current_objects_1

            for step in range(args.max_steps):
                real_observation = self.to_observation(processor.obs())
                dreamed = self.wants_to_dream(dreaming, view) and step > 0
                step_dict = self.agent.step_in_episode(
                    None if dreamed else real_observation,
                    use_posterior = not dreamed,
                    deterministic = deterministic)

                if user_action:
                    adjusted = adjust_action(step_dict['action'][ACTION_NAME].clone()).to(
                        step_dict['action'][ACTION_NAME].dtype)
                    # The world model's next prior must see the action actually taken.
                    self.agent.action[ACTION_NAME] = adjusted
                    step_dict['action'][ACTION_NAME] = adjusted
                    values = [critic(self.agent.hidden_states[0], self.agent.action) for critic in self.agent.critics]
                else:
                    values = step_dict['values']

                wheels_joints = step_dict['action'][ACTION_NAME]
                record(step_dict, real_observation, dreamed, step)
                episode_dict['action'].append(wheels_joints.detach().cpu().flatten())
                episode_dict['critic_predictions'].append([round(v.item(), 3) for v in values])

                show(step, step_dict, real_observation)

                reward, done, win = processor.step(
                    wheels_joints[0, 0].clone(), verbose = verbose, sleep_time = sleep_time)
                episode_dict['reward'].append(reward)
                if done:
                    break

            # The last observation, and what the agent expected it to be.
            final_observation = self.to_observation(processor.obs())
            dreamed = self.wants_to_dream(dreaming, view)
            step_dict = self.agent.step_in_episode(
                None if dreamed else final_observation, use_posterior = not dreamed, deterministic = deterministic)
            record(step_dict, final_observation, dreamed, step + 1)
            show(step + 1, step_dict, final_observation, last_step = True)
            processor.done()

        if verbose:
            print(f'\n{"WIN" if win else "No win"} after {step + 1} steps. '
                  f'Total reward: {round(sum(episode_dict["reward"]), 3)}.')
        return win



    #------------------
    # general_FEP_RL's LiveView: actual / prior / posterior / errors for every
    # observation, the wheels_joints bars, and per-modality MSE and DKL, in one window.
    #------------------

    def open_live_view(self, **kwargs):
        """The window is built on its first update and reused by later episodes.
        Keyword arguments go to LiveView the first time (or after close_live_view)."""
        if getattr(self, 'live_view', None) is None or not self.live_view_is_open():
            from general_FEP_RL.live_view import LiveView
            n_actions = self.args.wheels_joints_shape
            settings = dict(
                layout = {'vision' : 'rgbd'},           # RGB, then depth.
                action_labels = {ACTION_NAME : ['left wheel', 'right wheel'] + [f'joint {i}' for i in range(1, n_actions - 1)]},
                error_limit = 1.,                       # Every observation is in [0, 1].
                title = f'{self.args.arg_name}, agent {self.i}, epoch {self.epochs}')
            settings.update(kwargs)
            self.live_view = LiveView(**settings)
        return self.live_view

    def live_view_is_open(self):
        import matplotlib.pyplot as plt
        view = getattr(self, 'live_view', None)
        return view is not None and (view.fig is None or plt.fignum_exists(view.fig.number))

    def close_live_view(self):
        if getattr(self, 'live_view', None) is not None:
            self.live_view.close()
        self.live_view = None

    @staticmethod
    def wants_to_dream(dreaming, view):
        return dreaming or (view is not None and view.dreaming)

    @staticmethod
    def voices_as_probabilities(prediction_dict):
        """Voice decoders output logits; LiveView draws [0, 1] images."""
        return {name : value.softmax(-1) if name.endswith('voice') else value
                for name, value in prediction_dict.items()}

    def live_view_text(self, step, last_step = False, win = False):
        episode_dict = self.episode_dict
        rewards = episode_dict['reward']
        lines = [f'Step {step}' + (f'  (last: {"WIN" if win else "no win"})' if last_step else '')
                 + f'    goal: {episode_dict["goal"].human_text}'
                 + f'    reward: {round(rewards[-1], 3) if rewards else "-"} last, {round(sum(rewards), 3)} total']
        if not last_step:
            lines[0] += f'    Q: {episode_dict["critic_predictions"][step]}'
        obs = episode_dict['obs'][step]
        prior = episode_dict['prior_predictions'][step]
        posterior = episode_dict['posterior_predictions'][step]
        for voice in ['command_voice', 'feedback_voice']:
            line = f'{voice:<15} actual: {getattr(obs, voice).human_text:<28}'
            if prior is not None:
                line += f' prior: {getattr(prior, voice).human_text:<28}'
            if posterior is not None:
                line += f' post: {getattr(posterior, voice).human_text}'
            lines.append(line)
        return '\n'.join(lines)

    def update_live_view(self, view, step, step_dict, real_observation, last_step = False, win = False):
        prior = {} if step == 0 else self.voices_as_probabilities(step_dict['prior_predictions'][0])
        posterior = ({} if step_dict['dreamed'] else
                     self.voices_as_probabilities(step_dict['posterior_predictions'][0]))
        view.update(
            real_observation, step_dict, prior = prior, posterior = posterior,
            actions = {} if last_step else None,       # The last step's action is never taken.
            text = self.live_view_text(step, last_step = last_step, win = win))



    #------------------
    # One epoch, then the schedules.
    #------------------

    def epoch(self):
        args = self.args
        self.epochs += 1
        log_now = self.epochs == 1 or self.epochs >= args.epochs or self.epochs % args.keep_data == 0

        # Batch norm in training mode needs more than one episode in the batch.
        if len(self.agent.buffer) >= 2:
            epoch_dict, epoch_dict_actor = self.agent.epoch(batch_size = args.batch_size, add_to_log = log_now)
            if log_now and epoch_dict is not None:
                self.log_epoch(epoch_dict, epoch_dict_actor)

        fraction_done = self.epochs / args.epochs
        self.reward_inflation = schedule(args.reward_inflation_type, fraction_done, self.epochs, args.epochs)
        reduction_type = args.hidden_state_eta_feedback_voice_reduction_type
        if reduction_type == 'None':
            self.feedback_voice_eta_reduction = 1.
        else:                                       # Falling from 1 to 0, as in the original.
            self.feedback_voice_eta_reduction = 1. - schedule(reduction_type, fraction_done, self.epochs, args.epochs)
        self.agent.list_of_dict_of_inner_state_scalar_dicts[0]['feedback_voice']['eta'] = \
            self.base_etas['feedback_voice'] * self.feedback_voice_eta_reduction



    def log_epoch(self, epoch_dict, epoch_dict_actor):
        """The package's epoch results, under the original plot_dict's names."""
        pd = self.plot_dict
        layer = 'layer_0'
        scalars = self.agent.list_of_dict_of_inner_state_scalar_dicts[0]

        accuracy = 0.
        complexity = 0.
        for name in OBSERVATION_NAMES:
            loss = epoch_dict['accuracy_losses_prior'][layer][name] * scalars[name]['upsilon_prior']
            pd[f'{name}_loss'].append(loss)
            accuracy += loss
            complexity += epoch_dict['complexity_losses'][layer][name] * scalars[name]['beta']
            pd[f'{name}_hidden_state_curiosity'].append(epoch_dict['curiosities'][layer][name])
            pd[f'{name}_prediction_error_curiosity'].append(0.)
        pd['accuracy_loss'].append(accuracy)
        pd['complexity_loss'].append(complexity)
        pd['hidden_state_curiosity'].append(epoch_dict['curiosity'])
        pd['prediction_error_curiosity'].append(0.)
        pd['intrinsic_curiosity'].append(epoch_dict['curiosity'])
        pd['extrinsic'].append(epoch_dict['average_reward'])

        pd['alpha_loss'].append(
            sum(epoch_dict['alpha_losses'].values()) if self.args.alpha is None else None)
        pd['alpha_text_loss'].append(None)

        for k, critic_loss in enumerate(epoch_dict['critic_losses']):
            pd['critics_loss'][k].append(log(critic_loss) if critic_loss > 0 else critic_loss)   # Original logged log.

        actor = epoch_dict_actor or {}
        pd['actor_loss'].append(actor.get('actor_loss'))
        pd['q'].append(actor.get('Q_for_actor'))
        pd['intrinsic_entropy'].append(actor.get('entropy_for_actor'))



    #------------------
    # Called by main.py.
    #------------------

    def regular_checks(self, force = False):
        args = self.args
        if args.save_agents and self.i <= args.agents_per_agent_save:
            if force or self.epochs % args.epochs_per_agent_save == 0:
                self.save_agent()
        if force or self.epochs % args.epochs_per_gen_test == 0:
            self.gen_test()



    def training(self, q = None):
        args = self.args
        if self.processor is None:
            self.start_physics()
        try:
            self.regular_checks(force = True)
            while self.epochs < args.epochs:
                self.training_episode()
                self.epoch()
                self.check_pings()
                if q is not None:
                    q.put((self.i, self.epochs / args.epochs))
                if self.epochs < args.epochs:
                    self.regular_checks()
            self.regular_checks(force = True)
            self.plot_dict['division_epochs'].append(
                (self.epochs, self.processor.linestyle, self.processor.full_name))
        finally:
            self.stop_physics()

        self.save_dicts()
        if q is not None:
            q.put((self.i, 1.))



    #------------------
    # Saving.
    #------------------

    def save_dicts(self, temp = False):
        """value_dict_###.pickle and min_max_dict_###.pickle, or the _temp_ versions
        mid-training. finish_dicts.py turns every key into a list with one entry per agent."""
        tag = 'temp_' if temp else ''
        index = str(self.i).zfill(3)

        value_dict = dict(self.plot_dict)
        value_dict['accumulated_reward'] = list(accumulate(value_dict['reward']))
        value_dict['accumulated_gen_reward'] = list(accumulate(value_dict['gen_reward']))
        for task_name in task_name_list + ['all']:
            value_dict[f'rolled_wins_{task_name}'] = rolling_average(value_dict[f'wins_{task_name}'], window_size=500)
            value_dict[f'rolled_gen_wins_{task_name}'] = rolling_average(value_dict[f'gen_wins_{task_name}'], window_size=500)
        value_dict['training_log'] = plain_log(self.agent.training_log)
        value_dict['training_log_actor'] = plain_log(self.agent.training_log_actor)

        # Every key gets an entry, because utils.load_dicts looks each one up.
        min_max_dict = {}
        for key, values in value_dict.items():
            if key in ['args', 'arg_title', 'arg_name']:
                continue
            if key == 'critics_loss':
                min_max_dict[key] = [min_max_of(critic) for critic in values]
            elif key in NOT_SERIES or not isinstance(values, list):
                min_max_dict[key] = (None, None)
            else:
                min_max_dict[key] = min_max_of(values)

        with open(os.path.join(folder, f'value_dict_{tag}{index}.pickle'), 'wb') as handle:
            pickle.dump(value_dict, handle)
        with open(os.path.join(folder, f'min_max_dict_{tag}{index}.pickle'), 'wb') as handle:
            pickle.dump(min_max_dict, handle)



    def save_agent(self):
        """Saved without the replay buffer (hundreds of MB) or the simulator.
        main.py --load_agents True picks up the latest of these."""
        path = os.path.join(
            folder, 'agents', f'agent_{str(self.i).zfill(4)}_epoch_{str(self.epochs).zfill(6)}.pkl.gz')
        buffer, self.agent.buffer = self.agent.buffer, None
        try:
            with gzip.open(path, 'wb') as handle:
                pickle.dump(self, handle, protocol = pickle.HIGHEST_PROTOCOL)
        finally:
            self.agent.buffer = buffer



    def check_pings(self):
        """React to GUI.py's ping files, then delete them."""
        base = os.path.join(folder, f'save_{self.args.arg_name}_{self.i}')
        if os.path.exists(base + '_plot_dict'):
            self.save_dicts(temp = True)
            os.remove(base + '_plot_dict')
        if os.path.exists(base + '_agent'):
            self.save_agent()
            os.remove(base + '_agent')



    def resume(self, args):
        """After loading, follow this run's arguments (epochs, saving, logging...), not the
        ones it was pickled with. Model sizes must match the run that saved it."""
        self.args = args
        self.plot_dict['args'] = args
        self.plot_dict['arg_title'] = args.arg_title
        self.plot_dict['arg_name'] = args.arg_name



    def __getstate__(self):
        # The simulator can't be pickled; training() builds a new one.
        state = self.__dict__.copy()
        state['arena_1'] = None
        state['processor'] = None
        state['live_view'] = None           # A matplotlib window can't be pickled either.
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.agent.buffer is None:       # Saved without it: start a fresh, empty one.
            observation_shapes, action_shapes = shapes_from_world_model(self.agent.world_model)
            self.agent.buffer = Recurrent_Replay_Buffer(
                observation_shapes, action_shapes, self.args.capacity, self.args.max_steps)



def latest_agent_file(i):
    """The newest save_agent() file for agent i, or None."""
    files = glob.glob(os.path.join(folder, 'agents', f'agent_{str(i).zfill(4)}_epoch_*.pkl.gz'))
    return max(files) if files else None        # Epochs are zero-padded, so text order is epoch order.



if __name__ == '__main__':
    from utils import args
    agent = Agent(args = args, i = 1, verbose = True)
    agent.start_physics()
    total_reward, win, steps, _ = agent.episode()
    print(f'\nOne episode: {steps} steps, reward {total_reward}, win {win}. Buffer: {len(agent.agent.buffer)}.')
    agent.stop_physics()
# %%