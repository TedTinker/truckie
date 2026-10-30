#%%
#------------------
# Per-step figures for Agent.save_episodes (test_agent.py).
#
# Ported from the original communication project's plotting_episodes.py (plot_step)
# and plotting_for_video.py (plot_video_step), for the episode_dict that
# Agent.save_episodes now records:
#   obs[t], prior_predictions[t], posterior_predictions[t]   Obs of o_t (real / expected / reconstructed)
#   action[t], critic_predictions[t], reward[t]              a_t taken from o_t, and its result
#   {name}_dkl[t], {name}_curiosity[t]                       for each observation name
#   birds_eye[t], dreamed[t], goal, objects, dreaming
# Predictions are None where they don't exist (prior at step 0, posterior while dreaming).
#------------------

import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from utils import empty_goal
from pybullet_data.robots.robot_maker import robot_dict



OBSERVATION_NAMES = ['vision', 'touch', 'prop', 'command_voice', 'feedback_voice']
TITLES = {
    'vision' : 'Vision', 'touch' : 'Touch', 'prop' : 'Proprioception',
    'command_voice' : 'Command voice', 'feedback_voice' : 'Feedback voice'}



def goal_text(goal):
    return f'{goal.human_text} ({goal.char_text})'


def action_labels(n):
    return ['left wheel', 'right wheel'] + [f'joint {i}' for i in range(1, n - 1)]


def series(episode_dict, key, step):
    """Values up to and including this step, with None (no posterior) as gaps."""
    return [float('nan') if v is None else v for v in episode_dict[key][:step + 1]]



#------------------
# Large figure: real vs. prior vs. posterior for every observation.
#------------------

def plot_step(step, episode_dict, args, last_step = False, saving = False, save_folder = '.'):
    sensor_plotter, _ = robot_dict[args.robot_name]
    dreaming = episode_dict['dreamed'][step]

    obs = episode_dict['obs'][step]
    prior = episode_dict['prior_predictions'][step]
    posterior = episode_dict['posterior_predictions'][step]
    has_action = step < len(episode_dict['action'])

    columns = [('Real' + (' (not seen: dreaming)' if dreaming else ''), obs),
               ('Prior (expected)', prior),
               ('Posterior (reconstructed)', posterior)]

    def touch_image(touch):
        return sensor_plotter(touch.clamp(0, 1).tolist())[80:-70, 125:-100]

    rows = []       # (label, kind, contents, height)
    rows.append(('Goal', 'text', goal_text(episode_dict['goal']), .1))
    rows.append(("Bird's eye view", 'image_wide', episode_dict['birds_eye'][step], 1))
    rows.append(('', 'headers', [c for c, _ in columns], .1))
    rows.append(('Vision', 'row', [None if o is None else o.vision[:, :, :3] for _, o in columns], 1))
    rows.append(('Touch', 'touch_row', [None if o is None else touch_image(o.touch) for _, o in columns], 1))
    rows.append(('Proprioception', 'bars_row', [None if o is None else o.prop for _, o in columns], .4))
    rows.append(('Command voice', 'text_row', [None if o is None else goal_text(o.command_voice) for _, o in columns], .2))
    rows.append(('Feedback voice', 'text_row', [None if o is None else goal_text(o.feedback_voice) for _, o in columns], .2))
    rewards = episode_dict['reward'][:step]     # Frame t is drawn before a_t is taken.
    reward_text = ('—' if not rewards else
                   f'{round(rewards[-1], 3)} from the last action, {round(sum(rewards), 3)} in total')
    if has_action and not last_step:
        rows.append(('Wheels, joints', 'action', episode_dict['action'][step], .5))
        rows.append(('Critics, reward', 'text',
                     f'Q of this action: {episode_dict["critic_predictions"][step]}     Reward: {reward_text}', .1))
    else:
        rows.append(('Reward', 'text', reward_text, .1))
    for name in OBSERVATION_NAMES:
        rows.append((f'{TITLES[name]} DKL', 'line', series(episode_dict, f'{name}_dkl', step), .4))

    fig = plt.figure(figsize = (20, 3 * sum(h for *_, h in rows)))
    fig.suptitle(f'Step {step}' + (' (last)' if last_step else '') + (' — dreaming' if dreaming else ''), fontsize = 18)
    gs = gridspec.GridSpec(len(rows), 4, figure = fig, height_ratios = [h for *_, h in rows], hspace = .6)

    def text(ax, value, size = 12):
        ax.text(0.02, 0.5, value, fontsize = size, va = 'center', transform = ax.transAxes)
        ax.axis('off')

    def image(ax, value):
        ax.imshow(value)
        ax.set_xticks([])
        ax.set_yticks([])

    for r, (label, kind, contents, _) in enumerate(rows):
        text(fig.add_subplot(gs[r, 0]), label + (':' if label else ''), 14)

        if kind == 'text':
            text(fig.add_subplot(gs[r, 1:]), contents)
        elif kind == 'image_wide':
            image(fig.add_subplot(gs[r, 1]), contents)
        elif kind == 'headers':
            for c, header in enumerate(contents):
                text(fig.add_subplot(gs[r, c + 1]), header)
        elif kind in ('row', 'touch_row', 'text_row', 'bars_row'):
            for c, value in enumerate(contents):
                ax = fig.add_subplot(gs[r, c + 1])
                if value is None:
                    text(ax, '—')
                elif kind == 'text_row':
                    text(ax, value)
                elif kind == 'bars_row':
                    ax.bar(range(len(value)), value.tolist(), color = 'gray')
                    ax.set_ylim(0, 1)
                    ax.set_xticks([])
                    ax.tick_params(labelsize = 9)
                elif kind == 'touch_row':
                    image(ax, value)
                    ax.axis('off')
                else:
                    image(ax, value)
        elif kind == 'action':
            ax = fig.add_subplot(gs[r, 1:])
            numbers = contents.tolist()
            ax.bar(range(len(numbers)), numbers, color = ['red' if x < 0 else 'blue' for x in numbers])
            ax.axhline(0, color = 'black', linewidth = 1)
            ax.set_ylim(-1, 1)
            ax.set_xticks(range(len(numbers)))
            ax.set_xticklabels(action_labels(len(numbers)), fontsize = 10)
            ax.tick_params(axis = 'y', labelsize = 9)
        elif kind == 'line':
            ax = fig.add_subplot(gs[r, 1:])
            ax.plot(contents, marker = '.')
            ax.set_xlim(0, max(step, 1))
            ax.xaxis.set_major_locator(MaxNLocator(integer = True))
            ax.tick_params(labelsize = 9)

    if saving:
        os.makedirs(save_folder, exist_ok = True)
        plt.savefig(os.path.join(save_folder, f'Step {step}.png'))
    plt.show()
    plt.close()



#------------------
# Compact frame for videos: vision, touch, voices, curiosities.
#------------------

def plot_video_step(step, episode_dict, args, last_step = False, saving = True, save_folder = 'saved_deigo/thesis_pics/video_pics'):
    sensor_plotter, _ = robot_dict[args.robot_name]

    obs = episode_dict['obs'][step]
    vision = obs.vision[:, :, :3]
    touch = obs.touch.tolist()

    command = obs.command_voice
    feedback = obs.feedback_voice
    posterior = episode_dict['posterior_predictions'][step]
    prior = episode_dict['prior_predictions'][step]
    # What the agent made of the feedback voice: the reconstruction, or while dreaming, the prior.
    predicted = posterior if posterior is not None else prior
    predicted_feedback = empty_goal if predicted is None else predicted.feedback_voice

    def words(goal):
        return f'{goal.task.name} {goal.color.name} {goal.shape.name}.'

    fig = plt.figure(figsize = (4, 8), dpi = 100, facecolor = 'none')
    fig.patch.set_alpha(0)
    main_ax = fig.add_axes([0, 0, 1, 1])
    main_ax.set_axis_off()

    main_ax.text(
        .94, .98, f'Step {step}', fontsize = 15, transform = main_ax.transAxes, zorder = 3, ha = 'right', va = 'center',
        bbox = dict(facecolor = 'white', edgecolor = 'black', boxstyle = 'round,pad=0.3', linewidth = 2))
    if episode_dict['dreamed'][step]:
        main_ax.text(.06, .98, 'Dreaming', fontsize = 13, transform = main_ax.transAxes, ha = 'left', va = 'center')

    def framed(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(3)

    vision_ax = fig.add_axes([0.05, 0.27, 0.9, 0.9])
    vision_ax.imshow(vision)
    framed(vision_ax)

    touch_ax = fig.add_axes([0.05, -0.14, 0.9, 0.9])
    touch_ax.imshow(sensor_plotter(touch)[80:-70, 80:-60])
    framed(touch_ax)

    text_ax = fig.add_axes([0.05, -0.18, 0.9, 0.25])
    text_ax.set_axis_off()
    text_ax.text(0, .8, f'Command:\n{words(command)}', ha = 'left', va = 'center', fontsize = 12)
    text_ax.text(0, .45, f'Feedback:\n{words(feedback)}', ha = 'left', va = 'center', fontsize = 12)
    text_ax.text(0, .1, f'Predicted Feedback:\n{words(predicted_feedback)}', ha = 'left', va = 'center', fontsize = 12)

    curiosity_names = ['vision', 'touch', 'feedback_voice']
    curiosity_data = [[0.] + [0. if c is None else c for c in episode_dict[f'{name}_curiosity'][1:step + 1]]
                      for name in curiosity_names]
    everything = [c for data in curiosity_data for c in data] or [0.]
    low, high = min(everything) * .9, max(everything) * 1.1
    if high <= low:
        high = low + 1

    plot_height = .07
    for idx, (name, data) in enumerate(zip(curiosity_names, curiosity_data)):
        ax = fig.add_axes([0.1, -0.30 - idx * (plot_height + .03), 0.8, plot_height])
        if len(data) > 1:
            ax.plot(data, color = 'black', linewidth = 2)
        else:
            ax.plot([0], data, marker = 'o', markersize = 6, color = 'black')
        ax.set_ylim(low, high)
        ax.set_xlim(0, max(step, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{TITLES[name]} Curiosity', fontsize = 10)
        ax.patch.set_alpha(0)
        ax.text(-0.02, low, f'{low:.2g}', va = 'center', ha = 'right', fontsize = 8, transform = ax.get_yaxis_transform())
        ax.text(-0.02, high, f'{high:.2g}', va = 'center', ha = 'right', fontsize = 8, transform = ax.get_yaxis_transform())
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

    if saving:
        os.makedirs(save_folder, exist_ok = True)
        plt.savefig(os.path.join(save_folder, f'Goal {command.human_text} Step {step}.png'),
                    transparent = True, bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    plt.close()