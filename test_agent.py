#%%
import os
import gzip
import pickle

from utils import make_objects_and_task
from processor import Processor
from agent import Agent        # So pickle can find the class (and its newest methods).



# -------------------------------
#  INITIAL SETUP
# -------------------------------

# If the user has no specific goal in mind, leave this as None
set_goal = None

# Change these to the agent you would like to test
hyper_parameters = 'eh_3'
agent_num = '0001'
epochs = '030000'
saved_file = 'saved_deigo'



def load_agent(hyper_parameters, agent_num, epochs, saved_file = 'saved_deigo'):
    """Agents are saved whole by Agent.save_agent (without their replay buffer)."""
    load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
    with gzip.open(load_path, 'rb') as f:
        return pickle.load(f)



print('\n\nLoading default agent...', end = ' ')
agent = load_agent(hyper_parameters, agent_num, epochs, saved_file)
agent.start_physics(GUI = True)

episodes = 0
wins = 0

print('Ready to go!')



#%%

# -------------------------------
#  SWAP AGENT
# -------------------------------

hyper_parameters = 'd'
agent_num = '0002'
epochs = '020000'
saved_file = 'saved_deigo'



def change_agent(hyper_parameters, agent_num, epochs, saved_file = 'saved_deigo'):
    """Swap in another saved agent. Its own args come with it, so the arena is rebuilt
    with them (only one PyBullet GUI can be open, so the old one is closed first)."""
    global agent, episodes, wins
    print('\n\nLoading new agent...', end = ' ')
    agent.stop_physics()
    agent.close_live_view()
    agent = load_agent(hyper_parameters, agent_num, epochs, saved_file)
    agent.start_physics(GUI = True)
    episodes = 0
    wins = 0
    print('Ready to go!')

# change_agent(hyper_parameters, agent_num, epochs, saved_file)



#%%

"""# -------------------------------
#  DEFINE SPECIFIC GOAL
# -------------------------------

set_goal = make_objects_and_task(
    num_objects = agent.processor.objects,
    allowed_tasks_and_weights = agent.processor.tasks_and_weights,
    allowed_colors = agent.processor.colors,
    allowed_shapes = agent.processor.shapes,
    test_train_num = agent.args.test_train_num,
    test = None
)

# Print details about the goal
print(set_goal)
print(set_goal[0].name)
print(set_goal[1][0][0].name)
print(set_goal[1][0][1].name)
print(set_goal[1][1][0].name)
print(set_goal[1][1][1].name)"""



#%%
# -------------------------------
#  RUN ONE EPISODE
# -------------------------------

#   Tasks:
#   0 - Free Play
#   1 - Watch
#   2 - Be Near
#   3 - Touch the Top
#   4 - Push
#   5 - Push Left
#   6 - Push Right

#   Colors:
#   0 - Red
#   1 - Green
#   2 - Blue
#   3 - Cyan
#   4 - Magenta
#   5 - Yellow

#   Shapes:
#   0 - Pillar
#   1 - Pole
#   2 - Dumbbell
#   3 - Cone
#   4 - Hourglass

# Replaces the agent's processor (all of its allowed tasks) with a narrower one.
# Parenting only: there is no second arena.
agent.processor = Processor(
    agent.args,
    agent.arena_1,
    None,
    tasks_and_weights = [(6, 1)],
    objects = 2,
    colors = [0, 1, 2, 3, 4, 5],
    shapes = [0, 1, 2, 3, 4],
    parenting = True
)

episodes += 1

win = agent.save_episodes(
    test = False,                   # Use training or test objects?
    verbose = False,                # Print extra info?
    live_view = True,               # One live window: actual / prior / posterior / errors, actions, DKL.
                                    #   Its 'Dreaming' button toggles dreaming mid-episode.
    display = False,                # Full plotting of model internals (one figure per step)
    video_display = False,          # Plot and save observation video frames
    sleep_time = 0.25,              # Delay per step
    waiting = False,                # Wait for a button press after each step?
    user_action = False,            # Manual control?
    dreaming = False,               # Run hallucinated? (world model on its prior alone after step 0)
    deterministic = False,          # Use mean actions instead of samples?
    set_positions = None, #([0, 5], [0, -5]),  # Object positions
    set_goal = set_goal,            # Specific goal
    video_folder = f'{saved_file}/thesis_pics/video_pics'
)

if win:
    wins += 1

print(f'\tWIN RATE: {round(100 * (wins / episodes), 2)}% \t ({wins} wins out of {episodes} episodes)')

# The recorded episode (observations, predictions, actions, DKLs...) is in agent.episode_dict.
# The live view stays open for the next episode; agent.close_live_view() closes it.
# %%