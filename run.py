import os
import git
import gym
import carla_gym
import inspect
import argparse
import numpy as np
import os.path as osp
from pathlib import Path
currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
# sys.path.insert(1, currentPath + '/agents/stable_baselines/')
import shutil

from stable_baselines.bench import Monitor
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.ddpg.policies import CnnPolicy as DDPGCnnPolicy
from stable_baselines.common.policies import MlpPolicy as CommonMlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy as CommonMlpLstmPolicy
from stable_baselines.common.policies import CnnPolicy as CommonCnnPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines import PPO2
from stable_baselines import TRPO
from stable_baselines import A2C
from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy, sequence_1d_cnn, sequence_1d_cnn_ego_bypass_tc



from config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file


def parse_args_cfgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--env', help='environment ID', type=str, default='CarlaGymEnv-v1')
    parser.add_argument('--log_interval', help='Log interval (model)', type=int, default=100)
    parser.add_argument('--agent_id', type=int, default=None),
    parser.add_argument('--num_timesteps', type=float, default=1e7),
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play_mode', type=int, help='Display mode: 0:off, 1:2D, 2:3D ', default=0)
    parser.add_argument('--verbosity', help='Terminal mode: 0:Off, 1:Action,Reward 2:All', default=0, type=int)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--test_model', help='test model file name', type=str, default='')
    parser.add_argument('--test_last', help='test model best or last?', action='store_true', default=False)
    parser.add_argument('--carla_host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('-p', '--carla_port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--tm_port', default=8000, type=int, help='Traffic Manager TCP port to listen to (default: 8000)')
    parser.add_argument('--carla_res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')


    args = parser.parse_args()

    args.num_timesteps = int(args.num_timesteps)

    if args.test and args.cfg_file is None:
        path = 'logs/agent_{}/'.format(args.agent_id)
        conf_list = [cfg_file for cfg_file in os.listdir(path) if '.yaml' in cfg_file]
        args.cfg_file = path + conf_list[0]

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    # visualize all test scenarios
    if args.test:
        args.play_mode = True

    return args, cfg


if __name__ == '__main__':
    args, cfg = parse_args_cfgs()
    print('Env is starting')
    env = gym.make(args.env)
    if args.play_mode:
        env.enable_auto_render()
    env.begin_modules(args)
    n_actions = env.action_space.shape[-1]  # the noise objects for DDPG

    # --------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------Training----------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------
    if cfg.POLICY.NAME == 'DDPG':
        policy = {'MLP': DDPGMlpPolicy, 'CNN': DDPGCnnPolicy}   # DDPG does not have LSTM policy
    else:
        policy = {'MLP': CommonMlpPolicy, 'LSTM': CommonMlpLstmPolicy, 'CNN': CommonCnnPolicy}

    if not args.test:  # training
        if args.agent_id is not None:
            # create log folder
            os.mkdir(currentPath + '/logs/agent_{}/'.format(args.agent_id))                             # create agent_id folder
            os.mkdir(currentPath + '/logs/agent_{}/models/'.format(args.agent_id))
            save_path = 'logs/agent_{}/models/'.format(args.agent_id)
            env = Monitor(env, 'logs/agent_{}/'.format(args.agent_id))    # logging monitor

            # log commit id
            repo = git.Repo(search_parent_directories=False)
            commit_id = repo.head.object.hexsha
            with open('logs/agent_{}/reproduction_info.txt'.format(args.agent_id), 'w') as f:  # Use file to refer to the file object
                f.write('Git commit id: {}\n\n'.format(commit_id))
                f.write('Program arguments:\n\n{}\n\n'.format(args))
                f.write('Configuration file:\n\n{}'.format(cfg))
                f.close()

            # save a copy of config file
            original_adr = currentPath + '/tools/cfgs/' + args.cfg_file.split('/')[-1]
            target_adr = currentPath + '/logs/agent_{}/'.format(args.agent_id) + args.cfg_file.split('/')[-1]
            shutil.copyfile(original_adr, target_adr)

        else:
            save_path = 'logs/'
            env = Monitor(env, 'logs/', info_keywords=('reserved',))                                   # logging monitor
        model_dir = save_path + '{}_final_model'.format(cfg.POLICY.NAME)                               # model save/load directory

        if cfg.POLICY.NAME == 'DDPG':
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                        sigma=float(cfg.POLICY.ACTION_NOISE) * np.ones(n_actions))

            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(cfg.POLICY.PARAM_NOISE_STD),
                                                 desired_action_stddev=float(cfg.POLICY.PARAM_NOISE_STD))
            model = DDPG(policy[cfg.POLICY.NET], env, verbose=1, param_noise=param_noise, action_noise=action_noise,
                         policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)})
        elif cfg.POLICY.NAME == 'PPO2':
            model = PPO2(policy[cfg.POLICY.NET], env, verbose=1, model_dir=save_path, policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)})
        elif cfg.POLICY.NAME == 'TRPO':
            model = TRPO(policy[cfg.POLICY.NET], env, verbose=1, model_dir=save_path, policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)})
        elif cfg.POLICY.NAME =='A2C':
            model = A2C(policy[cfg.POLICY.NET], env, verbose=1, model_dir=save_path, policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)})
        else:
            print(cfg.POLICY.NAME)
            raise Exception('Algorithm name is not defined!')

        print('Model is Created')
        try:
            print('Training Started')
            if cfg.POLICY.NAME == 'DDPG':
                model.learn(total_timesteps=args.num_timesteps, log_interval=args.log_interval, save_path=save_path)
            else:
                model.learn(total_timesteps=args.num_timesteps, log_interval=args.log_interval)
        finally:
            print(100 * '*')
            print('FINISHED TRAINING; saving model...')
            print(100 * '*')
            # save model even if training fails because of an error
            model.save(model_dir)
            env.destroy()
            print('model has been saved.')

    # --------------------------------------------------------------------------------------------------------------------"""
    # ------------------------------------------------Test----------------------------------------------------------------"""
    # --------------------------------------------------------------------------------------------------------------------"""

    else:  # test
        if args.agent_id is not None:
            save_path = 'logs/agent_{}/models/'.format(args.agent_id)
        else:
            save_path = 'logs/'

        if args.test_model == '':
            best_last = 'best'
            if args.test_last:
                best_last = 'step'
            best_s = [int(best[5:-4])for best in os.listdir(save_path) if best_last in best]
            best_s.sort()
            args.test_model = best_last + '_{}'.format(best_s[-1])

        model_dir = save_path + args.test_model  # model save/load directory
        print('{} is Loading...'.format(args.test_model))
        if cfg.POLICY.NAME == 'DDPG':
            model = DDPG.load(model_dir)
            model.action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                              sigma=np.zeros(n_actions))
            model.param_noise = None
        elif cfg.POLICY.NAME == 'PPO2':
            model = PPO2.load(model_dir)
        elif cfg.POLICY.NAME == 'TRPO':
            model = TRPO.load(model_dir)
        elif cfg.POLICY.NAME == 'A2C':
            model = A2C.load(model_dir)
        else:
            print(cfg.POLICY.NAME)
            raise Exception('Algorithm name is not defined!')

        print('Model is loaded')
        try:
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render()
                if done:
                    obs = env.reset()
        finally:
            env.destroy()
