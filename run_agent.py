from argparse import ArgumentParser
import pickle
import gym

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from agent import MineRLAgent, ENV_KWARGS
import logging

import coloredlogs

coloredlogs.install(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)


def main(model, weights):
    env = gym.make("MineRLPunchCow-v0")
    # env = HumanSurvival(**ENV_KWARGS).make()
    print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs,
                        pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    print("---Launching MineRL enviroment (be patient)---")

    env.reset()

    # had to make this change because im pretty sure i fucked the return value of env.reset()
    # oops
    # doesnt rly matter tho
    obs, reward, done, info = env.step(env.action_space.noop())

    print("Starting episode")

    while not done:
        minerl_action = agent.get_action(obs)
        # THIS IS CRINGE! THIS MUST BE FIXED IN THE FUTURE!
        minerl_action["ESC"] = 0
        print(minerl_action)
        obs, reward, done, info = env.step(minerl_action)
        env.render()

    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the '.model' file to be loaded.")

    args = parser.parse_args()

    main(args.model, args.weights)
