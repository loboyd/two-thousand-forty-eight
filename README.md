a 2048 emulator, hopefully with very little code

# RL Agent to play 2048
* p(s',r|s,a) can be computed exactly
  * r(s,a) is deterministic
  * p(s'|s,a) is just a matter of what random tile gets placed in what random location
     * 2 with 90% probability, 4 with 10% probability; all available tiles equally likely
* with policy gradients methods described in andrej karpathy's blog post
  * policy net with 16 inputs, some hidden layer, 4 outputs
  * reward is how much score is increased between now and end of game
  * normalize rewards: subtract by mean and divide be standard deviation
  * compute gradient expected reward wrt log output (?)

# Notes
* During game play, the network is run with a single state at a time, but during training, a whole
  batch is run at once. Does this effect the log prob calculation?
* It's not clear how to make it faster.
  * inference: This actually seems to be pretty fast now. It takes ~1s to run the network 2000
    times.
  * game emulator: This seems to also not be a huge bottleneck since I replaced the Python version
    with a much faster Rust version and it's not any faster.
  * updating: This takes very little time compared with actually running the episode batches.

# Ideas
* Train a net to weight four states according to their relative quality. For each "move", do this
  several (like 16 or 32) times, and then combine the resulting distributions, similar to how is now
  done for symmetries. The idea is to sort of build a single search step into the agent.

# AlphaGo
* policy network trained with supervised learning on expert games
* policy network trained via self-play reinforcement learning
* fast policy network trained on hand-coded features (linear approximation)
* value network trained on self-play games by the self-play policy network
* tree-search limiting depth via the learned value function and breadth via learned policy

