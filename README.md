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

