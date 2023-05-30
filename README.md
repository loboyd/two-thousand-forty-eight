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

