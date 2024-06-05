# Energy Rank Alignment: Using Preference Optimization to Search Chemical Space at Scale

Official implementation of: 

**Energy Rank Alignment: Using Preference Optimization to Search Chemical Space at Scale**

Shriram Chennakesavalu, Frank Hu, Sebastian Ibarraran, and Grant M. Rotskoff 

https://arxiv.org/abs/2405.12961

**Abstract:** Searching through chemical space is an exceptionally challenging problem because the
number of possible molecules grows combinatorially with the number of atoms. Large,
autoregressive models trained on databases of chemical compounds have yielded powerful
generators, but we still lack robust strategies for generating molecules with desired properties.
This molecular search problem closely resembles the “alignment” problem for large language
models, though for many chemical tasks we have a specific and easily evaluable reward
function. Here, we introduce an algorithm called energy rank alignment (ERA) that
leverages an explicit reward function to produce a gradient-based objective that we use to
optimize autoregressive policies. We show theoretically that this algorithm is closely related
to proximal policy optimization (PPO) and direct preference optimization (DPO), but
has a minimizer that converges to an ideal Gibbs-Boltzmann distribution with the reward
playing the role of an energy function. Furthermore, this algorithm is highly scalable, does
not require reinforcement learning, and performs well relative to DPO when the number of
preference observations per pairing is small. We deploy this approach to align molecular
transformers to generate molecules with externally specified properties and find that it
does so robustly, searching through diverse parts of chemical space. While our focus here
is on chemical search, we also obtain excellent results on an AI supervised task for LLM
alignment, showing that the method is scalable and general.
