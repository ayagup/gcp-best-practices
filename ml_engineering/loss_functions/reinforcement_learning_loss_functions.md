# Reinforcement Learning Loss Functions

## 1. Policy Gradient Loss

**Formula:** L = -Σₜ log π(aₜ|sₜ) × Gₜ

**Pros:**
- Directly optimizes policy
- Works with continuous action spaces
- Can learn stochastic policies
- Theoretically grounded
- No need for value function
- Simple conceptual framework

**Cons:**
- High variance
- Sample inefficient
- Can be slow to converge
- Requires many trajectories
- Sensitive to reward scaling
- May get stuck in local optima

**When to Use:**
- REINFORCE algorithm
- Continuous control tasks
- When you want direct policy optimization
- Stochastic policy learning
- When Q-learning is impractical
- Robotics and control

**When NOT to Use:**
- When sample efficiency is critical (very sample-inefficient)
- For tasks requiring stable, low-variance learning (high variance)
- With limited interaction budget (needs many trajectories)
- When Q-learning or actor-critic methods are applicable (more efficient alternatives)
- For tasks with delayed or sparse rewards without variance reduction techniques

---

## 2. Value Function Loss

**Formula:** L = Σₜ [Vₜᵗᵃʳᵍᵉᵗ - V(sₜ)]²

**Pros:**
- Learns state value estimates
- Foundation for many RL algorithms
- Can use bootstrapping
- Lower variance than policy gradient
- Sample efficient
- Works with function approximation

**Cons:**
- Indirect policy improvement
- May need separate policy
- Can diverge with function approximation
- Requires careful target construction
- Hyperparameter sensitive
- Bootstrapping introduces bias

**When to Use:**
- Actor-Critic methods
- Value iteration algorithms
- When learning state values
- TD-learning
- As baseline for policy gradients
- Model-free RL

**When NOT to Use:**
- For direct policy optimization without value function (use policy gradient)
- With unstable function approximation (can diverge)
- When you need to learn Q-values instead of state values
- Without careful target construction (bootstrapping can be problematic)
- For model-based RL where planning is more appropriate

---

## 3. Q-Learning Loss

**Formula:** L = [r + γ max_a' Q(s', a') - Q(s,a)]²

**Pros:**
- Off-policy learning
- Sample efficient
- Can learn from any experience
- Well-studied and proven
- Works with discrete actions
- No policy needed during learning

**Cons:**
- Overestimation bias
- Doesn't work well with continuous actions
- Can be unstable with function approximation
- Requires experience replay for stability
- Target network often needed
- Max operator can be problematic

**When to Use:**
- DQN and variants
- Discrete action spaces
- Atari games
- When you have replay buffer
- Off-policy learning scenarios
- Value-based RL

**When NOT to Use:**
- For continuous action spaces (doesn't handle well - use actor-critic)
- Without experience replay and target networks (unstable)
- When overestimation bias is critical (use Double Q-Learning)
- For on-policy learning requirements (inherently off-policy)
- With very large or continuous action spaces (max operation impractical)

---

## 4. Actor-Critic Loss

**Formula:** L = L_actor + L_critic = -log π(a|s)A(s,a) + [V_target - V(s)]²

**Pros:**
- Combines policy and value learning
- Lower variance than pure policy gradient
- More sample efficient than REINFORCE
- Works with continuous actions
- Flexible framework
- Balances bias and variance

**Cons:**
- Two networks to train
- More hyperparameters
- Can be harder to stabilize
- Requires careful balancing
- Implementation complexity
- May need separate learning rates

**When to Use:**
- A3C, A2C algorithms
- Continuous control
- When you want sample efficiency
- Modern RL applications
- Robotics tasks
- Most state-of-the-art RL

**When NOT to Use:**
- When you only need value-based learning (Q-learning may be simpler)
- For very simple tasks where REINFORCE suffices
- Without careful hyperparameter tuning (many parameters)
- When debugging is critical (two networks complicate troubleshooting)
- For discrete actions where DQN works well (simpler alternative)

---

## 5. Temporal Difference Error

**Formula:** TD-Error = rₜ + γV(sₜ₊₁) - V(sₜ)

**Pros:**
- Bootstrap from current estimates
- Online learning
- No need for complete episodes
- Balances bias and variance
- Fundamental to many RL algorithms
- Computationally efficient

**Cons:**
- Biased estimates
- Can be high variance
- Sensitive to γ choice
- May be unstable with function approximation
- Requires careful implementation
- Convergence not always guaranteed

**When to Use:**
- TD-learning algorithms
- Continuous learning scenarios
- When episodes are very long
- As part of Actor-Critic
- Value function learning
- Most RL algorithms

**When NOT to Use:**
- When unbiased estimates are critical (TD has bias from bootstrapping)
- For episodic tasks where Monte Carlo is feasible and preferred
- With unstable function approximation (can diverge)
- When γ (discount factor) is difficult to set appropriately
- For model-based RL where planning is more suitable

---

## 6. Advantage Loss

**Formula:** L = -log π(a|s) × A(s,a) where A = Q(s,a) - V(s)

**Pros:**
- Reduces variance compared to returns
- Centers the rewards
- Better gradient estimates
- Used in state-of-the-art algorithms
- Improves policy gradient stability
- More sample efficient

**Cons:**
- Requires estimating both Q and V (or using n-step)
- More complex than simple policy gradient
- Implementation overhead
- Multiple estimation methods
- Can still have variance issues
- Hyperparameter tuning

**When to Use:**
- PPO, A3C, A2C
- When reducing variance is important
- Modern policy gradient methods
- Continuous control
- Most production RL systems
- Advantage Actor-Critic

**When NOT to Use:**
- For simple tasks where plain policy gradient works (unnecessary complexity)
- When Q and V estimation is problematic or expensive
- Without proper advantage estimation methods (GAE, n-step, etc.)
- For value-based methods that don't use policy gradients
- When implementation simplicity is paramount (plain returns are simpler)

---

## 7. PPO Clipped Loss

**Formula:** L = min[r(θ)A, clip(r(θ), 1-ε, 1+ε)A]

**Pros:**
- Prevents too large policy updates
- More stable than vanilla policy gradient
- Simple to implement
- State-of-the-art performance
- Robust across tasks
- Widely used in practice

**Cons:**
- Clipping introduces approximation
- Requires tuning ε
- Multiple epochs per batch
- Can be sample inefficient
- May limit exploration
- Not always optimal for all tasks

**When to Use:**
- Default choice for policy gradient RL
- Continuous control tasks
- Robotics
- When training stability is crucial
- OpenAI's standard RL algorithm
- Most modern RL applications

**When NOT to Use:**
- When you need maximum sample efficiency (other methods may be better)
- For discrete action spaces (DQN or simpler methods work)
- Without computational resources for multiple epochs per batch
- When ε tuning is impractical (critical hyperparameter)
- For tasks where exploration is heavily limited by clipping

---

## 8. DDPG Loss

**Formula:** Combined actor-critic loss for continuous control

**Pros:**
- Designed for continuous actions
- Off-policy learning
- Sample efficient
- Uses replay buffer
- Deterministic policy gradient
- Works with high-dimensional actions

**Cons:**
- Can be brittle
- Hyperparameter sensitive
- Requires careful tuning
- May not explore enough
- Overestimation issues
- Needs target networks

**When to Use:**
- Continuous control problems
- Robotics with continuous actions
- When DQN-like approach needed for continuous
- Manipulation tasks
- Locomotion
- When you have good simulator

**When NOT to Use:**
- For discrete action spaces (DQN is simpler and often better)
- Without careful hyperparameter tuning (very sensitive)
- When training stability is paramount (can be brittle)
- For exploration-heavy tasks (deterministic policy limits exploration)
- Without target networks and replay buffer (unstable)

---

## 9. A3C Loss

**Formula:** Asynchronous advantage actor-critic loss

**Pros:**
- Parallel training (multiple workers)
- No replay buffer needed
- Faster training with parallelism
- Diverse experience collection
- On-policy learning
- Simpler than off-policy methods

**Cons:**
- Requires multiple environments
- Can be harder to debug
- Synchronization overhead
- Not as sample efficient as off-policy
- Requires more compute
- Can have gradient staleness issues

**When to Use:**
- When you can run multiple environments
- Distributed RL training
- When replay buffer is problematic
- Atari games
- Continuous and discrete actions
- When you have computational resources

**When NOT to Use:**
- With single environment only (parallelism is the key advantage)
- When sample efficiency is critical (on-policy is less efficient)
- For debugging purposes (parallel training complicates debugging)
- When you have limited computational resources (requires multiple workers)
- If A2C (synchronous) provides similar performance with simpler implementation

---

## 10. Bellman Error

**Formula:** L = [r + γV(s') - V(s)]² or Q-learning variant

**Pros:**
- Foundation of value-based RL
- Theoretically grounded
- Enables dynamic programming
- Works with bootstrapping
- Can be computed incrementally
- Well-understood properties

**Cons:**
- Can be unstable with function approximation
- Bootstrapping introduces bias
- May diverge in some cases
- Requires careful target construction
- Sensitive to learning rate
- Overestimation/underestimation issues

**When to Use:**
- Value function approximation
- Q-learning and variants
- Fitted value iteration
- As part of most value-based RL
- TD-learning methods
- Foundational RL algorithms

**When NOT to Use:**
- With unstable function approximators (can lead to divergence)
- When policy-based methods are more suitable (e.g., continuous high-dimensional actions)
- In model-free settings requiring high sample efficiency (consider model-based methods)
- When the Markov assumption is violated (Bellman equation depends on it)
- With deadly triad present (function approximation + bootstrapping + off-policy) without safeguards
