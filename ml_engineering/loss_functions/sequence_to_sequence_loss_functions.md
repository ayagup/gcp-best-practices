# Sequence-to-Sequence Loss Functions

## 1. Cross-Entropy Loss (per timestep)

**Formula:** L = -Σₜ Σᵢ yₜᵢ log(ŷₜᵢ) where t is timestep, i is token

**Pros:**
- Standard loss for sequence generation
- Works well with teacher forcing
- Simple and well-understood
- Easy to implement
- Natural for token prediction
- Supports variable-length sequences

**Cons:**
- Treats each timestep independently
- Doesn't consider sequence-level metrics
- Exposure bias (train vs inference mismatch)
- Can lead to generic outputs
- Doesn't optimize for BLEU/ROUGE scores
- No explicit sequence structure modeling

**When to Use:**
- Machine translation
- Text generation
- Speech recognition
- Image captioning
- Any sequence-to-sequence task
- Default choice for seq2seq models

**When NOT to Use:**
- When sequence-level metrics (BLEU, ROUGE) are critical (consider RL-based training)
- For tasks with severe exposure bias issues (consider scheduled sampling)
- When you need to optimize alignment without frame-level labels (use CTC)
- For tasks requiring explicit sequence structure (consider structured prediction losses)
- When generic outputs are problematic (consider diversity-promoting losses)

---

## 2. Connectionist Temporal Classification (CTC) Loss

**Formula:** L = -log P(y|x) summed over all alignments

**Pros:**
- Handles variable-length input/output alignment
- No need for frame-level annotations
- Allows unsegmented sequence training
- Removes need for pre-alignment
- Works with RNNs naturally
- Proven effective in speech recognition

**Cons:**
- Assumes conditional independence between outputs
- Can produce repetitive outputs
- Requires special decoding (beam search)
- Computationally expensive
- Limited to monotonic alignments
- Can struggle with complex alignment patterns

**When to Use:**
- Speech recognition (audio to text)
- Handwriting recognition
- OCR (Optical Character Recognition)
- When you don't have frame-level alignments
- Sequence labeling without segmentation
- Time-series to sequence tasks

**When NOT to Use:**
- When you have frame-level alignment annotations (standard cross-entropy is simpler)
- For non-monotonic alignments (CTC assumes monotonicity)
- When conditional independence assumption is violated
- With attention-based models where alignment is learned (use cross-entropy)
- For very long sequences where computation becomes prohibitive

---

## 3. Sequence Loss

**Formula:** L = Σₜ wₜ × CrossEntropy(yₜ, ŷₜ) with sequence weights

**Pros:**
- Can weight different timesteps differently
- Handles padding naturally
- Flexible weighting scheme
- Easy to mask invalid positions
- Supports variable-length sequences
- Can focus on important parts of sequence

**Cons:**
- Still suffers from exposure bias
- Requires proper weight/mask specification
- Doesn't optimize sequence-level metrics
- Independence assumption between timesteps
- Can be tricky to set weights correctly

**When to Use:**
- When different parts of sequence have different importance
- Handling variable-length sequences with padding
- When you want to ignore certain timesteps
- Language modeling with masking
- Any seq2seq task with importance weighting

**When NOT to Use:**
- When all timesteps are equally important (standard cross-entropy is simpler)
- If you can't determine appropriate weights (arbitrary weighting can hurt)
- For tasks requiring sequence-level optimization (doesn't solve exposure bias)
- When mask specification is error-prone in your pipeline
- If simple uniform weighting works well (unnecessary complexity)

---

## 4. Edit Distance Loss

**Formula:** L = EditDistance(y, ŷ) = minimum operations to transform ŷ to y

**Pros:**
- Directly measures sequence similarity
- Intuitive metric (insertions, deletions, substitutions)
- Sequence-aware loss
- Better aligned with evaluation metrics
- Considers whole sequence structure
- Works well for string matching

**Cons:**
- Not differentiable (requires approximation or reinforcement learning)
- Computationally expensive (O(n²))
- Difficult to optimize directly
- May need differentiable approximations
- Can be unstable during training
- Requires special optimization techniques

**When to Use:**
- Spell correction
- DNA sequence alignment
- When edit distance is the evaluation metric
- String matching tasks
- When insertions/deletions are common errors
- OCR post-processing

**When NOT to Use:**
- For standard gradient-based training (not differentiable - use approximations or RL)
- With very long sequences (O(n²) computational complexity)
- When cross-entropy works well (much simpler and faster)
- For real-time applications (expensive computation)
- Without experience in RL or differentiable approximations (complex to implement)

---

## 5. BLEU Score Loss

**Formula:** L = 1 - BLEU = 1 - BP × exp(Σₙ wₙ log pₙ)

**Pros:**
- Standard machine translation metric
- Considers n-gram precision
- Accounts for brevity
- Better correlation with human judgment than per-token CE
- Sequence-level evaluation
- Well-established in NMT community

**Cons:**
- Not differentiable (requires approximation or RL)
- Corpus-level metric (not ideal for single sequences)
- Focuses on precision, not recall
- Can be gamed with common phrases
- Requires smoothing for zero counts
- Training can be unstable

**When to Use:**
- Machine translation training (with RL/approximation)
- When BLEU is the evaluation metric
- Text summarization
- Paraphrase generation
- Fine-tuning translation models
- Usually combined with cross-entropy

**When NOT to Use:**
- For initial model training (not differentiable - start with cross-entropy)
- When training stability is crucial (can be unstable)
- For single-sentence optimization (corpus-level metric)
- Without RL or differentiable approximation infrastructure
- When recall matters more than precision (BLEU focuses on precision)

---

## 6. Perplexity

**Formula:** PPL = exp(CrossEntropyLoss) = exp(-1/N Σ log P(wᵢ))

**Pros:**
- Interpretable (average branching factor)
- Standard language modeling metric
- Easy to understand and report
- Directly related to cross-entropy
- Good for model comparison
- Reflects model uncertainty

**Cons:**
- Not a loss function per se (evaluation metric)
- Exponential scale can be less intuitive for optimization
- Doesn't directly measure sequence quality
- Can be affected by vocabulary size
- May not correlate with downstream task performance
- Just a transformation of cross-entropy

**When to Use:**
- Language model evaluation (not training)
- Reporting model performance
- Comparing language models
- Understanding model confidence
- Academic papers and benchmarks
- As a secondary metric alongside cross-entropy loss

**When NOT to Use:**
- As a training loss (use cross-entropy directly - perplexity is just exp(CE))
- For optimization (exponential scale makes gradients less intuitive)
- When downstream task performance is what matters (may not correlate)
- For non-language modeling tasks (domain-specific metric)
- When you need interpretable loss values during training (cross-entropy is clearer)
