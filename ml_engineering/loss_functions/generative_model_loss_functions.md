# Generative Model Loss Functions

## 1. Adversarial Loss

**Formula:** L = E[log D(x)] + E[log(1 - D(G(z)))]

**Pros:**
- Core of GAN training
- Enables realistic sample generation
- No explicit density modeling needed
- Can generate sharp, realistic images
- Implicit learning of data distribution
- Powerful generative framework

**Cons:**
- Training instability (mode collapse)
- Difficult to balance generator and discriminator
- No direct likelihood evaluation
- Requires careful hyperparameter tuning
- Vanishing gradients problem
- Hard to diagnose training issues

**When to Use:**
- GAN training
- Image generation
- Data augmentation
- Style transfer
- Image-to-image translation
- Any adversarial training scenario

**When NOT to Use:**
- When training stability is critical and you can't handle mode collapse (use VAEs)
- For explicit density estimation (use autoregressive models or normalizing flows)
- Without experience in GAN training (very difficult to tune and debug)
- When you need guaranteed sample diversity (mode collapse is common)
- For small datasets where GANs may not converge well

---

## 2. Generator Loss

**Formula:** L_G = -E[log D(G(z))] (or variants)

**Pros:**
- Trains generator to fool discriminator
- Encourages realistic samples
- Simple formulation
- Part of adversarial framework
- Can use various formulations (non-saturating, etc.)
- Flexible objective

**Cons:**
- Can suffer from vanishing gradients
- Mode collapse issues
- Requires stable discriminator
- No guarantee of diversity
- Sensitive to discriminator quality
- Can be hard to balance

**When to Use:**
- In conjunction with discriminator in GANs
- Image generation tasks
- When training generative models adversarially
- StyleGAN, DCGAN, and variants
- Any GAN architecture
- Conditional generation

**When NOT to Use:**
- Without a properly trained discriminator (depends on discriminator quality)
- When you can't handle vanishing gradient issues (use non-saturating variant)
- For tasks where mode collapse is unacceptable (consider alternatives like VAE)
- Without strategies to maintain diversity (mode collapse risk)
- When training stability is the top priority (VAEs are more stable)

---

## 3. Discriminator Loss

**Formula:** L_D = -E[log D(x)] - E[log(1 - D(G(z)))]

**Pros:**
- Learns to distinguish real from fake
- Provides training signal to generator
- Binary classification objective
- Relatively stable compared to generator
- Well-defined task
- Can use various architectures

**Cons:**
- Can become too strong (blocks generator learning)
- May memorize training data
- Requires balancing with generator
- Can suffer from gradient saturation
- May need regularization
- Training dynamics are complex

**When to Use:**
- GAN training (paired with generator)
- As part of adversarial framework
- Conditional GANs
- In all GAN variants
- Image synthesis
- Domain adaptation

**When NOT to Use:**
- When the discriminator becomes too powerful (blocks generator learning - use techniques like one-sided label smoothing)
- Without regularization strategies (may overfit or memorize)
- For standalone classification (designed for adversarial setting)
- When you can't properly balance with generator training (complex dynamics)
- Without monitoring for discriminator dominance

---

## 4. Wasserstein Loss

**Formula:** W(Pᵣ, Pᵧ) = sup_{||f||_L≤1} E_x[f(x)] - E_z[f(G(z))]

**Pros:**
- More stable training than standard GAN
- Meaningful loss metric (Earth Mover's Distance)
- Reduces mode collapse
- Provides convergence indicator
- Better gradient flow
- Theoretically grounded

**Cons:**
- Requires weight clipping or gradient penalty
- More complex implementation
- Lipschitz constraint enforcement is tricky
- Computational overhead
- Still has training challenges
- Hyperparameter sensitivity (gradient penalty weight)

**When to Use:**
- WGAN and WGAN-GP
- When standard GAN training is unstable
- When you need stable convergence
- For better mode coverage
- In research and production GANs
- As improved GAN objective

**When NOT to Use:**
- Without proper Lipschitz constraint enforcement (weight clipping or gradient penalty required)
- When computational resources are very limited (higher overhead than standard GAN)
- For simple tasks where standard GAN works well (added complexity)
- Without understanding of Wasserstein distance and gradient penalty (complex theory)
- When implementation simplicity is paramount (more complex than vanilla GAN)

---

## 5. Least Squares GAN Loss

**Formula:** L = E[(D(x) - 1)²] + E[D(G(z))²]

**Pros:**
- More stable than standard GAN
- Penalizes samples far from decision boundary
- Reduces vanishing gradient problem
- Simpler than Wasserstein
- Better quality images
- Easier to optimize

**Cons:**
- Still can suffer mode collapse
- Requires careful architecture design
- Less theoretically motivated than Wasserstein
- May need additional regularization
- Not as widely used as standard or Wasserstein GAN

**When to Use:**
- As alternative to standard GAN
- When you want more stable training
- Image generation tasks
- When standard GAN gradients vanish
- LSGAN architecture
- Simpler alternative to WGAN

**When NOT to Use:**
- When mode collapse is a major concern (doesn't fully solve it)
- For tasks where WG AN provides better stability (WGAN is more theoretically grounded)
- Without proper architecture tuning (still requires careful design)
- When you need the best theoretical guarantees (Wasserstein is better)
- For widely-supported standard architectures (less common than vanilla or Wasserstein GAN)

---

## 6. Hinge Loss (for GANs)

**Formula:** L_D = E[max(0, 1-D(x))] + E[max(0, 1+D(G(z)))]

**Pros:**
- Margin-based objective
- Used in state-of-the-art GANs (e.g., BigGAN)
- More stable than standard GAN
- Better gradient properties
- Focuses on boundary samples
- Proven in large-scale generation

**Cons:**
- Requires careful implementation
- May need spectral normalization
- Less intuitive than binary cross-entropy
- Hyperparameter sensitivity
- Needs large-scale training to shine

**When to Use:**
- BigGAN and similar architectures
- Large-scale image generation
- When standard GAN is unstable
- State-of-the-art generation tasks
- With spectral normalization
- High-resolution image synthesis

**When NOT to Use:**
- For small-scale generation tasks (benefits mainly at large scale)
- Without spectral normalization or similar techniques (requires proper regularization)
- When you need simple, interpretable loss (less intuitive than cross-entropy)
- For beginner GAN projects (more advanced technique)
- Without substantial computational resources (designed for large-scale training)

---

## 7. Variational Loss (VAE)

**Formula:** L = E[log p(x|z)] - KL(q(z|x)||p(z))

**Pros:**
- Theoretically principled
- Provides explicit likelihood bound
- Learns structured latent space
- Enables interpolation
- Stable training
- No adversarial training needed

**Cons:**
- Generates blurry images
- Lower sample quality than GANs
- KL term can be hard to balance
- Posterior collapse issues
- May not capture all modes
- Reconstruction-generation trade-off

**When to Use:**
- VAE training
- When you need explicit likelihood
- Structured latent representations
- Disentangled representations
- When training stability is crucial
- Anomaly detection

**When NOT to Use:**
- When you need sharp, high-quality images (GANs generate sharper samples)
- For tasks where sample quality trumps everything (GANs outperform on quality)
- When posterior collapse is problematic and you can't mitigate it
- For unconditional high-resolution image generation (GANs are better)
- When you don't need the probabilistic framework (GANs may be simpler for generation only)

---

## 8. Evidence Lower Bound (ELBO)

**Formula:** ELBO = E[log p(x|z)] - KL(q(z|x)||p(z))

**Pros:**
- Foundation of VAE training
- Theoretically grounded
- Provides lower bound on likelihood
- Balances reconstruction and regularization
- Enables variational inference
- Well-understood optimization

**Cons:**
- Same as variational loss (they're equivalent)
- Blurry reconstructions
- KL balancing issues
- May not maximize likelihood tightly
- Requires careful tuning of β (if using β-VAE)

**When to Use:**
- VAE and variants (β-VAE, etc.)
- Variational inference
- Probabilistic modeling
- When you need likelihood bounds
- Representation learning
- Semi-supervised learning

**When NOT to Use:**
- For standalone generation without probabilistic framework (just use reconstruction loss)
- When you need tight likelihood estimates (ELBO is only a lower bound)
- For tasks where GANs provide better sample quality
- Without strategies to prevent posterior collapse
- When β-VAE tuning is impractical (adds complexity)

---

## 9. Reconstruction Loss

**Formula:** L = ||x - x̂||² or BCE(x, x̂)

**Pros:**
- Simple and intuitive
- Ensures output resembles input
- Used in autoencoders
- Differentiable
- Easy to implement
- Clear objective

**Cons:**
- Can lead to blurry outputs (with L2)
- Doesn't ensure realistic samples alone
- May not capture perceptual quality
- Can be pixel-wise myopic
- Doesn't enforce distribution matching

**When to Use:**
- Autoencoders
- VAE (as part of ELBO)
- Image denoising
- Anomaly detection
- Feature learning
- Combined with other losses in VAE/GAN

**When NOT to Use:**
- As standalone loss for generation (doesn't ensure realistic/diverse samples)
- When perceptual quality matters more than pixel-wise accuracy (use perceptual loss)
- For high-quality image generation alone (leads to blurry outputs with L2)
- Without regularization in generative models (won't learn good latent space)
- When you need distribution matching (doesn't enforce it alone)

---

## 10. KL Divergence Loss

**Formula:** KL(q||p) = Σ q(x) log(q(x)/p(x))

**Pros:**
- Measures distribution difference
- Regularizes latent space in VAEs
- Encourages prior matching
- Theoretically grounded
- Enables sampling from prior
- Analytical form for Gaussians

**Cons:**
- Can cause posterior collapse
- May be too strong (regularization vs reconstruction)
- Not symmetric
- Can be numerically unstable
- Requires careful weighting (β in β-VAE)

**When to Use:**
- VAE training (regularization term)
- Knowledge distillation
- Distribution matching
- Variational inference
- Enforcing prior distributions
- Disentanglement (β-VAE)

**When NOT to Use:**
- Without strategies to prevent posterior collapse (common issue in VAEs)
- When reconstruction quality is paramount and KL term interferes (may need β<1)
- For tasks not requiring distribution matching (unnecessary regularization)
- With poorly chosen prior that doesn't match true posterior (hurts performance)
- Without careful β tuning in β-VAE (balance is critical)

---

## 11. Feature Matching Loss

**Formula:** L = ||E_x[f(x)] - E_z[f(G(z))]||²

**Pros:**
- Matches intermediate features instead of outputs
- More stable than standard GAN
- Reduces mode collapse
- Encourages diversity
- Uses discriminator features meaningfully
- Improves training stability

**Cons:**
- Requires discriminator to extract features
- May produce less sharp images
- Architecture-dependent
- Not as direct as adversarial loss
- May need to be combined with other losses

**When to Use:**
- GAN training stabilization
- Combined with adversarial loss
- When mode collapse is an issue
- Image generation
- As auxiliary loss in GANs
- Video generation

**When NOT to Use:**
- When you need maximum sharpness (may reduce sharpness compared to pure adversarial)
- Without a good feature-extracting discriminator (architecture-dependent)
- As standalone loss (typically combined with adversarial loss)
- For tasks where direct adversarial training works well (added complexity)
- When discriminator features aren't meaningful for your task

---

## 12. Perceptual Loss

**Formula:** L = Σₗ ||φₗ(x) - φₗ(x̂)||²

**Pros:**
- Uses pretrained network features (e.g., VGG)
- Better perceptual quality than pixel loss
- Captures high-level structure
- Widely used in style transfer and super-resolution
- More aligned with human perception
- Proven effective

**Cons:**
- Requires pretrained network
- Computationally expensive
- Architecture-dependent (usually VGG)
- May not transfer to all domains
- Adds memory overhead
- Requires careful layer selection

**When to Use:**
- Style transfer
- Super-resolution
- Image-to-image translation
- When perceptual quality matters more than pixel accuracy
- pix2pix, CycleGAN variants
- Image restoration

**When NOT to Use:**
- When pixel-level accuracy is required (not pixel-wise loss)
- Without access to pretrained networks (requires VGG or similar)
- For domains very different from ImageNet (pretrained features may not transfer)
- When computational resources are limited (expensive forward passes)
- For tasks where simple L2 reconstruction works well (unnecessary overhead)

---

## 13. Style Loss

**Formula:** L = Σₗ ||G(φₗ(x)) - G(φₗ(x_style))||²

**Pros:**
- Captures texture and style
- Uses Gram matrices of features
- Core of neural style transfer
- Separates style from content
- Proven in artistic applications
- Flexible style representation

**Cons:**
- Computationally expensive
- Requires pretrained network
- Gram matrix computation overhead
- May not capture all style aspects
- Hyperparameter sensitive (layer weights)

**When to Use:**
- Neural style transfer
- Artistic image generation
- Texture synthesis
- Combined with content loss
- Image stylization applications
- Video style transfer

**When NOT to Use:**
- Without pretrained network (requires feature extractor like VGG)
- When computational budget is tight (Gram matrix computation is expensive)
- For non-texture/style tasks (designed specifically for style transfer)
- Without content loss (needs to be balanced with content preservation)
- When simpler texture losses suffice

---

## 14. Content Loss

**Formula:** L = ||φₗ(x) - φₗ(x_content)||²

**Pros:**
- Preserves semantic content
- Uses high-level features
- Complements style loss
- Simple formulation
- Works with pretrained networks
- Intuitive objective

**Cons:**
- Requires pretrained network
- Single-layer may not be enough
- Computational cost
- May not preserve low-level details
- Architecture-dependent

**When to Use:**
- Style transfer (with style loss)
- Super-resolution
- Image reconstruction
- When semantic content must be preserved
- Photo-realistic style transfer
- Image enhancement

**When NOT to Use:**
- Without access to pretrained feature extractors (requires networks like VGG)
- For tasks where low-level pixel details are critical (focuses on high-level features)
- As standalone loss without style loss in style transfer (need both)
- When computational resources are very limited (requires forward pass through network)
- For domains where pretrained features don't transfer well

---

## 15. Total Variation Loss

**Formula:** TV = Σᵢⱼ √((xᵢ₊₁,ⱼ - xᵢⱼ)² + (xᵢ,ⱼ₊₁ - xᵢⱼ)²)

**Pros:**
- Encourages spatial smoothness
- Reduces noise and artifacts
- Simple to implement
- No pretrained network needed
- Effective regularization
- Widely used in image processing

**Cons:**
- Can over-smooth images
- May remove fine details
- Can create piecewise-constant regions
- Hyperparameter tuning needed
- May not be suitable for all image types

**When to Use:**
- As regularization in style transfer
- Image denoising
- Super-resolution
- Reducing artifacts in generated images
- Combined with other losses
- Image reconstruction

**When NOT to Use:**
- For images with important fine details and textures (over-smooths)
- As strong regularization on natural images (can look unnatural/cartoon-like)
- When piecewise-constant appearance is undesirable
- Without careful weight tuning (can dominate other losses)
- For tasks where smoothness is not desired (e.g., preserving sharp edges)
