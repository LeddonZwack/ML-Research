# ML-Research

### MCNC: Manifold-Constrained Neural Compression

**Core Concepts**:

MCNC introduces a novel way to represent model parameters through a low-dimensional nonlinear manifold. Instead of searching for solutions in the full, high-dimensional parameter space, MCNC restricts the search to a curved (nonlinear) subspace shaped by a “random generator” network with sinusoidal activations.

1. **Wrapping a Lower Dimensional Space Around a Higher One**:
   - The idea: You start with a low-dimensional space (k dimensions) and use a nonlinear mapping φ: R^k → R^d that “wraps” or maps this k-dimensional space onto a d-dimensional hypersphere (Sd−1).
   - Why a hypersphere? A d-dimensional hypersphere Sd−1 is the set of points in Rd that all lie at the same distance (say radius 1) from a fixed center. The paper uses this because restricting weight directions to lie on a sphere is a natural way to ensure uniform coverage of directions. Any parameter vector in θ ∈ R^d can be decomposed into its amplitude β and direction u on the sphere: θ = θ0 + βu. By representing direction u via a nonlinear generator, you’re effectively exploring a manifold “wrapped” on the surface of the sphere.
   - What does the hypersphere represent? It represents all possible directions in parameter space. Since modern deep networks are often overparameterized, good solutions exist in many directions. Constraining parameters to lie on the hypersphere means you only vary directions (and a scale β), rather than searching the entire Rd blindly.
   - Data distribution on the hypersphere: The random generator network tries to evenly cover or “uniformly” parameterize the sphere. Essentially, you sample α in [−L, L]^k and map it through φ to a point on the sphere. This ensures that as α changes in a k-dimensional space, the mapped points are distributed broadly and fairly uniformly on the hypersphere.

2. **Linear Methods (like Low-Rank Approximations)**:
   - Linear methods for model compression or parameterization often involve approximating weight matrices by low-rank factorizations. For example, LoRA (in NOLA) or other PEFT methods factorize W into A*B where A and B have a small rank r, thus approximating a large parameter matrix with a product of two smaller ones.
   - Low-rank approximations are linear because you represent large parameter sets as a linear combination of fewer basis vectors/matrices. They do not capture nonlinear structure, potentially missing more intricate parameter patterns. MCNC tries to go beyond linearity by using nonlinear generators.

3. **Activations and Specifically Sinusoidal Ones**:
   - Activations are the nonlinear functions applied at the layers of a neural network (like ReLU, Sigmoid, or Tanh). Sinusoidal activations (Sine) are periodic functions, which means they map inputs to outputs in a wavelike pattern.
   - MCNC uses sinusoidal activations in the random generator network. Sine activations have infinite differentiability and allow for “smooth” and periodic coverage of the sphere. They help ensure good coverage and differentiability properties of the manifold.

4. **Random Networks and Their Usage**:
   - A “random network” here means a network whose weights are not trained. They are initialized randomly once and then fixed. The random generator in MCNC is such a network.
   - This network is used to map from the low-dimensional α space to the hypersphere. Because it’s random and uses sinusoidal activations, it creates a rich, nonlinear embedding that can represent a wide variety of directions on the sphere.
   - The randomness ensures that no a priori structure is assumed about what directions are “good” or “bad.” Combined with training α and β, MCNC finds a good low-dimensional manifold subspace that matches the target task’s solution.

---

### PRANC: Pseudo Random Networks for Compacting Deep Models

**Core Concepts**:

PRANC represents a trained model’s parameters as a linear combination of a small set of fixed, randomly generated “basis” models. Instead of storing the entire parameter vector θ, you store only the coefficients α_i that combine these random bases.

1. **Pseudo-Random Basis Vectors (or Basis Models)**:
   - These are randomly generated sets of parameters. Imagine you have k large random weight vectors θˆ_j. Each θˆ_j is fixed and not trained, only generated once from a pseudorandom seed.
   - They form a basis in a very high-dimensional space. Though not orthonormal in the strict linear algebra sense, the dimension is so high that random vectors are nearly orthogonal, making them a good “spread” of directions.

2. **“Learn only the linear combination coefficients”**:
   - Normally, training a model means adjusting all parameters W. PRANC says: Keep W forms as θ = Σ α_j θˆ_j. Instead of optimizing W directly, you only optimize the α_j’s.
   - Once you have α_j’s, you reconstruct the full model by α_j * θˆ_j. This drastically reduces the storage because you only store k coefficients (plus the seed for generation). The bulk of parameter data is implicit in the random generator seed.

3. **Recreating a Complex Model from a Linear Combination of Basis Vectors**:
   - High-dimensional spaces are surprisingly rich. Even randomly chosen vectors can span a set of “good solutions” for complex tasks.
   - By adjusting α_j, you find a point in the subspace spanned by these random vectors that corresponds to a well-performing model. Though each θˆ_j is random and by itself not special, their linear combinations can approximate trained solutions.

---

### NOLA: Compressing LoRA Using Linear Combination of Random Basis

**Core Concepts**:

NOLA builds on LoRA, which provides low-rank parameter increments for adapting large language models. NOLA then applies PRANC-like logic (linear combinations of random bases) to the LoRA parameters themselves.

1. **LoRA (Low-Rank Adaptation) Explained in Detail**:
   - When you have a large pretrained model (like a big LLM), fine-tuning all parameters for each new task is expensive.
   - LoRA says: Instead of fine-tuning the entire weight matrix W (which might be huge), represent the adaptation ΔW as a low-rank factorization ΔW = A * B, with A in R^(d×r) and B in R^(r×d), where r << d.
   - You freeze W and only learn A and B. This drastically reduces the trainable parameters. At inference, you combine W + A*B to adapt the model to the new task.
   - LoRA is linear and low-rank: it assumes the adaptation needed lives in a low-dimensional subspace.

2. **Pseudo-Random Matrices vs. Pseudo-Random Basis Vectors**:
   - In PRANC, you had random basis vectors for the entire parameter vector. For LoRA, you are dealing with matrices A and B. 
   - Pseudo-random matrices are just 2D versions of pseudo-random vectors. Instead of a single long vector, you generate random matrices. The principle is the same: random initialization with a fixed seed means they can be reconstructed easily.
   - Applying PRANC-like ideas to LoRA’s A and B means you represent each of these low-rank factors as a linear combination of smaller random matrices.

3. **Context: “Since LoRA modifies only a small subset of parameters, applying PRANC-like decomposition leads to extreme compression.”**
   - LoRA is already a compression technique: instead of updating all parameters, it updates a small, low-rank slice.
   - Now apply PRANC-like decomposition on that small slice (A and B). This reduces storage further since you represent A and B using a few random basis matrices and a set of coefficients.
   - The adaptability is preserved because LoRA’s low-rank parameters can still approximate task-specific updates. NOLA just makes these updates even more compact.

---

### SupSup: Supermasks in Superposition

**Core Concepts**:

SupSup trains a single randomly initialized network once and never changes its weights. For each new task, it finds a binary mask (supermask) that selects a subnetwork that can solve that task. Thus, each task is represented by a different pattern of zeros and ones over the same base parameters.

1. **Randomly Initialized, Fixed Base Networks**:
   - Think of having a large neural network (weights W), which you never train. You just sample it once from a distribution (like a normal initialization).
   - This network is large and overparameterized. The insight (from lottery ticket hypothesis and related works) suggests that inside this large random network are subnetworks that can perform well on many tasks, if only you find the right mask.

2. **“Implicitly Storing Them as Attractors in a Fixed-Sized Hopfield Network”**:
   - A Hopfield network is a type of recurrent network that can store a set of binary patterns as stable attractors. When you present partial or noisy input, the Hopfield network’s dynamics converge to the nearest stored pattern.
   - SupSup can store multiple supermasks as binary patterns in a Hopfield network’s weight matrix. Then, to recall a supermask for a particular task, you do gradient steps in the Hopfield energy landscape until you fall into the basin of attraction corresponding to that mask. This is a way to represent a large number of supermasks without storing them all explicitly as separate parameter sets.

3. **Training vs. Inference, Task Identity, and Why They Matter**:
   - Training: For each new task, you learn a binary mask M that, when applied to W, solves that task. You do this by optimizing mask scores and thresholding them.
   - Inference: When you get new input from an unknown task, you must figure out which mask M_i to use. That’s why task identity matters. If you know which task it is, just pick the right mask. If you don’t, SupSup proposes methods (like entropy minimization) to infer which mask would produce a confident prediction.
   - Identifiers (like task IDs) simplify the process of choosing the right mask at inference. Without them, you have to optimize or guess which mask leads to the best low-entropy output.
