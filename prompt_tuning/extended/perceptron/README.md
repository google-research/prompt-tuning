# Perceptron Training for Prompt Tuning

## Background

**Output Control can be Brittle**: We have observed that the soft prompts
learned using prompt tuning
[(Lester et al., 2021)](https://aclanthology.org/2021.emnlp-main.243/) tend to
have large norms. We also observed that when using checkpoints from T5 1.1,
trained with the Span Corruption objective
[(Raffel et al, 2020)](https://jmlr.org/papers/v21/20-074.html) it can be
difficult for the prompt to overrides the models strong bias toward generating a
sentinel (`<extra_id_0>`) as the first token. One possibility is that this high
norm is needed to make large enough changes to model activations to actually
influence the model's output.

We have seen in work on Prompt Design (GPT-3
[(Brown et al., 2019)](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
for example) that rank classification (restricting predictions to the set of
possible labels) is generally needed to control a model with text based prompts
(which have a much harder time influencing model output as they are fixed to
representations of words the model already knows).

Similarly, Prompt Recycling
[(Lester et al., 2022)](https://arxiv.org/abs/2208.05577) showed that transfer
of task knowledge across models via prompts was possible; however, transfer of
output control was not—as evidenced by their use of Rank Classification.

**Training does not Match Inference**: In FLAN
[(Wei et al, 2022)](https://arxiv.org/abs/2109.01652) rank classification was
used in conjunction with Prompt Tuning, but training still used a standard cross
entropy loss.

When training with a cross entropy loss, we are still trying to make the
example's target the **most** probable output, even through this is not what we
need at test time. We just need the correct target to the be most probable of
the possible labels. The model could still really want to output "Hi" but we
just ignore that. Having a training objective that still optimizes for most
probable output will still drive the model to use prompts with high norms.

This work explores if it is possible to train a prompt using a ranking loss so
that control of the model's actual output is less of a factor.

## Perceptron Loss

We can use a ranking loss, such as the Perceptron loss, to directly optimize for
out test time requirements, that the correct class $$y = C_i$$ is ranked higher
than any other class $$\tilde{y}_{\tilde{y}\neq y} = C_j$$.

The Perceptron algorithm is a way to train non-probabilistic global models. More
information about it (for training sequence models) can be found in
[these slides](https://www.phontron.com/class/nn4nlp2021/assets/slides/nn4nlp-14-structure.pdf).

In our case, the "global" aspect of the output is the full label string, even if
it has been broken into multiple Sentence Pieces
[(Kudo and Richardson, 2018)](https://aclanthology.org/D18-2012.pdf) or is a
multi-token string (for example an entity in ReCoRD
[(Zhang et al., 2018)](https://arxiv.org/abs/1810.12885))

The general Perceptron algorithm is:

1.  $$\hat{Y} = \text{argmax}_{\tilde{Y} \neq Y} S(\tilde{Y} | X ; \theta)$$
    Find the 1-best solution according to your model.
2.  If $$S(\tilde{Y} | X ; \theta) \geq S(Y|X;\theta)$$ then $$\theta \leftarrow
    \theta + \alpha (\frac{\partial S(Y|X;\theta)}{\partial \theta} -
    \frac{\partial S(\hat{Y} | X; \theta)}{\partial \theta})$$ If the 1-best
    score is better than the gold score, increase the score of the gold answer
    and decrease the score of the 1-best.

**Perceptron Loss in a Neural Network**:

We can express this as a loss function that we can plug into any neural network.

$$
\ell_{\text{percept}}(X, Y; \theta) = \max(0, S(\hat{Y}|X;\theta) - S(Y|X;\theta))
$$

If we look at the gradient, we can see that it looks just like the perceptron
update rule.

$$
\frac{\partial \ell_{\text{percept}}(X, Y)}{\partial \theta} = \begin{cases}
    \frac{\partial S(Y | X ; \theta)}{\partial \theta} - \frac{\partial S(\hat{Y} | X ; \theta)}{\partial \theta} & \text{if }  S(\hat{Y} | X ; \theta) \geq S(Y | X ; \theta) \\
    0 & \text{otherwise}
  \end{cases}
$$

An issue with this approach is that you need to do inference during training. In
the classification setting, inference is just scoring a set of possible labels
with the model and is therefore cheap, just a max over the scores assigned to
each class.

**Hinge Loss Extension**:

We can also extend this approach by adding a hinge loss. With a hinge, the gold
score has to be better than the 1-best score by some margin $m$. This results in
a more robust classifier, although the size of margin becomes another
hyper-parameter to tune.

$$
\ell_{\text{hinge}}(X, Y; \theta) = \max(0, m + S(\hat{Y}|X;\theta) - S(Y|X;\theta))
$$

## Implementation

The main idea behind the implementation is that instead of a training example
being a single `(input, target)` pair, it is now a `(List[input], List[target],
List[bool])` where all lists are the same length. The list of targets enumerates
all the possible targets. The list of inputs is the original input replicated,
once for each target. Finally there is one-(or multi)-hot feature marking which
target is correct, the `List[bool]`.

Shapes for important intermediate tensors include:

*   `B` :: The batch size
*   `C` :: The number of classes
*   `T` :: The max number of tokens
*   `T_t` :: The max number of target tokens.
*   `inputs` :: `[B, C, T]`
*   `targets` :: `[B, C, T_t]`
*   `is_correct` :: `[B, C]`
*   `scores_over_classes` :: `[B, C]`
*   `y_hat` :: `[B]`
*   `y` :: `[B]`

### Datasets

We evaluate on 7 tasks, 3 from GLUE
[(Wang et al., 2018)](https://aclanthology.org/W18-5446/) (SST2
[(Socher et al., 2013)](https://nlp.stanford.edu/sentiment/), MRPC
[(Dolan and Brockett, 2005)](https://www.microsoft.com/en-us/download/details.aspx?id=52398),
and QQP
[(Iyer et al., 2017)](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs))
and 4 from SuperGLUE
[(Wang et al., 2019)](https://papers.nips.cc/paper/2019/hash/4496bf24afe7fab6f046bf4923da8de6-Abstract.html)
(BoolQ [(Clark et al., 2019)](https://aclanthology.org/N19-1300/), CB
[(De Marneff et al., 2019)](https://github.com/mcdm/CommitmentBank), CoPA
[(Roemmele et al., 2011)](https://people.ict.usc.edu/~gordon/copa.html), WiC
[(Pilehvar and Camacho-Collados, 2019)](https://aclanthology.org/N19-1128/)).
Code for these tasks can be found in
[prompt_tuning/extended/perceeptron/data/tasks.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/extended/perceptron/data/tasks.py).

**Generation and Rank Classification Mixtures**:

As part of our experiments, we compare autoregressive generation to rank
classification when evaluating models trained with token-level cross-entropy
loss. For these we create a generative and a rank classification version of the
inference task in SeqIO. This is mixed together with the standard generative
training task to get results for both settings with a single model.

**Perceptron Mixtures**:

The perceptron datasets are made using the T5 rank classification preprocesssor
in `fewshot_eval` mode to keep everything as a single batch, instead of breaking
it into multiple examples—one for each possible target. We then do normal
tokenization and trimming, but then a bit of padding to make sure the input and
targets are dense tensors instead of ragged.

*Note: SeqIO and Non-Rank 1 Tensors* SeqIO currently (2022/03/09) has limited
support for 2D input tensors. To make sure that things like maximum length
inference works during evaluation, we needed to transpose our inputs so the
sequence dimension was first, i.e. `[T, C]` instead of `[C, T]`.

### Model

The implementation and be found in
[prompt_tuning/extended/perceptron/train/models.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/extended/perceptron/train/models.py).

The implementation makes heavy use of the folding trick where the batch and
class dimension are folded together to process all possibilities at once.

**Training**:

During training, score are calculated for each possible target and the best
score is found with `y_hat = jnp.max(scores, axis=-1)`. The score for correct
answer can be found by using the `is_correct` feature as a mask, `y =
jnp.sum(scores * is_correct, axis=-1)`.

**Inference**:

At inference time the model's prediction is calculated with `pred =
jnp.argmax(scores, axis=-1)`.

**Cross Entropy Extension**:

Given that we are limited to just $|C|$ possible outputs (instead of the
exponential number of outputs, $T^{|C|}$, one gets from a sequence prediction
problem), globally scoring all possible outputs is tractable. Therefore, there
is also a version of the model that uses cross-entropy for training, but instead
of it being at the token level, it is used at the sequence level. That is, the
inputs are the scores over classes used above. We have not done an extensive
study on the efficacy of this loss function.

## Experiments and Results

| Method     | SST2   | MRPC   | QQP    | BoolQ  | CB      | CoPA   | WiC    |
| ---------- | -----: | -----: | -----: | -----: | ------: | -----: | -----: |
| Perceptron | 93.0 ± | 80.6 ± | 84.4 ± | 69.4 ± | 82.4 ±  | 52.3 ± | 60.8 ± |
:            : 0.3    : 3.1    : 0.3    : 0.8    : 7.3     : 4.0    : 2.1    :
| Cross      | 94.2 ± | 81.9 ± | **85.6 | 71.3 ± | 78.22 ± | 48.4 ± | 61.1 ± |
: Entropy    : 0.6    : 0.5    : ±      : 0.3    : 4.7     : 5.5    : 1.4    :
:            :        :        : 0.1**  :        :         :        :        :
| Cross      | 94.2 ± | 82.5 ± | 85.6 ± | 71.3 ± | 78.22 ± | 48.3   | 60.7 ± |
: Entropy +  : 0.6    : 1.2    : 0.1    : 0.2    : 4.7     : ±5.5   : 2.0    :
: Rank       :        :        :        :        :         :        :        :

Above we see results from prompt tuning using Perceptron training compared with
normal Cross Entropy training. While the numbers for Perceptron training tend to
be lower, QQP is the only dataset where the difference is statistically
significant.

Additionally, the "Cross Entropy + Rank" row shows the results from training
with a standard cross entropy loss, but then using rank classification at
evaluation time instead of autoregressive generation. We see that there is some
slight noise, especially on datasets like MRPC where one label (`not_equivalent`
$\rightarrow$ `[59, 834, 15, 1169, 15592]`) is broken into many more pieces than
the other (`equivalent` $\rightarrow$ `[7072]`); however, this difference is not
statistically significant. For most datasets it is more likely due to numerical
noise based on the differences between how SeqIO/T5X
[(Roberts et al., 2022)](https://arxiv.org/abs/2203.17189) implements
autoregressive generation and rank classification (where inputs are replicated
for each label) rather than there being bias toward some label in the model's
generation.

## Analysis

**Prompt Norms**

Type                 | Mean Norm | Std Dev
-------------------- | --------: | ------:
Embeddings           | 297.35    |
Cross-Entropy Prompt | 1600.16   | 89.85
Perceptron Prompt    | 299.08    | 24.37

Prompt Norms are know to be large, one possible explanation is that these large
values are required to make enough changes to attention to completely overwrite
the original decoder logits. In Perceptron training, where you don't need to be
the *most* probable, these large values might not be required. We see in the
table above that the norms of prompts trained with perceptron loss are more in
line with the norms for the pre-trained token embeddings.

**Output Space**

While a Perceptron Prompt isn't explicit trained to control the generative
output space, it still makes a lot of changes. For example, on MRPC, the
unprompted model often begins generations with `sentence3:`. When a Perceptron
trained prompt is used, we see generations like `a 149mph serve` or `WD Energy
employee`. This suggests that there would still be composition issues if you try
to combine multiple perceptron prompts.

Prompt Tuning with standard cross entropy loss is known to be strong in this
evaluation setting, classification problems from GLUE/SuperGLUE, so there may be
other settings where Perceptron training is useful. One possible setting is the
low-resource setting. It may be easier to generalize if the prompt doesn't need
to take on such extreme values.


## Note

This is not an officially supported Google product. As this code supported an
experiment with negative results, there is no guarantee it will continue to
function as the rest of the code base evolves.
