# Transformer LLM — a decoder-only GPT in plain C++17

Built from scratch with no external libraries, in the same style as the
`LSTM_unit` and `SimpleRNN` projects: modular classes, heavy comments, math you
can read, clarity ahead of speed.

## Build and run

Get the data first:

```bash
mkdir -p data
curl -o data/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Then either:

```bash
mkdir -p build && cd build && cmake .. && make && ./main
```

or, without CMake:

```bash
g++ -O2 -std=c++17 -Iincludes -Iincludes/Math -Iincludes/Data \
    -Iincludes/Layers -Iincludes/Model \
    main.cpp src/Math/*.cpp src/Data/*.cpp src/Layers/*.cpp src/Model/*.cpp \
    -o transformer
./transformer
```

## Layout

```
includes/                       src/
  Math/                           Math/
    matrix.h                        matrix.cpp
    activation_functions.h          activation_functions.cpp
  Data/                           Data/
    char_tokenizer.h                char_tokenizer.cpp
    text_dataset.h                  text_dataset.cpp
  Layers/                         Layers/
    linear_layer.h                  linear_layer.cpp
    embedding_table.h               embedding_table.cpp
    layer_norm.h                    layer_norm.cpp
    multi_head_attention.h          multi_head_attention.cpp
    feed_forward.h                  feed_forward.cpp
    transformer_block.h             transformer_block.cpp
  Model/                          Model/
    model_config.h                  gpt_model.cpp
    gpt_model.h
main.cpp
```

Read them in that order — each file only depends on the ones above it.

## Stage 1 (done): the forward pass

`main.cpp` loads Tiny Shakespeare, builds a ~818k parameter model, runs a batch
through it, and checks the loss against `log(vocabulary_size)`.

An untrained model should assign roughly equal probability to all 65 symbols,
which gives a loss of `log(65) = 4.174`. Measured: `4.177`. That single number
confirms the embeddings, all four attention projections, both layer norms, the
feed-forward blocks, the residual connections and the output head are all
correctly wired and correctly scaled.

The printed attention matrix should be strictly lower-triangular. That triangle
is the causal mask — the model physically cannot see the future.

## Stage 2 (next): the backward pass

Every class already carries `gradient_*` buffers, `cached_*` activations, and a
`zero_gradients()` method, so backprop is a matter of adding a
`compute_backward()` to each layer, in reverse order of the forward pass:

1. cross entropy + softmax (they combine into a beautifully simple derivative)
2. `Linear_Layer`
3. `Feed_Forward` (needs `gelu_derivative`, already written)
4. `Layer_Norm` (the fiddly one)
5. `Multi_Head_Attention`
6. `Transformer_Block`, then `Gpt_Model`
7. a numerical gradient check, exactly like the RNN project

## Stage 3: training and generation

Adam optimizer, the training loop, and sampling with temperature and top-k.

## Stage 4: TinyStories

Needs a word-level or BPE tokenizer. Because everything downstream of
`Char_Tokenizer` only ever sees integers, that swap touches nothing else.
