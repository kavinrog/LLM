# AdaDecode ✨

**Faster Large Language Model (LLM) Decoding using Adaptive Layer Parallelism**
---

## 🚀 What is AdaDecode?

**AdaDecode** is a decoding technique for LLMs that makes text generation **faster** without changing the model or using extra helper models. It works by **predicting some tokens early** (before all layers are done) and starting the next token's processing right away.

> "If the model is already confident about the next word, let's not wait. Just go!"

---

## ✨ Key Features

* ✅ Faster decoding (up to **1.73x** speedup)
* ✅ No need for extra models (unlike speculative decoding)
* ✅ Keeps output the **same** as normal decoding (verified)
* ✅ Easy to plug into existing transformer models

---

## ⚖️ How It Works (Simple Explanation)

In normal decoding:

* The model thinks **very deeply** (all layers) before writing each word.
* It goes step by step: token → token → token.

In AdaDecode:

1. The model **guesses early** if it's confident (e.g., at layer 20 instead of 50).
2. It **starts the next word immediately**.
3. Meanwhile, it **finishes the skipped layers in the background**.
4. It checks: was the early guess correct?

   * If YES ✅: move on!
   * If NO ❌: redo properly (rollback).

### 🔹 Example:

Generating: `The cat is sleeping.`

Normal decoding:

```
Think 50 layers → 'The'
Think 50 layers → 'cat'
Think 50 layers → 'is' ...
```

AdaDecode:

```
Think 20 layers → (confident) → 'cat' → start next
Finish layers 21-50 for 'cat' in background
Verify later
```

---

## ⚠️ Downsides

* Might rollback if early guess was wrong (some overhead)
* More complex than standard decoding
* Works best when tokens are predictable

---

## 📆 Citation

If you use AdaDecode, please cite:

```
@article{wei2025adadecode,
  title={AdaDecode: Accelerating LLM Decoding with Adaptive Layer Parallelism},
  author={Wei, Zhepei and Chen, Wei-Lin and Zhu, Xinyu and Meng, Yu},
  journal={arXiv preprint arXiv:2506.03700},
  year={2025}
}
```

---

## 🔗 Links

* [📄 Paper on arXiv](https://arxiv.org/abs/2506.03700v1)
* [📁 Official Code (GitHub)](https://github.com/weizhepei/AdaDecode)

---

Made with ❤️ by the community. Contributions welcome!

