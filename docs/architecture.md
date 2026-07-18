# Architecture

VHSS separates five concerns:

1. **Task specification** defines inputs, targets, permitted operators, and data splits.
2. **Work receipt validation** verifies SHA-256d digests against Bitcoin-style targets.
3. **Seed derivation** removes target-induced prefix bias using domain separation.
4. **Symbolic search** performs bounded local mutation, crossover, caching, and selection.
5. **Evaluation** measures candidate behavior independently of proof-of-work validity.

This separation is deliberate. A low hash proves work under a declared target;
it does not imply that a candidate program is accurate. Conversely, an accurate
candidate does not prove that work was performed. The protocol binds these two
facts without treating them as equivalent.

Large datasets and model weights are not stored in block headers. The current
release is a local software laboratory, not a blockchain implementation.
