import tensorflow as tf
import numpy as np
from nNetsEnsemble import ConstrainedSoftmax, test_constraint_layer

# Test the constraint layer
print("Testing ConstrainedSoftmax layer implementation")
print("="*60)

# Test with max_weight = 0.7
test_constraint_layer(max_weight=0.7, n_models=7)

print("\n" + "="*60)
print("Additional test: Direct layer usage")
print("="*60)

# Create the layer
layer = ConstrainedSoftmax(max_weight=0.7)

# Test with extreme inputs
test_input = tf.constant([[100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tf.float32)
output = layer(test_input)

print(f"Input logits: {test_input.numpy()[0]}")
print(f"Output weights: {output.numpy()[0]}")
print(f"Max weight: {tf.reduce_max(output).numpy():.4f}")
print(f"Sum of weights: {tf.reduce_sum(output).numpy():.4f}")

if tf.reduce_max(output).numpy() <= 0.7 + 1e-6:
    print("\n✅ SUCCESS: Constraint is properly enforced!")
else:
    print("\n❌ FAILURE: Constraint is NOT enforced!") 