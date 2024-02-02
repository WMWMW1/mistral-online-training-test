ResNet, short for Residual Network, is a type of neural network architecture that was designed to enable the training of very deep neural networks. The core idea behind ResNet is the introduction of a so-called "residual block" that helps to combat the vanishing gradient problem in deep networks.

Here's a simplified mathematical representation of a ResNet:

1. **Residual Block**: The key component of ResNet is the residual block. A standard residual block has the following form:

   \[ \mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x} \]

   Here, \(\mathbf{x}\) and \(\mathbf{y}\) are the input and output vectors of the layers considered. \(\mathcal{F}(\mathbf{x}, \{W_i\})\) represents the residual mapping to be learned. For example, in a block with two layers, this would be \(\mathcal{F} = W_2 \sigma(W_1 \mathbf{x})\), where \(\sigma\) denotes the ReLU activation function, and \(W_1, W_2\) are weights of the two layers. The operation \(\mathbf{x} + \mathcal{F}(\mathbf{x}, \{W_i\})\) is performed by a shortcut connection and element-wise addition.

2. **Stacking Residual Blocks**: In a ResNet, several of these residual blocks are stacked together:

   \[ \mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \{W_l\}) \]

   Here, \(\mathbf{x}_{l+1}\) is the output of the \(l\)-th layer, and \(\mathbf{x}_l\) is its input. This equation is recursively applied for each layer.

3. **Identity Shortcut Connection**: The identity shortcut connections (when the input and output are of the same dimensions) help in propagating gradients back through the network without attenuation. When the dimensions increase, a linear projection \(W_s\) by shortcut connections is used to match the dimensions:

   \[ \mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s\mathbf{x} \]

4. **Overall Network Structure**: The overall structure of ResNet includes an initial convolutional layer, followed by stacking of residual blocks, and typically ends with a fully connected layer for classification tasks.

5. **Batch Normalization**: In practice, each layer in \(\mathcal{F}\) is usually followed by a batch normalization operation.

6. **Activation Function**: After each addition operation (like \(\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}\)), an activation function (commonly ReLU) is applied.

This architecture allows ResNet to train very deep networks (with hundreds of layers) effectively. The residual connections essentially make it easier to train the network by allowing the gradient to flow through the layers without being diminished rapidly.