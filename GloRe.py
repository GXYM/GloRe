import tensorflow as tf
slim = tf.contrib.slim

def GloRe(X, C,  N, activation_fn=None):
    imput_chancel = X.get_shape().as_list()[-1]
    inputs_shape = tf.shape(X)

    B = slim.conv2d(X, N, [1, 1])
    B = tf.reshape(B, [inputs_shape[0], -1, N]) # [B, H*W, N]

    x_reduced = slim.conv2d(X, C, [1, 1])
    x_reduced = tf.reshape(x_reduced, [inputs_shape[0], -1, C]) # [B,  H*W, C]
    x_reduced = tf.transpose(x_reduced, perm=[0, 2, 1])  # [B, C, H*W]

    # [B, C, H * W] * [B, H*W, N] —>#[B, C, N]
    v = tf.matmul(x_reduced, B) # [B, C, N]
    v = tf.expand_dims(v, axis=1)  # [B, 1, C, N]

    def GCN(Vnode, nodeN, mid_chancel):

        net = slim.conv2d(Vnode, nodeN, [1, 1], ) # [B, 1, C, N]

        net = Vnode -net  #(I-Ag)V

        net = tf.transpose(net, perm=[0, 3, 1, 2]) # [B, N, 1, C]

        net = slim.conv2d(net, mid_chancel, [1, 1]) # [B, N, 1, C]

        return net

    z = GCN(v, N, C) # [B, N, 1, C]
    z = tf.reshape(z, [inputs_shape[0], N, C])  # [B, N, C]

    # [B, H*W, N] * [B, N, C] => [B, H*W, C]
    y = tf.matmul(B, z) # [B, H*W, C]
    y = tf.expand_dims(y, axis=1)  #[B, 1, H*W, C]
    y = tf.reshape(y, [inputs_shape[0], inputs_shape[1], inputs_shape[2], C])  # [B, H, W, C]
    x_res = slim.conv2d(y, imput_chancel, [1, 1])

    return X + x_res
