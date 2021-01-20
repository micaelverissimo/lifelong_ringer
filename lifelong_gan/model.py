import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import Model
from tensorlayer import logging

from params import *

lrelu = lambda x: tl.act.lrelu(x, 0.2)

class DeConv1d(Layer):
    """Simplified version of :class:`DeConv1dLayer`, see `tf.nn.conv1d_transpose <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/conv1d_transpose>`__.

    Parameters
    ----------
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size width.
    strides : tuple of int
        The stride step width.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    act : activation function
        The activation function of this layer.
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    dilation_rate : int of tuple of int
        The dilation rate to use for dilated convolution
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([5, 100, 100, 32], name='input')
    >>> deconv2d = tl.layers.DeConv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), in_channels=32, name='DeConv2d_1')
    >>> print(deconv2d)
    >>> tensor = tl.layers.DeConv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), name='DeConv2d_2')(net)
    >>> print(tensor)

    """

    def __init__(
        self,
        n_filter=32,
        filter_size=3,
        strides=2,
        act=None,
        padding='SAME',
        dilation_rate=1,
        data_format='channels_last',
        W_init=tl.initializers.truncated_normal(stddev=0.02),
        b_init=tl.initializers.constant(value=0.0),
        in_channels=None,
        name=None  # 'decnn2d'
    ):
        super().__init__(name, act=act)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        # Attention: To build, we need not only the in_channels!
        # if self.in_channels:
        #     self.build(None)
        #     self._built = True

        logging.info(
            "DeConv1d {}: n_filters: {} strides: {} padding: {} act: {} dilation: {}".format(
                self.name,
                str(n_filter),
                str(strides),
                padding,
                self.act.__name__ if self.act is not None else 'No Activation',
                dilation_rate,
            )
        )

        if type(strides) != int:
            raise ValueError("type(strides) should be int... Like in tensorflow")

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
            ', strides={strides}, padding={padding}'
        )
        if self.dilation_rate != 1:
            s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        self.layer = tf.keras.layers.Conv1DTranspose(
            filters=self.n_filter,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.act,
            use_bias=(True if self.b_init is not None else False),
            kernel_initializer=self.W_init,
            bias_initializer=self.b_init,
            # dtype=tf.float32,
            name=self.name,
        )
        if self.data_format == "channels_first":
            self.in_channels = inputs_shape[1]
        else:
            self.in_channels = inputs_shape[-1]
        _out = self.layer(
            tf.convert_to_tensor(np.random.uniform(size=inputs_shape), dtype=np.float32)
        )  #np.random.uniform([1] + list(inputs_shape)))  # initialize weights
        outputs_shape = _out.shape
        self._trainable_weights = self.layer.weights

    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs

tf.config.experimental.set_memory_growth(gpu, True)
def residual(R, n_f, f_s):
	w_init = tl.initializers.truncated_normal(stddev=0.01)
	R_tmp = R
	R = BatchNorm1d(act=tf.nn.relu)(Conv1d(n_f, f_s, 1, W_init=w_init)(R))
	R = BatchNorm1d(act=None)(Conv1d(n_f, f_s, 1, W_init=w_init)(R))
	R_tmp = Conv1d(n_f, 1, 1)(R_tmp)
	return Elementwise(tf.add, act=tf.nn.relu)([R_tmp, R])


def Discriminator(input_shape, prefix = ""):
	I = Input(input_shape)
	D = Conv1d(
		64, 4, 2, padding='SAME', act=lrelu, b_init=None, name=prefix+'D_conv_1')(I)
	D = InstanceNorm1d(act=lrelu)(Conv1d(
		128, 4, 2, padding='SAME', b_init=None, name=prefix+'D_conv_2')(D))
	D = InstanceNorm1d(act=lrelu)(Conv1d(
		256, 4, 2, padding='SAME', b_init=None, name=prefix+'D_conv_3')(D))
	#D = InstanceNorm2d(act=lrelu)(Conv2d(
	#	512, (4, 4), (2, 2), padding='SAME', b_init=None, name=prefix+'D_conv_4')(D))
	#D = InstanceNorm2d(act=lrelu)(Conv2d(
	#	512, (4, 4), (2, 2), padding='SAME', b_init=None, name=prefix+'D_conv_5')(D))
	#D = InstanceNorm2d(act=lrelu)(Conv2d(
	#	512, (4, 4), (2, 2), padding='SAME', b_init=None, name=prefix+'D_conv_6')(D))
	D = Conv1d(1, 4, 1, name=prefix+'D_conv_7')(D)
	D = GlobalMeanPool1d()(D)
	D_net = Model(inputs=I, outputs=D, name=prefix+'Discriminator')
	return D_net


def Generator(input_shape, z_dim, prefix = ""):
	w_init = tl.initializers.truncated_normal(stddev=0.01)
	I = Input(input_shape)
	Z = Input([input_shape[0], z_dim])
	# read reshape and tile
	#z = Reshape((input_shape[0], 1, 1, -1))(Z)
	z = Reshape((input_shape[0], 1, -1))(Z)
	z = Tile([1, input_shape[1], 1])(z)

	print('MICAEL: I, z', I.shape, z.shape)
	conv_layers = []
	G = Concat(concat_dim=-1)([I, z])
	print('MICAEL: I, z, G', I.shape, z.shape, G.shape)
	filters = [64, 128, 256]#, 512, 512, 512, 512]
	if image_size == 256:
		filters.append(512)
	G = Conv1d(
		filters[0], 4, 2, act=lrelu, W_init=w_init, b_init=None, name=prefix+'G_conv_1')(G)
	conv_layers.append(G)
	for i, n_f in enumerate(filters[1:]):
		G = BatchNorm1d(act=lrelu)(Conv1d(
			n_f, 4, 2, W_init=w_init, b_init=None, name=prefix+'G_conv_{}'.format(i + 2))(G))
		conv_layers.append(G)

	filters.pop()
	filters.reverse()
	conv_layers.pop()
	for i, n_f in enumerate(filters):
		G = BatchNorm1d(act=tf.nn.relu)(DeConv1d(
			n_f, 4, 2, W_init=w_init, b_init=None, name=prefix+'G_deconv_{}'.format(len(filters)+1-i))(G))
		G = Concat(concat_dim=-1)([G, conv_layers.pop()])
	G = DeConv1d(3, 4, 2, act=tf.nn.tanh, W_init=w_init, b_init=None, name=prefix+'G_deconv_1')(G)
	G_net = Model(inputs=[I, Z], outputs=G, name=prefix+'Generator')
	return G_net


def Encoder(input_shape, z_dim, prefix=""):
	I = Input(input_shape)
	print('ENCODER, I ', I.shape)
	E = Conv1d(64, 4, 2, act=lrelu, name=prefix+'E_conv_1')(I)
	print('E shape: ', E.shape)
	E = MeanPool1d(2, 2, 'SAME')(residual(E, 128, 3))
	E = MeanPool1d(2, 2, 'SAME')(residual(E, 256, 3))
	#E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 512, 3))
	#E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 512, 3))
	#E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 512, 3))
	E = Flatten()(MeanPool1d(8, 8, 'SAME')(E))
	mu = Dense(z_dim)(E)
	log_sigma = Dense(z_dim)(E)
	z = Elementwise(tf.add)(
		[mu, Lambda(lambda x:tf.random.normal(shape=[z_dim]) * tf.exp(x))(log_sigma)])
	E_net = Model(inputs=I, outputs=[z, mu, log_sigma], name=prefix+'Encoder')
	return E_net

class BicycleGAN(object):
	count = 0

	def __init__(self, LOAD = False, load_tag = model_tag):
		BicycleGAN.count += 1
		self.G = Generator(input_shape, z_dim, "model_{}/".format(self.count))
		self.D = Discriminator(input_shape, "model_{}/".format(self.count))
		self.E = Encoder(input_shape, z_dim, "model_{}/".format(self.count))
		if LOAD:
			self.load(load_tag)
		self.G.train()
		self.D.train()
		self.E.train()

	def load(self, model_tag):
		tl.files.load_and_assign_npz(os.path.join(models_dir, "G_weights_{}.npz".format(model_tag)), self.G)
		tl.files.load_and_assign_npz(os.path.join(models_dir, "D_weights_{}.npz".format(model_tag)), self.D)
		tl.files.load_and_assign_npz(os.path.join(models_dir, "E_weights_{}.npz".format(model_tag)), self.E)
		print("Model weights has been loaded.")

	def save(self, model_tag):
		tl.files.save_npz(self.G.all_weights, os.path.join(models_dir, "G_weights_{}.npz".format(model_tag)))
		tl.files.save_npz(self.D.all_weights, os.path.join(models_dir, "D_weights_{}.npz".format(model_tag)))
		tl.files.save_npz(self.E.all_weights, os.path.join(models_dir, "E_weights_{}.npz".format(model_tag)))

	def calc_loss(self, image_A, image_B, z):
		print('MICA, imagem_B', image_B.shape)
		encoded_z, encoded_mu, encoded_log_sigma = self.E(image_B)
		vae_img = self.G([image_A, encoded_z])
		lr_img = self.G([image_A, z])
		self.vae_img = vae_img
		self.lr_img = lr_img

		reconst_z, reconst_mu, reconst_log_sigma = self.E(lr_img)

		P_real = self.D(image_B)
		P_fake = self.D(lr_img)
		P_fake_encoded = self.D(vae_img)
		self.P_real = P_real
		self.P_fake = P_fake
		self.P_fake_encoded = P_fake_encoded

		loss_vae_D = (tl.cost.mean_squared_error(P_real, 0.9, is_mean=True) +
					  tl.cost.mean_squared_error(P_fake_encoded, 0.0, is_mean=True))
		loss_lr_D = (tl.cost.mean_squared_error(P_real, 0.9, is_mean=True) +
					 tl.cost.mean_squared_error(P_fake, 0.0, is_mean=True))
		loss_vae_G = tl.cost.mean_squared_error(P_fake_encoded, 0.9, is_mean=True)
		loss_lr_G = tl.cost.mean_squared_error(P_fake, 0.9, is_mean=True)
		self.loss_G = loss_vae_G + loss_lr_G
		self.loss_D = loss_vae_D + loss_lr_D
		self.loss_vae_L1 = tl.cost.absolute_difference_error(
			image_B, vae_img, is_mean=True, axis=[1, 2, 3])
		self.loss_latent_L1 = tl.cost.absolute_difference_error(z, reconst_z, is_mean=True)
		self.loss_kl_E = 0.5 * tf.reduce_mean(
			-1 - 2 * encoded_log_sigma + encoded_mu**2 +
			tf.exp(2 * tf.clip_by_value(encoded_log_sigma, -10, 10)))
		loss = self.loss_D - reconst_C * self.loss_vae_L1 - latent_C * self.loss_latent_L1 - kl_C * self.loss_kl_E
		return loss

if __name__ == '__main__':
	x = BicycleGAN()
	print(x.G)
	print(x.D)
	print(x.E)
	y = BicycleGAN()
	print(y.G)
	print(y.D)
	print(y.E)

