import multiprocessing

import numpy as np
import pandas as pd
import tensorflow as tf
from imageio import imwrite

from loader import load_images
from params import *

dataset_dir = "../data/raw"

aux_dict = {
	'low_mu'  : ('cond_low_mu_data17_trn_data.csv', 'low_mu_data17_trn_data.csv'),
	'high_mu' : ('cond_high_mu_data17_trn_data.csv', 'high_mu_data17_trn_data.csv'),

}

class DataGenerator(object):
	def __init__(self, dataset_name, mode):
		self.mode = mode
		self.df_list = [[], []]
		if isinstance(dataset_name, list):
			for sub_name in dataset_name:
				A_name, B_name = aux_dict[sub_name]
				self.df_list[0].append(pd.read_csv(os.path.join(dataset_dir, A_name)))
				self.df_list[1].append(pd.read_csv(os.path.join(dataset_dir, B_name)))
			
			# transform in an unique df
			self.A_df = pd.concat(self.df_list[0])
			self.B_df = pd.concat(self.df_list[1])

			idx = np.random.permutation(self.A_df.index)
			self.A_df = self.A_df.loc[idx]
			self.B_df = self.B_df.loc[idx]
		else:
			A_name, B_name = aux_dict[dataset_name]
			self.A_df = pd.read_csv(os.path.join(dataset_dir, A_name))
			self.B_df = pd.read_csv(os.path.join(dataset_dir, B_name))

			# shuffle
			idx = np.random.permutation(self.A_df.index)
			self.A_df = self.A_df.loc[idx]
			self.B_df = self.B_df.loc[idx]

	def __len__(self):
		return len(self.A_df)

	def generator(self):
		for (_, Asample), (_, Bsample) in zip(self.A_df.iterrows(), self.B_df.iterrows()):
			yield Asample.values[None, :].T, Bsample.values[None, :].T

	def _map_fn(self, image_A, image_B):
		patch_pnum = image_size // patch_size
		patch_num = patch_pnum * patch_pnum
		sample_origins = list(zip(np.random.choice(range(image_size - patch_size + 1), patch_num),
								  np.random.choice(range(image_size - patch_size + 1), patch_num)))
		aux_A = [tf.slice(image_A, [x, y, 0], [patch_size, patch_size, 3]) for x, y in sample_origins]
		aux_B = [tf.slice(image_B, [x, y, 0], [patch_size, patch_size, 3]) for x, y in sample_origins]
		aux_A = [tf.concat(aux_A[r*patch_pnum:(r+1)*patch_pnum], 1) for r in range(patch_pnum)]
		aux_A = tf.concat(aux_A, 0)
		aux_B = [tf.concat(aux_B[r*patch_pnum:(r+1)*patch_pnum], 1) for r in range(patch_pnum)]
		aux_B = tf.concat(aux_B, 0)
		return image_A, image_B, aux_A, aux_B

	def __call__(self, batch_size, shuffle = True, repeat = True, use_aux = False):
		data = tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32))
		if use_aux:
			data = data.map(self._map_fn, num_parallel_calls=multiprocessing.cpu_count())
		if shuffle == "train":
			data = data.shuffle(100 * batch_size)
		data = data.prefetch(10 * batch_size)
		data = data.batch(batch_size)
		if repeat:
			data = data.repeat()
		return data

if __name__ == '__main__':
	data = DataGenerator(['low_mu','high_mu'], 'train')
	print("*")
	for i, ( a, b) in zip(range(20), data(1, use_aux=False)):
		print(i)
		print(np.shape(a))
		#imwrite("samples/aux_AB{}.png".format(i), np.concatenate([c[0], d[0]], 1))
