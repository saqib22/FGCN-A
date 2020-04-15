
class Parameters():
	def __init__(self):
		self.neighborNumber = 40
		self.outputClassN = 13
		self.pointNumber = 4096
		self.gcn_1_filter_n = 1000
		self.gcn_2_filter_n = 1000
		self.gcn_3_filter_n = 1000
		self.gcn_4_filter_n = 1000
		self.fc_1_in = 1000
		self.fc_1_out = 600
		self.chebyshev_1_Order = 4
		self.chebyshev_2_Order = 3
		self.keep_prob_1 = 0.9 #0.9 original
		self.keep_prob_2 = 0.55
		self.batchSize = 16
		self.testBatchSize = 16
		self.max_epoch = 50
		self.learningRate = 1e-3
		self.dataset = 'ModelNet40'
		self.weighting_scheme = 'uniform'
		self.modelDir = '/home/saqibalikhan/Graph-GCN/global_pooling_model/model/'
		self.logDir = '/work/ge75zan/Graph-CNN/global_pooling_model/log/'
		self.fileName = '0112_1024_40_cheby_4_3_modelnet40_max_var_first_second_layer'
		self.weight_scaler = 40#50
