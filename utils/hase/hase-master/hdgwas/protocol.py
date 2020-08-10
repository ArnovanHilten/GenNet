

import os
import sys
import string




class Protocol: #TODO (mid) write class

	def __init__(self, path):
		self.name=None
		self.model=None
		self.covariates=None
		self.MAF=None
		self.QC=None
		self.genotype_format=None
		self.phenotype_format=None
		self.type=None #meta,single etc
		self.enable=False
		self.path=path
		if os.path.isfile(self.path):
			self.enable=True
			self.parse()

	def parse(self):
		print('Not implemented!')

	def regression_model(self):
		pass