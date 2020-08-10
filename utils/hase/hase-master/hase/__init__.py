
import sys
import os
import hdgwas.data
from  hdgwas.tools import Timer, Checker, study_indexes, Mapper,HaseAnalyser, merge_genotype, Reference, timing, check_np
from hdgwas.converter import  GenotypePLINK, GenotypeMINIMAC, GenotypeVCF
from hdgwas.data import Reader, MetaParData, MetaPhenotype
from hdgwas.fake import Encoder
from hdgwas.hdregression import A_covariates, A_tests, B_covariates, C_matrix, A_inverse,B4
from hdgwas.pard import partial_derivatives
from hdgwas.regression import haseregression
import time
from hdgwas.protocol import Protocol
from config import MAPPER_CHUNK_SIZE, basedir,CONVERTER_SPLIT_SIZE, PYTHON_PATH

class HASE:
	def __init__(self):
		self.cluster = 'n'
		self.effect_intercept = False
		self.hdf5 = True
		self.id = False
		self.interaction = None
		self.intercept = 'y'
		self.maf = 0.0
		self.mapper_chunk = None
		self.node = None
		self.np = True
		self.pd_full = False
		self.permute_ph = False
		self.protocol = None
		self.ref_name = '1000Gp1v3_ref'
		self.thr = None
		self.vcf = False

		self.ph_id_exc = None
		self.ph_id_inc = None
		self.snp_id_inc = None
		self.snp_id_exc = None

		self.mapper = None
		self.mode = None

		self.derivatives = []
		self.encoded = []
		self.genotype = []
		self.phenotype = []
		self.covariates = []
		self.study_name = []

		#os.environ['HASEDIR'] = ''
		#os.environ['HASEOUT'] = ''

		self.flags = {}

	def add_study(self, study):

		if study.__class__.__name__=='Study':

			if study.study_name not in self.study_name:
				self.study_name.append(study.study_name)
				if study.derivatives is not None:
					self.derivatives.append(study.derivatives)
					if study.encoded is not None:
						self.encoded.append(study.encoded)
					else:
						raise ValueError('There is no info about study encode status!')

				elif study.covariates is not None:
					self.covariates.append(study.covariates)
				else:
					raise ValueError('Study should have covariates or derivatives!')
				if study.genotype is not None:
					self.genotype.append(study.genotype)
				if study.phenotype is not None:
					self.phenotype.append(study.phenotype)


			else:
				raise ValueError('You already added study with name {}'.format(study.study_name))

		else:
			raise ValueError('You can add only Study object!')


class Study:
	def __init__(self, name):
		self.name = name
		self.genotype = None
		self.phenotype = None
		self.derivatives = None
		self.covariates = None
		self.study_name = None
		self.encoded = None

	def add_genotype(self, genotype_path, hdf5=True):
		self.genotype = Reader('genotype')
		self.genotype.start(genotype_path, hdf5=hdf5, study_name=self.study_name, ID=False)

	def add_phenotype(self, phenotype_path):
		self.phenotype = Reader('phenotype')
		self.phenotype.start(phenotype_path)

	def add_derivatives(self, derivatives_path):
		self.derivatives = Reader('partial')
		self.derivatives.start(derivatives_path, study_name=self.study_name)
		self.derivatives.folder.load()

	def add_covariates(self, covariates_path):
		self.covariates = Reader('covariates')
		self.covariates.start(covariates_path)