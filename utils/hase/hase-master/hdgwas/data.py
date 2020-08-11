import h5py
import numpy as np
import pandas as pd
import os
import sys
from .tools import study_indexes
import gc
from hdgwas.hdregression import A_inverse
from numpy import genfromtxt
from subprocess import Popen,PIPE
import bitarray as ba
import subprocess
from hdgwas.tools import Mapper, timing,Timer
import glob
import shutil
from collections import OrderedDict

class MINIMACPool(object):

	def __init__(self):
		self.paths={}
		self.id=None
		self.max_index=None
		self.chunk_size=None
		self.finish=0
		self.split_size=None


	def get_chunk(self, indices=None):

		if self.max_index is None and len(self.path)==0:
			raise ValueError('There is no data in pool.')
		else:
			sample=h5py.File(self.paths[self.id[0]], 'r')[self.id[0]][...]
			self.max_index=np.max(sample.shape)

		print(self.finish)
		result=[]
		if indices is None:
			if self.finish!=self.max_index:
				n=list(range(self.finish, self.finish + self.chunk_size)) if (self.finish + self.chunk_size)<self.max_index else list(range(self.finish,self.max_index))
				n=np.array(n)
				n_s=np.argsort(n)
				n_s_s=np.argsort(n_s)
				self.finish=np.max(n) + 1

			else:
				return None
		else:
			n_s=np.argsort(indices)
			n_s_s=np.argsort(n_s)

		for k in self.id:
			d=h5py.File(self.paths[k], 'r')[k][ list(n[n_s]) ]
			result.append(d[n_s_s])
		result=np.array(result)
		result=result.T
		return result

	def remove(self, type=None):
		if type is None:
			raise ValueError('Need to set type to pool remove function')
		if self.finish!=self.max_index:
			print ('WARNING! Removing files before fully read them!')
		print ('Start to remove MINIMAChdf5 id files...')
		for i in self.id:
			os.remove(self.paths[i])
		print ('Finished')

	def move(self,dst,type=None):
		if type is None:
			raise ValueError('Need to set type to pool move type')
		if self.finish!=self.max_index:
			print ('WARNING! Moving files before fully read them!')
		print ('Start to move MINIMAChdf5 id files to id_genotype folder...')
		for i in self.id:
			shutil.move(self.paths[i], os.path.join(dst,i+'.h5') )
		print ('Finished')


class PhenPool(object):

	def __init__(self, folder):
		self.paths={}
		self.loaded={}
		self.readed={}
		self.inmem=0
		self.limit=2
		self.split_size=None
		self.folder=folder
		self.keys=self.folder.files
		self.len_dict = {k: len(self.folder.data_info[k]) for k in self.keys}

	#@timing
	def link(self,n):
		r = np.zeros((len(n), 2), dtype=np.int64)
		n = np.array(n)
		ind = 0
		for j, k in enumerate(self.keys):
			n = n - self.len_dict[k]
			index = np.where(n < 0)[0]
			if (len(index)) != 0:
				r[index, 0] = j
				r[index, 1] = n[index] + self.len_dict[k]
				n[index] = 10 ** 9 # replacing by huge number, should be bigger than actual number of phenotypes
				ind += len(index)
				if ind == len(n):
					break
		return r

	#@timing
	def get_chunk(self,indices):
		indices=self.link(indices)
		indices=np.array(indices)
		keys, ind = np.unique(indices[:,0], return_inverse=True)
		result=None
		#gc.collect()
		for i,k in enumerate(keys):
			r=None
			if k in self.loaded:
				n=indices[(ind==i),1].flatten()
				r=self.loaded[self.keys[k]][n,:]
			else:
				self.folder.read(self.keys[k])
				r=self.folder._data._data
				n=indices[(ind==i),1].flatten()
				r=r[:,n]
			l=r.shape[0]
			if result is None:
				result=np.empty((l,indices.shape[0]))
			result[:,(ind==i)]=r
		return result


class Pool(object):

	def __init__(self):
		self.paths={}
		self.loaded={}
		self.keys=[]
		self.readed={}
		self.inmem=0
		self.limit=2
		self.split_size=None
	#@timing
	def link(self,n):
		chunks = n / self.split_size
		ind=n - self.split_size * chunks
		r=np.zeros((len(n),2),dtype=np.int)
		r[:,0]=chunks
		r[:,1]=ind
		return r
	#@timing
	def get_data(self,key,index):
		if key in self.loaded:
			r=self.loaded[key][index,:]
			self.readed[key]-=1
			if self.readed[key]==0:
				self.inmem-=1
				self.loaded[key]=None
				del self.loaded[key]
				gc.collect()

		else:
			if self.inmem<self.limit:
				self.loaded[key]=h5py.File(self.paths[key], 'r')['genotype'][...]
				self.inmem+=1
				self.readed[key]=self.loaded[key].shape[0]
				r=self.loaded[key][index,:]
				self.readed[key]-=1

			else:
				f=h5py.File(self.paths[key], 'r')
				r=f['genotype'][index,:]
				f.close()
		return r
	#@timing
	def get_chunk(self,indices, impute):
		indices=self.link(indices)
		indices=np.array(indices)
		keys, ind = np.unique(indices[:,0], return_inverse=True)
		result=None
		for i,k in enumerate(keys):
			r=None
			if k in self.loaded:
				n=indices[(ind==i),1].flatten()
				r=self.loaded[k][n,:]
			else:
				f=h5py.File(self.paths[k], 'r')
				n=indices[(ind==i),1].flatten()
				r=f['genotype'][...]
				r=r[n,:]
				f.close()

			l=r.shape[1]
			if result is None:
				print(indices.shape[0], l)
				result=np.empty((indices.shape[0],l))

			result[(ind==i),:]=r
		if impute: # default True
			if np.sum(result==9)!=0:
				print ('Imputing missing genotype to mean...')
				m=np.where(result == 9)
				result[m] = np.nan
				result[m] = np.take(np.nanmean(result, axis=1), m[0])
				#result[m]=np.nanmean(result[m[0], :], axis=1)
				print(result)

			if np.sum( np.isnan(result) )!=0:
				print ('Imputing missing genotype to mean...')
				m=np.where(np.isnan(result))
				result[m] = np.take(np.nanmean(result, axis=1), m[0])
				#print result
				#result[np.where(np.isnan(result))]=np.nanmean(result[np.where(np.isnan(result))[0], :], axis=1)

		return result

class Data(object):

	def __init__(self):
		self.id=None
		self.shape=None
		self.names=None
		self.type=None
		self.format=None
		self._data=None
		self.chunk_size=None
		self.processed=0
		self.start=None
		self.finish=None
		self.filename=None


	def test(self):
		if isinstance(self.id, type(None)) or isinstance(self.names, type(None)) or isinstance(self._data, type(None)):
			raise ValueError('Read data first!')
		else:
			N,M=self._data.shape
			uniq_id=np.unique(self.id)
			if len(uniq_id)!=N:
				print(uniq_id)
				raise ValueError('Should be uniq ID, but there are {} only uniq from {}'.format(len(uniq_id),N))
			uniq_names=np.unique(self.names)
			if len(uniq_names)!=M:
				print(uniq_names)
				raise ValueError('Should be uniq NAMES, but there are {} only uniq from {}'.format(len(uniq_id),M))


	def get_id(self):
		if not isinstance(self.id,type(None)):
			return np.copy(np.array(self.id))
		else:
			raise ValueError('No id defined!')

	def get_names(self):
		if not isinstance(self.names,type(None)):
			return (np.array(self.names))
		else:
			raise ValueError('No names defined!')


	def get_next(self, index=None, select_columns=None):

		if isinstance(self._data,type(None)):
			return None

		if self.processed==self.shape[1]:
			return None
		else:
			if select_columns is not None:
				if not isinstance(index,type(None)):
					return self._data[index,select_columns]
				else:
					return self._data[:,select_columns]
			else:
				start=self.processed
				finish=self.processed+self.chunk_size if (self.processed+self.chunk_size)<=self.shape[1] else self.shape[1]
				self.processed=finish
				if not isinstance(index,type(None)):
					self.start=start
					self.finish=finish
					return self._data[index,start:finish]
				else:
					self.start=start
					self.finish=finish
					return self._data[:,start:finish]


class Hdf5Data(Data):
	def __init__(self, data_path , name, type='PLINK'):
		super(Hdf5Data, self).__init__()
		self.name=name
		self.id=np.array(pd.read_hdf(os.path.join(data_path,'individuals',self.name+'.h5'),'individuals').individual.tolist())
		if "MAPPER" in os.environ:
			print ('Reading ID from probes....')
			self.names=pd.read_hdf(os.path.join(data_path,'probes', self.name +'.h5'),'probes', where='columns=[ID]').ID
			self.shape=(len(self.id), len(self.names))
		else:
			print ('Use ID from mapper')
			self.names=pd.HDFStore(os.path.join(data_path,'probes', self.name +'.h5'),'r')
			self.shape=(len(self.id), self.names.get_storer('probes').nrows)
		if type=='PLINK':
			l=os.listdir(os.path.join(data_path,'genotype'))
			a,b=h5py.File(os.path.join(data_path,'genotype',l[0]), 'r')['genotype'][...].shape
			try:
				c,d=h5py.File(os.path.join(data_path,'genotype',l[1]), 'r')['genotype'][...].shape
				self.gen_split_size=np.max((a,c)) #TODO (middle) not efficient
			except:
				self.gen_split_size =a

		print(('There are %d ids' %(self.shape[0])))


class ParData(Data):
	def __init__(self):
		super(ParData, self).__init__()
		self.a_inv=None
		self.a_cov=None
		self.b_cov=None
		self.C=None
		self.a_test=None
		self.metadata=None
		self.b4=None

	def check(self): #TODO (middle) place to implement the check of analysis protocol and log summary from PD
		print(('Number of subjects {}'.format(int(self.a_cov[0,0]))))

	def get(self,gen_order=None, phen_order=None,cov_order=None):

		if gen_order is None or phen_order is None or cov_order is None:
			raise ValueError('PD order is not define!')
		if isinstance(self.a_inv, type(None)):
			return self.a_test[np.ix_(gen_order,np.append(cov_order,self.a_test.shape[1]-1))], self.b_cov[np.ix_(cov_order,phen_order)], \
				   self.C[phen_order], self.a_cov[np.ix_(cov_order,cov_order)]
		else:
			return self.a_inv[np.ix_(gen_order,np.append(cov_order,self.a_test.shape[1]-1))], self.b_cov[np.ix_(cov_order,phen_order)], \
				   self.C[phen_order], self.a_cov[np.ix_(cov_order,cov_order)]

class MetaPhenotype(object):

	def __init__(self, phen, protocol=None, include=None, exclude=None):
		#@timing
		def _check(map,keys,index=None):
			values=np.array(list(map.dic.values()) )
			result={}
			r=(values==-1).any(axis=1)
			if index is not None:
				r[np.where(index==False)[0] ]=True
			if np.sum(r)==values.shape[0]:
				raise ValueError('There is no common names between studies')
			for i,k in enumerate(keys):
				result[k]=values[~r,i]
			return result, np.array(list(map.dic.keys()))[~r]
		self.chunk_size=10000
		self.exclude=None
		self.include=None
		self.name=None
		self.mapper=Mapper()
		self.keys=[]
		self.keep_index=None
		if include is not None:
			self.include = pd.DataFrame.from_csv(include, index_col=None)
			print('Include:')
			print(self.include.head())
			if 'ID' not in self.include.columns:
				raise ValueError('{} table does not have ID column for phenotypes'.format(include))
		if exclude is not None:
			self.exclude = pd.DataFrame.from_csv(exclude, index_col=None)
			print('Exclude:')
			print(self.exclude.head())
			if 'ID' not in self.exclude.columns:
				raise ValueError('{} table does not have ID column for phenotypes'.format(exclude))
		if protocol is None:
			for i,k in enumerate(phen):
				phen_names=[]
				if i==0:
					for j in k.folder.files:
						if j!='info_dic.npy':
							phen_names=phen_names+list(k.folder.data_info[j])
					self.mapper.fill(phen_names,i, reference=False)
				else:
					for j in k.folder.files:
						if j!='info_dic.npy':
							phen_names=phen_names+list(k.folder.data_info[j])
					self.mapper.push(phen_names,name=i, new_id=False)

				self.keys.append(i)
			if self.exclude is not None or self.include is not None:
				phen_names=pd.Series(list(self.mapper.dic.keys()))
				phen_names=phen_names[phen_names.isin(self.include.ID)] if self.include is not None else phen_names
				phen_names = phen_names[~phen_names.isin(self.exclude.ID)] if self.exclude is not None else phen_names
				self.keep_index=pd.Series(list(self.mapper.dic.keys())).isin(phen_names)
			print(np.sum(self.keep_index))
			self.order, self.phen_names =_check(self.mapper,self.keys,index=self.keep_index)
			self.n_phenotypes=len(self.order[self.keys[0]])
			print(('Loaded {} common phenotypes for meta-analysis'.format( self.n_phenotypes  )))
			self.processed=0
		else:
			if not protocol.enable:
				protocol.parse()

		self.pool={i:PhenPool(j.folder) for i,j in enumerate(phen) }


	def get(self):
		if self.processed==self.n_phenotypes:
			return None,None
		else:
			start=self.processed
			finish=self.processed+self.chunk_size if (self.processed+self.chunk_size)<=self.n_phenotypes else self.n_phenotypes
			self.processed=finish
			phenotype = np.zeros( (   np.sum([ len(self.pool[i].folder._data.id) for i in self.pool ] ) , len(list(range(start,finish)))      )  )
		for i,j in enumerate(self.keys):

			if i==0:
				a,b=0,self.pool[j].folder._data.id.shape[0]
				phenotype[a:b,:]=self.pool[j].get_chunk(self.order[j][start:finish])
				a=b
				N = np.random.randint(0, phenotype.shape[1], 10)
				print(phenotype.mean(axis=0)[N])
			else:
				ph_tmp=self.pool[j].get_chunk(self.order[j][start:finish])
				print(self.pool[j].folder.path)
				print(ph_tmp.mean(axis=0)[N])
				b+=ph_tmp.shape[0]
				phenotype[a:b,:]=ph_tmp
				a=b


		return phenotype, self.phen_names[start:finish]

class MetaParData(object):

	def __init__(self,pd,study_names,protocol=None):
		#@timing
		def _check(map,keys):
			values=np.array(list(map.dic.values()) )
			result={}
			r=(values==-1).any(axis=1)
			if np.sum(r)==values.shape[0]:
				raise ValueError('There is no common names between studies')
			for i,k in enumerate(keys):
				result[k]=values[~r,i]
			return result

		self.name=None
		self.study_names=study_names
		self.phen_mapper=Mapper()
		self.cov_mapper=Mapper()
		self.covariates = OrderedDict()
		self.pd =OrderedDict()
		for i in pd:
			self.pd[i.folder.name]=i
		keys=[]
		if protocol is None:
			for i,k in enumerate(pd):
				self.covariates[k]=[n.split(self.study_names[i]+'_')[1] for n in k.folder._data.metadata['names']]
				if i==0:
					self.phen_mapper.fill(k.folder._data.metadata['phenotype'],k.folder.name, reference=False)
					self.cov_mapper.fill(self.covariates[k],k.folder.name, reference=False)
				else:
					self.phen_mapper.push(k.folder._data.metadata['phenotype'],name=k.folder.name,new_id=False)
					self.cov_mapper.push(self.covariates[k],name=k.folder.name, new_id=False)
				keys.append(k.folder.name)

			self.phen_order =_check(self.phen_mapper,keys)
			self.cov_order =_check(self.cov_mapper, keys)

		if protocol is not None:
			if not protocol.enable:
				protocol.parse()

	def check_pd(self, old, new):

		np.set_printoptions(precision=3, suppress=True)
		print ("*******PD CHECK*********")
		print ('A covariates...')
		print(old[3] / old[3][0, 0])
		print(new[3] / new[3][0, 0])
		print ('FREQ TESTs')
		N=np.random.randint(0, np.min( (old[0].shape[0],new[0].shape[0]) )  ,10)
		print(np.array(old[0][:, 0] / old[3][0, 0]/ 2)[N])
		print(np.array(new[0][:, 0] / new[3][0, 0] / 2)[N])
		print ('FREQ PHENO')
		M=np.random.randint(0, np.min( (old[1].shape[1],new[1].shape[1]) )  ,10)
		print(np.array(old[1][0, :] / old[3][0, 0])[M])
		print(np.array(new[1][0, :] / new[3][0, 0])[M])
		print ("****************")

	#@timing
	def check_maf(self, SNPs_index):
		maf=np.zeros( ( SNPs_index[0].shape[0], len(self.pd)) )

		for i,j in enumerate(self.pd):
			maf[:,i]=np.array(self.pd[j].folder._data.metadata['MAF'])[ SNPs_index[i].astype(np.int64) ]

		if (np.std(maf,axis=1)>0.1).any():
			raise ValueError('MAF is not consistent between PD data!')


	def get(self, SNPs_index=None, B4=False, regression_model=None,random_effect_intercept=False ):

		if self.pd is None:
			raise ValueError('Data not defined!')
		k=list(self.pd.keys())
		if SNPs_index is not None:
			if len(self.pd)!=len(SNPs_index):
				raise ValueError('There are not equal number od PD and SNPs indexes {}!={}'.format(len(self.pd), len(SNPs_index)))
			a_test, b_cov, C, a_cov = self.pd[k[0]].get(gen_order=SNPs_index[0], phen_order=self.phen_order[k[0]], cov_order=self.cov_order[k[0]] )
			if random_effect_intercept and len(self.pd)>1: #TODO (high)
				a_test_effect=a_test[:,0:1]
				b_cov_effect=b_cov[0:1,:]
				a_cov_effect=a_cov[0:1,:]
		else:
			raise ValueError('Indexes are not defined!')

		if B4:
			b4=self.pd[k[0]].folder._data.b4[SNPs_index[0],:]
			b4=b4[:,self.phen_order[k[0]]]

		#self.check_maf(SNPs_index)

		for i in range(1, len(self.pd)):
			a,b,c,a_c=self.pd[k[i]].get(gen_order=SNPs_index[i], phen_order=self.phen_order[k[i]], cov_order=self.cov_order[k[i]])
			self.check_pd([a_test, b_cov, C, a_cov], [a,b,c,a_c])
			a_test=a_test + a
			b_cov=b_cov + b
			C=C+c
			a_cov=a_cov + a_c
			if random_effect_intercept and i<(len(self.pd)-1) :
				a_test_effect=np.hstack((a_test_effect,a[:,0:1]))
				b_cov_effect=np.vstack((b_cov_effect,b[0:1,:]))
				a_cov_effect=np.vstack((a_cov_effect,a_c[0:1,:]))
			if B4:
				b4_tmp=self.pd[k[i]].folder._data.b4[SNPs_index[i], :	]
				b4=b4+b4_tmp[:,  self.phen_order[k[i]]]

		if random_effect_intercept and len(self.pd)>1:
				a_cov_I=np.zeros((len(self.pd)-1,len(self.pd)-1))
				np.fill_diagonal(a_cov_I,a_cov_effect[:,0])
				a_effect=np.hstack((a_cov_I,a_cov_effect))
				a_test=np.hstack((a_test_effect,a_test))
				b_cov=np.vstack((b_cov_effect,b_cov))
				a_cov=np.vstack((a_cov_effect,a_cov))
				a_cov=np.hstack((a_effect.T,a_cov))

		if B4:
			return a_test, b_cov, C, a_cov, b4
		else:
			return a_test, b_cov, C, a_cov


	def maf_pard(self,SNPs_index=None):

		samples=0
		maf=np.zeros( len(SNPs_index[0]) )

		for j,i in enumerate(self.pd):
			n=len(self.pd[i].folder._data.metadata['id'])
			samples+=n
			maf=maf+n*np.array(self.pd[i].folder._data.metadata['MAF'])[SNPs_index[j]]
		maf=maf/np.float(samples)
		return maf

	def get_n_id(self):
		return np.sum([len(self.pd[i].folder._data.metadata['id']) for i in self.pd  ])

	def get_phenotype_names(self):
		n={}
		for i in self.pd:
			n[i]=self.pd[i].folder._data.metadata['phenotype']


class Folder(object):

	def __init__(self,path):
		self.path=path
		if not os.path.isdir(self.path):
			raise ValueError('{} is not a folder'.format(self.path))
		self.format=None
		self.protected=[]
		self.n_files=None
		self.read_files={}
		self.files=None
		self.processed=0
		self._data=Data()
		self._scan()
		self.folder_cache={}
		self.folder_cache_flag=True
		self.cache_buffer_size=10
		self._id=None

	def _scan(self):
		l=os.listdir(self.path)
		if len(l)==0:
			raise ValueError('There is no files in {}'.format(self.path))
		exts=[os.path.splitext(i)[1] for i in l ]
		s=set(exts)
		if not len(s) <= 1:
			if s=={'.fam','.bed','.bim'} :
				self.format='PLINK'
			else:
				raise ValueError('There are different data format {} in {} folder'.format(set(exts) ,self.path))
		else:
			N=len(l)
			self.n_files=N
			self.format=exts[0]
			self.files=l


	def get_next(self,**kwargs):
		d=self._data.get_next(**kwargs)
		if isinstance(d,type(None)):
			file=next(self)
			if isinstance(file,type(None)):
				return None
			self.read(file)
			return self._data.get_next(**kwargs)
		else:
			return d


	def read(self,*args,**kwargs):
		pass

	def __next__(self):
		while True:
			try:
				if self.processed==self.n_files:
					print('read all', self.path)
					return None
				f=self.files[self.processed]
				self.processed+=1
				if f not in self.protected:
					self.read_files[f]=0
					break
				else:
					self.read_files[f]=1
			except:
				return None
		return f

	def get(self,**kwargs):
		return self._data.get(**kwargs)


class HDF5Folder(Folder):

	def __init__(self,path, name):
		super(HDF5Folder, self).__init__(path)

	def _scan(self):
		pass

	def get_next(self,**kwargs):
		pass

	def read(self, file, index=None, MAF=None):
		pass

	def get(self, index):
		pass


class PLINKHDF5Folder(HDF5Folder):

	def __init__(self,path,name):
		super(HDF5Folder, self).__init__(path)
		self.name=name
		self.path=path
		self.format='.h5'
		self.pool=Pool()
		self.iter=-1
		if not os.path.isdir(os.path.join(self.path,'genotype')) or not os.path.isdir(os.path.join(self.path,'probes')) or not os.path.isdir(os.path.join(self.path,'individuals')):
			raise ValueError('in genotype folder should be /genotype; /probes; /individuals folders')

		self.f_names=os.listdir(os.path.join(self.path,'genotype')) #TODO (high) remove this extra attribute
		self.files = self.f_names
		self.n_files=len(self.f_names)
		self.processed=-1
		self.pool.paths={i:os.path.join(self.path,'genotype',str(i)+'_'+self.name +'.h5') for i in range(self.n_files)}

		self._data=Hdf5Data(self.path,self.name)
		self._id=self._data.id
		self.pool.split_size=self._data.gen_split_size

	def _scan(self):
		pass

	def get_next(self,**kwargs):

		self.iter+=1
		if self.iter==self.n_files:
			return None
		file=os.path.join(self.path,'genotype',str(self.iter)+'_'+self.name +'.h5')
		return self.read(file,**kwargs)

	def read(self, file, index=None, MAF=None):
		print('reading file {}'.format(file))

		f=h5py.File(file, 'r')

		d=f['genotype'][...]

		if isinstance(index,type(None)): #TODO (mid) check
			return d
		else:
			return d[:,index]

	def get(self, index, impute=True):
		self.processed+=len(index)
		result=[]
		if True:
		#if self.pool.inmem==self.pool.limit: #TODO (high) check the bellow methods, remove get_data from pool?
			result=self.pool.get_chunk(index, impute)
		else:
			result=[self.pool.get_data(i,j) for i,j in index]
		return np.array(result)

	def get_info(self,file):
		info={}
		f = h5py.File(file, 'r')
		info['shape']= f['genotype'].shape
		f.close()
		return info

class MINIMACHDF5Folder(HDF5Folder):

	def __init__(self,path, name):
		super(HDF5Folder, self).__init__(path)
		self.name=name
		self.path=path
		self.format='.h5'
		self.pool=MINIMACPool()
		self.pool.path=self.path
		self.f_names=os.listdir(os.path.join(self.path,'genotype'))
		self.n_files=len(self.f_names)
		self._data=Hdf5Data(self.path,self.name, type='MINIMAC')
		self._id=self._data.id
		self.pool.id=self._id
		self.pool.paths={i:os.path.join(self.path,'genotype',i + '.h5') for i in self._id}



	def _scan(self):
		pass

	def get_next(self,**kwargs):
		return self.pool.get_chunk()

	def get(self, index):
		return self.pool.get_chunk(indices=index)

	def read(self, file, index=None, MAF=None):
		pass

	def get(self, index):
		pass


class CSVFolder(Folder):

	def __init__(self,path):
		super(CSVFolder,self).__init__(path)
		try:
			self.read(next(self))
		except Exception as e:
			raise (e)
		self.data_info={}
		for i in self.files:
			for j in ['\t', ' ']:
					df=pd.read_csv(os.path.join(self.path,i), sep=j, index_col=None)
					if df.shape[1] > 1:
						break
			self.data_info[i]=np.array(df.columns[1:])


	def read(self,file):

		if self.folder_cache_flag and self.folder_cache.get(file) is not None and len(self.folder_cache)<self.cache_buffer_size:
			self._data=self.folder_cache[file]
			self._data.processed=0
			print(('There are samples and columns {} in cache {}'.format(self._data.shape, file)))

		else:
			print('reading file {}'.format(file))
			for i in ['\t', ' ']:
					df=pd.read_csv(os.path.join(self.path,file), sep=i, index_col=None)
					if df.shape[1]>1:
						break
			else:
				raise ValueError('Cant read {} file; default settings: index=None, header=True; sep=tab or space '.format(file))


			self._data=Data()
			self._data.chunk_size=1000
			self._data._data=df[df.columns[1:]].as_matrix()
			self._data.id=np.array(df[df.columns[0]])
			if self._id is None:
				self._id=self._data.id
			else:
				if (self._id!=self._data.id).any(): #TODO (middle) check for not overlapped id (AttributeError: 'bool' object has no attribute 'any')
					raise ValueError('id in {} different from previous!'.format(file))
			self._data.type=np.ndarray
			self._data.names=np.array(df.columns[1:])
			self._data.shape=self._data._data.shape
			self._data.format='csv'
			self._data.filename=file

			self._data.test() # check the dim of data

			if self.folder_cache_flag and len(self.folder_cache)<self.cache_buffer_size:
				self.folder_cache[file]=self._data

			print(('There are %d ids and %d columns ' %(self._data.shape)))

class NPFolder(Folder):

	def __init__(self,path, protected=[]):
		super(NPFolder, self).__init__(path)
		self._data=Data()
		self._data.chunk_size=1000
		self._data.type=np.ndarray
		self._data.format='npy'
		self.protected=protected

		try:
			file_path=os.path.dirname(path)
			self.data_info=np.load(os.path.join(file_path, 'info_dic.npy')).item()
			self._data.id=np.array(self.data_info['id'])
			self.files=[k for k in list(self.data_info.keys()) if k!='id']

		except Exception as e:
			raise ValueError('in directory {} should be data info file info_dic.npy!'.format(self.path) + str(e)  )


		try:
			self.read(next(self))
		except Exception as e:
			raise ValueError("Failed to init NPFolder;" + str(e))
	#@timing
	def read(self,file):

		if self.folder_cache_flag and self.folder_cache.get(file) is not None and len(self.folder_cache)<self.cache_buffer_size:
			print('start cache {}'.format(file))
			self._data._data=self.folder_cache[file]
			self._data.processed=0
			self._data.type=np.ndarray
			self._data.shape=self._data._data.shape
			self._data.format='npy'
			self._data.filename=file
			self._data.names=np.array(self.data_info[file])
		#print 'end cache'
		else:
			#print 'reading file {}'.format(file)
			d=np.load(os.path.join(self.path,file))
			if not isinstance(d, np.ndarray):
				raise ValueError('File {} saved not as numpy array'.format(file))
			self._data.processed=0
			self._data._data=d
			self._data.type=np.ndarray
			self._data.shape=d.shape
			self._data.format='npy'
			self._data.filename=file
			self._data.names=np.array(self.data_info[file])

			self._data.test() # check the dim of data
			if self.folder_cache_flag and len(self.folder_cache)<self.cache_buffer_size:
				self.folder_cache[file]=self._data._data

class PDFolder(Folder):

	def __init__(self,path,name):
		super(PDFolder, self).__init__(path)
		self.name=name
		files=[self.name+'_a_cov.npy',self.name+'_b_cov.npy',self.name+'_C.npy',self.name+'_metadata.npy']
		for i in files:
			if i not in self.files:
				raise ValueError('There is not {} file in directory {}'.format(i,self.path))
		self._data=ParData()
		self.loaded=False

	def scan(self):
		pass

	def get_next(self,**kwargs):
		pass

	def get(self,**kwargs):
		return self._data.get(**kwargs)

	def read(self,file):
		pass

	def load(self):

		if self.name+'_b4.npy' in self.files:
			self._data.b4=np.load(os.path.join(self.path,self.name+'_b4.npy'))


		self._data.a_cov=np.load(os.path.join(self.path,self.name+'_a_cov.npy'))
		self._data.b_cov=np.load(os.path.join(self.path,self.name+'_b_cov.npy'))
		self._data.C=np.load(os.path.join(self.path,self.name+'_C.npy'))
		self._data.metadata=np.load(os.path.join(self.path,self.name+'_metadata.npy')).item()

		if self.name+'_a_inv.npy' in self.files: #TODO (low) current version does not save inv matrix
			self._data.a_inv=np.load(os.path.join(self.path,self.name+'_a_inv.npy'))

		elif self.name+'_a_test.npy' in self.files:
			self._data.a_test=np.load(os.path.join(self.path,self.name+'_a_test.npy'))

		else:
			raise ValueError('There is not a_inv.npy or a_test.npy file in directory {}'.format(self.path))

		self.loaded=True

	def summary(self): #TODO (middle) write function
		if not self.loaded:
			raise ValueError('call summary PD before load data!')




class PLINKFolder(Folder):

	def __init__(self, path):
		super(PLINKFolder, self).__init__(path)
		self.scan()
		self.names=None
		self.bim=None
		self.bed=None
		self.fam=None
		self.N_probes=0
		self.n_probes=0
		self.n_probes_dic={}
		self.n_ind_dic={}
		self.n_ind=0
		self.N_ind=0
		self._currentSNP = 0

		self._bedcode = {
			2: ba.bitarray('11'),
			9: ba.bitarray('10'), #TODO (high) NA data handle
			1: ba.bitarray('01'),
			0: ba.bitarray('00')
		}

		self.scan()

	def scan(self):

		l=os.listdir(self.path)
		self.names=np.unique([os.path.splitext(i)[0] for i in l ])

		for i in self.names:
			if i+'.fam' not in l or  i+'.bed' not in l  or i+'.bim' not in l:
				raise ValueError('There are not all plink files for {} in {} folder'.format(i, self.path))

		self.n_files=len(l)
		self.format='PLINK'
		self.files=self.names

	def get_fam(self):
		"""Read the FAM file to get information about individuals
		Family ID
		Individual ID
		Paternal ID
		Maternal ID
		Sex (1=male; 2=female; other=unknown)
		Label - phenotype should be in separate file
		"""

		individuals=None

		for i in self.names:
			ind = genfromtxt(open(os.path.join(self.path,i+'.fam'), 'r')  , delimiter=' ',
							 dtype={'names': ['family', 'individual', 'paternal', 'maternal', 'sex', 'label'],
									'formats': ['S10', 'S16', int, int, int, int]})

			if isinstance(individuals, type(None)):
				individuals=ind

			else:
				individuals=np.append(individuals,ind)
			self.n_ind_dic[i]=len(ind)
		self.N_ind=len(individuals)
		print(('Number of Individuals: %d' % self.N_ind))

		return individuals


	def get_bim(self, chunk_size):

		if isinstance(self.bim,type(None)):
			file=next(self)
			self.read_bim(file)
		try:
			d=self.bim.get_chunk(chunk_size)
			return d
		except:
			file=next(self)
			if isinstance(file,type(None)):

				return None
			self.read_bim(file)
			d=self.bim.get_chunk(chunk_size)
			return d


	def read_bim(self,file):

		N=int(Popen(['wc','-l',os.path.join(self.path,file +'.bim')], stdout=PIPE).communicate()[0].split(' ')[0])

		print(('Number of Probes {} in {}'.format(N,file+'.bim')))
		self.N_probes+=N

		self.n_probes_dic[file]=N

		self.bim = pd.read_table(os.path.join(self.path,file +'.bim'), sep='\t', header=None, names=['CHR', 'ID', 'distance', 'bp', 'allele1', 'allele2'],
								 iterator=True)


	def get_bed(self,chunk_size):

		d=self.nextSNPs(chunk_size)

		if isinstance(d, type(None)):
			file=next(self)
			if isinstance(file,type(None)):
				return None
			else:
				self.read_bed(file)
				return self.get_bed(chunk_size)
		else:
			if d.shape[0]==chunk_size:
				return d
			else:
				d_append=self.get_bed(chunk_size-d.shape[0])
				if isinstance(d_append,type(None)):
					return d
				else:
					return np.vstack((d,d_append))

	def read_bed(self,file):

		self._currentSNP=0
		self.n_ind=self.n_ind_dic[file]
		self.n_probes=self.n_probes_dic[file]
		n=self.n_ind
		self.bed = open(os.path.join(self.path,file +'.bed'), 'rb')
		magicNumber = ba.bitarray(endian="little")
		magicNumber.fromfile(self.bed, 2)
		bedMode = ba.bitarray(endian="little")
		bedMode.fromfile(self.bed, 1)
		e = (4 - n % 4) if n % 4 != 0 else 0
		nru = self.n_ind + e
		self.nru = nru
		# check magic number
		if magicNumber != ba.bitarray('0011011011011000'):
			raise IOError("Magic number from Plink .bed file not recognized")

		if bedMode != ba.bitarray('10000000'):
			raise IOError("Plink .bed file must be in default SNP-major mode")


	def nextSNPs(self, b):
		'''
		Unpacks the binary array of genotypes and returns an n x b matrix of floats of
		normalized genotypes for the next b SNPs, where n := number of samples.

		Parameters
		----------
		b : int
			Number of SNPs to return.
		Returns
		-------
		X : np.array with dtype float64 with shape (n, b), where n := number of samples

		'''
		if self._currentSNP==self.n_probes:
			return None
		if self._currentSNP + b > self.n_probes:
			b=(self.n_probes-self._currentSNP)

		print(('next {} SNPs, from {}, need to convert {}'.format(b, self.n_probes, (self.n_probes-self._currentSNP-b))))

		c = self._currentSNP
		n = self.n_ind
		nru = self.nru
		slice= ba.bitarray(endian="little")

		bit_number=((2*(c+b)*nru)-(2*c*nru))/8
		slice.fromfile(self.bed, bit_number )

		X = np.array(slice.decode(self._bedcode), dtype="float64").reshape((b, nru)).T
		X = X[0:n, :]

		self._currentSNP += b
		gc.collect()
		return X.T



	def get_next(self,**kwargs):
		pass

	def read(self):
		pass


class MINIMACFolder(Folder):

	def __init__(self, path):
		super(MINIMACFolder, self).__init__(path)
		self.scan()

	def scan(self):
		if len(glob.glob(self.path + '*dose.gz'))==0 or len(glob.glob(self.path + '*info.gz'))==0:
			raise ValueError('There is no dose.gz or info.gz files in {}, check folder or compression flag'.format(self.path))

	def get_next(self,**kwargs):
		pass


class VCFFolder(Folder):

	def __init__(self, path):
		super(VCFFolder, self).__init__(path)
		self.scan()

	def scan(self): #TODO (middle) add checking for correct VCF format (? checkVCF.py) before start shell scripts
		pass

	def get_next(self,**kwargs):
		pass


class Reader(object):

	def __init__(self, name):
		self.name=name
		self.path=None
		self._data=None
		self.ext=['.npy', '.csv', '.txt','.h5', 'PLINK','.gz','VCF']
		self.pool=None
		self.folder=None
		self.processed=0
		self.kwargs=None
		self.permutation=False


	def start(self, path, **kwargs):

		self.kwargs=kwargs
		if path is None:
			raise ValueError('Not defined path for {}'.format(self.name))
		if os.path.isdir(path):

			self.folder=Folder(path)
			self.path=path

			self.format=self.folder.format

			if kwargs.get('vcf',0) and kwargs['vcf']:
				self.format='VCF'

			if self.name=='genotype' and self.format!='PLINK' and self.format!='.gz' and self.format!='VCF':
				self.format='.h5'

			if self.format not in self.ext:
				raise ValueError(' {} is not supported format (only {} supported)!'.format(self.format, self.ext))

			elif self.format=='.npy' and self.name!='partial':
				self.folder=NPFolder(path,protected=['info_dic.npy'] )
			#self.folder.protected=['info_dic.npy']

			elif self.format=='.npy' and self.name=='partial':
				self.folder=PDFolder(path,name=kwargs['study_name'])


			elif self.format in ['.csv', '.txt']:
				self.folder=CSVFolder(path)

			elif self.format=='.h5' and self.name=='genotype':
					self.folder=PLINKHDF5Folder(path,kwargs['study_name'])

			elif self.format=='PLINK':
				self.folder=PLINKFolder(path)

			elif self.format=='.gz':
				self.folder=MINIMACFolder(path)
				self.format="MINIMAC"

			elif self.format=='VCF':
				self.folder = VCFFolder(path)

			elif self.format=='.h5' and self.name!='genotype':
				raise ValueError('hdf5 format implemented only for converted from genotype data...Sorry')

			elif self.format!='.h5' and self.name=='genotype':
				raise ValueError('genotype data should be in hdf5 format')

			self._get=self.folder.get_next

		else:
			raise ValueError('{} is not a directory'.format(path))


	def get(self,*args,**kwargs):
		if self.format!='.h5' and self.name!='partial':
			raise ValueError()
		return self.folder.get(*args,**kwargs)


	def get_next(self,**kwargs):
		return self._get(**kwargs)
