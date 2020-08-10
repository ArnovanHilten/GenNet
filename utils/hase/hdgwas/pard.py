

from hdgwas.hdregression import HASE, A_covariates, A_tests, B_covariates, C_matrix, A_inverse,B4
from hdgwas.tools import study_indexes, Timer
import numpy as np
import os
import time

def merge_PD(path, max_node, study_name):
    print ('Merging PD...')
    while True:
        time.sleep(10)
        if np.sum( [ os.path.isfile(os.path.join(path,'node_{}_{}_metadata.npy'.format(i,study_name)   )) for i in range(1,max_node+1)  ] )==max_node:
            if np.sum( [ os.path.isfile(os.path.join(path,'node_{}_{}_a_test.npy'.format(i,study_name)   )) for i in range(1,max_node+1)  ] )==max_node:
                break
    print('All files found...')
    b4_flag=os.path.isfile(os.path.join(path,'node_{}_{}_b4.npy'.format(1,study_name) ) )
    for i in range(1,max_node+1):
        print('node_{}'.format(i))
        if i==1:
            metadata=np.load(os.path.join(path,'node_{}_{}_metadata.npy'.format(i,study_name)) ).item()
            a_test=np.load( os.path.join(path,'node_{}_{}_a_test.npy'.format(i,study_name)  ))
            if b4_flag:
                b4=np.load( os.path.join(path,'node_{}_{}_b4.npy'.format(i,study_name)  ))


        else:
            metadata_tmp=np.load((os.path.join(path,'node_{}_{}_metadata.npy'.format(i,study_name)) )).item()
            a_test_tmp=np.load( (os.path.join(path,'node_{}_{}_a_test.npy'.format(i,study_name)  )))
            metadata['MAF']=metadata['MAF'] + metadata_tmp['MAF']
            a_test=np.vstack((  a_test,   a_test_tmp     ) )
            if b4_flag:
                b4_tmp=np.load((os.path.join(path, 'node_{}_{}_b4.npy'.format(i, study_name))))
                b4 = np.vstack((b4,   b4_tmp  ))


    np.save(os.path.join(path, study_name + '_a_test.npy'), a_test)
    np.save(os.path.join(path, study_name + '_metadata.npy'), metadata)
    if b4_flag:
        np.save(os.path.join(path, study_name + '_b4.npy'),b4)

    for i in range(1,max_node+1):
        os.remove( os.path.join(path,'node_{}_{}_metadata.npy'.format(i,study_name))  )
        os.remove(os.path.join(path,'node_{}_{}_a_test.npy'.format(i,study_name)))
        if b4_flag:
            os.remove(os.path.join(path, 'node_{}_{}_b4.npy'.format(i, study_name)))



def partial_derivatives(save_path=None,COV=None,PHEN=None, GEN=None,
                        MAP=None, MAF=None, R2=None, B4_flag=False, study_name=None,intercept=True):

    row_index, ids =  study_indexes(phenotype=PHEN.folder._data,genotype=GEN.folder._data,covariates=COV.folder._data)

    metadata={}



    #TODO (mid) add parameter to compute PD only for new phenotypes or cov
    metadata['id']=ids
    metadata['MAF']=[]
    metadata['filter']=[]
    metadata['names']=[] #TODO (low) change to cov_names
    metadata['phenotype']=[]
    b_cov=[]
    C=[]
    a_test=[]
    b4=[]

    covariates=COV.get_next(index=row_index[2])

    if MAP.cluster == 'n' or MAP.node[1] == 1:
        if intercept:
            metadata['names'].append(study_name+ '_intercept')
        metadata['names']=metadata['names']+[ study_name+ '_' + i for i in COV.folder._data.get_names() ]

        a_cov=A_covariates(covariates,intercept=intercept)
        np.save(os.path.join(save_path,study_name+'_a_cov.npy'),a_cov)


        with Timer() as t_phen:

            while True:

                phenotype=PHEN.get_next(index=row_index[1])
                if isinstance(phenotype, type(None)):
                    b_cov=np.concatenate(b_cov, axis=1)
                    C=np.concatenate(C, axis=0)
                    np.save(os.path.join(save_path,study_name+'_b_cov.npy'),b_cov)
                    np.save(os.path.join(save_path,study_name+'_C.npy'),C)
                    break

                metadata['phenotype']=metadata['phenotype']+ list(PHEN.folder._data.get_names())
                b_cov.append(B_covariates(covariates,phenotype,intercept=intercept))
                C.append(C_matrix(phenotype))

        print(('Time to PD phenotype {} is {} s'.format(np.array(C).shape, t_phen.secs)))

    if MAP.cluster == 'y':
        f_max=np.max([ int(f.split('_')[0]) for f in GEN.folder.files  ])
        files2read=[ '{}_{}.h5'.format(i,study_name) for i in np.array_split(list(range(f_max+1)),MAP.node[0])[MAP.node[1] -1  ]  ][::-1]
        filesdone=[]
        for i in range(MAP.node[1] -1):
            filesdone=filesdone + [ '{}_{}.h5'.format(i,study_name) for i in np.array_split(list(range(f_max+1)),MAP.node[0])[ i  ]  ]

        N_snps_read=0
        for f in filesdone:
            file = os.path.join(GEN.folder.path, 'genotype', f)
            N_snps_read+=GEN.folder.get_info(file)['shape'][0]
    else:
        N_snps_read=0
    while True:
        with Timer() as t_gen:
            if MAP.cluster == 'y':
                if len(files2read)!=0:
                    file = os.path.join(GEN.folder.path, 'genotype',files2read.pop() )
                    genotype=GEN.folder.read(file)
                else:
                    genotype=None
            else:
                genotype=GEN.get_next()
            if isinstance(genotype, type(None)):
                if MAP.cluster == 'y':

                    np.save(os.path.join(save_path,'node_{}_'.format(MAP.node[1]) + study_name + '_a_test.npy'), np.concatenate(a_test).astype(np.float64))
                    np.save(os.path.join(save_path,'node_{}_'.format(MAP.node[1]) +  study_name + '_metadata.npy'), metadata)
                    if B4_flag:
                        b4 = np.concatenate(b4, axis=0)
                        np.save(os.path.join(save_path, 'node_{}_'.format(MAP.node[1]) + study_name + '_b4.npy'), b4.astype(np.float64))
                    if MAP.node[1]==MAP.node[0]:
                        merge_PD(save_path, MAP.node[0],study_name)

                else:
                    np.save(os.path.join(save_path,study_name+'_a_test.npy'), np.concatenate(a_test) )
                    np.save(os.path.join(save_path,study_name+'_metadata.npy'),metadata)
                    if B4_flag:
                        b4=np.concatenate(b4, axis=0)
                        np.save(os.path.join(save_path, study_name + '_b4.npy'), b4)
                break
            flip = MAP.flip[GEN.folder.name][N_snps_read:N_snps_read + genotype.shape[0]]
            N_snps_read += genotype.shape[0]
            flip_index=(flip==-1)
            genotype=np.apply_along_axis(lambda x: flip*(x-2*flip_index) ,0,genotype)
            genotype=genotype[:,row_index[0]]
            maf=np.mean(genotype, axis=1)/2
            metadata['MAF']=metadata['MAF']+list(maf)

            #TODO (low) add interaction
            a_test.append(A_tests(covariates,genotype,intercept=intercept))

            if B4_flag:
                #works only when all phenotypes in one chunk, if not, do not use this option!
                #it would use to much disk space anyway
                if len([f for f in PHEN.folder.files if f!='info_dic.npy' ])>1:
                    print('pd_full flag disabled!')
                    B4_flag=False
                    continue
                PHEN.folder.processed=0
                phenotype=PHEN.get_next(index=row_index[1])
                b4.append(B4(phenotype,genotype))

        print(('Time to PD genotype {} is {} s'.format(genotype.shape, t_gen.secs)))







