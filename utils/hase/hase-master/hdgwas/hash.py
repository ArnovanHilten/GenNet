import numpy as np
import pickle as pickle
import os



class HashTablePOS(object):
    def __init__(self):
        self.hash_dic={i:{} for i in range(23)}
        self.protected=False
        self.data_frame=None
        self.collisions={i:[] for i in range(23)}
        self.type='POS'
        self.n_keys=0
        self._print=False

    def fill_id(self,id):
        if self.hash_dic[id.CHR].get(id.bp) is not None:
            if isinstance(self.hash_dic[id.CHR][id.bp], tuple):
                if self._print:
                    print((
                    "Collision! You are trying to fill {} again with A1_{}/A2_{},"
                    " it is already inserted with A1_{}/A2_{}. "
                    "".format(
                        "CHR "+str(id.CHR) + " bp "+ str(id.bp),
                        id.allele1,id.allele2,
                        self.hash_dic[id.CHR][id.bp][1],self.hash_dic[id.CHR][id.bp][2])
                                           ))

                self.hash_dic[id.CHR][id.bp]=[self.hash_dic[id.CHR][id.bp],(self.n_keys,id.allele1, id.allele2)   ]
                self.n_keys+=1
            else:
                self.hash_dic[id.CHR][id.bp].append((self.n_keys,id.allele1, id.allele2))
                self.n_keys+=1
            self.collisions[id.CHR].append(id.bp)
        else:
            self.hash_dic[id.CHR][id.bp]=(self.n_keys, id.allele1, id.allele2)
            self.n_keys+=1


    def fill(self,data_frame):
        self.data_frame=data_frame
        if self.data_frame is None:
            raise ValueError('Genotype data frame is not defined!')
        for i in self.data_frame.iterrows():
            self.n_keys+=1
            if self.hash_dic[i[1].CHR].get(i[1].bp) is not None:
                if isinstance(self.hash_dic[i[1].CHR][i[1].bp], tuple):
                    if self._print:
                        print((
                        "Collision! You are trying to fill {} again with A1_{}/A2_{},"
                        " it is already inserted with A1_{}/A2_{}. "
                        "".format(
                            "CHR "+str(i[1].CHR) + " bp "+ str(i[1].bp),
                            i[1].allele1,i[1].allele2,
                            self.hash_dic[i[1].CHR][i[1].bp][1],self.hash_dic[i[1].CHR][i[1].bp][2])
                                                   ))

                    self.hash_dic[i[1].CHR][i[1].bp]=[self.hash_dic[i[1].CHR][i[1].bp],(i[0],i[1].allele1, i[1].allele2)   ]
                else:
                    self.hash_dic[i[1].CHR][i[1].bp].append((i[0],i[1].allele1, i[1].allele2))
                self.collisions[i[1].CHR].append(i[1].bp)
            else:
                self.hash_dic[i[1].CHR][i[1].bp]=(i[0], i[1].allele1, i[1].allele2)
        self.protected=True

    def get_index(self,chr, bp):

        r=self.hash_dic[chr].get(bp)
        if r is None:
            return -1,'R','R'
        else:
            return r

    def get_map(self,info):

        def _map(r,a1,a2):
            ind,A1,A2=r
            if ind==-1:
                result=[-1,0]
            elif A1==a1 and A2==a2:
                result=[ind, 1]
            elif A1==a2 and A2==a1:
                result=[ ind, -1]
            else:
                result=[-1,0]
            return result

        r=self.get_index(info.CHR, info.bp)
        if not isinstance(r[0],tuple):
            return _map(r,info.allele1,info.allele2)
        else:
            for i in r:
                m=_map(i,info.allele1,info.allele2)
                if m[0]!=-1:
                    return m
            return [-1,0]

class HashTableRSID(object):

    def __init__(self):
        self.hash_table=np.zeros([250*10**6])
        self.list=[]
        self.hash_dic={}
        self.protected=False
        self.type='RSID'
        self.n_keys=0


    def fill_id(self,id):
        #print id
        if not self.protected:
            #self.n_keys+=1
            try:
                if id.ID[:2]=='rs':
                    m=int(id.ID[2:])
                    if self.hash_table[m]!=0:
                        print('collision {}!'.format(id))
                        if isinstance(self.list[self.hash_table[m]],tuple):
                            self.list[self.hash_table[m]]=[self.list[self.hash_table[m]], (self.n_keys,id['allele1'],id['allele2'])         ]
                            self.n_keys+=1
                        else:
                            self.list[self.hash_table[m]].append( (self.n_keys,id['allele1'],id['allele2'])  )
                            self.n_keys+=1
                    else:
                        self.list.append( ( self.n_keys,id['allele1'],id['allele2'] )  )
                        self.hash_table[m]=len(self.list)
                        self.n_keys+=1
                else:
                    if self.hash_dic.get(id.ID) is None:
                        self.hash_dic[id.ID]=(self.n_keys,id['allele1'],id['allele2'])
                        self.n_keys+=1
                    else:
                        if isinstance(self.hash_dic[id.ID],tuple):
                            if self._print:
                                print((
                                "Collision! You are trying to fill {} again with A1_{}/A2_{},"
                                " it is already inserted with A1_{}/A2_{}. "
                                    "You can resolve the collisions, use genotype data with CHR and bp info".format(
                                    id.ID,
                                    id['allele1'],id['allele2'],
                                    self.hash_dic[id.ID][1],self.hash_dic[id.ID][2])

                                            ))
                            self.hash_dic[id.ID]=[self.hash_dic[id.ID], (self.n_keys,id['allele1'],id['allele2'])]
                            self.n_keys+=1
                        else:
                            self.hash_dic[id.ID].append((self.n_keys,id['allele1'],id['allele2']))
                            self.n_keys+=1
            except:

                if self.hash_dic.get(id.ID) is None:
                    self.hash_dic[id.ID]=[self.n_keys,id['allele1'],id['allele2']]
                    self.n_keys+=1
                else:
                    if isinstance(self.hash_dic[id[1].ID],tuple):
                        if self._print:
                            print((
                            "Collision! You are trying to fill {} again with A1_{}/A2_{},"
                            " it is already inserted with A1_{}/A2_{}. "
                                "You can resolve the collisions, use genotype data with CHR and bp info".format(
                                id.ID,
                                id['allele1'],id['allele2'],
                                self.hash_dic[id.ID][1],self.hash_dic[id.ID][2])

                                        ))
                        self.hash_dic[id.ID]=[self.hash_dic[id.ID], (self.n_keys,id['allele1'],id['allele2'])]
                        self.n_keys+=1
                    else:
                        self.hash_dic[id.ID].append((self.n_keys,id['allele1'],id['allele2']))
                        self.n_keys+=1

            #self.protected=True
        else:
            raise ValueError('Hash table already filled!')

    def fill(self,df):
        if not self.protected:
            for i in df.iterrows():
                try:
                    if i[1].ID[:2]=='rs':
                        m=int(i[1].ID[2:])
                        if self.hash_table[m]!=0:
                            print('collision {}!'.format(i))
                            if isinstance(self.list[self.hash_table[m]],tuple):
                                self.list[self.hash_table[m]]=[self.list[self.hash_table[m]], (i[0],i[1]['allele1'],i[1]['allele2'])         ]
                            else:
                                self.list[self.hash_table[m]].append( (i[0],i[1]['allele1'],i[1]['allele2'])  )
                        else:
                            self.list.append( ( i[0],i[1]['allele1'],i[1]['allele2'] )  )
                            self.hash_table[m]=len(self.list)
                    else:
                        if self.hash_dic.get(i[1].ID) is None:
                            self.hash_dic[i[1].ID]=(i[0],i[1]['allele1'],i[1]['allele2'])
                        else:
                            if isinstance(self.hash_dic[i],tuple):
                                print((
                                "Collision! You are trying to fill {} again with A1_{}/A2_{},"
                                " it is already inserted with A1_{}/A2_{}. "
                                    "You can resolve the collisions, use genotype data with CHR and bp info".format(
                                    i[1].ID,
                                    i[1]['allele1'],i['allele2'],
                                    self.hash_dic[i[1].ID][1],self.hash_dic[i[1].ID][2])

                                            ))
                                self.hash_dic[i[1].ID]=[self.hash_dic[i[1].ID], (i[0],i[1]['allele1'],i[1]['allele2'])]
                            else:
                                self.hash_dic[i[1].ID].append((i[0],i[1]['allele1'],i[1]['allele2']))
                except:
                    if self.hash_dic.get(i[1].ID) is None:
                        self.hash_dic[i[1].ID]=[i[0],i[1]['allele1'],i[1]['allele2']]
                    else:
                        if isinstance(self.hash_dic[i[1].ID],tuple):
                            print((
                            "Collision! You are trying to fill {} again with A1_{}/A2_{},"
                            " it is already inserted with A1_{}/A2_{}. "
                                "You can resolve the collisions, use genotype data with CHR and bp info".format(
                                i[1].ID,
                                i[1]['allele1'],i[1]['allele2'],
                                self.hash_dic[i[1].ID][1],self.hash_dic[i[1].ID][2])

                                        ))
                            self.hash_dic[i[1].ID]=[self.hash_dic[i[1].ID], (i[0],i[1]['allele1'],i[1]['allele2'])]
                        else:
                            self.hash_dic[i[1].ID].append((i[0],i[1]['allele1'],i[1]['allele2']))

            self.protected=True
        else:
            raise ValueError('Hash table already filled!')


    def get_index(self,value):
        try:
            if value[:2]=='rs':
                v=int(value[2:])
                r=int(self.hash_table[v])
                if r==0:
                    return -1,'R','R'
                else:
                    return self.list[r-1]
            else:
                return self.hash_dic[value]
        except:
            return -1,'R','R'

    def get_map(self,info):

        def _map(r,a1,a2):
            ind,A1,A2=r
            if ind==-1:
                result=[-1,0]
            elif A1==a1 and A2==a2:
                result=[ind, 1]
            elif A1==a2 and A2==a1:
                result=[ ind, -1]
            else:
                result=[-1,0]
            return result

        r=self.get_index(info.ID)
        if not isinstance(r[0],tuple):
            return _map(r,info.allele1,info.allele2)
        else:
            for i in r:
                m=_map(i,info.allele1,info.allele2)
                if m[0]!=-1:
                    return m
            return [-1,0]

    def save(self,path, filename):
        with open(os.path.join(path,filename), 'wb') as output:
            pickle.dump(self, output, -1)

if __name__=="__main__":
    print('Hash')


