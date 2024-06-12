###########################################################################################################
#   This file contains method to featurize feature of local environment,                                  #
#   each structure can be described as a graph, and the descriptors contains information                  #
#   of local environment are generate by graph convlution: Aggragate features of each node(atom) and its  #
#   neighbors, then use an operation 'readout' to aggragate all the nodes to a single value.              #
###########################################################################################################
# cartesian coordinate system is used
 
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure
from ast import literal_eval
import numpy as np
import scipy.stats as st
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

element_csv_dir=os.path.join(os.path.dirname(__file__),'element_property.csv')

def parallel_Featurize_enhance(featureFunc, s_list, n_jobs):
    pool = ThreadPool(processes=n_jobs)
    Feature_list = pool.starmap_async(func=featureFunc, iterable=s_list).get()
    pool.close()
    pool.join()
    return Feature_list

def parallel_Featurize(featureFunc, s_list, n_jobs):
    pool = mp.Pool(processes=n_jobs)
    Feature_list = pool.starmap_async(func=featureFunc, iterable=s_list).get()
    pool.close()
    pool.join()
    return Feature_list

def loadDictionary(df, str):
    labels = df.columns
    if str in labels and 'element' in labels:
        elements = df['element'].values
        x = df[str].values
    else:
        print(str)
        print('ERROR input csv')
        exit(0)
    x_dict = {}
    nElements = len(elements)
    for i in range(nElements):
        x_dict[elements[i]] = x[i]
    return x_dict

def dump_error_structure(s):
    string=s.composition.reduced_formula
    filename=string+'.cif'
    s.to(filename=filename)

def site_property(s,p_key,condition):
    '''
    s: pymatgen structure
    p_key: site_properties is a dictionary
    '''
    try:
        L=s.site_properties[p_key]
    except KeyError:
        print('this structure has no key, use all sites')
        return None
    else:
        indexs=[]
        for i in range(len(L)):
            if L[i]==condition:
                indexs.append(i)
            else:
                continue
        return indexs


class StructureNeighbors:

    def __init__(self,site_index_lst=None,element=None):
        """
        site_index_list: list: specify the index of sites to find nn, if it is None, all the sites are calculated
        element: str: specify the element, if it is None, all the elements are calculated, length is 1
        """
        self.if_error=False
        self.specify_element=element
        self.site_index=site_index_lst
        self.elements=[] # The elements in the list are str
        self.elements_coords=[] # The elements in the list are Cartesian coordinates
        self.neighbors_lst=[] # Tessellation list: The elements in the list are str
        self.neighbors_coords=[] # Tessellation list: The elements in the list are Cartesian coordinates
    
    def is_empty(self):
        if len(self.elements)==0 or len(self.neighbors_lst)==0:
            return False
        elif self.if_error==True:
            return False
        else:
            return True
    
    def get_all_neighbors_by_cutoff(self,s,cutoff=None):
        """
        s: pymatgen.core.sturcture.Structure: a pymatgen structure
        cutoff: float:determine radius
        """
        if cutoff is None:
            SUM=0
            for Site in s:
                try:
                    SUM+=Site.species.elements[0].atomic_radius
                except TypeError:
                    SUM+=1
                finally:
                    pass
            avg_atom_radius=SUM/len(s)
            radius=3.6*avg_atom_radius
            pass
        else:
            radius=cutoff
            pass
        if self.site_index is None:
            if self.specify_element is None:
                self.elements=self.elements + [site.species.elements[0].symbol for site in s]
                self.elements_coords=self.elements_coords + [site.coords for site in s]
                neighbor_lst=s.get_all_neighbors(r=radius,include_index=True)
                #sorted by distance
                neighbor_lst=[sorted(nbrs, key=lambda x: x[1]) for nbrs in neighbor_lst]
                #Just get the nearest 12 neighbor
                #check if any error
                for n in neighbor_lst:
                    #Just get the nearest 12 neighbor
                    if len(n)==0:
                        self.if_error=True
                        self.neighbors_lst.append([])
                        self.neighbors_coords.append([])
                    elif len(n)>12:
                        self.neighbors_lst.append([site.species.elements[0].symbol for site in n[:12]])
                        self.neighbors_coords.append([site.coords for site in n[:12]])
                    else:
                        self.neighbors_lst.append([site.species.elements[0].symbol for site in n])
                        self.neighbors_coords.append([site.coords for site in n])
            else:
                site_lst=[]
                for site in s:
                    if site.species.elements[0].symbol==self.specify_element:
                        site_lst.append(site)
                if len(site_lst)==0:
                    self.if_error=True
                    pass
                else:
                    self.elements=self.elements + [site.species.elements[0].symbol for site in site_lst]
                    self.elements_coords=self.elements_coords + [site.coords for site in site_lst]
                    neighbor_lst=s.get_all_neighbors(r=radius,sites=site_lst,include_index=True)
                    #sorted by distance
                    neighbor_lst=[sorted(nbrs, key=lambda x: x[1]) for nbrs in neighbor_lst]
                    #check if any error
                    for n in neighbor_lst:
                        if len(n)==0:
                            self.if_error=True
                            self.neighbors_lst.append([])
                            self.neighbors_coords.append([])
                        elif len(n)>12:
                            self.neighbors_lst.append([site.species.elements[0].symbol for site in n[:12]])
                            self.neighbors_coords.append([site.coords for site in n[:12]])
                        else:
                            self.neighbors_lst.append([site.species.elements[0].symbol for site in n])
                            self.neighbors_coords.append([site.coords for site in n])
        else:
            if self.specify_element is None:
                site_lst=[]
                for index in self.site_index:
                    site_lst.append(s[index])
                self.elements=self.elements + [site.species.elements[0].symbol for site in site_lst]
                self.elements_coords=self.elements_coords + [site.coords for site in site_lst]
                neighbor_lst=s.get_all_neighbors(r=radius,sites=site_lst,include_index=True)
                #sorted by distance
                neighbor_lst=[sorted(nbrs, key=lambda x: x[1]) for nbrs in neighbor_lst]
                #check if any error
                for n in neighbor_lst:
                    if len(n)==0:
                        self.if_error=True
                        self.neighbors_lst.append([])
                        self.neighbors_coords.append([])
                    elif len(n)>12:
                        self.neighbors_lst.append([site.species.elements[0].symbol for site in n[:12]])
                        self.neighbors_coords.append([site.coords for site in n[:12]])
                    else:
                        self.neighbors_lst.append([site.species.elements[0].symbol for site in n])
                        self.neighbors_coords.append([site.coords for site in n])
            else:
                site_lst=[]
                for index in self.site_index:
                    if s[index].species.elements[0].symbol==self.specify_element:
                        site_lst.append(s[index])
                    else:
                        continue
                if len(site_lst)==0:
                    self.if_error=True
                    pass
                else:
                    self.elements=self.elements + [site.species.elements[0].symbol for site in site_lst]
                    self.elements_coords=self.elements_coords + [site.coords for site in site_lst]
                    neighbor_lst=s.get_all_neighbors(r=radius,sites=site_lst,include_index=True)
                    #sorted by distance
                    neighbor_lst=[sorted(nbrs, key=lambda x: x[1]) for nbrs in neighbor_lst]
                    #check if any error
                    for n in neighbor_lst:
                        if len(n)==0:
                            self.if_error=True
                            self.neighbors_lst.append([])
                            self.neighbors_coords.append([])
                        elif len(n)>12:
                            self.neighbors_lst.append([site.species.elements[0].symbol for site in n[:12]])
                            self.neighbors_coords.append([site.coords for site in n[:12]])
                        else:
                            self.neighbors_lst.append([site.species.elements[0].symbol for site in n])
                            self.neighbors_coords.append([site.coords for site in n])
    
    def get_site_neighbors(self,s,i):
        """
        i: int: specify the site index of the strcuture
        """
        nn=CrystalNN(distance_cutoffs=None,x_diff_weight=0.0,porous_adjustment=False)
        try:
            nn_list_per_atom=nn.get_nn(s,i)
        except ValueError:
            print('Check this structure...')
            print(s.species)
            print('Check this site')
            print(s[i].species)
            print('Ignore and dump this structure')
            dump_error_structure(s)
            self.if_error=True
        else:
            self.elements.append(s[i].species.elements[0].symbol)
            self.elements_coords.append(s[i].coords)
            self.neighbors_lst.append([site.species.elements[0].symbol for site in nn_list_per_atom])
            self.neighbors_coords.append([site.coords for site in nn_list_per_atom])
    
    def get_all_neighbors(self,s):
        if self.site_index is None:
            #No marked site
            natoms=len(s)
            if self.specify_element is None:
                #No specify element
                for i in range(natoms):
                    self.get_site_neighbors(s,i)
            else:
                for i in range(natoms):
                    #only find neighbors of speicify element
                    if s[i].species.elements[0].symbol==self.specify_element:
                        self.get_site_neighbors(s,i)
                    else:
                        pass
        else:
            natoms=len(s)
            if self.specify_element is None:
                #No Specify element
                for i in self.site_index:
                    if i>=natoms:
                        print('Out of range, ignore this!')
                        continue
                    else:
                        self.get_site_neighbors(s,i)
            else:
                #only find specify element
                for i in self.site_index:
                    if i>=natoms:
                        print('Out of range, ignore this!')
                        continue
                    else:
                        if s[i].species.elements[0].symbol==self.specify_element:
                            self.get_site_neighbors(s,i)
                        else:
                            continue
    
    def __call__(self,s):
        self.get_all_neighbors(s)


class nn_batch:
    
    def __init__(self,p_key=None,condition=None,element=None,neighbors_method='Voronoi',cutoff=None):
        self.cutoff=cutoff
        self.neighbors_method=neighbors_method
        self.specify_element=element
        self.condition=condition
        self.p_key=p_key
        self.structnn_lst=[]
    
    def site_property(self,s):
        '''
        s: pymatgen structure
        p_key: site_properties is a dictionary
        '''
        indexs=site_property(s,self.p_key,self.condition)
        return indexs

    def find_nn(self,s,site_index_list=None):
        nn=StructureNeighbors(site_index_lst=site_index_list,element=self.specify_element)
        if self.neighbors_method=='Voronoi':
            nn.get_all_neighbors(s)
        elif self.neighbors_method=='cutoff':
            nn.get_all_neighbors_by_cutoff(s,self.cutoff)
        return nn

    def find_nn_in_dataframe(self,df,n_jobs,local_lst=None,global_lst=None): 
        try:
            structures=df['structures']
        except KeyError:
            print('The DataFrame must have a column with structure dictionary')
        else:
            s_lst=[]
            for s in structures:
                S=Structure.from_dict(literal_eval(s))
                if self.p_key is None:
                    s_lst.append([S])
                else:
                    if self.condition is None:
                        print('Find no condition,ignore the key!')
                        s_lst.append([S])
                    else:
                        indexs=self.site_property(S)
                        s_lst.append([S,indexs])
            nn_lst=parallel_Featurize(self.find_nn,s_lst,n_jobs)
            self.structnn_lst=[[s,local_lst,global_lst] for s in nn_lst]


#base Featurize
class basefeaturizer:
    def __init__(self,local_key='avg',global_key='avg',structnn_lst=None,p_key=None,condition=None,element=None,neighbors_methods='Voronoi',cutoff=None):
        self.local_keyName=local_key
        self.global_keyName=global_key
        self.structnn_lst=structnn_lst
        self.p_key=p_key
        self.condition=condition
        self.specify_element=element
        self.neighbors_method=neighbors_methods
        self.cutoff=cutoff
        self.features_type=['_'] #rewrite this attribute
        self.feature_type='_' #rewrite this attribute
        self.element_csv_df=pd.read_csv(element_csv_dir)        
    
    def featurize_local(self,nn):
        '''
        rewrite this function
        '''
        local_feature_lst=[]
        return local_feature_lst
    
    def featurize_global(self,nn):
        local_feature_lst=self.featurize_local(nn)
        if local_feature_lst is None:
            return np.nan
        else:
            local_feature_lst=np.array(local_feature_lst)
            if self.global_keyName=='avg':
                return np.mean(local_feature_lst)
            elif self.global_keyName=='std':
                return np.std(local_feature_lst)
            elif self.global_keyName=='ra':
                return max(local_feature_lst)-min(local_feature_lst)
            elif self.global_keyName=='sk':
                return st.skew(local_feature_lst)
            elif self.global_keyName=='ku':
                return st.kurtosis(local_feature_lst)
            elif self.global_keyName=='sum':
                return np.sum(local_feature_lst)
            else:
                print('error key!!')
                return np.nan
    
    def featurize(self,s):
        '''
        S: a pymatgen Structure object
        '''
        if self.p_key is None:
            site_index_lst=None
        else:
            if self.condition is None:
                site_index_lst=None
            else:
                site_index_lst=site_property(s,self.p_key,self.condition)

        nn=StructureNeighbors(site_index_lst,self.specify_element)

        if self.neighbors_method=='Voronoi':
            nn.get_all_neighbors(s)
        else:
            nn.get_all_neighbors_by_cutoff(s,self.cutoff)

        if nn.is_empty()==False:
            return np.nan
            
        feature=self.featurize_global(nn)

        return feature
    
    def featurize_dataframe(self,df,n_jobs):
        '''
        df: a pandas DataFrame containing structure information
        its column name should be 'structures'
        n_jobs: Finding Neighbors needs long time, using multi cpu cores to acclerate
        '''

        S_lst=[[Structure.from_dict(literal_eval(s))] for s in df['structures']]
        # print(S_lst[0])
        result_lst=parallel_Featurize(self.featurize,S_lst,n_jobs)
        cols_name="_".join([self.feature_type,self.local_keyName,self.global_keyName])
        df[cols_name]=result_lst
        return df

    def featurize_structnn(self,nn,local_lst=None,global_lst=None):
        # key list
        feature_lst = self.features_type
        if local_lst is None:
            local_lst = ['avg','std', 'ra','sk', 'ku', 'sum'] #统一顺序
        else:
            pass
        if global_lst is None:
            global_lst = ['avg', 'std', 'ra','sk', 'ku', 'sum']
        else:
            pass
        output_dict={}

        #check if empty
        if nn.is_empty()==False:
            for f in feature_lst:
                for l in local_lst:
                    for g in global_lst:
                        output_dict['_'.join([f,l,g])]=np.nan
        else:
            #Featurize
            for i in range(len(feature_lst)):
                for j in range(len(local_lst)):
                    for k in range(len(global_lst)):
                        self.feature_type=feature_lst[i]
                        self.local_keyName=local_lst[j]
                        self.global_keyName=global_lst[k]                        
                        output_dict['_'.join([feature_lst[i],local_lst[j],global_lst[k]])] = self.featurize_global(nn)
        return output_dict
    
    def featurize_all(self,s,local_lst=None,global_lst=None):
        '''
        S: a pymatgen structure object
        This function will return a dictionary containing all kinds of electronegative
        '''
        #get nn
        if self.p_key is None:
            site_index_lst=None
        else:
            if self.condition is None:
                site_index_lst=None
            else:
                site_index_lst=site_property(s,self.p_key,self.condition)

        nn=StructureNeighbors(site_index_lst,self.specify_element)

        if self.neighbors_method=='Voronoi':
            nn.get_all_neighbors(s)
        else:
            nn.get_all_neighbors_by_cutoff(s,self.cutoff)

        D=self.featurize_structnn(nn,local_lst,global_lst)
        return D
    
    def featurize_all_dataframe(self,n_jobs,df,local_lst=None,global_lst=None):
        '''
        df: a pandas dataframe containing structures,the name of the column containing structures should
        be 'structures'
        '''
        s_list=[[Structure.from_dict(literal_eval(s)),local_lst,global_lst] for s in df['structures']]
        D_list=parallel_Featurize(self.featurize_all,s_list,n_jobs)
        total_dict={}
        for item in D_list:
            for key,value in item.items():
                total_dict.setdefault(key,[]).append(value)
        df=pd.DataFrame(total_dict)
        return df

    def featurize_neighbor_lst_enhance(self,n_jobs,local_lst=None,global_lst=None):
        '''
        Because finding the neighbors is time wasting, When featurize more than one features in this file, use this function
        '''
        #check nnlist
        if self.structnn_lst is None:
            print('finding the neighbors first')
            # exit(0)
        else:
            parameter_lst = [[s[0],local_lst,global_lst] for s in tqdm(self.structnn_lst)]
            dict_lst=parallel_Featurize_enhance(self.featurize_structnn,parameter_lst,n_jobs)            
            total_dict={}
            for d in dict_lst:
                for key,value in d.items():
                    total_dict.setdefault(key,[]).append(value)
            return pd.DataFrame(total_dict)

    def featurize_neighbor_lst(self,local_lst=None,global_lst=None):
        '''
        Because finding the neighbors is time wasting, When featurize more than one features in this file, use this function
        '''
        #check nnlist
        if self.structnn_lst is None:
            print('finding the neighbors first')
            # exit(0)
        else:
            dict_lst=[self.featurize_structnn(s[0],local_lst,global_lst) for s in tqdm(self.structnn_lst)]
            total_dict={}
            for d in dict_lst:
                for key,value in d.items():
                    total_dict.setdefault(key,[]).append(value)
            return pd.DataFrame(total_dict)


#Featurize Electronegative
class ElectronegativeFeatures(basefeaturizer):
    '''
    'Pa_EN':Pauling_Electronegativity
    'MB_EN':MB_electonegativity
    'Go_EN':Gordy_electonegativity
    'Mu_EN':Mulliken_electonegativity
    'AR_EN':Allred_Rockow_electronegativity
    'Mi_EN':Miedema_electronegativity
    'Mi_ED':Miedema_electron_density

    '''
    def __init__(self,en='Pa_EN',local_key='avg',global_key='avg',structnn_lst=None,p_key=None,condition=None,element=None,neighbors_methods='Voronoi',cutoff=None):
        basefeaturizer.__init__(self,local_key,global_key,structnn_lst,p_key,condition,element,neighbors_methods,cutoff)
        self.feature_type=en
        self.features_type=['Pa_EN','MB_EN','Go_EN','Mu_EN','AR_EN','Mi_EN','Mi_ED']
        self.property_dict=None
    
    def load_property(self):
        #EN_dict
        if self.feature_type=='Pa_EN':
            self.property_dict=loadDictionary(self.element_csv_df,'Pauling_Electronegativity')
        elif self.feature_type=='MB_EN':
            self.property_dict=loadDictionary(self.element_csv_df,'MB_electonegativity')
        elif self.feature_type=='Go_EN':
            self.property_dict=loadDictionary(self.element_csv_df,'Gordy_electonegativity')
        elif self.feature_type=='Mu_EN':
            self.property_dict=loadDictionary(self.element_csv_df,'Mulliken_electonegativity')
        elif self.feature_type=='AR_EN':
            self.property_dict=loadDictionary(self.element_csv_df,'Allred_Rockow_electronegativity')
        elif self.feature_type=='Mi_EN':
            self.property_dict=loadDictionary(self.element_csv_df,'Miedema_electronegativity')
        elif self.feature_type=='Mi_ED':
            self.property_dict=loadDictionary(self.element_csv_df,'Miedema_electron_density')
        else:
            print('{} is invalid key'.format(self.feature_type))
            exit(0)

    def gen_site_feature(self,nn):
        #elements and its neighbors
        elements=nn.elements
        nn_lst=nn.neighbors_lst

        l=len(elements)
        self.load_property()
        nbrs_property=[]
        for i in range(l):
            try:
                x_self_EN=self.property_dict[elements[i]]
            except KeyError:
                print('This Element {} has no specific properties!!!'.format(elements[i]))   
                return None
            else:
                if x_self_EN != x_self_EN:
                    print('This element {} has no EN data,returns NaN'.format(elements[i]))
                    return None
                nn_EN=[]
                for e in nn_lst[i]:
                    try:
                        x_nn_EN=self.property_dict[e]
                    except KeyError:
                        print('This Element {} has no specific properties!!!'.format(e))
                        return None
                    else:
                        if x_nn_EN != x_nn_EN:
                            print('This neighbor {} has no EN data,returns NaN'.format(e))
                            return None
                        nn_EN.append(x_nn_EN)
                nbrs_property.append(np.abs(np.array(nn_EN)-x_self_EN))
        return nbrs_property
    
    def featurize_local(self,nn):
        #gen feature for each site
        nbrs_property=self.gen_site_feature(nn)
        if nbrs_property is None:
            return None
        #aggregate
        local_feature_lst=[]
        for delta_EN in nbrs_property:
            #delta_EN=np.abs(nn_property-element_property[i])
            if self.local_keyName=='avg':
                avg=np.mean(delta_EN)
                local_feature_lst.append(avg)
            elif self.local_keyName=='std':
                std=np.std(delta_EN)
                local_feature_lst.append(std)
            elif self.local_keyName=='sk':
                sk=st.skew(delta_EN)
                local_feature_lst.append(sk)
            elif self.local_keyName=='ku':
                ku=st.kurtosis(delta_EN)
                local_feature_lst.append(ku)
            elif self.local_keyName=='ra':
                ra=max(delta_EN)-min(delta_EN)
                local_feature_lst.append(ra)
            elif self.local_keyName=='sum':
                SUM=np.sum(delta_EN)
                local_feature_lst.append(SUM)
            else:
                print('Error local key!')
                return None
        return local_feature_lst


#Chemical Bonds 
class BondSphericalCoordinatesFeatures(basefeaturizer):
    '''
    Bond features describe the length and the angle of a bond
    Bond_features has choices below:
    Radius: length of bond
    theta: angle between bond and xoy plane, return the value of cosine
    Phi: angle between bond and yoz plane,return the value of cosine
    '''
    def __init__(self,Bond_feature='Radius',global_keyName='avg',local_keyName='avg',structnn_lst=None,p_key=None,condition=None,element=None,neighbors_methods='Voronoi',cutoff=None):
        basefeaturizer.__init__(self,global_keyName,local_keyName,structnn_lst,p_key,condition,element,neighbors_methods,cutoff)
        self.feature_type=Bond_feature
        self.features_type=['R','T','P']        
    
    def transform_sphere(self,vec):
        r=np.linalg.norm(vec)
        theta=abs(vec[2])/r
        phi=abs(vec[0])/r
        return np.array([r,theta,phi])
    
    def Range_vector(self,vector):
        '''
        vector is a 2D array like data
        '''
        V=np.array(vector)
        if len(V.shape)==1:
            return np.max(V)-np.min(V)
        else:
            L=V.shape[1]
            return np.array([max(V[:,i])-min(V[:,i]) for i in range(L)])
    
    def gen_site_feature(self,nn,all=False):
        '''
        element_coords: attribution of class Structure_NN
        nn_coords: attribution of class Structure_NN
        '''
        element_coords=nn.elements_coords
        nn_coords=nn.neighbors_coords
        #Featurize
        nAtoms=len(element_coords)
        nbr_property=[]
        for i in range(nAtoms):
            spheres_per_atom=[]
            if len(nn_coords[i])==0:
                print("This site has no neighbors")
                nbr_property.append(np.array([]))
            else:
                for site_coord in nn_coords[i]:
                    deltaR=site_coord-element_coords[i]
                    if deltaR is None:
                        print('Error,dump this structure for debug')
                        return None
                    else:
                        sphere=self.transform_sphere(deltaR)
                        spheres_per_atom.append(sphere)
                spheres_per_atom=np.array(spheres_per_atom)
                if all==True:
                    nbr_property.append(spheres_per_atom)
                else:
                    if self.feature_type=='Radius':
                        nbr_property.append(spheres_per_atom[:,0])
                    elif self.feature_type=='Theta':
                        nbr_property.append(spheres_per_atom[:,1])
                    elif self.feature_type=='Phi':
                        nbr_property.append(spheres_per_atom[:,2])
        return nbr_property
    
    def featurize_local(self,nn,all=False):
        nbr_property=self.gen_site_feature(nn,all)
        if nbr_property is None:
            return None
        local_feature_lst=[]
        for spheres_per_atom in nbr_property:
            if self.local_keyName=='avg':
                local_feature_lst.append(np.mean(spheres_per_atom,axis=0))
            elif self.local_keyName=='std':
                local_feature_lst.append(np.std(spheres_per_atom,axis=0))
            elif self.local_keyName=='sk':
                local_feature_lst.append(st.skew(spheres_per_atom,axis=0))
            elif self.local_keyName=='ku':
                local_feature_lst.append(st.kurtosis(spheres_per_atom,axis=0))
            elif self.local_keyName=='sum':
                local_feature_lst.append(np.sum(spheres_per_atom,axis=0))  
            elif self.local_keyName=='ra':
                local_feature_lst.append(self.Range_vector(spheres_per_atom))
            else:
                print('Error Key! ')
                return None
        local_feature_lst=np.array(local_feature_lst)
        return local_feature_lst
              
    def featurize_structnn(self,nn,local_lst=None,global_lst=None):
        #all the keyname
        if local_lst is None:
            local_lst = ['avg','std', 'ra','sk', 'ku', 'sum'] #统一顺序
        else:
            pass
        if global_lst is None:
            global_lst = ['avg', 'std', 'ra','sk', 'ku', 'sum']
        else:
            pass
        rtp=self.features_type

        D={}
        #check error
        if nn.is_empty() == False:
            for l in local_lst:
                for g in global_lst:
                    for r in rtp:
                        D['_'.join([r,l,g])]=np.nan
            return D
        
        #Featurize
        for l in local_lst:
            for g in global_lst:
                self.local_keyName=l
                local_feature_lst=self.featurize_local(nn,all=True)
                if local_feature_lst is None:
                    return np.nan
                else:
                    local_feature_lst=np.array(local_feature_lst)
                    if g=='avg':
                        Bond_vec=np.mean(local_feature_lst,axis=0)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
                    elif g=='std':
                        Bond_vec=np.std(local_feature_lst,axis=0)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
                    elif g=='sk':
                        Bond_vec=st.skew(local_feature_lst,axis=0)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
                    elif g=='ku':
                        Bond_vec=st.kurtosis(local_feature_lst,axis=0)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
                    elif g=='ra':
                        Bond_vec=self.Range_vector(local_feature_lst)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
                    elif g=='sum':
                        Bond_vec=np.sum(local_feature_lst,axis=0)
                        for i in range(3):
                            D['_'.join([rtp[i],l,g])]=Bond_vec[i]
        return D


#Coordination Number  
class CoordinationNumeber(basefeaturizer):
    '''
    This feature describe the number neighbors of each atom in a structure
    '''
    def __init__(self,local_key='avg',global_key='avg',structnn_lst=None,p_key=None,condition=None,element=None,neighbors_methods='Voronoi',cutoff=None):
        basefeaturizer.__init__(self,local_key,global_key,structnn_lst,p_key,condition,element,neighbors_methods,cutoff)
        self.feature_type='CN'
        self.features_type=['CN']        
    
    def featurize_local(self,nn):
        '''
        return a list
        '''
        elements=nn.elements
        nn_lst=nn.neighbors_lst
        nAtoms=len(elements)
        local_feature_list=[]
        for i in range(nAtoms):
            local_feature_list.append(len(nn_lst[i]))
        return local_feature_list

    def featurize_structnn(self,nn,local_lst=None,global_lst=None):
        #all the keyname
        if global_lst is None:
            global_lst = ['avg', 'std', 'ra','sk', 'ku', 'sum']
        else:
            pass

        D={}
        #check if empty
        if nn.is_empty()==False:
            for g in global_lst:
                 D['_'.join(['CN',g])]=np.nan
        else:
            for g in global_lst:
                self.global_keyName=g
                D['_'.join(['CN',g])]=self.featurize_global(nn)
        return D


class CompositionDescriptor(basefeaturizer):
    '''
    'AN':Atomic_number
    'AW':Atomic_weight
    'De':Density
    'MVo':Miedema_volume
    'Gr':Group
    'Me':Metal
    'NMe':Nonmetal
    'MN':Mendeleev_number
    'Pcs':Pcs
    'EA':Electron_affinity
    'AR':Atomic_radius
    'IR':Ionic_radius
    'TMR':Teatum_metalic_radii
    'VDWR':Van_der_waals_radii
    'ZRS':Zunger_radii_sum
    'MV':Metallic_valence
    'LQN':L_quantum_number
    'NVE':Number_of_valence_electrons
    'GNVE':Gilmor_number_of_valence_electron
    'VS':Valence_s
    'VP':Valence_p
    'VD':Valence_d
    'VF':Valence_f
    'NVS':Number_of_unfilled_s_valence_electrons
    'NVP':Number_of_unfilled_p_valence_electrons
    'NVD':Number_of_unfilled_d_valence_electrons
    'NVF':Number_of_unfilled_f_valence_electrons
    'OSE':Outer_shell_electrons
    'MP':Melting_point
    'BP':Boiling_point
    'FE':Fusion_enth
    'VE':Vap_enth
    'SHC':Specific_heat_capacity
    'TC':Thermal_conductivity
    'FIP':1st_ionization_potential
    '''
    def __init__(self,cd='AN',local_key='avg',global_key='avg',structnn_lst=None,p_key=None,condition=None,element=None,neighbors_methods='Voronoi',cutoff=None):
        basefeaturizer.__init__(self,local_key,global_key,structnn_lst,p_key,condition,element,neighbors_methods,cutoff)
        self.feature_type=cd
        self.features_type=['AN','AW','De','MVo','Gr','Me','NMe','MN','Pcs','EA','AR','IR','TMR','VDWR','ZRS','MV','LQN','NVE','GNVE','VS','VP','VD','VF',
                            'NVS','NVP','NVD','NVF','OSE','MP','BP','FE','VE','SHC','TC','FIP']
        self.property_dict=None        
    
    def load_property(self):
        #cd_dict
        if self.feature_type=='AN':
            self.property_dict=loadDictionary(self.element_csv_df,'Atomic_number')
        elif self.feature_type=='AW':
            self.property_dict=loadDictionary(self.element_csv_df,'Atomic_weight')
        elif self.feature_type=='De':
            self.property_dict=loadDictionary(self.element_csv_df,'Density')
        elif self.feature_type=='MVo':
            self.property_dict=loadDictionary(self.element_csv_df,'Miedema_volume')
        elif self.feature_type=='Gr':
            self.property_dict=loadDictionary(self.element_csv_df,'Group')
        elif self.feature_type=='Me':
            self.property_dict=loadDictionary(self.element_csv_df,'Metal')
        elif self.feature_type=='NMe':
            self.property_dict=loadDictionary(self.element_csv_df,'Nonmetal')
        elif self.feature_type=='MN':
            self.property_dict=loadDictionary(self.element_csv_df,'Mendeleev_number')
        elif self.feature_type=='Pcs':
            self.property_dict=loadDictionary(self.element_csv_df,'Pcs')
        elif self.feature_type=='EA':
            self.property_dict=loadDictionary(self.element_csv_df,'Electron_affinity')
        elif self.feature_type=='AR':
            self.property_dict=loadDictionary(self.element_csv_df,'Atomic_radius')
        elif self.feature_type=='IR':
            self.property_dict=loadDictionary(self.element_csv_df,'Ionic_radius')
        elif self.feature_type=='TMR':
            self.property_dict=loadDictionary(self.element_csv_df,'Teatum_metalic_radii')
        elif self.feature_type=='VDWR':
            self.property_dict=loadDictionary(self.element_csv_df,'Van_der_waals_radii')
        elif self.feature_type=='ZRS':
            self.property_dict=loadDictionary(self.element_csv_df,'Zunger_radii_sum')
        elif self.feature_type=='MV':
            self.property_dict=loadDictionary(self.element_csv_df,'Metallic_valence')
        elif self.feature_type=='LQN':
            self.property_dict=loadDictionary(self.element_csv_df,'L_quantum_number')
        elif self.feature_type=='NVE':
            self.property_dict=loadDictionary(self.element_csv_df,'Number_of_valence_electrons')
        elif self.feature_type=='GNVE':
            self.property_dict=loadDictionary(self.element_csv_df,'Gilmor_number_of_valence_electron')
        elif self.feature_type=='VS':
            self.property_dict=loadDictionary(self.element_csv_df,'Valence_s')
        elif self.feature_type=='VP':
            self.property_dict=loadDictionary(self.element_csv_df,'Valence_p')
        elif self.feature_type=='VD':
            self.property_dict=loadDictionary(self.element_csv_df,'Valence_d')
        elif self.feature_type=='VF':
            self.property_dict=loadDictionary(self.element_csv_df,'Valence_f')
        elif self.feature_type=='NVS':
            self.property_dict=loadDictionary(self.element_csv_df,'Number_of_unfilled_s_valence_electrons')
        elif self.feature_type=='NVP':
            self.property_dict=loadDictionary(self.element_csv_df,'Number_of_unfilled_p_valence_electrons')
        elif self.feature_type=='NVD':
            self.property_dict=loadDictionary(self.element_csv_df,'Number_of_unfilled_d_valence_electrons')
        elif self.feature_type=='NVF':
            self.property_dict=loadDictionary(self.element_csv_df,'Number_of_unfilled_f_valence_electrons')
        elif self.feature_type=='OSE':
            self.property_dict=loadDictionary(self.element_csv_df,'Outer_shell_electrons')
        elif self.feature_type=='MP':
            self.property_dict=loadDictionary(self.element_csv_df,'Melting_point')
        elif self.feature_type=='BP':
            self.property_dict=loadDictionary(self.element_csv_df,'Boiling_point')
        elif self.feature_type=='FE':
            self.property_dict=loadDictionary(self.element_csv_df,'Fusion_enth')
        elif self.feature_type=='VE':
            self.property_dict=loadDictionary(self.element_csv_df,'Vap_enth')
        elif self.feature_type=='SHC':
            self.property_dict=loadDictionary(self.element_csv_df,'Specific_heat_capacity')
        elif self.feature_type=='TC':
            self.property_dict=loadDictionary(self.element_csv_df,'Thermal_conductivity')
        elif self.feature_type=='FIP':
            self.property_dict=loadDictionary(self.element_csv_df,'1st_ionization_potential')       
        else:
            print('{} is invalid key'.format(self.feature_type))
            exit(0)

    def featurize_local(self,nn):
        #elements
        elements=nn.elements

        l=len(elements)
        self.load_property()
        local_feature_list=[]
        for i in range(l):
            try:
                x_self_cd=self.property_dict[elements[i]]
            except KeyError:
                print('This Element {} has no specific properties!!!'.format(elements[i]))   
                return None
            else:
                if x_self_cd != x_self_cd:
                    print('This element {} has no cd data,returns NaN'.format(elements[i]))
                    return None
                local_feature_list.append(x_self_cd)
        return local_feature_list
    
    def featurize_structnn(self,nn,local_lst=None,global_lst=None):
        # key list
        feature_lst = self.features_type

        #all the keyname
        if global_lst is None:
            global_lst = ['avg', 'std', 'ra','sk', 'ku', 'sum']
        else:
            pass

        D={}
        #check if empty
        if nn.is_empty()==False:
            for f in feature_lst:
                for g in global_lst:
                 D['_'.join([f,g])]=np.nan
        else:
            for i in range(len(feature_lst)):
                for j in range(len(global_lst)):
                    self.feature_type=feature_lst[i]
                    self.global_keyName=global_lst[j]
                    D['_'.join([feature_lst[i],global_lst[j]])]=self.featurize_global(nn)
        return D
    

if __name__=='__main__':
    bf = basefeaturizer()
    print(bf.element_csv_df.info())
    ef = ElectronegativeFeatures()
    ef.load_property()