import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import LocalProperty as LR
from pymatgen.core import Structure
from xgboost import XGBRegressor as XGBR
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def compoundType(compound):
    trans_elements = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Lu','Hf','Ta','W','Re','Os'
                     ,'Ir','Pt','Au','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb']
    left_elements = ['H','Li','Be','Na','Mg','K','Ca','Cs','Ba','Rb','Sr']
    right_elements = ['Zn','Cd','Hg','B','C','N','Al','Si','P','Ga','Ge','As','In','Sn','Sb','Tl','Pb','Bi']
    elements = compound.split('-')
    if all(element in trans_elements for element in elements):
        return 0
    elif any(element in trans_elements for element in elements) and any(element in left_elements for element in elements):
        return 1
    elif any(element in trans_elements for element in elements) and any(element in right_elements for element in elements):
        return 2
    else:
        return 3

class Enthalpy_Calculator():

    def __init__(self):

        self.poscar_path = sys.argv[1]
        self.structure = Structure.from_file(self.poscar_path)  # structure
        self.density = self.structure.density  # density 
        self.volume = self.structure.lattice.volume / len(self.structure)  # volume per atom
        self.datasets = pd.DataFrame(data=[(self.density, self.volume)], columns=['density', 'volume/atom'])  # initial dataframe storing eigenvalues        

        # Extract element information
        self.elements = [str(site.specie) for site in self.structure.sites]
        self.element_counts = {element: self.elements.count(element) for element in set(self.elements)}
        self.element_type = compoundType('-'.join(sorted(set(self.elements))))

        # Load the appropriate model
        model_path = self.select_model_path()
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

    def select_model_path(self):
        model_dir = os.path.dirname(__file__)
        model_dir = os.path.join(model_dir, 'models')
        element_set = set(self.element_counts.keys())
        
        if len(element_set) == 2:
            element_type = self.element_type
            if element_type == 0:
                return os.path.join(model_dir, 'Trans.dat')
            elif element_type == 1:
                return os.path.join(model_dir, 'Left.dat')
            elif element_type == 2:
                return os.path.join(model_dir, 'Right.dat')
            else:
                return os.path.join(model_dir, 'Nontrans.dat')
        else:
            print("Warning: please check the input file")  
            sys.exit(1) 

    def model_run(self):

        EF = LR.ElectronegativeFeatures()
        BSC = LR.BondSphericalCoordinatesFeatures()
        CN = LR.CoordinationNumeber()
        CD = LR.CompositionDescriptor()
        my_lst = [CD, EF, BSC, CN]
        df_lst = [self.datasets]
        df = pd.DataFrame(index=[0], columns=['structures'])
        df.iloc[0, 0] = str(SpacegroupAnalyzer(self.structure).get_primitive_standard_structure().as_dict())
        for item in my_lst:
            df_lst.append(item.featurize_all_dataframe(n_jobs=1, df=df, local_lst=['avg']))
        self.datasets = pd.concat(df_lst, axis=1)
        # self.datasets.to_csv('result.csv')

        predict_value = self.model.predict(self.datasets)[0]
        return predict_value

if __name__ == '__main__':

    if len(sys.argv) != 2:  
        print("Usage: python predict.py <path_to_poscar_file>")  
        sys.exit(1)        

    En_cal = Enthalpy_Calculator()
    
    # Print element counts and type
    # print("Element counts:", En_cal.element_counts)
    # print("Element type:", En_cal.element_type)
    
    enthalpy = En_cal.model_run()
    print('%.4f eV/atom' % enthalpy)
