from datetime import datetime
import LocalProperty as LP
from tqdm import tqdm
import pandas as pd 
import argparse

def main():
    parse=argparse.ArgumentParser()
    parse.add_argument('--input',help="Specify the csv file path",type=str)   
    parse.add_argument('--n_jobs',help='Specify the cpu cores you used',type=int, default=1)
    parse.add_argument('--descriptor',help="Specify the descriptor, available choice:Electronegative,Bond,CoordinationNumber,CompositionDescriptor",choices=['Electronegative','Bond','CoordinationNumber','CompositionDescriptor','All'])
    parse.add_argument('--output',help='output filename',type=str,default='output.csv')
    #load_data
    args=parse.parse_args()
    file_path=args.input
    try:
        data=pd.read_csv(file_path)
    except FileNotFoundError:
        print('Can not find your file')
    else:
        try:
            s=data['structures']
        except KeyError:
            print('This file has no structures')
        else:
            #check element
            n_jobs=args.n_jobs            
            outputfilename=args.output
            print("No Specify element, featurize all for you")
            D=args.descriptor
            EF=LP.ElectronegativeFeatures()
            BSC=LP.BondSphericalCoordinatesFeatures()
            CN=LP.CoordinationNumeber()
            CD=LP.CompositionDescriptor()
            if D=='Electronegative':
                start_time=datetime.now()
                df=EF.featurize_all_dataframe(n_jobs=n_jobs, df=data, local_lst=['avg'])
                end_time = datetime.now()
                print(f'TOTAL TIME TO Featurize = {(end_time - start_time).seconds} seconds')
            elif D=='Bond':
                start_time=datetime.now()
                df=BSC.featurize_all_dataframe(n_jobs=n_jobs, df=data, local_lst=['avg'])
                end_time = datetime.now()
                print(f'TOTAL TIME TO Featurize = {(end_time - start_time).seconds} seconds')
            elif D=='CoordinationNumber':
                start_time=datetime.now()
                df=CN.featurize_all_dataframe(n_jobs=n_jobs, df=data, local_lst=['avg'])
                end_time = datetime.now()
                print(f'TOTAL TIME TO Featurize = {(end_time - start_time).seconds} seconds')
            elif D=='CompositionDescriptor':
                start_time=datetime.now()
                df=CD.featurize_all_dataframe(n_jobs=n_jobs, df=data, local_lst=['avg'])
                end_time = datetime.now()
                print(f'TOTAL TIME TO Featurize = {(end_time - start_time).seconds} seconds')
            elif D=='All':
                start_time=datetime.now()
                df_list = []
                for D in tqdm([CD,EF,BSC,CN]):
                    df_list.append(D.featurize_all_dataframe(n_jobs=n_jobs, df=data, local_lst=['avg']))
                df = pd.concat(df_list,axis=1)
                end_time = datetime.now()
                print(f'TOTAL TIME TO Featurize = {(end_time - start_time).seconds} seconds')
            df=pd.concat([data, df], axis=1)
            df.to_csv(outputfilename)
            
if __name__=='__main__':
    main()    