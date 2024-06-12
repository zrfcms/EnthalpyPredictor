import os
import argparse
import pandas as pd
import multiprocessing as mp
from pymatgen.core import Structure

def structure_info(path):
    structure = Structure.from_file(path)
    s = structure.as_dict()
    density = structure.density
    volume = structure.lattice.volume
    volume_per_atom = volume / len(structure)

    return [s, density, volume_per_atom]

def parallel_run(myfunc, mylist: list, n_jobs: int):
    pool = mp.Pool(processes=n_jobs)
    re = pool.map_async(func=myfunc, iterable=mylist).get()
    pool.close()
    pool.join()

    return re

def main():
    parser = argparse.ArgumentParser(description='Process structure files.')
    parser.add_argument('--input', type=str, help='Path to the input file containing structure file paths')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--output', type=str, default='output.csv', help='Output CSV file')

    args = parser.parse_args()
    path = args.input
    n_jobs = args.n_jobs

    file_lst = os.listdir(path)
    file_lst.sort(key= lambda x: int(x.split('.')[0]))
    file_lst = [os.path.join(path, filename) for filename in file_lst]
    
    results = parallel_run(structure_info, file_lst, n_jobs)
    
    df = pd.DataFrame(results, columns=['structures', 'density', 'volume/atom'])
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
