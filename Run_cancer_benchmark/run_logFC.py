import os
import pandas as pd
import numpy as np
from grea import GREA, read_gmt,preprocess_signature,benchmark_parallel,entropy2gene, HVGs, entropy_HVGs

data_list = [
    {
        "dataset_name": "COAD_iv",
        "path": "data/COAD/COAD",
        "group": "stage",
        "sample_name": "sample_id",
        "perturbation": "iv",
    },
    {
        "dataset_name": "COAD_iia",
        "path": "data/COAD/COAD",
        "group": "stage",
        "sample_name": "sample_id",
        "perturbation": "iia",
    },
    {
        "dataset_name": "KIRC_iv",
        "path": "data/KIRC/KIRC",
        "group": "stage",
        "sample_name": "sample_id",
        "perturbation": "iv",
    },
    {
        "dataset_name": "KIRC_iii",
        "path": "data/KIRC/KIRC",
        "group": "stage",
        "sample_name": "sample_id",
        "perturbation": "iii",
    },
    {
        "dataset_name": "LUAD_iv",
        "path": "data/LUAD/LUAD",
        "group": "stage",
        "sample_name": "sample_id",
        "perturbation": "iv",
    },
    {
        "dataset_name": "LUAD_iib",
        "path": "data/LUAD/LUAD",
        "group": "stage",
        "sample_name": "sample_id",
        "perturbation": "iib",
    },
    {
        "dataset_name": "STAD_iv",
        "path": "data/STAD/STAD",
        "group": "stage",
        "sample_name": "sample_id",
        "perturbation": "iv",
    },
    {
        "dataset_name": "STAD_iiia",
        "path": "data/STAD/STAD",
        "group": "stage",
        "sample_name": "sample_id",
        "perturbation": "iiia",
    },
    {
        "dataset_name": "THCA_iii",
        "path": "data/THCA/THCA",
        "group": "stage",
        "sample_name": "sample_id",
        "perturbation": "iii",
    },
    {       
        "dataset_name": "THCA_iva",
        "path": "data/THCA/THCA",
        "group": "stage",
        "sample_name": "sample_id",
        "perturbation": "iva",
    },
]



def run_entropy_5000_FC_benchmark(data_list, library_path,output_dir="result"):
    library = read_gmt(library_path)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for data_obj in data_list:
        dataset_name = data_obj.get("dataset_name")
        data_path = data_obj.get("path")

        print(f"Processing dataset: {dataset_name}")

        signature_file = f"{data_path}_entropy.csv"
        group_file = f"{data_path}_phenotype.csv"
        if not os.path.exists(signature_file):
            print(f"Warning: {signature_file} not found, skipping dataset {dataset_name}.")
            continue

        signature = pd.read_csv(signature_file)
        group = pd.read_csv(group_file)

        signature['sig_name'] = signature["node1"] + "_" + signature["node2"]
        signature.drop(columns=["node1", "node2"], inplace=True)
        signature.set_index("sig_name", inplace=True)

        # entropy_5000
        signature_5000, _ = HVGs(signature,base_name = dataset_name,output_dir="Processed Data")

        group = group[[data_obj.get("sample_name"), data_obj.get("group")]]

        signature_5000_FC = preprocess_signature(signature_5000,group,base_name = dataset_name,output_dir="Processed Data")

        sub_signature_5000_FC = signature_5000_FC[data_obj.get("perturbation")]
        sub_signature_5000_FC.drop(columns=["group"], inplace=True)

        benchmark_parallel(signature=sub_signature_5000_FC, library=library,output_dir=f"{output_dir}/{dataset_name}")  

run_entropy_5000_FC_benchmark(data_list,library_path="DATA/Enrichr.KEGG_2021_Human.gmt",output_dir="result_entropy_5000_logfc")

def run_entropy_5000_sg_FC_benchmark(data_list,library_path,output_dir="result"):
    library = read_gmt(library_path)
    sub_library = {}
    # for i, key in enumerate(library.keys()):
    #     if i > 20:
    #         break
    #     sub_library[key] = library[key]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for data_obj in data_list:
        dataset_name = data_obj.get("dataset_name")
        data_path = data_obj.get("path")

        print(f"Processing dataset: {dataset_name}")

        signature_file = f"{data_path}_entropy.csv"
        group_file = f"{data_path}_phenotype.csv"
        if not os.path.exists(signature_file):
            print(f"Warning: {signature_file} not found, skipping dataset {dataset_name}.")
            continue

        signature = pd.read_csv(signature_file)
        group = pd.read_csv(group_file)

        signature['sig_name'] = signature["node1"] + "_" + signature["node2"]
        signature.drop(columns=["node1", "node2"], inplace=True)
        signature.set_index("sig_name", inplace=True)

        # entropy_5000
        signature_5000, _ = HVGs(signature,base_name = dataset_name,output_dir="Processed Data")
        #break to single gene
        signature_5000 = signature_5000.reset_index()
        signature_5000["node1"] = signature_5000['sig_name'].str.split("_").str[0]
        signature_5000["node2"] = signature_5000['sig_name'].str.split("_").str[1]
        signature_5000.drop(columns=["sig_name"], inplace=True)
        single_5000 = entropy2gene(signature_5000,base_name = dataset_name,output_dir="Processed Data")
        #FC
        group = group[[data_obj.get("sample_name"), data_obj.get("group")]]
        single_5000_FC = preprocess_signature(single_5000,group)
        #Symtomic group FC
        sub_single_5000_FC = single_5000_FC[data_obj.get("perturbation")]
        sub_single_5000_FC ['names'] = sub_single_5000_FC['names'].str.upper()
        sub_single_5000_FC.drop(columns=["group"], inplace=True)

        benchmark_parallel(signature=sub_single_5000_FC, library=library,output_dir=f"{output_dir}/{dataset_name}",rep_n = 3, perm_list = [500,1000,2000])  

run_entropy_5000_sg_FC_benchmark(data_list,library_path="data/Enrichr.KEGG_2019_Mouse.gmt",output_dir="result_entropy_5000_sg_logfc")

def run_exp_5000_FC_benchmark(data_list, library_path,output_dir="result"):
    library = read_gmt(library_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for data_obj in data_list:
        dataset_name = data_obj.get("dataset_name")
        data_path = data_obj.get("path")

        print(f"Processing dataset: {dataset_name}")
        
        group_file = f"{data_path}_phenotype.csv"
        exp_file = f"{data_path}_expr.csv"

        
        if not os.path.exists(exp_file):
            print(f"Warning: {exp_file} not found, skipping dataset {dataset_name}.")
            continue
        group = pd.read_csv(group_file)

        #-------------------
        # exp = pd.read_csv(exp_file,index_col=0)

        exp = pd.read_csv(exp_file)

        exp.index = exp.iloc[:, 1]

        exp = exp.iloc[:, 2:]

        #-------------------


        # exp_5000
        exp_5000, _ = HVGs(exp,base_name = dataset_name,output_dir="Processed Data")
        #FC
        group = group[[data_obj.get("sample_name"), data_obj.get("group")]]
        exp_5000_FC = preprocess_signature(exp_5000,group,base_name = dataset_name,output_dir="Processed Data")
        

        sub_exp_5000_FC = exp_5000_FC[data_obj.get("perturbation")]

        sub_exp_5000_FC['names'] = sub_exp_5000_FC['names'].str.upper()
        sub_exp_5000_FC.drop(columns=["group"], inplace=True)

        benchmark_parallel(signature=sub_exp_5000_FC, library=library,output_dir=f"{output_dir}/{dataset_name}",rep_n = 3, perm_list = [500,1000,2000])  

run_exp_5000_FC_benchmark(data_list,library_path="data/Enrichr.KEGG_2019_Mouse.gmt",output_dir="result_exp_5000_logfc")

def run_entropy_5000_FC_sg_benchmark(data_list,library_path,output_dir="result"):
    library = read_gmt(library_path)
    sub_library = {}
    for i, key in enumerate(library.keys()):
        if i > 20:
            break
        sub_library[key] = library[key]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for data_obj in data_list:
        dataset_name = data_obj.get("dataset_name")
        data_path = data_obj.get("path")

        print(f"Processing dataset: {dataset_name}")

        signature_file = f"{data_path}_entropy.csv"
        group_file = f"{data_path}_phenotype.csv"
        if not os.path.exists(signature_file):
            print(f"Warning: {signature_file} not found, skipping dataset {dataset_name}.")
            continue

        signature = pd.read_csv(signature_file)
        group = pd.read_csv(group_file)

        signature['sig_name'] = signature["node1"] + "_" + signature["node2"]
        signature.drop(columns=["node1", "node2"], inplace=True)
        signature.set_index("sig_name", inplace=True)

        # entropy_5000
        signature_5000, _ = HVGs(signature,base_name = dataset_name,output_dir="Processed Data")
        group = group[[data_obj.get("sample_name"), data_obj.get("group")]]
        #FC
        signature_5000_FC = preprocess_signature(signature_5000,group,base_name = dataset_name,output_dir="Processed Data")
        sub_signature_5000_FC = signature_5000_FC[data_obj.get("perturbation")]

        sub_signature_5000_FC['names']= sub_signature_5000_FC['names'].str.upper()


        sub_signature_5000_FC.drop(columns=["group"], inplace=True)

        #break to single gene
        sub_signature_5000_FC["node1"] = sub_signature_5000_FC['names'].str.split("_").str[0]
        sub_signature_5000_FC["node2"] = sub_signature_5000_FC['names'].str.split("_").str[1]
        sub_signature_5000_FC.drop(columns=["names"], inplace=True)
        single_gene = entropy2gene(sub_signature_5000_FC,base_name = dataset_name,output_dir="Processed Data")
        single_gene.reset_index(inplace=True)
        benchmark_parallel(signature=single_gene, library=library,output_dir=f"{output_dir}/{dataset_name}",rep_n = 3, perm_list = [500,1000,2000])  
  

run_entropy_5000_FC_sg_benchmark(data_list,library_path="data/Enrichr.KEGG_2019_Mouse.gmt",output_dir="result_entropy_5000_FC_sg")

def run_entropy_sg_5000_FC_benchmark(data_list,library_path,output_dir="result"):
    library = read_gmt(library_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for data_obj in data_list:
        dataset_name = data_obj.get("dataset_name")
        data_path = data_obj.get("path")

        print(f"Processing dataset: {dataset_name}")

        signature_file = f"{data_path}_entropy.csv"
        group_file = f"{data_path}_phenotype.csv"
        if not os.path.exists(signature_file):
            print(f"Warning: {signature_file} not found, skipping dataset {dataset_name}.")
            continue

        signature = pd.read_csv(signature_file)
        group = pd.read_csv(group_file)

        #break to single gene
        single_gene = entropy2gene(signature,base_name = dataset_name,output_dir="Processed Data")

        # sg_5000
        single_gene_5000, _ = HVGs(single_gene,base_name = dataset_name,output_dir="Processed Data")

        #FC
        group = group[[data_obj.get("sample_name"), data_obj.get("group")]]
        single_gene_5000_FC = preprocess_signature(single_gene_5000,group,base_name = dataset_name,output_dir="Processed Data")
        sub_single_gene_5000_FC = single_gene_5000_FC[data_obj.get("perturbation")]
        sub_single_gene_5000_FC['names'] = sub_single_gene_5000_FC ['names'].str.upper()

        sub_single_gene_5000_FC.drop(columns=["group"], inplace=True)

        benchmark_parallel(signature=sub_single_gene_5000_FC, library=library,output_dir=f"{output_dir}/{dataset_name}",rep_n = 3, perm_list = [500,1000,2000])  


run_entropy_sg_5000_FC_benchmark(data_list,library_path="data/Enrichr.KEGG_2019_Mouse.gmt",output_dir="result_entropy_sg_5000_FC")

def align_entropy_FC_benchmark(data_list,library_path,output_dir="result"):
    library = read_gmt(library_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for data_obj in data_list:
        dataset_name = data_obj.get("dataset_name")
        data_path = data_obj.get("path")

        print(f"Processing dataset: {dataset_name}")

        signature_file = f"{data_path}_entropy.csv"
        group_file = f"{data_path}_phenotype.csv"
        exp_file = f"{data_path}_expr.csv"
        if not os.path.exists(signature_file):
            print(f"Warning: {signature_file} not found, skipping dataset {dataset_name}.")
            continue

        signature = pd.read_csv(signature_file)
        group = pd.read_csv(group_file)
        exp = pd.read_csv(exp_file,index_col=0)

        # exp_5000
        _, gene_list = HVGs(exp,base_name = dataset_name,output_dir="Processed Data")
        #align
        entropy_5000 = entropy_HVGs(signature,gene_list,base_name = dataset_name,output_dir="Processed Data")
        #FC
        entropy_5000['sig_name'] = entropy_5000["node1"] + "_" + entropy_5000["node2"]
        entropy_5000.drop(columns=["node1", "node2"], inplace=True)
        entropy_5000.set_index("sig_name", inplace=True)
        group = group[[data_obj.get("sample_name"), data_obj.get("group")]]
        signature_5000_FC = preprocess_signature(entropy_5000,group,base_name = dataset_name,output_dir="Processed Data")
        sub_signature_5000_FC = signature_5000_FC[data_obj.get("perturbation")]
        sub_signature_5000_FC.drop(columns=["group"], inplace=True)

        benchmark_parallel(signature=sub_signature_5000_FC, library=library,output_dir=f"{output_dir}/{dataset_name}")  

align_entropy_FC_benchmark(data_list,library_path="DATA/Enrichr.KEGG_2021_Human.gmt",output_dir="align_entropy_5000_logfc")

def align_entropy_sg_FC_benchmark(data_list,library_path,output_dir="result"):
    library = read_gmt(library_path)
    sub_library = {}
    for i, key in enumerate(library.keys()):
        if i > 20:
            break
        sub_library[key] = library[key]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for data_obj in data_list:
        dataset_name = data_obj.get("dataset_name")
        data_path = data_obj.get("path")

        print(f"Processing dataset: {dataset_name}")

        signature_file = f"{data_path}_entropy.csv"
        group_file = f"{data_path}_phenotype.csv"
        exp_file = f"{data_path}_expr.csv"
        if not os.path.exists(signature_file):
            print(f"Warning: {signature_file} not found, skipping dataset {dataset_name}.")
            continue

        signature = pd.read_csv(signature_file)
        group = pd.read_csv(group_file)

        #-------------------

        # exp = pd.read_csv(exp_file,index_col=0)

        exp = pd.read_csv(exp_file)

        # 把第二列设为 index
        exp.index = exp.iloc[:, 1]

        # 丢掉前两列，只保留后面的表达矩阵
        exp = exp.iloc[:, 2:]

        #-------------------


        # exp_5000
        _, gene_list = HVGs(exp,base_name = dataset_name,output_dir="Processed Data")
        #align
        entropy_5000 = entropy_HVGs(signature,gene_list,base_name = dataset_name,output_dir="Processed Data")
        
        #break to single gene
        entropy_5000.reset_index(inplace=True)
        single_gene = entropy2gene(entropy_5000,base_name = dataset_name,output_dir="Processed Data")

        #FC
        group = group[[data_obj.get("sample_name"), data_obj.get("group")]]
        single_gene_FC = preprocess_signature(single_gene,group,base_name = dataset_name,output_dir="Processed Data")
        sub_single_gene_FC = single_gene_FC[data_obj.get("perturbation")]
        sub_single_gene_FC['names'] = sub_single_gene_FC['names'].str.upper()
        sub_single_gene_FC.drop(columns=["group"], inplace=True)

        benchmark_parallel(signature=sub_single_gene_FC, library=library,output_dir=f"{output_dir}/{dataset_name}",rep_n = 3, perm_list = [500,1000,2000])  
  
align_entropy_sg_FC_benchmark(data_list,library_path="data/Enrichr.KEGG_2019_Mouse.gmt",output_dir="align_entropy_sg_FC")
