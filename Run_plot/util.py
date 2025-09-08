import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def process_file(file_path, enrichment_method, i, preprocess_folder, disease_folder, column_mapping, sep=","):
    df = pd.read_csv(file_path)

    if "Lead_genes" in df.columns:
        df.rename(columns={"Lead_genes": "leading_edge"}, inplace=True)

    if "leading_edge" in df.columns:
        df['leading_edge_n'] = df['leading_edge'].apply(
            lambda x: len(x.split(sep)) if isinstance(x, str) and sep in x else 0
        )

    mapping = column_mapping.get(enrichment_method, column_mapping["default"])
    df.rename(columns=mapping, inplace=True)

    df['rep'] = i
    df['Preprocess_Method'] = preprocess_folder
    df['Disease'] = disease_folder
    return df

def collect_data(
    disease_folders,
    preprocess_folders,
    file_patterns,
    column_mapping,
    base_data_dir="./result",
    skip_methods_for_entropy=("blitz", "gseapy"),
    method_mode="full"  # full: preprocess+method, simple: method only
):
    all_data = []

    for preprocess_folder in preprocess_folders:
        for disease_folder in disease_folders:
            base_path = os.path.join(base_data_dir, preprocess_folder, disease_folder)
            if not os.path.exists(base_path):
                continue

            for enrichment_method, files in file_patterns.items():

                if preprocess_folder in ["align_entropy_5000_logfc", "result_entropy_5000_logfc"]:
                    if enrichment_method in skip_methods_for_entropy:
                        continue

                for i, file in enumerate(files, start=1):
                    file_path = os.path.join(base_path, file)
                    if not os.path.exists(file_path):
                        continue

                    df = process_file(file_path, enrichment_method, i, preprocess_folder, disease_folder, column_mapping)
                    
                    if method_mode == "full":
                        df["Method"] = f"{preprocess_folder}_{enrichment_method}"
                    elif method_mode == "simple":
                        df["Method"] = enrichment_method
                    else:
                        raise ValueError("Unknown method_mode")
                    all_data.append(df)

    if not all_data:
        raise ValueError("No data collected.")

    return pd.concat(all_data, ignore_index=True)


def compute_variance_and_plot(
    disease_folders,
    file_patterns,
    preprocess_folders,
    column_mapping,
    base_data_dir="./result",
):

    final_df = collect_data(
        disease_folders, preprocess_folders, file_patterns, column_mapping, base_data_dir
    )
    results_var = []

    for (method, disease), group in final_df.groupby(['Method', 'Disease']):
        for perm_num, group_perm in group.groupby('Perm_num') if 'Perm_num' in group.columns else [(1, group)]:
            wide_df = group_perm.pivot(index='Term', columns='rep', values='pval')
            cor = wide_df.corr(method='spearman')
            np.fill_diagonal(cor.values, np.nan)
            var = 1 - cor.mean()
            mean = cor.mean()
            n = wide_df.shape[1]
            results_var.append({
                "Method": method,
                "Disease": disease,
                "Perm_num": perm_num,
                "Var": var,
                "Mean": mean,
                "n": n,
            })

    data = []
    for entry in results_var:
        for value in entry["Var"]:
            data.append({
                "Method": entry["Method"],
                "Disease": entry["Disease"],
                "Perm_num": entry["Perm_num"],
                "Var": value,
                "Mean": entry["Mean"],
                "n": entry["n"]
            })

    df = pd.DataFrame(data)
    return df

def load_and_compute_average_correlation_filtered(
    disease_folders,
    preprocess_folders,
    file_patterns,
    column_mapping,
    short_method_order,
    method_name_mapping,
    base_data_dir="./result",
):

    final_df = collect_data(
        disease_folders, preprocess_folders, file_patterns, column_mapping, base_data_dir,method_mode="simple"
    )
    final_df['combo'] = final_df['Preprocess_Method'] + "_" + final_df['Method']

    correlation_matrices = []

    for rep in final_df["rep"].unique():
        for perm_num in final_df["Perm_num"].unique():
            subset = final_df[(final_df["rep"] == rep) & (final_df["Perm_num"] == perm_num)]
            subset = subset[['Term', 'pval', 'Preprocess_Method', 'Method', 'combo']]
            subset['-logp'] = -np.log10(subset['pval'])
            df_correlation = subset.pivot_table(index="Term", columns="combo", values="-logp", aggfunc="first")
            df_correlation = df_correlation.fillna(1)
            ordered_cols = [col for col in method_name_mapping.keys() if col in df_correlation.columns]
            df_correlation = df_correlation[ordered_cols]
            correlation_matrix = df_correlation.corr(method='spearman')
            correlation_matrices.append(correlation_matrix)

    if not correlation_matrices:
        raise ValueError("No valid data found to calculate correlations.")

    average_correlation_matrix = np.mean(correlation_matrices, axis=0)
    correlation_df = pd.DataFrame(average_correlation_matrix, index=short_method_order, columns=short_method_order)

    return correlation_df

def sort_key(col):
    if col.startswith('grea'):
        return (0, col)
    elif 'blitz' in col:
        return (1, col)
    elif 'gseapy' in col:
        return (2, col)
    else:
        return (3, col)
    
def load_and_final_df(
    disease_folders,
    preprocess_folders,
    file_patterns,
    column_mapping,
    short_method_order,
    method_name_mapping,
    base_data_dir="./result",
):

    final_df = collect_data(
        disease_folders, preprocess_folders, file_patterns, column_mapping, base_data_dir,method_mode="simple"
    )
    final_df['combo'] = final_df['Preprocess_Method'] + "_" + final_df['Method']

    final_df["log_pval"] = -np.log10(final_df["pval"]) 
    p_value_acc = final_df[['Term','pval','Preprocess_Method','rep','Method','log_pval','Perm_num']]

    p_value_acc['combo'] = p_value_acc['Preprocess_Method']  + "_" + p_value_acc['Method']

    idx_min_pval = p_value_acc.groupby('combo')['pval'].idxmin()
    p_value_acc_min = p_value_acc.loc[idx_min_pval].copy()

    select_term = p_value_acc_min['Term'].unique()


    p_value_acc_filter = p_value_acc[p_value_acc['Term'].isin(select_term)]

    result_df = p_value_acc_filter.set_index('Term')[['combo', 'pval']].pivot_table(
        index='Term',
        columns='combo',
        values='pval'
    )

    custom_order = sorted(result_df.columns, key=sort_key)


    sorted_df = result_df[custom_order]


    columns_to_use = [col for col in sorted_df.columns if 'blitz' not in col and 'gseapy' not in col]
    filtered_df = sorted_df[columns_to_use]

    term_sum = filtered_df.sum(axis=1)

    sorted_terms = term_sum.sort_values(ascending=True).index
    sorted_df = sorted_df.loc[sorted_terms]

    sorted_df.columns = sorted_df.columns.map(method_name_mapping)
    sorted_df = sorted_df[short_method_order]

    log_df = -np.log10(sorted_df+1e-10)



    return log_df


def data_input( disease_folders,
    file_patterns,
    preprocess_folders,
    column_mapping,
    base_data_dir="./result",
    top_n=50):
    
    final_df = collect_data(
        disease_folders, preprocess_folders, file_patterns, column_mapping, base_data_dir, method_mode="simple"
    )
    final_df['Tool'] = final_df['Preprocess_Method'] + "_" + final_df['Method']
    final_df['combo'] = final_df['Preprocess_Method'] + "_" + final_df['Method']
    pathway_filter = final_df[['Tool','enrichment_signal','normalized_enrichment_signal','Method', 'pval','Term','Disease']]
    pathway_filter = pathway_filter.groupby(['Tool','Term','Method','Disease']).mean().reset_index()
    pathway_filter = pathway_filter[pathway_filter['pval'] <0.05]
    result = pathway_filter.groupby('Tool', group_keys=False).apply(lambda group: group.sort_values('enrichment_signal', ascending=False).head(top_n))
    return result


def plot_variance_for_dataset(
    df,
    method_name_mapping,
    short_method_order,
    label_colors,
    dataset_name,
    output_dir="p_value_bench_figures",
):

    df["Short_Method"] = df["Method"].map(method_name_mapping)
    df["Short_Method"] = pd.Categorical(df["Short_Method"], categories=short_method_order, ordered=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    macaron_palette = [
        (1.0, 0.7, 0.7), (1.0, 0.9, 0.7),
        (0.7, 1.0, 0.8), (0.7, 0.9, 1.0),
        (0.85, 0.7, 1.0)
    ]

    plt.figure(figsize=(30, 10))
    sns.boxplot(
        data=df,
        x="Short_Method",
        y="Var",
        hue="Perm_num",
        palette=macaron_palette,
        order=short_method_order
    )

    plt.title(f"Variance - Boxplot for {dataset_name} P-values", fontsize=20)
    plt.xlabel("Method")
    plt.ylabel("Variance")
    plt.xticks(rotation=90)

    ax = plt.gca()
    xtick_labels = ax.get_xticklabels()
    for label in xtick_labels:
        method_name = label.get_text()
        if method_name in label_colors:
            font_color, bg_color = label_colors[method_name]
            label.set_color(font_color)
            label.set_backgroundcolor(bg_color)

    output_path = os.path.join(output_dir, f"boxplot_{dataset_name}")
    plt.tight_layout()
    plt.savefig(f"{output_path}.pdf", format="pdf")

    plt.show()

    return output_path


def plot_correlation_heatmap(
    correlation_df,
    short_method_order,
    label_colors,
    output_dir="p_value_bench_figures",
    dataset_name=""
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    correlation_df = correlation_df.loc[short_method_order, short_method_order]

    col_methods = correlation_df.columns
    row_methods = correlation_df.index
    col_colors = col_methods.map(lambda m: label_colors.get(m, ('black', '#DDDDDD'))[1])
    row_colors = row_methods.map(lambda m: label_colors.get(m, ('black', '#DDDDDD'))[1])

    g = sns.clustermap(
    correlation_df,
    annot=True,
    # fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"shrink": 0.2},
    cbar_pos=(-0.1, .3, .03, .5),
    linecolor="gray",
    annot_kws={"size": 3},
    figsize=(15,15),
    # row_cluster=False,
    # col_cluster=False,
    dendrogram_ratio=(0.1, 0.1),
    col_colors=col_colors,
    row_colors=row_colors,
)

    plt.suptitle(f"Average Spearman Correlation Heatmap of P-values({dataset_name})", fontsize=16, y=1.02)

    ax = g.ax_heatmap
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)

    xtick_labels = ax.get_xticklabels()
    ytick_labels = ax.get_yticklabels()
    for label in xtick_labels:
        method_name = label.get_text()
        if method_name in label_colors:
            font_color, bg_color = label_colors[method_name]
            label.set_color(font_color)
            label.set_backgroundcolor(bg_color)
    for label in ytick_labels:
        method_name = label.get_text()
        if method_name in label_colors:
            font_color, bg_color = label_colors[method_name]
            label.set_color(font_color)
            label.set_backgroundcolor(bg_color)

    # png_path = os.path.join(output_dir, f"average_spearman_correlation_heatmap_{dataset_name}.png")
    pdf_path = os.path.join(output_dir, f"average_spearman_correlation_heatmap_{dataset_name}.pdf")

    plt.tight_layout()
    # plt.savefig(png_path, format="png", bbox_inches="tight", dpi=600)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=600)
    plt.show()

    return 


def plot_min_heatmap(log_df,label_colors,dataset_name="",output_dir="p_value_bench_figures"):
    mask = log_df.T.isna()

    plt.figure(figsize=(20, 16),dpi=600)

    cmap = sns.cubehelix_palette(as_cmap=True)
    cmap.set_bad(color='lightgray')  

    sns.heatmap(
        log_df.T,
        mask=mask, 
        # annot=True,
        fmt=".4f",
        cmap=cmap,
        cbar_kws={"shrink": 0.8, "label": "-log10(p-value)"},
        linewidths=0.5,
        linecolor="gray",
        vmin=0,      
        vmax = 3, 
    )

    plt.xticks(fontsize=7, rotation=75)


    plt.yticks(fontsize=8, rotation=0)


    ax = plt.gca()
    ytick_labels = ax.get_yticklabels()

    for label in ytick_labels:
        method_name = label.get_text()
        if method_name in label_colors:
            font_color, bg_color = label_colors[method_name]
            label.set_color(font_color)                
            label.set_backgroundcolor(bg_color)        

    plt.title(f"Maximum -log10(p-value) of Each Method for pathway({dataset_name})", fontsize=14)
    plt.xlabel("Pathway")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Maximum_-log10(p-value)_of_Each_Method_for_pathway_{dataset_name}.pdf", format="pdf")

    plt.show()

