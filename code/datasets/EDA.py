import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8')

def combine_artist_datasets(file_paths):
    """
    Combine all individual artist datasets into one unified dataframe
    """
    all_dfs = []
    
    for file_path in file_paths:
        try:
            file_name = os.path.basename(file_path)
            df = pd.read_csv(file_path)
            
            artist_name = file_name.replace('artworks_', '').replace('.csv', '').replace('_', ' ').title()
            df['artist'] = artist_name
            
            df.columns = [col.lower().strip() for col in df.columns]
            all_dfs.append(df)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def analyze_dataset(file_name):
    """
    Comprehensive analysis of a single dataset
    """
    try:
        df = pd.read_csv(file_name)
        return df
    except Exception as e:
        print(f"ERROR reading {file_name}: {e}")
        return None

def main():
    """Main function to execute EDA pipeline"""
    
    # Dataset Configuration
    file_list = [
        'artDataset.csv',
        'artists.csv', 
        'artistsBAT.csv',
        'artworks.csv'
    ]

    artist_dfs = [
        'artworks_andy_warhol.csv',
        'artworks_claude_monet.csv', 
        'artworks_david_hockney.csv',
        'artworks_gerhard_richter.csv',
        'artworks_gerhard_richter1.csv',
        'artworks_jean_michel_basquiat.csv',
        'artworks_mark_rothko.csv',
        'artworks_piccaso4.csv',
        'artworks_zao_wou_ki.csv'
    ]

    base_path = "../../data/row/"
    artist_files_full = [os.path.join(base_path, file_name) for file_name in artist_dfs]

    # Combine artist datasets
    combined_artists_df = combine_artist_datasets(artist_files_full)
    
    # Save combined dataset
    output_path = os.path.join("../../data/processed/", "all_artists_artworks.csv")
    combined_artists_df.to_csv(output_path, index=False)
    
    file_list.append("all_artists_artworks.csv")
    file_list_with_path = [os.path.join(base_path, file_name) for file_name in file_list]

    # Analyze all datasets
    dataframes = {}
    for file_name in file_list_with_path:
        df = analyze_dataset(file_name)
        if df is not None:
            key_name = file_name.replace('.csv', '').replace("../../data/processed/", '')
            dataframes[key_name] = df
            
            # Save processed version
            output_filename = f"{os.path.basename(file_name)}"
            output_path = os.path.join("../../data/processed/", output_filename)
            df.to_csv(output_path, index=False)
    
    print("EDA completed successfully!")
    print(f"Processed {len(dataframes)} datasets")

if __name__ == "__main__":
    main()