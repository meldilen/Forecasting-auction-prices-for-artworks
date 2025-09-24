import pandas as pd
import numpy as np
import re
from datetime import datetime
import os

def extract_price(price_str):
    """Extract numeric price from string format like '28.500 USD'"""
    if pd.isna(price_str):
        return np.nan
    price_clean = re.sub(r'[^\d.]', '', str(price_str))
    try:
        return float(price_clean) if price_clean else np.nan
    except:
        return np.nan

def extract_year(year_str):
    """Extract year from various formats including textual descriptions"""
    if pd.isna(year_str):
        return np.nan
    
    year_str = str(year_str).strip()
    
    year_match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', year_str)
    if year_match:
        return int(year_match.group())
    
    year_str_lower = year_str.lower()
    period_mapping = {
        'first half 20th century': 1925,
        'second half 20th century': 1975,
        'early 20th century': 1910,
        'mid 20th century': 1950,
        'late 20th century': 1980,
        '19th century': 1850,
        '18th century': 1750,
        'contemporary': 2010,
    }
    
    for pattern, approx_year in period_mapping.items():
        if pattern in year_str_lower:
            return approx_year
    
    return np.nan

def categorize_signature(signed_str):
    return 0 if pd.isna(signed_str) else 1

def extract_lifespan(years_str):
    """Extract birth and death years from string like '1884 - 1920'"""
    if pd.isna(years_str):
        return np.nan, np.nan
    years_match = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', str(years_str))
    if len(years_match) >= 2:
        return int(years_match[0]), int(years_match[1])
    elif len(years_match) == 1:
        return int(years_match[0]), np.nan
    return np.nan, np.nan

def calculate_lifespan(birth_year, death_year):
    """Calculate lifespan only if both years are available"""
    if pd.isna(birth_year) or pd.isna(death_year):
        return np.nan
    return death_year - birth_year

def calculate_years_since_death(death_year):
    """Calculate years since death only if death year is available"""
    if pd.isna(death_year):
        return np.nan
    return datetime.now().year - death_year

def extract_avg_dimension(dimensions_str):
    """Extract average dimension from string format"""
    if pd.isna(dimensions_str):
        return np.nan
    
    cm_matches = re.findall(r'(\d+\.?\d*)\s*cm', str(dimensions_str), re.IGNORECASE)
    if cm_matches:
        numeric_dims = [float(x) for x in cm_matches]
        return np.mean(numeric_dims) if numeric_dims else np.nan
    
    inch_matches = re.findall(r'(\d+\.?\d*)\s*"', str(dimensions_str))
    if inch_matches:
        numeric_dims = [float(x) * 2.54 for x in inch_matches]
        return np.mean(numeric_dims) if numeric_dims else np.nan
    
    return np.nan

def categorize_size(avg_dim):
    if pd.isna(avg_dim):
        return 'Unknown'
    elif avg_dim < 30:
        return 'Small'
    elif avg_dim < 100:
        return 'Medium'
    else:
        return 'Large'

def clean_market_price(price_str):
    """Convert market prices like '$2.2K' to numeric USD"""
    if pd.isna(price_str):
        return np.nan
    
    price_str = str(price_str).upper().replace('$', '').replace(',', '')
    
    multiplier = 1
    if 'K' in price_str:
        multiplier = 1000
        price_str = price_str.replace('K', '')
    elif 'M' in price_str:
        multiplier = 1000000
        price_str = price_str.replace('M', '')
    
    try:
        return float(price_str) * multiplier
    except:
        return np.nan

def main():
    """Main function to execute preprocessing pipeline"""
    
    # Load datasets
    art_price_df = pd.read_csv('../../data/processed/artDataset.csv')
    artists_df = pd.read_csv('../../data/processed/artists.csv')
    artists_bat_df = pd.read_csv('../../data/processed/artistsBAT.csv')
    artworks_df = pd.read_csv('../../data/processed/artworks.csv')
    market_df = pd.read_csv('../../data/processed/all_artists_artworks.csv')

    # Step 1: Preprocess Art Price Dataset
    columns_to_drop = []
    if 'Unnamed: 0' in art_price_df.columns:
        columns_to_drop.append('Unnamed: 0')
    if 'condition' in art_price_df.columns:
        columns_to_drop.append('condition')

    if columns_to_drop:
        art_price_df = art_price_df.drop(columns_to_drop, axis=1)

    art_price_df['price_usd'] = art_price_df['price'].apply(extract_price)
    art_price_df = art_price_df.drop('price', axis=1)

    art_price_df['creation_year'] = art_price_df['yearCreation'].apply(extract_year)
    art_price_df = art_price_df.drop('yearCreation', axis=1)

    art_price_df['signed'] = art_price_df['signed'].str.strip().replace('[nan]', np.nan)
    art_price_df['signed_binary'] = art_price_df['signed'].apply(categorize_signature)
    art_price_df = art_price_df.drop('signed', axis=1)

    art_price_df['period'] = art_price_df['period'].str.strip().replace('[nan]', np.nan)
    art_price_df['movement'] = art_price_df['movement'].str.strip()

    # Step 2: Preprocess Artists Dataset
    artists_df.columns = [col.strip().lower().replace(' ', '_') for col in artists_df.columns]
    artists_df_clean = artists_df[['artist_id', 'name']].copy()
    artists_df_clean['name'] = artists_df_clean['name'].str.strip().str.title()

    # Step 3: Preprocess Artists BAT Dataset
    artists_bat_df[['birth_year', 'death_year']] = pd.DataFrame(
        artists_bat_df['years'].apply(extract_lifespan).tolist(), 
        index=artists_bat_df.index
    )

    artists_bat_df['lifespan'] = artists_bat_df.apply(
        lambda row: calculate_lifespan(row['birth_year'], row['death_year']), 
        axis=1
    )

    artists_bat_df['years_since_death'] = artists_bat_df['death_year'].apply(calculate_years_since_death)
    artists_bat_df['is_living'] = artists_bat_df['death_year'].isna()
    artists_bat_df['name'] = artists_bat_df['name'].str.strip().str.title()

    useful_columns = ['name', 'lifespan', 'years_since_death', 'is_living', 'paintings']
    artists_bat_clean = artists_bat_df[useful_columns].copy()

    # Step 4: Preprocess Artworks Dataset
    artworks_df.columns = [col.strip().lower().replace(' ', '_') for col in artworks_df.columns]
    artworks_df['avg_dimension_cm'] = artworks_df['dimensions'].apply(extract_avg_dimension)
    artworks_df['size_category'] = artworks_df['avg_dimension_cm'].apply(categorize_size)

    essential_columns = ['artist_id', 'title', 'avg_dimension_cm', 'size_category']
    artworks_df = artworks_df[essential_columns].copy()

    # Step 5: Preprocess Market Dataset
    market_df['purchase_price_usd'] = market_df['purchase_price'].apply(clean_market_price)
    market_df['sale_price_usd'] = market_df['sale_price'].apply(clean_market_price)

    market_df['appreciation_multiplier'] = np.where(
        (market_df['purchase_price_usd'].notna()) & 
        (market_df['sale_price_usd'].notna()) & 
        (market_df['purchase_price_usd'] > 0),
        market_df['sale_price_usd'] / market_df['purchase_price_usd'],
        np.nan
    )

    market_df['artist'] = market_df['artist'].str.strip().str.title()

    columns_to_drop = [
        'description', 'purchase_price', 'sale_price', 'gross_appreciation_multiplier',
        'gross_appreciation_period', 'url', 'has_image', 'image_url'
    ]

    market_df_clean = market_df.drop(columns=columns_to_drop, errors='ignore')

    # Step 6: Create Final Merged Dataset
    final_df = art_price_df.copy()
    final_df['artist'] = final_df['artist'].str.strip().str.title()

    # Merge datasets
    final_df = final_df.merge(
        artists_df_clean[['name', 'artist_id']],
        left_on='artist',
        right_on='name',
        how='left'
    )

    final_df['artist_id'] = final_df['artist_id'].astype(str)
    artworks_df['artist_id'] = artworks_df['artist_id'].astype(str)

    artworks_aggregated = artworks_df.groupby('artist_id').agg({
        'avg_dimension_cm': 'mean',
        'size_category': lambda x: x.mode()[0] if not x.mode().empty else np.nan
    }).reset_index()

    final_df = final_df.merge(artworks_aggregated, on='artist_id', how='left')

    final_df = final_df.merge(
        artists_bat_clean[['name', 'lifespan', 'years_since_death', 'is_living', 'paintings']],
        left_on='artist',
        right_on='name',
        how='left'
    )

    artist_market_stats = market_df_clean.groupby('artist').agg({
        'purchase_price_usd': 'mean',
        'sale_price_usd': 'mean',
        'appreciation_multiplier': 'mean'
    }).reset_index()

    final_df = final_df.merge(artist_market_stats, on='artist', how='left')

    # Create artist popularity based on MoMA collection
    artist_moma_count = artworks_df['artist_id'].value_counts().reset_index()
    artist_moma_count.columns = ['artist_id', 'moma_artwork_count']
    artist_moma_count['artist_id'] = artist_moma_count['artist_id'].astype(str)

    # Select final columns
    final_columns = [
        'artist_id', 'artist', 'title', 'price_usd', 'creation_year',
        'period', 'movement', 'size_category',
        'avg_dimension_cm', 'lifespan', 'years_since_death', 'paintings',
        'purchase_price_usd', 'sale_price_usd', 'appreciation_multiplier', 
        'moma_artwork_count', 'signed_binary', 'is_living'
    ]

    available_columns = [col for col in final_columns if col in final_df.columns]
    final_df = final_df[available_columns]

    # Save final dataset
    output_path = "../../data/processed/final_art_dataset.csv"
    final_df.to_csv(output_path, index=False)
    
    print("Preprocessing completed successfully!")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Dataset saved to: {output_path}")

if __name__ == "__main__":
    main()