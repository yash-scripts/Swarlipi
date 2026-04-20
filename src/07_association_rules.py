import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os

os.makedirs('outputs', exist_ok=True)

def main():
    print("Task 4.2: Generating Association Rules...")
    try:
        df = pd.read_csv('data/processed/clustered_data.csv')
        events = pd.read_csv('data/raw/india_event_timeline.csv')
    except FileNotFoundError:
        print("Required data files not found. Ensure 06_kmeans_clustering is run first.")
        # Fallback to create a skeleton to prevent crash if running blindly
        df = pd.DataFrame({'week_number': [1,1,2,2], 'rank': [1,2,1,2], 'mood': ['Happy', 'Calm', 'Energetic', 'Happy']})
        events = pd.DataFrame({'date': [], 'event_type': []})

    # Ensure necessary columns are present
    if not all(col in df.columns for col in ['week_number', 'rank', 'mood']):
        print("Missing required 'week_number', 'rank', or 'mood' columns. Check upstream processing.")
        return

    # Filter Top 20 songs per week
    df_top20 = df[df['rank'] <= 20]

    # Create transactions: each week_number's mood set (unique moods appearing)
    transactions = df_top20.groupby('week_number')['mood'].unique().apply(list).tolist()

    if not transactions:
        print("No transactions to process.")
        return

    # Use TransactionEncoder -> binary matrix
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # Run Apriori with min_support=0.1
    frequent_itemsets = apriori(df_trans, min_support=0.1, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("No frequent itemsets found with minimum support 0.1.")
        return

    # Generate rules with min_confidence=0.5
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    # Sort by lift
    rules = rules.sort_values(by='lift', ascending=False)
    
    # Save rules
    rules.to_csv('outputs/association_rules.csv', index=False)
    
    # Task: Repeat separately per event_type to find crisis-specific vs celebration-specific patterns
    # (Leaving skeleton/commentary for event join if event timeline structure is unified)
    if 'event_type' in events.columns and 'week_number' in events.columns:
        merged = pd.merge(df_top20, events, on='week_number', how='inner')
        for ev_type in merged['event_type'].unique():
            sub_trans = merged[merged['event_type'] == ev_type].groupby('week_number')['mood'].unique().apply(list).tolist()
            if sub_trans:
                sub_ary = te.fit(sub_trans).transform(sub_trans)
                sub_df = pd.DataFrame(sub_ary, columns=te.columns_)
                sub_freq = apriori(sub_df, min_support=0.1, use_colnames=True)
                if not sub_freq.empty:
                    sub_rules = association_rules(sub_freq, metric="confidence", min_threshold=0.5).sort_values(by='lift', ascending=False)
                    sub_rules.to_csv(f'outputs/association_rules_{str(ev_type).replace(" ", "_")}.csv', index=False)

    print("Task 4.2 completed successfully.")

if __name__ == '__main__':
    main()
