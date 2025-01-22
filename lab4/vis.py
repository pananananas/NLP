import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

colors={
    "block_output": "Purples",
    "mlp_activation": "Greens",
    "attention_output": "Reds"
} 

def create_enhanced_plot(df, stream, question, ground_truth, input_ids, tokenizer, output_prefix):
    """Create an enhanced visualization showing both Yes/No probabilities and token effects"""
    # Decode all tokens for y-axis labels
    tokens = [tokenizer.decode(tid) for tid in input_ids['input_ids'][0]]
    
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Calculate probability changes from base
    df['yes_change'] = df.apply(lambda x: x['prob_yes'] - x['base_prob_yes'], axis=1)
    df['no_change'] = df.apply(lambda x: x['prob_no'] - x['base_prob_no'], axis=1)
    
    # Get max absolute change for consistent color scaling
    # max_change = max(
    #     abs(df['yes_change']).max(),
    #     abs(df['no_change']).max()
    # )
    
    # Plot Yes probability changes
    ax1 = plt.subplot(gs[0])
    sns.heatmap(
        df.pivot(index='pos', columns='layer', values='yes_change'),
        cmap=colors[stream],
        center=0,
        vmin=0,
        vmax=1,
        ax=ax1,
        yticklabels=tokens,
        cbar_kws={'label': 'Change in Yes Probability'}
    )
    ax1.set_title(f'Yes Probability Changes\nBase: {df["base_prob_yes"].iloc[0]:.4f}')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Token')
    
    # Plot No probability changes
    ax2 = plt.subplot(gs[1])
    sns.heatmap(
        df.pivot(index='pos', columns='layer', values='no_change'),
        cmap=colors[stream],  # Reversed colormap for No
        center=0,
        vmin=0,
        vmax=1,
        ax=ax2,
        yticklabels=tokens,
        cbar_kws={'label': 'Change in No Probability'}
    )
    ax2.set_title(f'No Probability Changes\nBase: {df["base_prob_no"].iloc[0]:.4f}')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Token')
    
    # Mark noised positions
    noised_positions = df[df['noised']]['pos'].unique()
    for ax in [ax1, ax2]:
        for pos in noised_positions:
            ax.axhline(y=pos, color='black', linestyle='--', alpha=0.3)
            ax.axhline(y=pos+1, color='black', linestyle='--', alpha=0.3)
    
    # Add overall title
    plt.suptitle(
        f"Activation Analysis for Question: '{question}'\n"
        f"Stream: {stream}, Ground Truth: {ground_truth}",
        fontsize=12
    )
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_prefix}_{stream}_plot.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Create a separate plot for probability changes in noised positions
    if len(noised_positions) > 0:
        plt.figure(figsize=(12, 6))
        noised_df = df[df['noised']]
        
        # Plot both Yes and No changes
        plt.plot(noised_df.groupby('layer')['yes_change'].mean(), 
                label='Yes', color='green', marker='o')
        plt.plot(noised_df.groupby('layer')['no_change'].mean(), 
                label='No', color='red', marker='o')
        
        plt.title('Average Probability Changes at Noised Positions')
        plt.xlabel('Layer')
        plt.ylabel('Change in Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{stream}_changes.png", bbox_inches='tight', dpi=300)
        plt.close()

        plt.show()
