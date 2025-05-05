# output_processing.py
from src.commonconst import *

def ensure_plot_dir():
    """Ensure that the directory to save plots exists."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

def load_evaluation_scores():
    """Load evaluation scores from the CSV file."""
    return pd.read_csv(OUTPUT_CSV_PATH)

def plot_bar_chart(df, metric):
    """Generate and save a bar chart for a given evaluation metric."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Chatbot', y=metric, palette='coolwarm')
    plt.title(f'{metric} by Chatbot')
    plt.xlabel('Chatbot')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, f'{metric.replace(" ", "_").lower()}.png')
    plt.savefig(filename)
    plt.close()

def generate_all_bar_charts(df):
    """Generate bar charts for all defined evaluation metrics."""
    for metric in VISUALIZATION_METRICS:
        if metric in df.columns:
            plot_bar_chart(df, metric)

def generate_plots():
    """Primary function called from main.py to trigger plot generation."""
    ensure_plot_dir()
    generate_all_bar_charts(load_evaluation_scores())