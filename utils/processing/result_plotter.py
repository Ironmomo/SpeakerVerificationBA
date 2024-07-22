import pandas as pd
import matplotlib.pyplot as plt

# Configure Plot
# Setting global parameters for Matplotlib to use LaTeX rendering
plt.rcParams.update({
    'text.usetex': True,  # Enable LaTeX rendering
    'text.latex.preamble': r'\usepackage{lmodern}\usepackage{amsmath}',  # Use Latin Modern font and include amsmath
    'font.family': 'serif',  # Use serif font for consistency with LaTeX document
    'font.serif': ['Latin Modern Roman'],  # Specify Latin Modern Roman
    'pdf.fonttype': 42,  # Ensures fonts are embedded as TrueType
    'savefig.dpi': 400,  # Lower DPI setting for non-text elements
    'font.size': 11,  # Adjust font size to match document (you may need to tweak this)
    'axes.labelsize': 9.0,  # Size of the x and y labels
    'axes.titlesize': 11,  # Size of the plot title
    'xtick.labelsize': 7.5,  # Size of the x-axis tick labels
    'ytick.labelsize': 7.5,  # Size of the y-axis tick labels
    'legend.fontsize': 9,  # Size of the legend font
    'figure.titlesize': 12.0  # Size of the figure's main title if any
})

# Path to csv data
results = '/home/fassband/ba/SpeakerVerificationBA/finetuning/exp/finetuned-20240529-174239-original-base-f128-t2-b128-lr1e-4-m390-finetuning_avg_v1-asli/result.csv'
# Read the CSV file into a DataFrame
df = pd.read_csv(results, header=None)

# Selecting columns where the values in the second column are not equal to 0
train_loss_avg = df[df.iloc[:, 1] != 0][0]
test_loss_avg = df[df.iloc[:, 1] != 0][1]

plt.plot(df[0])

# Plotting
plt.figure(figsize=(6.0,3.2))
plt.plot(train_loss_avg, label='Train Loss Avg')
plt.plot(test_loss_avg, label='Test Loss Avg')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

#plt.savefig("/home/fassband/ba/SpeakerVerificationBA/plots_and_audios/plots/Finetuning/eer_finetuned_model_batch128.pdf")

plt.show()
