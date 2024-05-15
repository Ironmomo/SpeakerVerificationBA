import pandas as pd
import matplotlib.pyplot as plt

# Path to csv data
results = '/home/fassband/ba/SpeakerVerificationBA/finetuning/exp/finetuned-20240514-111049-original-base-f128-t2-b128-lr1e-4-m390-finetuning_avg-asli/result.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(results, header=None)

# Selecting columns where the values in the second column are not equal to 0
train_loss_avg = df[df.iloc[:, 1] != 0][0]
test_loss_avg = df[df.iloc[:, 1] != 0][1]

plt.plot(df[0])

# Plotting
plt.figure()
plt.plot(train_loss_avg, label='Train Loss Avg')
plt.plot(test_loss_avg, label='Test Loss Avg')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()
