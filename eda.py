"""
Author: Alex (Tai-Jung) Chen

This code visualize the distributions of the features.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
	"""
	Call out the distribution show function.
	"""
	feature = "air.temp"
	include_sub = True
	integrate = True

	show_distribution(feature, include_sub, integrate)


def show_distribution(feature, include_sub, integrate):
	"""
	Visualize the given feature's distribution based on the modeling type and plot type

	:param feature: The feature to be visualized
	:type feature: String

	:param include_sub: Whether the minority class is broken into subclasses or not.
	:type include_sub: Boolean

	:param integrate: Whether to plot the feature by labels, or integrate them into one single plot.
	:type integrate: Boolean
	"""
	# read data
	df = pd.read_csv("datasets/preprocessed/mpmc.csv")

	if include_sub:
		df_pass = df[df['failure.type'] == 0]
		df_fail_1 = df[df['failure.type'] == 1]
		df_fail_2 = df[df['failure.type'] == 2]
		df_fail_3 = df[df['failure.type'] == 3]
		df_fail_4 = df[df['failure.type'] == 4]
		df_fail_5 = df[df['failure.type'] == 5]

		if integrate:
			# sns.histplot(data=df, x=feature_name, bins=30, kde=True,
			# 			 label='All Samples')
			sns.histplot(data=df_pass, x=feature, bins=30, kde=True, label='No Failure', color='black')
			sns.histplot(data=df_fail_1, x=feature, bins=30, kde=True, label='Heat Dissipation', color='cyan')
			sns.histplot(data=df_fail_2, x=feature, bins=30, kde=True, label='Power Failure', color='green')
			sns.histplot(data=df_fail_3, x=feature, bins=30, kde=True, label='Overstrain Failure', color='orange')
			sns.histplot(data=df_fail_4, x=feature, bins=30, kde=True, label='Tool Wear Failure', color='brown')
			sns.histplot(data=df_fail_5, x=feature, bins=30, kde=True, label='Random Failure', color='magenta')

			plt.title(f'Distribution of {feature}')  # Set the title
			plt.ylabel('Frequency')  # Set y-axis label
			plt.xlabel(feature)  # Set x-axis label
			plt.legend()

			# Set new axis limits to zoom in
			if feature == "torque":
				# plt.xlim(0, 80)  # Set x-axis limits
				plt.ylim(0, 30)  # Set y-axis limits
			elif feature == "air.temp":
				pass
				# plt.xlim(0, 80)  # Set x-axis limits
				# plt.ylim(0, 20)  # Set y-axis limits
			plt.show()

		else:
			# All samples
			sns.histplot(data=df, x=feature, bins=30, kde=True, label='All Samples')
			plt.xlabel(feature)  # Set x-axis label
			plt.ylabel('Frequency')  # Set y-axis label
			plt.title(f'Distribution of {feature}')  # Set the title
			plt.show()

			# Pass samples
			sns.histplot(data=df_pass, x=feature, bins=30, kde=True, label='Pass Samples', color='black')
			plt.xlabel(feature)  # Set x-axis label
			plt.ylabel('Frequency')  # Set y-axis label
			plt.title(f'Distribution of {feature} on passed samples')  # Set the title
			plt.show()

			# Failure 1 samples
			sns.histplot(data=df_fail_1, x=feature, bins=30, kde=True, label='Failure 1 Samples', color='cyan')
			plt.xlabel(feature)  # Set x-axis label
			plt.ylabel('Frequency')  # Set y-axis label
			plt.title(f'Distribution of {feature} on Failure 1 samples')  # Set the title
			plt.show()

			# Failure 2 samples
			sns.histplot(data=df_fail_2, x=feature, bins=30, kde=True, label='Failure 2 Samples', color='green')
			plt.xlabel(feature)  # Set x-axis label
			plt.ylabel('Frequency')  # Set y-axis label
			plt.title(f'Distribution of {feature} on Failure 2 samples')  # Set the title
			plt.show()

			# Failure 3 samples
			sns.histplot(data=df_fail_3, x=feature, bins=30, kde=True, label='Failure 3 Samples', color='orange')
			plt.xlabel(feature)  # Set x-axis label
			plt.ylabel('Frequency')  # Set y-axis label
			plt.title(f'Distribution of {feature} on Failure 3 samples')  # Set the title
			plt.show()

			# Failure 4 samples
			sns.histplot(data=df_fail_4, x=feature, bins=30, kde=True, label='Failure 4 Samples', color='brown')
			plt.xlabel(feature)  # Set x-axis label
			plt.ylabel('Frequency')  # Set y-axis label
			plt.title(f'Distribution of {feature} on Failure 4 samples')  # Set the title
			plt.show()

			# Failure 5 samples
			sns.histplot(data=df_fail_5, x=feature, bins=30, kde=True, label='Failure 5 Samples', color='magenta')
			plt.xlabel(feature)  # Set x-axis label
			plt.ylabel('Frequency')  # Set y-axis label
			plt.title(f'Distribution of {feature} on Failure 5 samples')  # Set the title
			plt.show()

	# binary view
	else:
		df_pass = df[df['target'] == 0]
		df_fail = df[df['target'] == 1]

		if integrate:
			# All samples
			# sns.histplot(data=df, x=feature_name, bins=30, kde=True,
			# 			 label='All Samples')  # Adjust the number of bins as needed
			# Pass samples
			sns.histplot(data=df_pass, x=feature, bins=30, kde=True, label='Pass Samples', color='black')
			# Failure samples
			sns.histplot(data=df_fail, x=feature, bins=30, kde=True, label='Failed Samples', color='red')

			plt.title(f'Distribution of {feature}')  # Set the title
			plt.ylabel('Frequency')  # Set y-axis label
			plt.xlabel(feature)  # Set x-axis label
			plt.legend()

			# Set new axis limits to zoom in
			if feature == "torque":
				plt.xlim(0, 80)  # Set x-axis limits
				plt.ylim(0, 40)  # Set y-axis limits
			plt.show()

		else:
			# All samples
			sns.histplot(data=df, x=feature, bins=30, kde=True, label='All Samples')
			plt.xlabel(feature)  # Set x-axis label
			plt.ylabel('Frequency')  # Set y-axis label
			plt.title(f'Distribution of {feature}')  # Set the title
			plt.show()

			# Pass samples
			sns.histplot(data=df_pass, x=feature, bins=30, kde=True, label='Pass Samples', color='black')
			plt.xlabel(feature)  # Set x-axis label
			plt.ylabel('Frequency')  # Set y-axis label
			plt.title(f'Distribution of {feature} on passed samples')  # Set the title
			plt.show()

			# Failure samples
			sns.histplot(data=df_fail, x=feature, bins=30, kde=True, label='Failed Samples', color='red')
			plt.xlabel(feature)  # Set x-axis label
			plt.ylabel('Frequency')  # Set y-axis label
			plt.title(f'Distribution of {feature} on failure samples')  # Set the title
			plt.show()


if __name__ == "__main__":
	main()
