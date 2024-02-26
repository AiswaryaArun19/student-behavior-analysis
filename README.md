# 1.Problem Statement
A study of the segmentation of the the intricate relationships between student behaviors and their academic performance.
#  2.Introduction
In the realm of education, understanding the intricacies of student behavior and its correlation with academic performance is crucial for effective teaching strategies and tailored interventions. This exploratory data analysis (EDA) and clustering endeavor delve into a dataset encompassing diverse aspects of student engagement and performance. The dataset includes information on demographics, nationality, educational levels, and various behavioral metrics such as raised hands, visited resources, and discussion participation.

## Dataset Overview:

Categorical Variables: The dataset covers a range of categorical variables, including gender, nationality, educational stage, grade, section, and more.
Behavioral Metrics: Numerical metrics like the frequency of raising hands, visiting resources, announcements viewed, and discussion participation offer insights into student engagement.
# 3.Methodology
The methodology encompasses a systematic approach to analyzing a dataset on student behavior and academic performance. It begins with understanding the data structure and types of variables. Exploratory Data Analysis (EDA) follows, revealing insights through visualizations and numerical summaries. Correlation analysis highlights relationships between variables. Preprocessing prepares the data for K-means clustering, identifying student groups based on behavior. Cluster analysis delves into the characteristics of each group, exploring correlations with academic performance. Derived insights lead to actionable recommendations. Further analysis considers additional factors and validates findings. The entire process is meticulously documented, culminating in a comprehensive report that informs educators and policymakers. This methodology aims to provide nuanced insights into the intricate dynamics of student behavior and academic outcomes.

### Data Analysis Setup

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```
### Reading Data into a Pandas DataFrame

```python


# df_class = pd.read_csv("/content/survey_data.csv")

df_class = pd.read_csv("KMEANS_SAMPLE.csv")
```
### Displaying the First Few Rows of the DataFrame

```python
# Use the `head()` method to display the first few rows of the DataFrame 'df_class'.
df_class.head()
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/69d68f3d-93ce-4aa0-a05a-d4e9c5197029)
### Percentage Analysis of RP-wise Distribution of Gender Data

```python
# Calculate the percentage distribution of the 'gender' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution = round(df_class["gender"].value_counts(normalize=True) * 100, 2)
percentage_distribution
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/3d76af3f-520b-4df3-9e78-564dacf16353)
### Percentage Analysis of RP-wise Distribution of Nationality Data

```python
# Calculate the percentage distribution of the 'NationalITy' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_nationality = round(df_class["NationalITy"].value_counts(normalize=True) * 100, 2)
percentage_distribution_nationality
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/54c9b036-ae09-4685-81bc-70989f515bc7)

### Percentage Analysis of RP-wise Distribution of Place of Birth Data

```python
# Calculate the percentage distribution of the 'PlaceofBirth' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_birthplace = round(df_class["PlaceofBirth"].value_counts(normalize=True) * 100, 2)
percentage_distribution_birthplace
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/29ad28f1-c15e-4b50-8981-abd78854b779)
### Percentage Analysis of RP-wise Distribution of Stage ID Data

```python
# Calculate the percentage distribution of the 'StageID' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_stage = round(df_class["StageID"].value_counts(normalize=True) * 100, 2)
percentage_distribution_stage
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/0c5235a1-4783-49cd-8aa6-aac234e5219c)
### Percentage Analysis of RP-wise Distribution of Grade ID Data

```python
# Calculate the percentage distribution of the 'GradeID' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_grade = round(df_class["GradeID"].value_counts(normalize=True) * 100, 2)
percentage_distribution_grade
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/465188fd-a5ac-4639-b3a3-f596aae1d757)
### Percentage Analysis of RP-wise Distribution of Section ID Data

```python
# Calculate the percentage distribution of the 'SectionID' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_section = round(df_class["SectionID"].value_counts(normalize=True) * 100, 2)
percentage_distribution_section
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/15f2df16-ac04-424b-a4ed-09aeadda90ca)


### Percentage Analysis of RP-wise Distribution of Topic Data

```python
# Calculate the percentage distribution of the 'Topic' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_topic = round(df_class["Topic"].value_counts(normalize=True) * 100, 2)
percentage_distribution_topic
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/7dd2b6b3-b74c-4d43-9073-d96f5491b0b8)

### Percentage Analysis of RP-wise Distribution of Semester Data

```python
# Calculate the percentage distribution of the 'Semester' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_semester = round(df_class["Semester"].value_counts(normalize=True) * 100, 2)
percentage_distribution_semester
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/8bacc939-a2fa-4faa-9eca-7bfd71476ce3)
### Percentage Analysis of RP-wise Distribution of Relation Data

```python
# Calculate the percentage distribution of the 'Relation' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_relation = round(df_class["Relation"].value_counts(normalize=True) * 100, 2)
percentage_distribution_relation
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/d26ce297-732d-4ca8-9873-a243560e4523)



### Percentage Analysis of RP-wise Distribution of Raised Hands Data

```python
# Calculate the percentage distribution of the 'raisedhands' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_raisedhands = round(df_class["raisedhands"].value_counts(normalize=True) * 100, 2)
percentage_distribution_raisedhands
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/9512e1eb-c7e3-4b09-bdef-7a1200f8ddf3)
### Percentage Analysis of RP-wise Distribution of Visited Resources Data

```python
# Calculate the percentage distribution of the 'VisITedResources' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_visited_resources = round(df_class["VisITedResources"].value_counts(normalize=True) * 100, 2)
percentage_distribution_visited_resources
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/f76197dc-3a82-4173-a03c-f5e856360a64)


### Percentage Analysis of RP-wise Distribution of Announcements View Data

```python
# Calculate the percentage distribution of the 'AnnouncementsView' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_announcements_view = round(df_class["AnnouncementsView"].value_counts(normalize=True) * 100, 2)
percentage_distribution_announcements_view
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/0809fb87-221b-4695-bcf9-48f5aed61b11)

### Percentage Analysis of RP-wise Distribution of Discussion Data

```python
# Calculate the percentage distribution of the 'Discussion' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_discussion = round(df_class["Discussion"].value_counts(normalize=True) * 100, 2)
percentage_distribution_discussion
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/173db4de-12b0-4208-bd7c-d7c5587d857c)

### Percentage Analysis of RP-wise Distribution of Parent Answering Survey Data

```python
# Calculate the percentage distribution of the 'ParentAnsweringSurvey' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_parent_answering_survey = round(df_class["ParentAnsweringSurvey"].value_counts(normalize=True) * 100, 2)
percentage_distribution_parent_answering_survey
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/8d1a9e48-60ad-4129-8b5b-0029ef6e3898)

### Percentage Analysis of RP-wise Distribution of Parent School Satisfaction Data

```python
# Calculate the percentage distribution of the 'ParentschoolSatisfaction' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_parent_school_satisfaction = round(df_class["ParentschoolSatisfaction"].value_counts(normalize=True) * 100, 2)
percentage_distribution_parent_school_satisfaction
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/bcc0e138-ecef-45b6-a523-c38a1c180b3d)

### Percentage Analysis of RP-wise Distribution of Student Absence Days Data

```python
# Calculate the percentage distribution of the 'StudentAbsenceDays' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_student_absence_days = round(df_class["StudentAbsenceDays"].value_counts(normalize=True) * 100, 2)
percentage_distribution_student_absence_days
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/a2de3f5d-6a69-48a2-8b2b-44523fa5c06d)

### Percentage Analysis of RP-wise Distribution of Class Data

```python
# Calculate the percentage distribution of the 'Class' column in the DataFrame 'df_class'.
# The result is rounded to two decimal places for clarity.

percentage_distribution_class = round(df_class["Class"].value_counts(normalize=True) * 100, 2)
percentage_distribution_class
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/4c200f9d-b50b-40e3-846b-f18188e310e3)

### Creating a Feature Matrix (X) from Selected Columns

```python
# Define a list of column names to be used as input features.
input_col = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]

# Create a feature matrix (X) by extracting the values from the specified columns in the DataFrame 'df_class'.
X = df_class[input_col].values
```
### Finding Optimal Number of Clusters using Elbow Method

```python
# Import necessary library
from sklearn.cluster import KMeans

# Initialize an empty list to store the within-cluster sum of squares (WCSS)
wcss = []

# Try different values of k (number of clusters)
for k in range(1, 11):
    # Create a KMeans instance with the current value of k
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    
    # Fit the model to the feature matrix 'X'
    kmeans.fit(X)
    
    # Append the within-cluster sum of squares (inertia) to the list
    wcss.append(kmeans.inertia_)
```

### Elbow Method for Optimal Number of Clusters (k)

```python
# Initialize an empty list to store the within-cluster sum of squares
from sklearn.cluster import KMeans
wcss = []

# Try different values of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # Inertia calculates the sum of square distance in each cluster

# Plot the within-cluster sum of squares for different values of k
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/f575acf1-d493-4748-8ad0-7ca633607dd2)

### Grid Search for Optimal Number of Clusters (k) in KMeans

```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto', random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```
### Displaying the Best Parameters and Best Score from Grid Search

```python
# Print the best parameters and best score obtained from the grid search
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/6a33795c-edcb-4a02-8f74-514c1f3165d6)

### Perform k-means clustering
```python
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/522933c0-16b0-4bc9-8ece-3fdbe2497073)

#### Get the cluster labels and centroids
```
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
```
### Add the cluster labels to the DataFrame
```
df_class['Cluster'] = labels
df_class.head()
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/53663ba2-6e92-4a1d-b490-9e4a5ad23bfc)

### Visualize the clusters
```
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')
plt.scatter(centroids[:,1], centroids[:, 2], marker='X', s=200, c='red')
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')
plt.show()
```
![image](https://github.com/AiswaryaArun19/student-behavior-analysis/assets/106422393/5de34ea2-cbbf-4eea-bcde-d421108b928f)

# 4. EDA
The Exploratory Data Analysis (EDA) revealed several noteworthy insights into the dataset on student behavior and academic performance. The distribution of categorical variables provided a balanced representation of gender, a diverse range of educational stages, and an even distribution across semesters. Parental survey participation indicated a significant engagement level, and there was a mix of parental satisfaction levels.

In terms of numerical variables, the average number of times students raised their hands was approximately 46.78, showcasing varied levels of classroom participation. Students accessed course materials an average of 54.80 times, indicating a significant range in resource utilization. The average number of announcements viewed was 37.92, suggesting moderate engagement with school communications. Discussion participation had an average of 43.28 times, with a wide range of participation rates.

Correlation analysis provided insights into relationships between numerical variables. Notably, there was a moderate positive correlation between raised hands and visited resources, announcements viewed, and a weaker correlation with discussion participation. This suggests an interconnectedness of these behaviors, emphasizing the importance of active engagement in various academic activities.

These EDA results set the stage for further analyses, such as clustering, to identify patterns and groups within the dataset. The findings from EDA will inform subsequent steps in the analysis, contributing to a comprehensive understanding of student behavior and its impact on academic performance.
# 5.Machine Learning model to study segmentation :K-means Clustering
The application of K-means clustering on the dataset of student behavior and academic performance revealed distinct patterns and groups among the students. The clustering algorithm identified three main clusters based on various behavioral and demographic features, shedding light on nuanced distinctions in student engagement. The scatter plot visualization, focusing on raised hands and visited resources, vividly showcased the variability in student behaviors.

Cluster Summary:

Cluster 0 (140 students):

Likely represents students with lower engagement in classroom activities and resource utilization.
Potential candidates for targeted interventions to enhance participation and resource utilization.
Cluster 1 (187 students):

The largest group, suggesting a moderate level of engagement in both raising hands and visiting resources.
Represents a diverse range of behaviors and may require tailored strategies based on individual needs.
Cluster 2 (153 students):

Comprises potentially the most engaged students in terms of classroom participation and resource access.
Offers insights into high-performing students and their distinctive behavioral characteristics.
Insights:

The clustering provides a holistic view of student engagement, highlighting varying degrees of participation and resource utilization.
Educators can leverage these insights to tailor teaching strategies, interventions, and support mechanisms based on the identified clusters.
The clusters offer a foundation for personalized educational approaches, recognizing the diversity in student behaviors and preferences.
# 6.Results and Conclusion

The application of K-means clustering to the dataset on student behavior and academic performance revealed distinct clusters based on behavioral characteristics. The clustering analysis focused on features such as gender, educational stage, raised hands, visited resources, announcements viewed, discussion participation, and absence days.

The optimal number of clusters was determined using the elbow method, with three clusters identified. The clusters were visualized in a scatter plot based on two key features, raised hands, and visited resources. Each cluster demonstrated unique characteristics, indicating varying levels of student engagement and behavior.

**Cluster Summary:**
1. **Cluster 0 (140 students):** This cluster likely represents students with lower engagement in classroom activities and resource utilization. They may require targeted interventions to enhance participation.

2. **Cluster 1 (187 students):** The largest group, indicating a moderate level of engagement in both raising hands and visiting resources. This group could be considered as having a balanced level of participation.

3. **Cluster 2 (153 students):** Comprising potentially the most engaged students in terms of classroom participation and resource access. This group might benefit from advanced or challenging academic programs.

**Conclusion:**
The clustering results offer valuable insights into the diverse behavioral patterns among students. Each cluster represents a distinct profile, suggesting that students exhibit varying levels of engagement and participation in academic activities. These findings can guide educators and policymakers in tailoring strategies to meet the specific needs of each cluster.

Moreover, the clustering analysis provides a foundation for understanding how certain behaviors correlate with academic outcomes. Further analysis could explore the performance levels (Class) within each cluster, helping to identify the influence of behavior on academic success.

In conclusion, the K-means clustering analysis enhances our understanding of student behavior in the context of academic performance, paving the way for targeted interventions and personalized approaches to education. These insights contribute to a more nuanced and informed decision-making process in educational settings.
