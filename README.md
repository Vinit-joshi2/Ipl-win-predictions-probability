# 🏏 IPL Win Prediction Probability

<h1>
  
📌 Problem Statement
</h1>
In this project, we aim to predict the probability of a team winning an IPL match based on  match conditions. Using past IPL match data, we train a machine learning model that takes match features like current score, cureent run rate , require run rate, wickets, overs, and target score to calculate winning probabilities for both teams.

<h3>
  
📂 Dataset
</h3>
The dataset used for this project includes detailed ball-by-ball data from IPL matches, containing information such as:

 - Batting team and bowling team

- Current score and target score

- Wickets fallen

- Overs and balls completed

- Venue and match conditions

📁 Source:  <a href = "https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set">Ipl DataSet</a>

<h1>📄 Dataset Description</h1>

<p>
This dataset contains <strong>IPL match-level data</strong>, where each row represents one completed match. It includes information about the teams, toss results, match outcomes, player awards, and umpire details.
</p>

<table>
  <thead>
    <tr>
      <th>Column Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>Season</code></td>
      <td>The IPL season (e.g., <code>IPL-2017</code>).</td>
    </tr>
    <tr>
      <td><code>city</code></td>
      <td>The city where the match was played.</td>
    </tr>
    <tr>
      <td><code>date</code></td>
      <td>The date of the match (<code>DD-MM-YYYY</code> format).</td>
    </tr>
    <tr>
      <td><code>team1</code></td>
      <td>The first team (usually bats first).</td>
    </tr>
    <tr>
      <td><code>team2</code></td>
      <td>The second team (usually bowls first).</td>
    </tr>
    <tr>
      <td><code>toss_winner</code></td>
      <td>The team that won the toss.</td>
    </tr>
    <tr>
      <td><code>toss_decision</code></td>
      <td>The toss-winning team's decision: <code>bat</code> or <code>field</code>.</td>
    </tr>
    <tr>
      <td><code>result</code></td>
      <td>The match result type (e.g., <code>normal</code>, <code>tie</code>, <code>no result</code>).</td>
    </tr>
    <tr>
      <td><code>dl_applied</code></td>
      <td>Indicates whether the Duckworth–Lewis method was applied (1 = Yes, 0 = No).</td>
    </tr>
    <tr>
      <td><code>winner</code></td>
      <td>The team that won the match.</td>
    </tr>
    <tr>
      <td><code>win_by_runs</code></td>
      <td>Margin of victory by runs (non-zero if defending team won).</td>
    </tr>
    <tr>
      <td><code>win_by_wickets</code></td>
      <td>Margin of victory by wickets (non-zero if chasing team won).</td>
    </tr>
    <tr>
      <td><code>player_of_match</code></td>
      <td>The player awarded <em>Player of the Match</em>.</td>
    </tr>

</table>
      



<h1>🧠 Model & Approach</h1>

We used the following machine learning pipeline:

- Data cleaning and preprocessing

- Feature engineering

    - Current Run Rate
    - Required Run Rate
    - Required Run Rate
    - Wickets left
    - Balls left
    - Runs left

- One-hot encoding for categorical variables

- Splitting data into training and test sets

- Training a classification model (e.g., Logistic Regression)

<h3>
  
🧪 Evaluation
</h3>
We used metrics like:

- Accuracy

- Loss
  
| Metric        | Value        |
| ------------- | ------------ |
| Accuracy      | *e.g., 0.80* |
| Log Loss      | *e.g., 0.38* |

<h1>
  
📈 Visualization
</h1>
We used matplotlib and seaborn to visualize:

- Win probabilities over time

- Loss probabilities over time
  
- Wickets in each over

- Runs after each over

```
import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))
```
<img src = "https://github.com/Vinit-joshi2/Ipl-win-predictions-probability/blob/main/Image1.png">


<h1>✅ How to Run </h1>
<h3>
  
Step 1: Git clone
</h3>

```
git clone url
````

<h3>Step 2: Install Require Library 
</h3>

```
pip install pandas
pip install numpy
pip install seaborn 
pip install matplotlib
pip install scikit-learn
pip install streamlit
```

<h3>
  
 Step 3:  Run Streamlit app
</h3>

```
streamlit run app.py
```
