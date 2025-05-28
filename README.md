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
      ```
      # current_runrate = runs / overs
        delivery_df["crr"] = (delivery_df["current_score"]*6)/(120 - delivery_df["balls_left"])
      ```
    - Required Run Rate
      ```
        # required runrate
          delivery_df["rrr"] = (delivery_df["runs_left"] * 6) / delivery_df["balls_left"]
      ```
    - Wickets left
      ```
        delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
        delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
        delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
        wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
        delivery_df['wickets'] = 10 - wickets

      ```
    - Balls left
      ```
        delivery_df["balls_left"] = 126 - (delivery_df["over"]*6 + delivery_df["ball"])
      ```
    - Runs left
      ```
        delivery_df["runs_left"] = delivery_df["total_runs_x"] -  delivery_df["current_score"]
      ```
    - Current Score
      ```
        delivery_df["current_score"] = delivery_df.groupby("match_id")["total_runs_y"].cumsum()
      ```

- One-hot encoding for categorical variables
  ```
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    trf = ColumnTransformer([
        ('trf',OneHotEncoder(sparse_output=False,drop='first'),['batting_team','bowling_team','city'])
    ]
    ,remainder='passthrough')
  
  ```

- Splitting data into training and test sets
  ```
    x = final_df.iloc[: ,:-1]
    y  = final_df.iloc[: , -1]

    x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state=1)
     
  ```

- Training a classification model (e.g., Logistic Regression)
  ```
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(steps=[
    ("step1" , trf),
    ("step2" , LogisticRegression(solver="liblinear"))

    ])

    pipe.fit(x_train , y_train)
     
  ```

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

<h1>Let's test the web app</h1>

<h3>Let select batting team</h3>
<img src = "https://github.com/Vinit-joshi2/Ipl-win-predictions-probability/blob/main/web1.png">

<h3>Let select bowling team</h3>
<img src = "https://github.com/Vinit-joshi2/Ipl-win-predictions-probability/blob/main/web2.png">

<h3>Let select host city </h3>
<img src = "https://github.com/Vinit-joshi2/Ipl-win-predictions-probability/blob/main/web3.png">
 
<h3>Target made by bowling team</h3>
<img src = "https://github.com/Vinit-joshi2/Ipl-win-predictions-probability/blob/main/web4.png">
 
<h3>Other requirement</h3>
<img src = "https://github.com/Vinit-joshi2/Ipl-win-predictions-probability/blob/main/web5.png">

<h3>Winning Probability of each team</h3>
<img src = "https://github.com/Vinit-joshi2/Ipl-win-predictions-probability/blob/main/web6.png">

<h3>Prediction</h3>
<img src = "https://github.com/Vinit-joshi2/Ipl-win-predictions-probability/blob/main/web7.png">


 





