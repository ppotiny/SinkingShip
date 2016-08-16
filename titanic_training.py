__author__ = 'Praneetha'

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Import titanic files
train_df = "titanic_train.csv"
test_df = "titanic_test.csv"

# Read files into pandas dataframes
train = pd.read_csv(train_df)
test = pd.read_csv(test_df)

def get_labels(ticket):
    # Tickets with labels are stripped of punctuation so that similar ones do not appear twice
    # Unique labels are returned
    number = ticket.split()[-1].strip()
    ticket_name = ticket.replace(number, "").replace(".", "").replace("/", "").strip(). lower()
    return ticket_name

ticket_labels = train["Ticket"].apply(func=get_labels).unique().tolist() + test["Ticket"].apply(func=get_labels).unique().tolist()

def convert_ticket(ticket):
    # Each ticket label is separated by 10,000 and added to it's ticket number
    number = ticket.split()[-1].strip()
    ticket_name = ticket.replace(number, "").replace(".", "").replace("/", "").strip(). lower()
    if ticket_name == "": return number
    else: return (4000000 + ticket_labels.index(ticket_name)*1000000 + int(number))

def sibage(sibs, sex, age):
    # Based on graphs of the data, a feature is given a lower number if they are
    # more likely to survive. The number of siblings is then added to this number.
    # Assuming no one has more than 10 siblings and spouses combined, each category
    # is separated by 10.
    if sex == "female":
        if age < 20:
            return sibs
        elif age > 30:
            return 10 + sibs
        else: return 20 + sibs  # age between 20 and 30
    else:
        if age < 20:
            return 30 + sibs
        elif age > 30:
            return 40 + sibs
        else: return 50 + sibs

# Feature that combines sex with age and number of siblings
train["SibAge"] = train.apply(lambda x: sibage(x["SibSp"], x["Sex"], x["Age"]), axis=1)
test["SibAge"] = test.apply(lambda x: sibage(x["SibSp"], x["Sex"], x["Age"]), axis=1)

# Gets converted ticket number of each passenger
train["TicketNum"] = train["Ticket"].apply(func=convert_ticket, convert_dtype=True).convert_objects(convert_numeric=True)
test["TicketNum"] = test["Ticket"].apply(func=convert_ticket, convert_dtype=True).convert_objects(convert_numeric=True)

# Gets titles of each passenger
train["Titles"] = train["Name"].str.split(".").str[0].str.split(" ").str[-1].str.strip()
test["Titles"] = test["Name"].str.split(".").str[0].str.split(" ").str[-1].str.strip()

# Drop Unnecessary Features

X_train = train.drop(['PassengerId', 'Name', 'Cabin', 'Survived', 'Embarked', 'Parch', 'Ticket', 'Fare'], axis=1)
y_train = train["Survived"]
X_test = test.drop(['Name', 'Cabin', 'Embarked', 'Parch', 'Ticket', 'Fare'], axis=1)

# Fill values for age, fare, and ticket

X_train["Age"] = X_train["Age"].fillna((X_train["Age"].mean() + X_test["Age"].mean())/2)
X_train["Fare"] = X_train["Fare"].fillna((X_train["Fare"].mean() + X_test["Fare"].mean())/2)
X_train["TicketNum"] = X_train["TicketNum"].fillna(0)

X_test["Age"] = X_test["Age"].fillna((X_train["Age"].mean() + X_test["Age"].mean())/2)
X_test["Fare"] = X_test["Fare"].fillna((X_train["Fare"].mean() + X_test["Fare"].mean())/2)
X_test["TicketNum"] = X_test["TicketNum"].fillna(0)

# Return numerical labels for features that are not numerical

le = LabelEncoder()
le2 = LabelEncoder()

le.fit(X_train["Titles"])

X_test["Titles"] =  X_test["Titles"].map(lambda x: 'N/A' if x not in le.classes_ else x)
le.classes_ = np.append(le.classes_, 'N/A')
X_test["Titles"] = le.transform(X_test["Titles"])
X_train["Titles"] = le.transform(X_train["Titles"])

le2.fit(X_train["Sex"])

X_test["Sex"] = le2.transform(X_test["Sex"])
X_train["Sex"] = le2.transform(X_train["Sex"])

# Create a Random Forest classifier that will train the data
rf = RandomForestClassifier(n_estimators=100, oob_score = True, max_features=None)
rf.fit(X_train, y_train)

# Get prediction
y_pred = rf.predict(X_test.drop(['PassengerId'], axis=1))

print rf.score(X_train, y_train)
print rf.oob_score_

# Place predictions and passenger IDs in a csv to submit
submission = pd.DataFrame({
        'PassengerId': X_test['PassengerId'],
        'Survived': y_pred
    })
submission.to_csv('titanic_solution.csv', index=False)