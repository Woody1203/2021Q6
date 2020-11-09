import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# df = pd.read_csv('file_name.csv', engine='python')
out = pd.read_csv('Aggy-en-test.csv', encoding='utf-8')
# expression = pd.DataFrame(out, columns = ['expression'])
intents = ["Blackberry new", "vending_machines_unknown", "Headset issue", "Badge loader", "CiC config", "Print paper", "Change Language", "Blackberry incident", "Problem known", "Headset change model", "Airco issue"]
train_final = pd.DataFrame(columns=['intent','expression'])
test_final = pd.DataFrame(columns=['intent','expression'])

for selected_intent in intents:
    # for selected_intent in intents:

    single_intent_dataset = out.loc[out["intent"] == selected_intent]
    print("this is selected intent",selected_intent)
    print("this is single_intent_dataset",single_intent_dataset)

    shuffled_single_intent_dataset_expression_only = pd.DataFrame(single_intent_dataset, columns = ['intent','expression'])
    # print(shuffled_single_intent_dataset_expression_only)

    train, test = train_test_split(shuffled_single_intent_dataset_expression_only, test_size=0.5, random_state = 42, shuffle = True)
    print("Intent:",selected_intent)

    print("the length of trainset",len(train))
    print("the length of testset",len(test))

    # Stack the DataFrames on top of each other
    train_final = pd.concat([train_final, train], axis=0)
    test_final = pd.concat([test_final, test], axis=0)

    print("type of train", type(train))
    print("type of test", type(test))

print("the length of trainset",len(train_final))
print("the length of testset",len(test_final))

# print("this is out")
# print(out.loc[out["intent"] == "Problem known"])
#
# result = train_final.loc[train_final["intent"] == "Problem known"]
# print(result)


train_final.to_csv('final_training.csv',index=False)
test_final.to_csv('final_testing.csv',index=False)
