import re
import json
import pandas as pd

# Take json of golden queries from CoreScience codebase
# Clean and send to tsv

dest = '../data/golden_queries_preprocessed.tsv'

df = pd.DataFrame(columns=['domain', 'intent', 'golden_queries'])
ix = 0

with open('../data/golden_query_guide.json') as f:
    data = json.load(f)

for domain in data['domains']:
    domain_name = domain['domainName']
    print ('\n', domain_name)

    for intent in domain['intents']:
        intent_name = intent['intentName']
        print (intent_name)
        gqs = []

        for example in intent['positiveExamples']:
            query = example['query']
            if query is not None:
                print ('\t', query)
                query = re.sub(r'[^\w\s]', '', query)
                gqs.append(query)

        df.loc[ix] = pd.Series({'domain': domain_name, 'intent': intent_name, 'golden_queries': gqs})
        ix += 1

df.to_csv(dest, sep='\t', index=False)
print ('###\nSaved dataframe to', dest)
