import numpy as np
from sklearn.metrics import confusion_matrix

def accuracy(predictions, targets):
    list_acc = [x == y for x,y in zip(targets, predictions)]
    if len(list_acc) == 0:
        return 0, []
    return (sum(list_acc)/len(list_acc)), list_acc

def domain_classification_pra(predictions, targets):
    targets = np.array([t[:t.index('.')] for t in targets])
    predictions = np.array([p[:p.index('.')] for p in predictions])
    
    dom_acc, _ = accuracy(predictions, targets)
    
    picked_up = predictions != 'other'
    in_domains = targets != 'other'

    precision,_ = accuracy(predictions[picked_up], targets[picked_up])
    recall,_ = accuracy(predictions[in_domains], targets[in_domains])
    
    return dom_acc, precision, recall

def domain_specific_pr(predictions, targets, domain):
    d_targets = np.array([t[:t.index('.')] for t in targets])
    d_predictions = np.array([p[:p.index('.')] for p in predictions])
    
    picked_up = d_predictions == domain
    in_domain = d_targets == domain

    d_precision,_ = accuracy(d_predictions[picked_up], d_targets[picked_up])
    d_recall,_ = accuracy(d_predictions[in_domain], d_targets[in_domain])

    # Calculate intent level P/R. Using only properly domain-classified examples
    domain_classification_success = picked_up & in_domain
    predictions = predictions[domain_classification_success]
    targets = targets[domain_classification_success]
    
    # Get intent-wise precision recall numbers, print per intent
    print()
    intents = list(set(targets))
    prec_agg = 0.0
    rec_agg = 0.0
    total = targets.shape[0]
    for i in intents:
        picked_up = predictions == i
        in_intent = targets == i

        precision,_ = accuracy(predictions[picked_up], targets[picked_up])
        recall,_ = accuracy(predictions[in_intent], targets[in_intent])
        
        prec_agg += precision * sum(in_intent)
        rec_agg += recall * sum(in_intent)

        print (i, precision, recall, sum(in_intent))
    prec_agg /= total
    rec_agg /= total

    # Get cross-intent confusion
    intent_confusion_matrix = confusion_matrix(targets, predictions, labels=intents)
    np.set_printoptions(linewidth=1000000000)
    print ('\n', intent_confusion_matrix, '\n')

    return d_precision, d_recall, prec_agg, rec_agg
