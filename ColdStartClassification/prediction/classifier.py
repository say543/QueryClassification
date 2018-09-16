import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def classify(query_array, option_embeddings, option_names, gq_map, metric_name, classifier, intent_pools=None):
    # Calculate similarity metrics
    correlations = np.inner(query_array, option_embeddings)
    angular_sim = 1 - (np.arccos(correlations)/np.pi)
    euc_sim = -euclidean_distances(query_array, option_embeddings) # Negative so it is similarity

    if metric_name == 'cosine':
        metric = correlations
    elif metric_name == 'angular':
        metric = angular_sim
    else:
        metric = euc_sim

    # When doing intent only, limit to pools per domain
    if intent_pools is not None:
        metric[~intent_pools] = -10 #Arbitrary bad similarity score, for all non-domain choices

    # Classification methods
    # When gq_map is NONE, we are doing domain classification
    if gq_map:
        try:
            other_ind = option_names.index('other. other')
        except:
            other_ind = -1
    else:
        other_ind = option_names.index('other')

    # argmax
    maxinds = np.argmax(metric, axis=1)
    maxes = np.max(metric, axis=1)
    avg_of_maxes = np.mean(maxes)

    # Threshold method
    def select_by_threshold(values, threshold):
        thresholded = metric > threshold
        first_in_range = np.argmax(thresholded, axis=1)
        summed = np.sum(thresholded, axis=1)
        no_match = summed == 0
        t_choice = first_in_range
        t_choice[no_match] = other_ind
        return t_choice

    # Default theshold method
    threshold = avg_of_maxes
    threshold_choices = select_by_threshold(metric, threshold)

    # P/R threshold method
    outer_threshold = threshold
    inner_threshold = threshold*1.3
    inner_choices = select_by_threshold(metric, inner_threshold)

    outer_threshold_map = np.array(metric > outer_threshold) & np.array(metric < inner_threshold)
    outer_threshold_map = np.sum(outer_threshold_map, axis=1) # Do any entries in column hit the outer theshold but not the inner?

    p_r_choices = [inner_choice if outer_map_val == 0 else argmax_choice for (inner_choice,argmax_choice,outer_map_val) in zip(inner_choices,maxinds,outer_threshold_map)]

    if classifier == 'argmax':
        choices = np.array(maxinds)
        #print ('Selected based on max similarities, average of ', avg_of_maxes)
    elif classifier == 'threshold':
        choices = np.array(threshold_choices)
    else:
        choices = np.array(p_r_choices)

    representatives = np.array([option_names[x] for x in choices])
    predictions = representatives
    if gq_map:
        predictions = [gq_map[r] for r in representatives]
    return np.array(predictions), np.array(representatives)
