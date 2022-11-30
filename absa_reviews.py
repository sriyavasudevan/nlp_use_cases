from pyabsa import ATEPCCheckpointManager, available_checkpoints
import pandas as pd
import numpy as np

"""
@author: Sriya Madapusi Vasudevan
"""

def read_data_to_list(filename):
    """
    Read intial reviews into the desired dataframe and list of sentences (reviews)
    :param filename: filename to reviews
    :return: dataframe and list of sentences
    """
    try:
        BERT_TOKEN_LIMIT = 400
        df = pd.read_csv(filename, delimiter='\t')
        id_list = range(0, len(df))
        df['review_ID'] = id_list

        # i know BERT tokens are tokenized /not/ character-wise, but its a lot to find how many actual tokens
        # this is just a quick dirty fix that works
        df['chunked_reviews'] = df['Review'].apply(lambda x: [x[i: i + BERT_TOKEN_LIMIT]
                                                              for i in range(0, len(x), BERT_TOKEN_LIMIT)])
        df = (df.set_index(['review_ID', 'Name', 'RatingValue', 'DatePublished', 'Review'])
                .apply(pd.Series.explode)
                .reset_index())

        list_of_sentences = df['chunked_reviews'].tolist()
        return df, list_of_sentences
    except FileNotFoundError:
        print(f"Error: {filename} is not found. Please check again.\n")
        return -1


def read_inference_result(filename):
    """
    Once PyABSA returns a json file, this method can process that file
    This decoupling allows for us to run this algorithm after the json file has been generated once
    :param filename:
    :return:
    """
    inference_df = pd.read_json(filename)

    results_df = pd.DataFrame({'Aspect': inference_df['aspect'],
                               'Sentiment': inference_df['sentiment'],
                               'review_ID': original_review_df['review_ID'],
                               'Name': original_review_df['Name']})

    # source - https://stackoverflow.com/questions/45846765/efficient-way-to-unnest-explode-multiple-list-columns-in-a-pandas-dataframe
    results_df = results_df.set_index(['review_ID', 'Name']).apply(pd.Series.explode).reset_index()

    results_df['Aspect'].replace('', np.nan, inplace=True)
    results_df.dropna(inplace=True)

    temp_df = pd.DataFrame({'Aspect': results_df['Aspect'],
                            'Sentiment': results_df['Sentiment'],
                            'review_ID': results_df['review_ID'],
                            'Name': results_df['Name']})

    return temp_df


def absa(list_sentences):
    """
    Perform aspect based sentiment analysis using PyABSA
    :param list_sentences: list of sentences
    :return: dataframe with aspect, sentiment and review number
    """

    """Comment out below block of code once the json has been generated once; using the json to process"""
    ##
    # source - https://github.com/yangheng95/PyABSA/blob/release/demos/aspect_term_extraction/extract_aspects_multilingual.py
    checkpoint_map = available_checkpoints(from_local=False)

    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='multilingual')

    atepc_result = aspect_extractor.extract_aspect(inference_source=list_sentences,  # list-support only, for current
                                                   print_result=True,  # print the result
                                                   pred_sentiment=True,
                                                   # Predict the sentiment of extracted aspect terms
                                                   )
    ##
    res_df = read_inference_result('atepc_inference.result.json')
    return res_df


if __name__ == '__main__':
    path_to_file = "reviews.csv"
    returned_components = read_data_to_list(path_to_file)

    original_review_df = returned_components[0]
    sentences = returned_components[1]

    output_df = absa(sentences)
    output_df.to_csv("absa_reviews.csv")
