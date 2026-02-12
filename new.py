# import nltk
# nltk.data.path.append("./nltk_data")

# nltk.data.find("tokenizers/punkt_tab/english")
# nltk.data.find("taggers/averaged_perceptron_tagger_eng")

# print("Success!")

import nltk
nltk.data.path.append("./nltk_data")

print("punkt_tab:", nltk.data.find("tokenizers/punkt_tab/english"))
# print("tagger:", nltk.data.find("taggers/averaged_perceptron_tagger_eng"))



from nltk.tokenize import word_tokenize

print(word_tokenize("This is a test."))
