# from google.colab import files
# uploaded = files.upload()
#These above two lines are only for Vivian running the code through google collab
import numpy as np
import torch
import re
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import math

##############################################
## N-gram LM SECTION
##############################################

# bigram counts used in trigram model
bigram_counts = dict()

# This function will compute bigram conditional probabilities
# as in Eq 3.11 of Jurafsky & Martin Ch 3.
def bigram_training(training_set, smoothing):

    global bigram_counts
    bigram_count = Counter()
    unigram_count = Counter()
    if smoothing:
        unique_bigrams = set()

    # count bigram and unigram occurences in corpus
    for line in training_set:
        word_list = line
        last_word = len(word_list) - 1

        for i,w in enumerate(word_list):
            # special begin-symbol
            if i == 0:
                bg = '<s> ' + w
            # every other case
            else:
                bg = word_list[i-1] + ' ' + w
            # increment counts
            bigram_count[bg] += 1
            unigram_count[w] += 1
        # special end-symbol
        bg = w + ' </s>'
        bigram_count[bg] += 1

        # every line has a special begin and end symbol
        unigram_count['<s>'] += 1
        unigram_count['</s>'] += 1

        if smoothing:
            # unique vocabulary
            unique_bigrams.update(set(word_list))

    # compute probabilities and add to probs dictionary
    probs = {}

    for bg in bigram_count:

        w1,w2 = bg.split()
        num = bigram_count[bg]
        denom = unigram_count[w1]
        bigram_counts['%s %s'%(w1,w2)] = bigram_count[bg]

        if smoothing:
            V = len(unique_bigrams) + 1 # adds in UNK
            num += 1
            denom += V

        if (num == 0 or denom == 0):
            probs['%s | %s'%(w2,w1)] = 0.0
        else:
            p = float(num)/float(denom)
            probs['%s | %s'%(w2,w1)] = p


    return probs

def bigram_testing(test_set, model):
   
    sentence_probs = dict()
   
    # calculate bigram sentence probabilities
    for line in test_set:
        
        bigram_prob_log_sum = 0
        word_list = line
        last_word = len(word_list) - 1
        num_bigrams = 0
        
        for i,w in enumerate(word_list):
            if i == 0:
                w1 = '<s>'
                w2 = w
            else:
                w1 = word_list[i-1]
                w2 = w
            try:
                bigram_prob_log_sum += math.log(model['%s | %s'%(w2,w1)], 2)
            except:
                bigram_prob_log_sum += 0
            num_bigrams += 1
        # omit beginning of sentence marker from total word tokens N
        w1 = w
        w2 = '</s>'
        try:
            bigram_prob_log_sum += math.log(model['%s | %s'%(w2,w1)], 2)
        except:
            bigram_prob_log_sum += 0
       
        # add to bigram sentence probabilities dictionary
        sentence = " ".join(line)
        # sentence cross-entropy (used in perplexity calculation below)
        sentence_probs['%s|%s' %(sentence, num_bigrams)] = bigram_prob_log_sum 
                
    return sentence_probs

def bigram_perplexity(sentence_probs):
    num_bigrams = 0
    bigram_prob_log_sum = 0

    for sentence, prob in sentence_probs.items():
        sentence,bigram_count = sentence.split("|")
        num_bigrams += int(bigram_count)
        try:
            bigram_prob_log_sum += prob 
        except:
            bigram_prob_log_sum += 0 

    # entropy definition of perplexity
    return math.pow(2, -(bigram_prob_log_sum / num_bigrams))

# This function will compute trigram conditional probabilities
# as in Eq 3.11 of Jurafsky & Martin Ch 3.
def trigram_training(training_set, smoothing):

    global bigram_counts
    trigram_count = Counter()
    if smoothing:
        unique_trigrams = set()
    
    # count trigram occurences in corpus
    for line in training_set:
       
        word_list = line
        last_word = len(word_list) - 1
       
        for i in range(last_word):
            # first word
            if i == 0:
                tg = '<s> ' + word_list[i] + ' ' + word_list[i+1]
            # last word
            elif i == (last_word - 1):
                tg = word_list[i] + ' ' + word_list[i+1] + ' </s>'
            # every other case
            else:
                tg = word_list[i-1] + ' ' + word_list[i] + ' ' + word_list[i+1]
            # increment counts
            trigram_count[tg] += 1
        
        if smoothing:
            # unique vocabulary
            unique_trigrams.update(set(word_list))

    # compute probabilities and add to probs dictionary
    probs = {}

    for tg in trigram_count:
      
        w1,w2,w3 = tg.split()
        num = trigram_count[tg]
        denom = bigram_counts['%s %s'%(w1,w2)]
        
        if smoothing:
            V = len(unique_trigrams) + 1 # adds in UNK
            num += 1
            denom += V
        
        if (num == 0 or denom == 0):
            probs['%s | %s %s'%(w3,w1,w2)] = 0.0
        else:
            p = float(num)/float(denom)
            probs['%s | %s %s'%(w3,w1,w2)] = p
    
    return probs

def trigram_testing(test_set, model):
    
    sentence_probs = dict()
    
    # calculate trigram sentence probabilities
    for line in test_set:
       
        trigram_prob_log_sum = 0
        word_list = line
        last_word = len(word_list) - 1
        # adjust for not adding the beginning of sentence marker
        num_trigrams = -1
       
        for i in range(last_word):
            if i == 0:
                w1 = '<s>'
                w2 = word_list[i]
                w3 = word_list[i+1]
            elif i == (last_word - 1):
                w1 = word_list[i]
                w2 = word_list[i+1] 
                w3 = '</s>'
            else:
                w1 = word_list[i-1]
                w2 = word_list[i]
                w3 = word_list[i+1]
            try: 
                trigram_prob_log_sum += math.log(model['%s | %s %s'%(w3,w1,w2)], 2)
            except:
                pass
            num_trigrams += 1
        
        # add to trigram sentence probabilities dictionary
        sentence = " ".join(line)
        # sentence cross-entropy (used in perplexity calculation below)
        sentence_probs['%s|%s' %(sentence, num_trigrams)] = trigram_prob_log_sum
    
    return sentence_probs

def trigram_perplexity(sentence_probs):
    num_trigrams = 0
    trigram_prob_log_sum = 0
    for sentence, prob in sentence_probs.items():
        sentence,trigram_count = sentence.split("|")
        num_trigrams += int(trigram_count)
        try:
            trigram_prob_log_sum += prob
        except:
            trigram_prob_log_sum += 0
    return math.pow(2, -(trigram_prob_log_sum / num_trigrams))

# preprocesses corpus
def corp_lines(file_name, novel):
    corpus_lines = []
    
    # read in and preprocess the corpus
    raw_corpus = open(file_name, "r")
    corpus_file = raw_corpus.readlines()
    raw_corpus.close()
    if (novel):
        raw_corpus = open(file_name, "r")
        corpus_file = raw_corpus.read()
        # avoid extraneous newlines
        corpus_file = corpus_file.replace("Mr.", "Mr")
        corpus_file = corpus_file.replace("Mrs.", "Mrs")
        # remove page endings
        for i in range(2, 349):
            corpus_file = corpus_file.replace("Page | " + str(i) + " Harry Potter and the Philosophers Stone - J.K. Rowling ", "")
        # remove chapters
        for i in range(1, 62):
            corpus_file = corpus_file.replace("Chapter " + str(i), "")
        # make lines given to model full sentences
        corpus_file = corpus_file.replace("\n", " ")
        # split novels by periods
        corpus_file = corpus_file.rsplit(".")

    check = open("check"+file_name, "w")

    for line in corpus_file:
        # remove ham and spam tags
        if (file_name == "SMSSpamCollection.txt"):
            line = line[3:]
        # -> all lower case
        line = line.lower()
        # remove tweet start and ends
        line = line.replace("<p>", " ")
        line = line.replace("</p>", " ")
        # remove punctuation
        line = re.sub("[^a-zA-Z]", " ", line)
        # remove extra spaces
        words = line.split()

        line = " ".join(words)
        if (line != ""):
            check.write(line + "\n")
            words = line.split()
            corpus_lines.append(words)

    raw_corpus.close()
    check.close()

    return corpus_lines

# find the minimum lines of all the files
def minimum_lines(genre, novel, poem):
    if (genre):
        news = corp_lines("eng_news_2016_10K-sentences.txt", 0)
        wiki = corp_lines("eng_wikipedia_2016_10K-sentences.txt", 0)
        texts = corp_lines("SMSSpamCollection.txt", 0)
        tweets = corp_lines("tweets3.txt", 0)
        novel = corp_lines("philosophers_stone.txt", 1)
        return min(len(news), len(wiki), len(texts), len(tweets), len(novel))
    if (novel):
        ps = corp_lines("philosophers_stone.txt", 1)
        pnp = corp_lines("pride_and_prejudice.txt", 1)
        a = corp_lines("arthur.txt", 1)
        return min(len(ps), len(pnp), len(a))
    if (poem):
        s = corp_lines("sonnets.txt", 0)
        w = corp_lines("leaves_of_grass.txt", 0)
        f = corp_lines("frost.txt", 0)
        return min(len(s),len(w),len(f))

# Takes in file name string of the corpus for training (baseline). Splits into 2:1.
# Returns the training array, testing array of baseline corpus, and size of testing array
def process_baseline(file_name, novel, poem):
    
    corpus_lines = corp_lines(file_name, novel)

    # get total number of corpus sentences
    num_corpus = len(corpus_lines)

    # split into training and test sets (2:1 split)
    corpus_training = []
    corpus_test = []
    corpus_first_test = round(num_corpus*2/3)

    # add in training lines to training array
    for training in range(corpus_first_test):
        corpus_training.append(corpus_lines[training])
    
    # take the minimum of testing lines in this corpus and lines in all corpuses
    end_test = num_corpus - corpus_first_test
    if (poem):
        min_lines = minimum_lines(0,0,1)
    elif (novel):
        min_lines = minimum_lines(0,1,0)
    else:
        min_lines = minimum_lines(1,0,0)

    num_test = min(min_lines, end_test)
    print(num_test)
    end_test = corpus_first_test+num_test

    # add in test lines to test array
    for test in range(corpus_first_test, end_test):
        corpus_test.append(corpus_lines[test])

    test_length = len(corpus_test)

    return corpus_training, corpus_test, test_length

# Takes in the file name of the other testing corpora, and size of test size
# Returns testing array of other corpus, matching size of baseline test array
def process_test_corpora(file_name, test_size, novel):
    
    corpus_lines = corp_lines(file_name, novel)

    # add appropriate number of elements to corpus_test
    corpus_test = []

    for test in range(test_size):
        corpus_test.append(corpus_lines[test])

    return corpus_test

def perplexity_generator(a, b, c, novel, poem):
    ##strings for names of txt files
    ##################################### General N-Gram Perplexity Generator ######################################
    a_training, a_test_abaseline, a_test_size = process_baseline(a, novel, poem)
    # all are with smoothing
    # train bigram and trigram model on novel
    bigram_model_a = bigram_training(a_training, 1)
    trigram_model_a = trigram_training(a_training, 1)

    # process testing corpora with novel baseline testing size
    b_test_abaseline = process_test_corpora(b, a_test_size, novel)
    c_test_abaseline = process_test_corpora(c, a_test_size, novel)

    # clean up names
    a = a[:-4]
    b = b[:-4]
    c = c[:-4]
    a = a.replace("_", " ")
    b = b.replace("_", " ")
    c = c.replace("_", " ")
    a = a.upper()
    b = b.upper()
    c = c.upper()

    print("\n===================== " + a + " CORPUS-TRAINED MODEL =====================\n")
    # bigrams
    # A perplexities trained on A
    bigram_test_probs_a = bigram_testing(a_test_abaseline, bigram_model_a)
    print(a + " bigram perplexity: " + str(bigram_perplexity(bigram_test_probs_a)))
    # B perplexities trained on A
    bigram_test_probs_b = bigram_testing(b_test_abaseline, bigram_model_a)
    print(b + " bigram perplexity: " + str(bigram_perplexity(bigram_test_probs_b)))
    # C perplexities trained on A
    bigram_test_probs_c = bigram_testing(c_test_abaseline, bigram_model_a)
    print(c + " bigram perplexity: " + str(bigram_perplexity(bigram_test_probs_c)) + "\n\n")
    # trigrams
    trigram_test_probs_a = trigram_testing(a_test_abaseline, trigram_model_a)
    print(a + " trigram perplexity: " + str(trigram_perplexity(trigram_test_probs_a)))
    trigram_test_probs_b = trigram_testing(b_test_abaseline, trigram_model_a)
    print(b + " trigram perplexity: " + str(trigram_perplexity(trigram_test_probs_b)))
    trigram_test_probs_c = trigram_testing(c_test_abaseline, trigram_model_a)
    print(c + " trigram perplexity: " + str(trigram_perplexity(trigram_test_probs_c)) + "\n")

def perplexity_generator_big(a, b, c, d, e, novel, poem):
    ##strings for names of txt files
    ##################################### General N-Gram Perplexity Generator ######################################
    a_training, a_test_abaseline, a_test_size = process_baseline(a, novel, poem)
    # all are with smoothing
    # train bigram and trigram model on novel
    bigram_model_a = bigram_training(a_training, 1)
    trigram_model_a = trigram_training(a_training, 1)
    b_novel = b == "philosophers_stone.txt"
    c_novel = c == "philosophers_stone.txt"
    d_novel = c == "philosophers_stone.txt"
    e_novel = e == "philosophers_stone.txt"

    # process testing corpora with novel baseline testing size
    b_test_abaseline = process_test_corpora(b, a_test_size, b_novel)
    c_test_abaseline = process_test_corpora(c, a_test_size, c_novel)
    d_test_abaseline = process_test_corpora(d, a_test_size, d_novel)
    e_test_abaseline = process_test_corpora(e, a_test_size, e_novel)

    # clean up names
    a = a[:-4]
    b = b[:-4]
    c = c[:-4]
    d = d[:-4]
    e = e[:-4]
    a = a.replace("_", " ")
    b = b.replace("_", " ")
    c = c.replace("_", " ")
    d = d.replace("_", " ")
    e = e.replace("_", " ")
    a = a.upper()
    b = b.upper()
    c = c.upper()
    d = d.upper()
    e = e.upper()

    print("\n===================== " + a + " CORPUS-TRAINED MODEL =====================\n")
    # bigrams
    # A perplexities trained on A
    bigram_test_probs_a = bigram_testing(a_test_abaseline, bigram_model_a)
    print(a + " bigram perplexity: " + str(bigram_perplexity(bigram_test_probs_a)))
    # B perplexities trained on A
    bigram_test_probs_b = bigram_testing(b_test_abaseline, bigram_model_a)
    print(b + " bigram perplexity: " + str(bigram_perplexity(bigram_test_probs_b)))
    # C perplexities trained on A
    bigram_test_probs_c = bigram_testing(c_test_abaseline, bigram_model_a)
    print(c + " bigram perplexity: " + str(bigram_perplexity(bigram_test_probs_c)))
    # D perplexities trained on A
    bigram_test_probs_d = bigram_testing(d_test_abaseline, bigram_model_a)
    print(d + " bigram perplexity: " + str(bigram_perplexity(bigram_test_probs_d)))
    # E perplexities trained on A
    bigram_test_probs_e = bigram_testing(e_test_abaseline, bigram_model_a)
    print(e + " bigram perplexity: " + str(bigram_perplexity(bigram_test_probs_e)) + "\n\n")
    # trigrams
    trigram_test_probs_a = trigram_testing(a_test_abaseline, trigram_model_a)
    print(a + " trigram perplexity: " + str(trigram_perplexity(trigram_test_probs_a)))
    trigram_test_probs_b = trigram_testing(b_test_abaseline, trigram_model_a)
    print(b + " trigram perplexity: " + str(trigram_perplexity(trigram_test_probs_b)))
    trigram_test_probs_c = trigram_testing(c_test_abaseline, trigram_model_a)
    print(c + " trigram perplexity: " + str(trigram_perplexity(trigram_test_probs_c)))
    trigram_test_probs_d = trigram_testing(d_test_abaseline, trigram_model_a)
    print(d + " trigram perplexity: " + str(trigram_perplexity(trigram_test_probs_d)))
    trigram_test_probs_e = trigram_testing(e_test_abaseline, trigram_model_a)
    print(e + " trigram perplexity: " + str(trigram_perplexity(trigram_test_probs_e)) + "\n")

if __name__ == '__main__':

    ##################################### News Baseline ####################################
    perplexity_generator_big("eng_news_2016_10K-sentences.txt", "eng_wikipedia_2016_10K-sentences.txt", "tweets3.txt", "SMSSpamCollection.txt", "philosophers_stone.txt", 0, 0)
    ##################################### Wiki Baseline ####################################
    perplexity_generator_big("eng_wikipedia_2016_10K-sentences.txt", "tweets3.txt", "SMSSpamCollection.txt", "philosophers_stone.txt", "eng_news_2016_10K-sentences.txt", 0, 0)
    #################################### Tweet Baseline ####################################
    perplexity_generator_big("tweets3.txt", "SMSSpamCollection.txt", "philosophers_stone.txt", "eng_news_2016_10K-sentences.txt", "eng_wikipedia_2016_10K-sentences.txt", 0, 0)
    #################################### Texts Baseline ####################################
    perplexity_generator_big("SMSSpamCollection.txt", "philosophers_stone.txt", "eng_news_2016_10K-sentences.txt", "eng_wikipedia_2016_10K-sentences.txt", "tweets3.txt", 0, 0)
    ##################################### Novels Baseline ####################################
    perplexity_generator_big("philosophers_stone.txt", "eng_news_2016_10K-sentences.txt", "eng_wikipedia_2016_10K-sentences.txt", "tweets3.txt", "SMSSpamCollection.txt", 1, 0)
    ##################################### Novels Over Time ####################################
    ##################################### Old Baseline ######################################
    perplexity_generator("arthur.txt", "pride_and_prejudice.txt", "philosophers_stone.txt", 1, 0)
    ##################################### Middle Baseline ###################################
    perplexity_generator("pride_and_prejudice.txt", "philosophers_stone.txt", "arthur.txt", 1, 0)
    ##################################### Young Baseline #####################################
    perplexity_generator("philosophers_stone.txt", "arthur.txt", "pride_and_prejudice.txt", 1, 0)
    ##################################### Poems Over Time ####################################
    perplexity_generator("sonnets.txt", "leaves_of_grass.txt", "frost.txt", 0, 1)
    perplexity_generator("leaves_of_grass.txt", "frost.txt", "sonnets.txt", 0, 1)
    perplexity_generator("frost.txt", "sonnets.txt", "leaves_of_grass.txt", 0, 1)
