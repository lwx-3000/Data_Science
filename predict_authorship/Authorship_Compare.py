import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

LINES = ['-', ':', '--']  # style of line for plots, treated as a Constant


def main():
    # prepare text for analysis
    strings_by_author = {'doyle': text_to_string('hound.txt')}
    strings_by_author.update({'wells': text_to_string('war.txt')})
    strings_by_author['unknown'] = text_to_string('lost.txt')

    print(strings_by_author['doyle'][:300])

    # text need to be the same word length for comparing (Normalization)
    words_by_author = make_word_dict(strings_by_author)
    len_shortest_corpus = find_shortest_corpus(words_by_author)
    word_length_test(words_by_author, len_shortest_corpus)
    parts_of_speech_test(words_by_author, len_shortest_corpus)
    vocab_test(words_by_author)  # test vocab similarity
    jaccard_test(words_by_author, len_shortest_corpus)
    # jaccard similarity test:
    # intersection-over-union for a set the area of overlap divided by the area of union


def text_to_string(filename):
    """Read txt return a string"""
    with open(filename, encoding="ISO-8859-1") as infile:
        return infile.read()


def make_word_dict(strings_by_author):
    """Return dictionary of tokenized words by corpus with author"""
    words_by_author = dict()
    for author in strings_by_author:
        tokens = nltk.word_tokenize(strings_by_author[author])
        words_by_author[author] = ([token.lower() for token in tokens if token.isalpha()])
    return words_by_author


def find_shortest_corpus(words_by_author):
    """return length of the shortest corpus"""
    word_count = []
    for author in words_by_author:
        word_count.append(len(words_by_author[author]))
        print('\nNumber of words for {} = {}\n'.format(author, len(words_by_author[author])))
    len_shortest_corpus = min(word_count)
    print('length shortest corpus = {}\n'.format(len_shortest_corpus))
    return len_shortest_corpus


def word_length_test(words_by_author, len_shortest_corpus):
    """plot word length freq by author, truncated to the shortest corpus length."""
    by_author_length_freq_dist = dict()
    plt.figure(1)
    plt.ion()
    for i, author in enumerate(words_by_author):
        word_lengths = [len(word) for word in words_by_author[author][:len_shortest_corpus]]
        by_author_length_freq_dist[author] = nltk.FreqDist(word_lengths)
        by_author_length_freq_dist[author].plot(15, linestyle=LINES[i], label=author, title='Word Length')
    plt.legend()
    plt.show()


def stopwords_test(words_by_author, len_shortest_corpus):
    """plot stopwords freq by author, truncated to shortest corpus length."""
    stopwords_by_author_freq_dist = dict()
    plt.figure(2)
    stop_words = set(stopwords.word('english'))
    print('Number of stopwords = {}\n'.format(len(stop_words)))
    print('Stopwords = {}\n'.format(stop_words))

    for i, author in enumerate(words_by_author):
        stopwords_by_author = [word for word in words_by_author[author][:len_shortest_corpus] if word in stop_words]
        stopwords_by_author_freq_dist[author] = nltk.FreqDist(stopwords_by_author)
        stopwords_by_author_freq_dist[author].plot(50, lable=author, linestyle=LINES[i],
                                                   title='50 Most Common Stopwords')
    plt.legend()
    plt.show()


def parts_of_speech_test(words_by_author, len_shortest_corpus):
    """Plot author use of parts-of-speech such as nouns, verbs, adverbs"""
    by_author_pos_freq_dist = dict()
    plt.figure(3)
    for i, author in enumerate(words_by_author):
        pos_by_author = [pos[1] for pos in nltk.pos_tag(words_by_author[author][:len_shortest_corpus])]
        by_author_pos_freq_dist[author] = nltk.FreqDist(pos_by_author)
        by_author_pos_freq_dist[author].plot(35, label=author, linestyle=LINES[i], title='Part of Speech')

    plt.legend()
    plt.show()


def vocab_test(words_by_author):  # chi-squared test for goodness of fit
    """compare author vocabularies using the chi-squared test"""
    chisquared_by_author = dict()
    for author in words_by_author:
        # create population data for test
        if author != 'unknown':
            combined_corpus = (words_by_author[author] + words_by_author['unknown'])
            author_proportion = (len(words_by_author[author]) / len(combined_corpus))
            combined_freq_dist = nltk.FreqDist(combined_corpus)
            most_common_words = list(combined_freq_dist.most_common(1000))
            chisquared = 0
        # chi-squared test
        for word, combined_count in most_common_words:
            observed_count_author = words_by_author[author].count(word)
            expected_count_author = combined_count * author_proportion
            chisquared += ((observed_count_author - expected_count_author) ** 2 / expected_count_author)
            chisquared_by_author[author] = chisquared
        print('Chi-squared for {} = {:.1f}'.format(author, chisquared))
    most_likely_author = min(chisquared_by_author, key=chisquared_by_author.get)
    print('Most-likely author by vocabulary is {}\n'.format(most_likely_author))


def jaccard_test(words_by_author, len_shortest_corpus):
    """Calculate Jaccard similarity of each known corpus to unknown corpus"""
    jaccard_by_author = dict()
    unique_words_unknown = set(words_by_author['unknown'][:len_shortest_corpus])
    authors = (author for author in words_by_author if author != 'unknown')
    for author in authors:
        unique_words_author = set(words_by_author[author][:len_shortest_corpus])
        shared_words = unique_words_author.intersection(unique_words_unknown)
        jaccard_sim = (float(len(shared_words)) / (
                    len(unique_words_author) + len(unique_words_unknown) - len(shared_words)))
        jaccard_by_author[author] = jaccard_sim
        print('Jaccard Similarity for {} = {}'.format(author, jaccard_sim))
    most_likely_author = max(jaccard_by_author, key=jaccard_by_author.get)
    print('Most likely author by similarity is {}'.format(most_likely_author))


if __name__ == '__main__':
    main()
