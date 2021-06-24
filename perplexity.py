import collections, nltk

# https://sjmielke.com/comparing-perplexities.htm

# here you construct the unigram language model
def unigram(tokens):
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model[f] = 1
            continue
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word] / N
    return model

#computes perplexity of the unigram model on a testset
def perplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N))
    return perplexity

def main():

    corpus = """
    Monty Python (sometimes known as The Pythons) were a British surreal comedy group who created the sketch comedy show Monty Python's Flying Circus,
    that first aired on the BBC on October 5, 1969. Forty-five episodes were made over four series. The Python phenomenon developed from the television series
    into something larger in scope and impact, spawning touring stage shows, films, numerous albums, several books, and a stage musical.
    The group's influence on comedy has been compared to The Beatles' influence on music."""

    # we first tokenize the text corpus
    tokens = nltk.word_tokenize(corpus)

    testset1 = "Monty"
    testset2 = "abracadabra gobbledygook rubbish"

    model = unigram(tokens)
    print(perplexity(testset1, model))
    print(perplexity(testset2, model))


if __name__ == '__main__':
    main()