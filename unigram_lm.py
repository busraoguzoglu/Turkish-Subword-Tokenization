from tokenizers import Tokenizer, models, trainers
import collections


def unigram(tokens):
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        model[f] += 1
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word] / N
    return model

def perplexity(testset, model):
    perplexity = 1
    N = len(testset)
    for i in range(80):
        word = testset[i]
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N))
    return perplexity

def main():

    tokenizer = Tokenizer(models.Unigram())
    trainer = trainers.UnigramTrainer(vocab_size=25000)
    tokenizer.train(files=["UD_Turkish-Penn/tr_penn-ud-train.txt", "UD_Turkish-Penn/tr_penn-ud-dev.txt"], trainer=trainer)

    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

    encoding = tokenizer.encode("Merhaba bu bir kelime")
    print("Encoded string: {}".format(encoding.tokens))

    decoded = tokenizer.decode(encoding.ids)
    print("Decoded string: {}".format(decoded))

    vocab = tokenizer.get_vocab()

    # Perplexity:
    # Tokenize training and test corpus:
    with open("UD_Turkish-Penn/tr_penn-ud-train.txt", encoding="utf-8") as f:
        train_contents = f.read()
    with open("UD_Turkish-Penn/tr_penn-ud-test.txt", encoding="utf-8") as f:
        test_contents = f.read()

    train_tokenized = tokenizer.encode(train_contents)
    test_tokenized = tokenizer.encode(test_contents)

    # Get unigram model to calculate perplexity
    model = unigram(train_tokenized.tokens)
    print('Perplexity:', perplexity(test_tokenized.tokens, model))

if __name__ == '__main__':
    main()