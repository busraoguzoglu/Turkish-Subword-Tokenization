from tokenizers import Tokenizer, models, trainers
import collections, nltk

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
    #testset = testset.split()
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N))
    return perplexity

def main():

    tokenizer = Tokenizer(models.Unigram())
    trainer = trainers.UnigramTrainer(
        unk_token="<unk>",
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    )
    tokenizer.train(files=["UD_Turkish-Penn/tr_penn-ud-train.txt", "UD_Turkish-Penn/tr_penn-ud-dev.txt"], trainer=trainer)

    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

    encoding = tokenizer.encode("Merhaba bu bir kelime")
    print("Encoded string: {}".format(encoding.tokens))

    decoded = tokenizer.decode(encoding.ids)
    print("Decoded string: {}".format(decoded))

    vocab = tokenizer.get_vocab()
    print(vocab)

    # Can get count of each word like this.
    print(vocab['kendi'])

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
    print(perplexity(test_tokenized.tokens, model))

if __name__ == '__main__':
    main()