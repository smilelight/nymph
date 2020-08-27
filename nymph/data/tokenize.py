# -*- coding: utf-8 -*-


def default_tokenizer(sentence: str):
    return sentence.split()


if __name__ == '__main__':
    sent = "曾经 沧海 难 为 水"
    print(default_tokenizer(sent))
