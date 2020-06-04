# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:27:07 2020

@author: Quang
"""

import string

# load doc into memory
def load_doc(filename):
    # open the file as read only
    with (open(filename, "r")) as file:
        text = file.read()
    return text


# extract descriptions  for images
def load_descriptions(doc):
    mapping = dict()
    # process lines
    for line in doc.split("\n"):
        # split line by white space
        tokens = line.split()
        # ignore line with less than two words
        if len(line) < 2:
            continue

        # take the first token as the image id
        # the rest as the desripton
        image_id, image_desc = tokens[0], tokens[1:]

        # remove filename from image id
        image_id = image_id.split(".")[0]

        # convert description tokens back to a full line
        image_desc = " ".join(image_desc)

        # create the list if need
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping


def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans("", "", string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]

            # tokenize
            desc = desc.split()

            # convert to lower case
            desc = [word.lower() for word in desc]

            # replace punctuation to null string from each token
            desc = [w.translate(table) for w in desc]

            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]

            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]

            # store as string
            desc_list[i] = " ".join(desc)


# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + " " + desc)
    data = "\n".join(lines)
    with open(filename, "w") as file:
        file.write(data)


def main():
    filename = "../Flickr8k_text/Flickr8k.token.txt"

    # load descriptions
    doc = load_doc(filename)

    # parse descriptions
    descriptions = load_descriptions(doc)
    print("Loaded: %d " % len(descriptions))

    # clean descriptions
    clean_descriptions(descriptions)

    # sumarize vocabulary
    vocabulary = to_vocabulary(descriptions)
    print("Vocabulary size: %d" % len(vocabulary))

    # save to file
    save_descriptions(descriptions, "descriptions.txt")


if __name__ == "__main__":
    main()
