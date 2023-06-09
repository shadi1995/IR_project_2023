#!/usr/bin/env python
import math
import os
import re
from collections import defaultdict
from pathlib import Path
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from linguistic_correcter import LinguisticCorrection
from shortcuts_dictionary import ShortcutsDictionary


class IndexCreator:
    shortcuts_dictionary = ShortcutsDictionary()
    linguistic_correction = LinguisticCorrection()
    monthDictionary = dict(jan='01', feb='02', mar='03', apr='04', may='05', jun='06', jul='07', aug='08', sep='09',
                           oct='10', nov='11', dec='12')

    def __init__(self, **args):
        self.corpusDir = args.get('corpusDir', 'D:/IR_Project/corpus')
        self.stopWordsFile = args.get('stopWordsFile', 'D:/IR_Project/stop words.txt')
        self.indexFile = args.get('indexFile', 'D:/IR_Project/index.txt')
        self.soundexIndexFile = args.get('soundexIndexFile', 'D:/IR_Project/soundex-index.txt')
        self.stopWords = []
        self.index = defaultdict(list)
        self.soundex_dic = defaultdict(set)
        self.tfIndex = defaultdict(list)
        self.dfIndex = defaultdict(int)
        self.numOfDocuments = 0
        self.porter_stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def extract_stop_words(self):
        with open(self.stopWordsFile) as file:
            self.stopWords = [line.strip().lower() for line in file if line.strip()]

    def extract_terms(self, text):
        text = self.shortcuts_dictionary.process_text(text)
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        dates = self.date_converter(text)
        text = nltk.word_tokenize(text)
        words = [word for word in text if word not in self.stopWords]
        terms = []
        for word in words:
            term = self.porter_stemmer.stem(self.lemmatizer.lemmatize(word, 'v'))
            terms.append(term)
            if not re.search(r'\W|\d', term):
                self.soundex_dic[self.linguistic_correction.get_soundex(term)].add(term)
        for date in dates:
            terms.append(date)
        return terms

    def word_to_num(self, word):
        s = word.lower()[:3]
        return self.monthDictionary[s]

    def date_converter(self, line):
        results = []
        day = None
        month = None
        year = None
        regex = re.search(r'([0]?\d|[1][0-2])[/-]([0-3]?\d)[/-]([1-2]\d{3}|\d{2})', line)
        month_regex = re.search(
            r'([0-3]?\d)\s*(Jan(?:uary)?(?:aury)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug('
            '?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?(?:emeber)?).?,?\s([1-2]\d{3})',
            line)
        rev_month_regex = re.search(
            r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct('
            '?:ober)?|Nov(?:ember)?|Dec(?:ember)?).?[-\s]([0-3]?\d)(?:st|nd|rd|th)?[-,\s]\s*([1-2]\d{3})',
            line)
        no_day_regex = re.search(
            r'(Jan(?:uary)?(?:aury)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?('
            '?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?(?:emeber)?).?,?[\s]([1-2]\d{3}|\d{2})',
            line)
        no_day_digits_regex = re.search(r'([0]?\d|[1][0-2])[/\s]([1-2]\d{3})', line)
        year_only_regex = re.search(r'([1-2]\d{3})', line)
        if regex:
            month = regex.group(1)
            day = regex.group(2)
            year = regex.group(3)
        elif month_regex:
            day = month_regex.group(1)
            month = self.word_to_num(month_regex.group(2))
            year = month_regex.group(3)
        elif rev_month_regex:
            day = rev_month_regex.group(2)
            month = self.word_to_num(rev_month_regex.group(1))
            year = rev_month_regex.group(3)
        elif no_day_regex:
            month = self.word_to_num(no_day_regex.group(1))
            year = no_day_regex.group(2)
        elif no_day_digits_regex:
            month = no_day_digits_regex.group(1)
            year = no_day_digits_regex.group(2)
        elif year_only_regex:
            year = year_only_regex.group(0)
        if day or month or year:
            year = year if year else '1900'
            month = month.zfill(2) if month else '01'
            day = day.zfill(2) if day else '01'
            if day == '00':
                day = '01'
            if len(year) == 2:
                year = '19' + year
            results.append(year + month + day)
        return results

    def process_query(self, query):
        terms = self.extract_terms(query)
        print(terms)
        if not len(terms):
            return 'Not found'

        union = set()
        for term in terms:
            if not self.index.get(term, None):
                continue
            term_documents = self.index[term]
            documents_names = {term_document[0] for term_document in term_documents}
            union |= documents_names

        return self.rank_documents(terms, list(union))

    def parse_corpus(self):
        files_list = os.listdir(self.corpusDir)
        for file in files_list:
            yield self.parse_document(os.path.join(self.corpusDir, file))

    def parse_document(self, document_file):
        with open(document_file) as file:
            lines = '\n'.join(file.readlines())
        document_name = Path(document_file).stem
        return {'name': document_name, 'terms': self.extract_terms(lines)} if lines else {}

    def create_index(self):
        self.extract_stop_words()
        for document in self.parse_corpus():
            if document:
                self.numOfDocuments += 1
                document_name = document['name']
                terms = document['terms']
                document_index = defaultdict(lambda: [document_name, []])
                for position, term in enumerate(terms):
                    document_index[term][1].append(position)

                norm = math.sqrt(
                    sum([len(positions_vector) ** 2 for term, (document_name, positions_vector) in
                         document_index.items()]))

                for term, (document_name, positions_vector) in document_index.items():
                    self.tfIndex[term].append('%.5f' % (len(positions_vector) / norm))
                    self.dfIndex[term] += 1

                for term, positions_vector in document_index.items():
                    self.index[term].append(positions_vector)
        self.write_index()

    def write_index(self):
        with open(self.indexFile, 'w') as file:
            print(self.numOfDocuments, file=file)
            for term in self.index.keys():
                term_documents = []
                for document_term_index in self.index[term]:
                    document_name = document_term_index[0]
                    positions_vector = document_term_index[1]
                    term_documents.append(':'.join([str(document_name), ','.join(map(str, positions_vector))]))

                term_documents_positions = ';'.join(term_documents)
                documents_tf = ','.join(map(str, self.tfIndex[term]))
                term_idf = '%.5f' % (self.numOfDocuments / self.dfIndex[term])
                print('|'.join((term, term_documents_positions, documents_tf, term_idf)), file=file)

        with open(self.soundexIndexFile, 'w') as file:
            print('\n'.join([code + ':' + ','.join(group) for code, group in self.soundex_dic.items()]), file=file)


def main():
    # corpus_dir = input('Enter corpus directory path')
    # stop_words_file = input('Enter stop words file path')
    # index_file = input('Enter index file output path')
    IndexCreator().create_index()


if __name__ == '__main__':
    main()
