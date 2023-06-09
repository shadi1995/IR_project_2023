#!/usr/bin/env python
import copy
import functools
import re
from collections import defaultdict
from operator import itemgetter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from linguistic_correcter import LinguisticCorrection
from shortcuts_dictionary import ShortcutsDictionary


class IndexQuery:
    shortcuts_dictionary = ShortcutsDictionary()
    linguistic_correction = LinguisticCorrection()
    monthDictionary = dict(jan='01', feb='02', mar='03', apr='04', may='05', jun='06', jul='07', aug='08', sep='09',
                           oct='10', nov='11', dec='12')

    def __init__(self, **args):
        self.stopWordsFile = args.get('stopWordsFile', 'D:/IR_Project/stop words.txt')
        self.indexFile = args.get('indexFile', 'D:/IR_Project/index.txt')
        self.soundexIndexFile = args.get('soundexIndexFile', 'D:/IR_Project/soundex-index.txt')
        self.stopWords = []
        self.index = defaultdict(list)
        self.soundex_dic = defaultdict(list)
        self.tfIndex = defaultdict(list)
        self.idfIndex = defaultdict(float)
        self.numOfDocuments = 0
        self.porter_stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.extract_stop_words()
        self.read_index()
        self.read_soundex_index()

    def extract_stop_words(self):
        with open(self.stopWordsFile) as file:
            self.stopWords = [line.strip().lower() for line in file if line.strip()]

    def extract_terms(self, text):
        text = self.shortcuts_dictionary.process_text(text)
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        dates = self.date_converter(text)
        text = nltk.word_tokenize(text)
        words = [word for word in text if word not in self.stopWords]
        terms = [self.porter_stemmer.stem(self.lemmatizer.lemmatize(word, 'v')) for word in words]
        for date in dates:
            terms.append(date)
        return terms

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

    def word_to_num(self, word):
        s = word.lower()[:3]
        return self.monthDictionary[s]

    def read_index(self):
        with open(self.indexFile) as file:
            self.numOfDocuments = float(file.readline().strip())
            for line in file:
                line = line.rstrip()
                term, term_documents, documents_tf, term_idf = line.split('|')
                term_documents = term_documents.split(';')
                term_documents = [x.split(':') for x in term_documents]
                term_documents = [[x[0], map(int, x[1].split(','))] for x in term_documents]
                self.index[term] = term_documents
                documents_tf = documents_tf.split(',')
                self.tfIndex[term] = [float(tf) for tf in documents_tf]
                self.idfIndex[term] = float(term_idf)

    def read_soundex_index(self):
        with open(self.soundexIndexFile) as file:
            for line in file:
                if not line:
                    continue
                split = line.strip().split(':')
                self.soundex_dic[split[0]] = split[1].split(',')

    def rank_documents(self, terms, documents):
        vector_space = defaultdict(lambda: [0] * len(terms))
        query_vector = [0] * len(terms)

        for term_index, term in enumerate(terms):
            if term not in self.index:
                continue

            query_vector[term_index] = self.idfIndex[term]

            for docIndex, (document_name, positions_vector) in enumerate(self.index[term]):
                if document_name in documents:
                    vector_space[document_name][term_index] = self.tfIndex[term][docIndex]

        doc_scores = [[document_name,
                       self.dot_product(document_vector, query_vector)]
                      for document_name, document_vector in vector_space.items()]

        doc_scores.sort(reverse=True, key=itemgetter(1))

        return [document_name[0] for document_name in doc_scores]

    @staticmethod
    def dot_product(vec1, vec2):
        if len(vec1) != len(vec2):
            return 0
        return sum([x * y for x, y in zip(vec1, vec2)])

    def try_term_correction(self, term):
        term_code = self.linguistic_correction.get_soundex(term)
        group = self.soundex_dic.get(term_code, None)
        if not group:
            return None
        return self.linguistic_correction.best_similarity(term, group)

    def process_any_query(self, query):
        terms = self.extract_terms(query)
        if not len(terms):
            return 'Not found'

        union = set()
        for term in terms:
            if not self.index.get(term, None):
                term_correction = self.try_term_correction(term)
                if not term_correction or term_correction[1] > 3:
                    continue
                term = term_correction[0]
            term_documents = self.index[term]
            documents_names = {term_document[0] for term_document in term_documents}
            union |= documents_names

        return self.rank_documents(terms, list(union))

    def process_quote_query(self, query):
        terms = self.extract_terms(query)
        if not len(terms):
            return 'Not found'

        for index, term in enumerate(terms):
            if term not in self.index:
                term_correction = self.try_term_correction(term)
                if not term_correction or term_correction[1] > 3:
                    # if a term doesn't appear in the index
                    # there can't be any document matching it
                    return []
                terms[index] = term_correction[0]

        docs_vector = [self.index[term] for term in terms]
        docs = [[x[0] for x in p] for p in docs_vector]

        docs = self.intersect_lists(docs)

        for i in range(len(docs_vector)):
            docs_vector[i] = [x for x in docs_vector[i] if x[0] in docs]

        docs_vector = copy.deepcopy(docs_vector)

        for i in range(len(docs_vector)):
            for j in range(len(docs_vector[i])):
                docs_vector[i][j][1] = [x - i for x in docs_vector[i][j][1]]

        result = []
        for i in range(len(docs_vector[0])):
            li = self.intersect_lists([x[i][1] for x in docs_vector])
            if not li:
                continue
            else:
                result.append(docs_vector[0][i][0])

        return self.rank_documents(terms, result)

    @staticmethod
    def intersect_lists(lists):
        if len(lists) == 0:
            return []
        # start intersecting from the smaller list
        lists.sort(key=len)
        return list(functools.reduce(lambda x, y: set(x) & set(y), lists))

    def query(self, search_query):
        return self.process_quote_query(search_query) if search_query.startswith('"') \
            else self.process_any_query(search_query)


def main():
    index_query = IndexQuery()
    while True:
        search_query = input('Enter something to find:\n')

        if search_query == 'exit':
            break

        result = index_query.query(search_query)
        print(f'Total number of results: {len(result)} document')
        print('\n'.join(result))


if __name__ == '__main__':
    main()
