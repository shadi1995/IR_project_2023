import re
import string
from collections import defaultdict


class LinguisticCorrection:

    @staticmethod
    def get_levenshtein_distance(word1, word2):
        word2 = word2.lower()
        word1 = word1.lower()
        matrix = [[0 for x in range(len(word2) + 1)] for x in range(len(word1) + 1)]

        for x in range(len(word1) + 1):
            matrix[x][0] = x
        for y in range(len(word2) + 1):
            matrix[0][y] = y

        for x in range(1, len(word1) + 1):
            for y in range(1, len(word2) + 1):
                if word1[x - 1] == word2[y - 1]:
                    matrix[x][y] = min(
                        matrix[x - 1][y] + 1,
                        matrix[x - 1][y - 1],
                        matrix[x][y - 1] + 1
                    )
                else:
                    matrix[x][y] = min(
                        matrix[x - 1][y] + 1,
                        matrix[x - 1][y - 1] + 1,
                        matrix[x][y - 1] + 1
                    )

        return matrix[len(word1)][len(word2)]

    @staticmethod
    def get_soundex(word):
        word = re.sub(r'\W|\d|_', '', word.lower())

        if not word:
            return "0000"

        '''
        letters to soundex map is
        ('b', 'f', 'p', 'v'): 1 
        ('c', 'g', 'j', 'k', 'q', 's', 'x', 'z'): 2
        ('d', 't'): 3, ('l',): 4
        ('m', 'n'): 5, ('r',): 6

        remaining letters ('a', 'e', 'i', 'o', 'u', 'y', 'h', 'w') are removed
        '''

        regex = r'[aeiouyhw]'

        char_to_soundex = str.maketrans(re.sub(regex, '', string.ascii_lowercase), "123122245512623122")

        digits = word[0].upper() + re.sub(regex, '', word[1:]).translate(char_to_soundex)

        code = digits[0]
        for d in digits[1:]:
            if code[-1] != d:
                code += d

        return code.ljust(4, '0')[:4]

    def best_similarity(self, word, words_bag):
        ranks = defaultdict(int)
        for w in words_bag:
            ranks[w] = self.get_levenshtein_distance(word, w)
        return sorted(ranks.items(), key=lambda item: item[1])[0]
