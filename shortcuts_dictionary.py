import re


class ShortcutsDictionary:
    shortcuts_dictionary = {
        'usa': 'united states of america',
        'us': 'united states of america',
        'un': 'united nations',
        'su': 'soviet union',
        'uk': 'united kingdom',
    }

    def process_text(self, text: str):
        return ' '.join(
            [self.shortcuts_dictionary.get(re.sub(r'\W+', '', word).strip(), word) for word in text.lower().split()])
