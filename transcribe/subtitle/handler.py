import string
import re
import threading
from contextlib import contextmanager
from functools import lru_cache
from typing import List, Dict, Any
import logging
import sys
try:
    import jieba
except ImportError:
    jieba = None

try:
    from spacy.tokens import Token
    from spacy.language import Language
except ImportError:
    Token = Any
    Language = Any

logger = logging.getLogger(__name__)

valid_punctuation = "".join([p for p in string.punctuation if p not in ["'", "%", "$"]])
good_punctuation_str = '.,?!:;<>[]+-*/="_'
good_punctuation_re = re.escape(good_punctuation_str)
omit_punctuation = "".join(
    [
        p
        for p in string.punctuation
        if p not in (list(good_punctuation_str) + ["'", "%"])
    ]
)
# string.punctuation = good_punctuation + omit_punctuation + "'" + "%"
# valid_punctuation = good_punctuation + omit_punctuation


def remove_punctuation(text: str):
    # remove all punctuation but instead "'" "%"
    text = re.sub(rf"[{re.escape(valid_punctuation)}]", " ", text)

    text = re.sub(r"--", "-", text)
    text = re.sub(r"—", "-", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r" {2,}", " ", text)

    text = re.sub(r"' ", " ", text)
    text = re.sub(r" '", " ", text)
    text = re.sub(r"\$", "\\$", text)

    return text.strip()


def add_punctuation(text: str, model):
    # model = PunctuationModel(model="kredor/punctuate-all")
    # model = PunctuationModel()
    logger.info("restore punction:")
    logger.info(f"  before text: {text}")
    text = model.restore_punctuation(text)
    logger.info(f"   after text: {text}")
    return re.sub(rf"{re.escape(omit_punctuation)}", "", text)


def split_text_by_word(text: str, lang: str = "zh") -> List[str]:
    if lang == "zh":
        words = jieba.lcut(text, cut_all=False) if jieba else list(text)
    else:
        words = text.split(" ")
    return words


def add_spaces_around_english(text: str) -> str:
    result = re.sub(
        r"[^\u4e00-\u9fff \u3002 \uff1f \uff01 \uff0c \u3001 \uff1b \uff1a]+",
        r" \g<0> ",
        text,
    )
    result = re.sub(r" *([。，！？：；、《》·（）“”\"]) *", r"\1", result).strip()
    result = re.sub(r" *([,.?!;:])", r"\1", result).strip()
    result = re.sub(r" +", r" ", result).strip()
    result = re.sub(r" ?\n ?", r"\n", result).strip()
    result = re.sub(r"‘\n", r"‘", result).strip()
    result = re.sub(r"\n’", r"’", result).strip()

    return result


class LanguageContext:
    _local = threading.local()

    @classmethod
    def set_language(cls, language):
        setattr(cls._local, "language", language)

    @classmethod
    def get_language(cls):
        return getattr(cls._local, "language", "en")


@contextmanager
def language_context(language):
    old_language = LanguageContext.get_language()
    LanguageContext.set_language(language)
    try:
        yield
    finally:
        LanguageContext.set_language(old_language)


def is_all_english(text: str) -> bool:
    return bool(re.match(r'^[a-zA-Z\s\.,?!:;\'"()\[\]{}-]+$', text))

def count_chinese_word(text: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff]', text))

class LanguageHandler:
    end_symbols = []
    mid_symbols = []
    fake_symbols = []

    def __init__(self):
        pass

    @staticmethod
    @lru_cache
    def get_handlers_map():
        _handlers: Dict[str, Any] = {
            "en": EnglishHandler,
            "da": EnglishHandler,
            "es": EnglishHandler,
            "zh": ChineseHandler,
        }
        return _handlers

    @staticmethod
    def get_handler(language):
        handlers = LanguageHandler.get_handlers_map()
        if language.lower() in handlers:
            return handlers[language.lower()]()
        else:
            raise ValueError(f"Unsupported language: {language}")

    @staticmethod
    def guess_handler(text):
        language = LanguageHandler.guess_language(text)
        return LanguageHandler.get_handler(language)

    @staticmethod
    def guess_language(text):
        if is_all_english(text):
            return "en"
        if count_chinese_word(text) > 0:
            return "zh"
        # langid.set_languages(["zh", "en"])
        # language = langid.classify(text)[0]
        language = "en"
        return language

    def is_end(self, text, use_end_punctuation):
        raise NotImplementedError

    def clean_punctuation(self, text):
        raise NotImplementedError

    def length(self, text):
        raise NotImplementedError

    def join(self, texts: List[str]):
        raise NotImplementedError

    def split(self, text, **kwargs):
        raise NotImplementedError

    def concat_length(self, text1, text2):
        return self.length(self.join([text1, text2]))

    def split_by_max_length(self, text, max_length: int, max_line: int = 1, **kwargs):
        current_line = ""
        lines = []
        if kwargs.get("order"):
            return self.split_by_max_length_order(text, max_length, max_line, **kwargs)
        for word in reversed(self.split(text, **kwargs)):
            current_len = self.concat_length(word, current_line)
            if current_line and current_len > max_length:
                lines = [current_line] + lines
                current_line = word
            else:
                current_line = self.join([word, current_line])
        if current_line:
            lines = [current_line] + lines

        if len(lines) > 1 and self.length(lines[0]) <= 3:
            lines[1] = self.join([lines[0], lines[1]])
            lines = lines[1:]

        multi_lines = []
        for idx in range(0, len(lines), max_line):
            join_line = lines[idx : idx + max_line]
            multi_lines.append("\n".join(join_line))

        return multi_lines

    def split_by_max_length_order(self, text, max_length: int, max_line: int = 1, **kwargs):
        current_line = ""
        lines = []
        for word in self.split(text, **kwargs):
            current_len = self.concat_length(current_line, word)
            if current_line and current_len > max_length:
                lines = lines + [current_line]
                current_line = word
            else:
                current_line = self.join([current_line, word])
        if current_line:
            lines = lines + [current_line]

        if len(lines) > 1 and self.length(lines[-1]) <= 3:
            lines[-2] = self.join([lines[-2], lines[-1]])
            lines = lines[:-1]

        multi_lines = []
        for idx in range(0, len(lines), max_line):
            join_line = lines[idx : idx + max_line]
            multi_lines.append("\n".join(join_line))

        return multi_lines

    def split_by_punctuation(self, text, use_end_punctuation=True):
        segments = []
        segment = []
        for word in self.split(text):
            word = word.strip()
            segment.append(word)
            if self.is_end(word, use_end_punctuation):
                segments.append(self.join(segment))
                segment = []
        if segment:
            segments.append(self.join(segment))
        return segments

    def is_preposition(self, token: Token) -> bool:
        # 定义介词规则
        # fmt: off
        prep_rules = {
            "en": ["in", "on", "at", "by", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "over", "under", "again"],
            "zh": ["在", "用", "关于", "通过", "对于", "直到"],
            "fr": ["dans", "sur", "chez", "avec", "par", "entre", "contre", "pour", "en", "sous"],
            "es": ["en", "con", "de", "a", "para", "por", "sobre", "entre", "hacia"],
            "de": ["in", "auf", "bei", "mit", "durch", "unter", "zwischen", "zu", "von", "aus"],
            "it": ["in", "su", "con", "di", "tra", "fra", "per", "da"],
            "ru": ["в", "на", "с", "под", "между", "над"],
            "ja": ["で", "に", "を", "から", "まで", "へ", "と", "の", "として"]
        }
        # fmt: on

        language = LanguageContext.get_language()
        prepositions = prep_rules.get(language, [])

        return token.text.lower() in prepositions and token.dep_ == "prep"

    def is_conjunction(self, token: Token) -> bool:
        noun_pos = ["NOUN", "PROPN"]
        verb_pos = "VERB"
        mark_dep = ["mark"]
        det_pron_deps = ["det", "pron"]
        # fmt: off
        rules = {
            "en": [["that", "which", "where", "what", "when", "because", "but", "and", "or", "if", "so"], verb_pos, [*mark_dep, "nsubj", "dobj"], noun_pos, det_pron_deps],
            "zh": [["因为", "所以", "但是", "而且", "虽然", "如果", "即使", "尽管"], verb_pos, mark_dep, noun_pos, det_pron_deps],
            "ja": [["けれども", "しかし", "だから", "それで", "ので", "のに", "ため"], verb_pos, mark_dep, noun_pos, ["case"]],
            "fr": [["que", "qui", "où", "quand", "parce que", "mais", "et", "ou"], verb_pos, mark_dep, noun_pos, det_pron_deps],
            "ru": [["что", "который", "где", "когда", "потому что", "но", "и", "или"], verb_pos, mark_dep, noun_pos, ["det"]],
            "es": [["que", "cual", "donde", "cuando", "porque", "pero", "y", "o"], verb_pos, mark_dep, noun_pos, det_pron_deps],
            "de": [["dass", "welche", "wo", "wann", "weil", "aber", "und", "oder"], verb_pos, mark_dep, noun_pos, det_pron_deps],
            "it": [["che", "quale", "dove", "quando", "perché", "ma", "e", "o"], verb_pos, mark_dep, noun_pos, det_pron_deps],
        }
        # fmt: on

        language = LanguageContext.get_language()
        conjunction, verb_pos, mark_dep, noun_pos, det_pron_deps = rules.get(language)
        if token.text.lower() not in conjunction:
            return False

        if language == "en" and token.text.lower() == "that":
            if token.dep_ in mark_dep and token.head.pos_ == verb_pos:
                return True
            else:
                return False
        elif token.dep_ in det_pron_deps and token.head.pos_ in noun_pos:
            return False
        else:
            return True

    def split_by_nlp_model(
        self, text, context_words: int = 5, nlp: Language = None, by_conjunction=True
    ):
        # from: https://github.com/Huanshere/VideoLingo/blob/core/spacy_utils/split_by_connector.py#L84
        root_doc = nlp(text)
        sentences = [root_doc.text]

        while True:
            split_occurred = False
            new_sentences = []

            for sent in sentences:
                doc = nlp(sent)
                start = 0

                for i, token in enumerate(doc):
                    if by_conjunction:
                        split_word = self.is_conjunction(token)
                    else:
                        split_word = self.is_preposition(token)

                    if not split_word or (
                        i + 1 < len(doc) and doc[i + 1].text[0] == "'"
                    ):
                        continue

                    left_words = doc[max(0, token.i - context_words) : token.i]
                    right_words = doc[
                        token.i + 1 : min(len(doc), token.i + context_words + 1)
                    ]

                    left_words = [word.text for word in left_words if not word.is_punct]
                    right_words = [
                        word.text for word in right_words if not word.is_punct
                    ]

                    if (
                        len(left_words) >= context_words
                        and len(right_words) >= context_words
                    ):
                        new_sentences.append(doc[start : token.i].text.strip())
                        start = token.i
                        split_occurred = True
                        break

                if start < len(doc):
                    new_sentences.append(doc[start:].text.strip())

            if not split_occurred:
                break

            sentences = new_sentences

        return sentences

    def merge_by_max_length(self, texts, max_length):
        new_texts = []
        last_text = ""
        for text in texts:
            if last_text and self.concat_length(last_text, text) > max_length:
                new_texts.append(last_text)
                last_text = text
            else:
                last_text = self.join([last_text, text])
        if last_text:
            new_texts.append(last_text)

        return new_texts

    def calculate_phrase_timestamps(self, phrase: str, word_timestamps: "Elements"):
        # 这里可能的问题是，word_timestamps可能不是单词的时间戳
        words = self.split(self.join([i.strip() for i in phrase.split("\n")]))
        phrase_length = len(words)
        for i in range(len(word_timestamps) - phrase_length + 1):
            window = word_timestamps[i : i + phrase_length]
            if all(
                self.clean_punctuation(window[j].text)
                == self.clean_punctuation(words[j])
                for j in range(phrase_length)
            ):
                word_timestamps[i : i + phrase_length] = []
                return window[0].start, window[-1].end
        raise ValueError(
            f"The phrase ({phrase}) does not match a continuous sequence in word_timestamps ({word_timestamps})."
        )

    def calculate_word_timestamps(self, sentence: "Elements"):
        from .subtitle import Elements, Element
        timestamps = Elements()
        for element in sentence:
            words = self.split(element.text)
            durations = element.end - element.start

            word_lengths = [self.length(word) for word in words]
            total_length = sum(word_lengths)
            word_durations = [
                durations * (length / total_length) for length in word_lengths
            ]

            current_time = element.start
            for word, duration in zip(words, word_durations):
                timestamps.append(
                    Element(
                        round(current_time, 3), round(current_time + duration, 3), word
                    )
                )
                current_time += duration
        return timestamps

    def correct_incorrect_restore_punctuation(self, before_text, after_text):
        raise NotImplementedError

    def correct_sentence_punctuation_by_words_list(
        self, sentence: str, words_list: List
    ):
        # 这里可能的问题是，word_timestamps可能不是单词的时间戳
        words = self.split(self.join([i.strip() for i in sentence.split("\n")]))
        sentence_length = len(words)
        for i in range(len(words_list) - sentence_length + 1):
            window = words_list[i : i + sentence_length]
            if all(
                self.clean_punctuation(window[j]) == self.clean_punctuation(words[j])
                for j in range(sentence_length)
            ):
                words_list[i : i + sentence_length] = []
                return self.join(window)
        raise ValueError(
            f"The phrase '{sentence}' does not match a continuous sequence in word_timestamps."
        )

    def check_and_correct_split_sentence(
        self,
        split_sentence: List[str],
        sentence: str,
        can_reverse: bool = True,
    ):
        words_list = self.split(sentence)
        fixed_texts = []
        assert (
            len(split_sentence) == 2
        ), f"check_and_correct_split_sentence() first arg length should be 2, but got {len(split_sentence)}"
        for i in split_sentence:
            try:
                fixed = self.correct_sentence_punctuation_by_words_list(i, words_list)
            except Exception as e:
                fixed = ""
            fixed_texts.append(fixed)
        if all(not i for i in fixed_texts):
            return False, fixed_texts

        # 下面应当尽全力去匹配结果
        # 一定存在匹配项，匹配项可能匹配的是中间，或者匹配两边
        first = fixed_texts[0]
        second = fixed_texts[1]
        if first and (not second):
            fixed_texts = [first, re.sub(re.escape(first), "", sentence)]
            second = fixed_texts[1]
        elif (not first) and second:
            fixed_texts = [re.sub(re.escape(second), "", sentence), second]
            first = fixed_texts[0]

        sent = re.sub(" *", "", sentence)
        if (
            re.sub(" *", "", self.join([first, second])) == sent
            or re.sub(" *", "", self.join([second, first])) == sent
        ):
            good = True
        else:
            # 说明只匹配到了sentence的一部分
            good = False
            if first:
                fixed_texts = [first, re.sub(re.escape(first), "", sentence)]
                if re.sub(" *", "", self.join(fixed_texts)) == sent:
                    good = True
            if (not good) and second:
                fixed_texts = [re.sub(re.escape(second), "", sentence), second]
                if re.sub(" *", "", self.join(fixed_texts)) == sent:
                    good = True
            logger.warning(
                f"good: {good}; first: {first}; second: {second}; sentence: {sentence}; fixed_texts: {fixed_texts}"
            )
        if good:
            if can_reverse:
                if (
                    second
                    and first
                    and re.sub(" *", "", self.join([second, first])) == sent
                    and (first[-1] in ["。", "？", "！"])
                    and (second[-1] in ["，", "、"])
                ):
                    schar = second[-1]
                    fchar = first[-1]
                    second = second[:-1] + fchar
                    first = first[:-1] + schar
                    fixed_texts = [first, second]
            elif re.sub(" *", "", self.join([second, first])) == sent:
                fixed_texts = [second, first]

        return good, [i.strip() for i in fixed_texts]


class EnglishHandler(LanguageHandler):
    end_symbols = [".", "?", "!", "]"]
    mid_symbols = [",", ":", ";", "-"]
    fake_symbols = ["?]", "?[", "]?", "[?", "Mr.", "Mrs.", "Dr.", "Ms.", "etc.", "i.e.", "e.g.", "et al.", "a.m.", "p.m.", "Prof.", "Ph.D."]

    def __init__(self):
        super().__init__()

    def is_end(self, text, use_end_punctuation):
        if not text:
            return False
        check_symbols = self.end_symbols if use_end_punctuation else self.mid_symbols
        if text[-1] in check_symbols:
            if len(text) > 2 and text.endswith(tuple(self.fake_symbols)):
                return False
            else:
                return True
        return False

    def clean_punctuation(
        self, text, left_strip: bool = True, right_strip: bool = True
    ):
        strip_punc = ",.?!;:'\"-_“”‘’[]/() "
        if left_strip:
            text = text.lstrip(strip_punc)
        if right_strip:
            text = text.rstrip(strip_punc)
        return text

    def length(self, text):
        return len(text)

    def split(self, text: str, **kwargs):
        return text.split()

    def join(self, texts: List[str]):
        texts = [text for text in texts if text]
        return " ".join(texts)

    def correct_incorrect_restore_punctuation(self, before_text, after_text):
        pattern = r"\S+[,\.-:/?=]\S+"
        special_words = re.findall(pattern, before_text)
        if not special_words:
            return after_text

        before_word = before_text.split()
        after_word = after_text.split()
        if len(before_word) != len(after_word):
            logger.error(f"before_word: {before_word}")
            logger.error(f"after_word: {after_word}")
            logger.error(f"len(before_word): {len(before_word)}")
            logger.error(f"len(after_word): {len(after_word)}")
        assert len(before_word) == len(
            after_word
        ), "punctuation restore corner case, should fix code."

        fixed_text = []
        for before, after in zip(before_word, after_word):
            if re.findall(pattern, before):
                fixed_text.append(before)
            else:
                after = re.sub(r",+", ",", after)
                fixed_text.append(after)

        return " ".join(fixed_text)

    def find_half_join_balance(self, texts: List[str]):
        if len(texts) <= 2:
            return [1]
        weights = [1 if self.is_end(text, False) else sys.maxsize for text in texts]
        lengths = [self.length(text) for text in texts]
        diffs = [
            (weights[i - 1] * abs(sum(lengths[:i]) - sum(lengths[i:])))
            for i in range(1, len(texts))
        ]
        sorted_indices = sorted(range(len(diffs)), key=lambda i: diffs[i])
        sorted_indices = [i + 1 for i in sorted_indices]
        return sorted_indices


class ChineseHandler(LanguageHandler):
    end_symbols = ["。", "？", "！", "…"]
    mid_symbols = ["，", "：", "；", "、"]
    right_punc = "。？！，、；：’”）】》〉…"
    left_punc = "【（〈《“‘"
    fake_symbols = []

    def __init__(self):
        super().__init__()
        self.punctuation = self.right_punc + self.left_punc

    def is_end(self, text, use_end_punctuation):
        check_symbols = self.end_symbols if use_end_punctuation else self.mid_symbols
        if not text:
            return False
        if text[-1] in check_symbols:
            return True
        return False

    def clean_punctuation(
        self, text: str, left_strip: bool = True, right_strip: bool = True
    ):
        strip_punc = "。？！、，；：（）【】《》‘’”“-—.?!,;:'\"-_·… "
        if left_strip:
            text = text.lstrip(strip_punc)
        if right_strip:
            text = text.rstrip(strip_punc)
        return text

    def length(self, text):
        num = 0
        for char in self.split(text):
            if len(char) == 1:
                num += len(char)
            else:
                if any(ord(i) > 255 for i in char):
                    num += len(char)
                else:
                    num += (len(char) + 1) // 2
        return num

    def norm_and_beauty(self, text):
        return self.join(self.split(text))

    def split(self, text, **kwargs):
        # split
        last_char = ""
        tokens = []
        if kwargs.get("jieba_preprocess", False):
            text = jieba.lcut(text, cut_all=False)
        for i in text:
            if i.strip() and (
                self.isascii(i) or i[0] in (self.right_punc + self.left_punc + "—")
            ):
                if (
                    last_char
                    and (last_char[-1] in (self.right_punc + self.left_punc))
                    and (i not in (self.right_punc + self.left_punc))
                ):
                    tokens.append(last_char)
                    last_char = ""
                # 连续的标点和英文全部放在一组，"—"不会前后向合并
                last_char += i
            else:
                tokens.append(last_char)
                last_char = ""
                tokens.append(i)
        tokens.append(last_char)
        tokens = [i for i in tokens if i.strip()]

        # merge
        idx = 1
        while idx < len(tokens):
            # 前向和中文合并
            while tokens[idx] and tokens[idx][0] in self.right_punc:
                tokens[idx - 1] += tokens[idx][0]
                tokens[idx] = tokens[idx][1:]
            idx += 1
        tokens = [i for i in tokens if i]

        # merge
        idx = 1
        while idx < len(tokens):
            # 后向和中文合并
            while tokens[idx - 1] and tokens[idx - 1][-1] in self.left_punc:
                tokens[idx] = tokens[idx - 1][-1] + tokens[idx]
                tokens[idx - 1] = tokens[idx - 1][:-1]
            idx += 1

        return [i for i in tokens if i]

    # def isascii(self, chr: str):
    #     return chr.isascii() or (
    #         chr[0] in "⋯·ϕπ√βσΣθΘαγζρΣδΔεηκλμνξτφΦχψΩω−×÷≠≈<>≤≥√∞∫∑∝∩∏∂∇∈∉∩∪⊂⊃∅∠∥⊥"
    #     )
    def isascii(self, chr: str):
        # fmt: off
        punc = [
            0x3002, 0xFF1F, 0xFF01, 0x3010, 0x3011, 0xFF0C, 0x3001, 0xFF1B,
            0xFF1A, 0x300C, 0x300D, 0x300E, 0x300F, 0x2019, 0x201C, 0x201D,
            0x2018, 0xFF08, 0xFF09, 0x3014, 0x3015, 0x2026, 0x2013, 0xFF0E,
            0x2014, 0x300A, 0x300B, 0x3008, 0x3009
        ]
        # fmt: on
        unicode_ranges = [
            (0x4E00, 0x9FFF),  # 中日韩统一表意文字
            (0x3040, 0x309F),  # 平假名
            (0x30A0, 0x30FF),  # 片假名
            (0xAC00, 0xD7AF),  # 韩文音节
            (0x0E00, 0x0E7F),  # 泰文
            (0x0600, 0x06FF),  # 阿拉伯文
            (0x0400, 0x04FF),  # 西里尔字母（俄文等）
            (0x0590, 0x05FF),  # 希伯来文
            (0x1E00, 0x1EFF),  # 越南文
            (0x3130, 0x318F),  # 韩文兼容字母
        ]
        code_point = ord(chr[0])
        if code_point in punc:
            return False
        for start, end in unicode_ranges:
            if start <= code_point <= end:
                return False
        return True

    def join(self, texts: List[str]):
        # 需要和 self.split 匹配使用
        texts = [text for text in texts if text]
        text = ""
        for i in texts:
            if text and (
                ((text[-1] not in self.punctuation) and self.isascii(i[0]))
                or (
                    self.isascii(text[-1])
                    and (i[0] not in (self.right_punc + self.left_punc))
                )
                or (text[-1] in "—" or i[0] in "—")  # "—" 前后有空格
            ):
                text += " " + i
            else:
                text += i
        return text

    def correct_incorrect_restore_punctuation(self, before_text, after_text):
        return after_text
