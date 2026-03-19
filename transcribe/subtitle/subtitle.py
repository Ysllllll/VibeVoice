import copy
import json
import logging
import os
import re
import sys
import threading
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Union

import psutil

from .handler import (
    LanguageContext,
    LanguageHandler,
    language_context,
    omit_punctuation,
)

logger = logging.getLogger(__name__)

# --- Compiled Regular Expressions for Performance ---
TIME_PATTERN = re.compile(r"(\d+:\d+:\d+[.,]\d+) --> (\d+:\d+:\d+[.,]\d+)")
TIME_STRIP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3} --> \d{2}:\d{2}:\d{2}[,.]\d{3}")
BLOCK_SPLIT_PATTERN = re.compile(r"\n{2,}[第]?\s*\d+\s*[章号]?\n")
CLEAN_H_PATTERN = re.compile(re.escape(r"\h"))
CLEAN_SPACE_PUNC_PATTERN = re.compile(r"\s([,.!?:;])")

class PunctuationModel:
    def restore_punctuation(self, text: str) -> str:
        # Dummy implementation for testing
        return text

def float_equal(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) < tol

def extend_filename(filename: Path, extension: str) -> Path:
    return filename.with_name(f"{filename.stem}{extension}{filename.suffix}")

def format_srt_timestamp(start: float, end: float) -> str:
    def format_time(seconds: float) -> str:
        ms = round(seconds * 1000)
        s, ms = divmod(ms, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{format_time(start)} --> {format_time(end)}"

def parse_timestamp(time_str: str) -> float:
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:
        raise ValueError(f"Invalid timestamp format: {time_str}")
    return int(h) * 3600 + int(m) * 60 + float(s)

def parse_srt_timestamp(timestamp_line: str):
    start_str, end_str = timestamp_line.split(" --> ")
    return parse_timestamp(start_str.strip()), parse_timestamp(end_str.strip())


@dataclass(slots=True)
class Element:

    start: float = -1.0
    end: float = -1.0
    text: str = ""

    def __bool__(self):
        return bool(self.start != -1.0) and bool(self.end != -1.0) and bool(self.text)

    def to_dict(self):
        return {"start": self.start, "end": self.end, "text": self.text}

    @classmethod
    def from_dict(cls, json_dict):
        return cls(json_dict["start"], json_dict["end"], json_dict["text"])

    def to_str(self):
        return f"{format_srt_timestamp(self.start, self.end)}: {self.text}"

    @classmethod
    def from_str(cls, text: str):
        match = TIME_PATTERN.search(text)
        if not match:
            raise ValueError(f"Cannot parse timestamp from: {text}")
        start_str, end_str = match.group(1, 2)
        
        return Element(
            start=parse_timestamp(start_str),
            end=parse_timestamp(end_str),
            text=TIME_STRIP_PATTERN.sub("", text).strip("-: "),
        )

    def __eq__(self, other: "Element") -> bool:
        if not isinstance(other, Element):
            return False
        return (
            float_equal(self.start, other.start)
            and float_equal(self.end, other.end)
            and self.text == other.text
        )


@dataclass
class Elements:
    # maybe refactor name to Sentence and Sentences

    elements: List[Element] = field(default_factory=list)
    handler: LanguageHandler = field(default=None)

    def __post_init__(self):
        if not self.elements:
            self.elements = []
        elif isinstance(self.elements, Element):
            self.elements = [copy.deepcopy(self.elements)]
        elif isinstance(self.elements, Elements):
            self.elements = [copy.deepcopy(i) for i in self.elements.elements]
        elif isinstance(self.elements, list):
            new_elements = []
            for i in self.elements:
                if isinstance(i, dict):
                    new_elements.append(Element.from_dict(i))
                elif isinstance(i, Element):
                    new_elements.append(copy.deepcopy(i))
                else:
                    raise ValueError(f"Invalid element type in list: {type(i)}")
            self.elements = new_elements
        else:
            raise ValueError(
                f"elements type should be 'List[Element]', 'Element' or 'Elements', but got {type(self.elements)}"
            )
        self.elements.sort(key=lambda x: x.start)
        self.handler = LanguageHandler.get_handler(LanguageContext.get_language())

    # def split_by_llm(self):
    #     implement by llm
    #     pass

    def split_by_nlp_model(
        self, max_length: int, nlp=None, word_timestamps: "Elements" = None
    ) -> "Elements":
        _word_timestamps = self.get_word_timestamps(word_timestamps)

        ret_elements = Elements()
        for element in self.elements:
            if self.handler.length(element.text) > max_length:
                segments = self.handler.split_by_nlp_model(element.text, nlp=nlp)
                __word_timestamps = _word_timestamps.subelements(
                    element.start, element.end
                )
                for segment in segments:
                    start, end = self.handler.calculate_phrase_timestamps(
                        segment, __word_timestamps
                    )
                    ret_elements.append(Element(start, end, segment))
            else:
                ret_elements.append(element)
        return ret_elements

    def split_by_nlp_model_sentences(
        self, max_length: int, nlp=None, word_timestamps: "Elements" = None
    ) -> "Elements":
        _word_timestamps = self.get_word_timestamps(word_timestamps)
        ret_elements = Elements()

        split_elements = self.split_by_mid_punctuation(_word_timestamps)
        for sentence in split_elements.iter_sentence(True, _word_timestamps):
            elements = sentence.merge_by_punctuation(False, False)
            new_sentence = Elements()
            for element in elements:
                if element.handler.length(element.text) > max_length:
                    __word_timestamps = _word_timestamps.subelements(
                        element.start, element.end
                    )
                    segments = element.handler.split_by_nlp_model(element.text, nlp=nlp)
                    for segment in segments:
                        start, end = element.handler.calculate_phrase_timestamps(
                            segment, __word_timestamps
                        )
                        new_sentence.append(Element(start, end, segment))
                else:
                    new_sentence.extend(element)
            if len(new_sentence) > 0:
                for i in new_sentence.merge_by_max_length(max_length, False):
                    ret_elements.extend(i)
        return ret_elements

    def split_by_max_length(
        self,
        max_length: int,
        max_line: int,
        word_timestamps: "Elements" = None,
        **kwargs,
    ) -> "Elements":
        # every line length() <= max_length
        _word_timestamps = self.get_word_timestamps(word_timestamps)

        ret_elements = Elements()
        for element in self.elements:
            segments = self.handler.split_by_max_length(
                element.text, max_length, max_line, **kwargs
            )
            for segment in segments:
                start, end = self.handler.calculate_phrase_timestamps(
                    segment, _word_timestamps
                )
                ret_elements.append(Element(start, end, segment))
        return ret_elements

    def split_by_end_punctuation(
        self, word_timestamps: "Elements" = None
    ) -> "Elements":
        # every line end with (".?!..." punctuation or None)
        return self.split_by_punctuation(word_timestamps, True)

    def split_by_mid_punctuation(
        self, word_timestamps: "Elements" = None
    ) -> "Elements":
        # every line end with (",:;" punctuation or None)
        return self.split_by_punctuation(word_timestamps, False)

    def get_segments_timestamps(
        self, segments: List[str], word_timestamps: "Elements" = None
    ):
        _word_timestamps = self.get_word_timestamps(word_timestamps)
        ret_elements = Elements()
        for segment in segments:
            start, end = self.handler.calculate_phrase_timestamps(
                segment, _word_timestamps
            )
            ret_elements.append(Element(start, end, segment))
        return ret_elements

    def split_by_punctuation(
        self,
        word_timestamps: "Elements" = None,
        use_end_punctuation=True,
    ) -> "Elements":
        # 优化点：word_timestamps 天然切割好了，可以直接用
        _word_timestamps = self.get_word_timestamps(word_timestamps)

        ret_elements = Elements()
        for element in self.elements:
            segments = self.handler.split_by_punctuation(
                element.text, use_end_punctuation
            )
            for segment in segments:
                start, end = self.handler.calculate_phrase_timestamps(
                    segment, _word_timestamps
                )
                ret_elements.append(Element(start, end, segment))
        return ret_elements

    def merge_by_max_length(
        self, max_length: int, detail: bool = False
    ) -> List["Elements"]:
        ret_elements_list: List[Elements] = []
        last_sentence = Elements()

        for element in self.elements:
            current_len = self.handler.concat_length(last_sentence.text, element.text)
            if last_sentence.text and current_len > max_length:
                self._append(ret_elements_list, last_sentence, detail)
                last_sentence = Elements(element)
            else:
                last_sentence.append(element)
        if last_sentence:
            self._append(ret_elements_list, last_sentence, detail)
        return ret_elements_list

    def merge_by_end_punctuation(self, detail: bool = False) -> List["Elements"]:
        # every line end with (".?!..." punctuation or None)
        return self.merge_by_punctuation(detail, True)

    def merge_by_mid_punctuation(self, detail: bool = False) -> List["Elements"]:
        # every line end with (",:;" punctuation or None)
        return self.merge_by_punctuation(detail, False)

    def merge_by_punctuation(
        self, detail: bool = False, use_end_punctuation: bool = True
    ) -> List["Elements"]:
        ret_elements_list: List[Elements] = []
        last_sentence = Elements()

        for element in self.elements:
            last_sentence.append(element)
            if self.handler.is_end(element.text, use_end_punctuation):
                self._append(ret_elements_list, last_sentence, detail)
                last_sentence = Elements()
        if last_sentence:
            self._append(ret_elements_list, last_sentence, detail)
        return ret_elements_list

    def restore_punctuation(
        self, model: Any, threshold: int = 180, line_threshold: int = None
    ) -> "Elements":
        # only for single sentence
        # merged_elements = self.remove_repeat_element()
        merged_elements = copy.deepcopy(self)

        if not line_threshold:
            line_threshold = threshold
        texts = [i.text for i in merged_elements]
        before_text = merged_elements.handler.join(texts)
        length = merged_elements.handler.length(before_text)
        if length < threshold and all(
            merged_elements.handler.length(i) < line_threshold for i in texts
        ):
            return merged_elements

        after_text = model.restore_punctuation(before_text)
        after_text = re.sub(rf"{re.escape(omit_punctuation)}", "", after_text)
        fixed_text = self.handler.correct_incorrect_restore_punctuation(
            before_text, after_text
        )
        if fixed_text:
            tail_punc = before_text[
                len(self.handler.clean_punctuation(before_text, left_strip=False)) :
            ]
            fixed_text = (
                fixed_text[
                    : len(self.handler.clean_punctuation(fixed_text, left_strip=False))
                ]
                + tail_punc
            )
        logger.info(
            f"restore punction:\n  originx: {before_text}\n  restore: {after_text}\n  refixed: {fixed_text}"
        )
        fixed_words = self.handler.split(fixed_text)
        for i in merged_elements:
            i.text = self.handler.correct_sentence_punctuation_by_words_list(
                i.text, fixed_words
            )

        return merged_elements

    def restore_punctuation_and_align_sentence(
        self,
        nlp: Any,
        model: Any,
        threshold: int = 180,
        line_threshold: int = None,
        word_timestamps: "Elements" = None,
    ):
        if not line_threshold:
            line_threshold = threshold // 2
        if word_timestamps:
            new_word_timestamps = word_timestamps.subelements(self.start, self.end)
            restore_elements = new_word_timestamps.restore_punctuation(
                model, threshold, line_threshold
            )
            word_timestamps.replace(restore_elements)
            new_word_timestamps = restore_elements
        else:
            merged_elements = Elements()
            for i in self.iter_sentence(True, True, word_timestamps):
                merged_elements.extend(i)
            restore_elements = merged_elements.restore_punctuation(
                model, threshold, line_threshold
            )
            new_word_timestamps = None
        elements_list = [
            i for i in restore_elements.iter_sentence(True, True, new_word_timestamps)
        ]

        ret_elements: List[Elements] = []
        for elements in elements_list:
            new_elements = Elements()
            split_elements = elements.split_by_nlp_model(
                line_threshold, nlp, new_word_timestamps
            )
            for i in split_elements.merge_by_max_length(line_threshold, False):
                new_elements.extend(i)
            ret_elements.append(new_elements)

        return ret_elements

    def _append(
        self, result: List["Elements"], last_sentence: "Elements", detail: bool = False
    ):
        if detail:
            result.append(last_sentence)
        else:
            e = Element(last_sentence.start, last_sentence.end, last_sentence.text)
            result.append(Elements(e))

    def get_word_timestamps(self, word_timestamps: "Elements" = None):
        if not word_timestamps:
            _word_timestamps = self.handler.calculate_word_timestamps(self)
        else:
            _word_timestamps = word_timestamps.subelements(self.start, self.end)
            assert (
                _word_timestamps.start == self.start
            ), "get_word_timestamps corner case"
            assert _word_timestamps.end == self.end, "get_word_timestamps corner case"
        return _word_timestamps

    def iter_align_sentence(
        self, max_length, nlp: None, word_timestamps: "Elements" = None
    ):
        # every line, comma align by rearrange, auto
        _word_timestamps = self.get_word_timestamps(word_timestamps)
        sentences_list = _word_timestamps.merge_by_end_punctuation(detail=True)
        for sentence in sentences_list:
            elements_list = sentence.merge_by_mid_punctuation(detail=False)
            new_elements = Elements()
            for elements in elements_list:
                new_elements.extend(
                    elements.split_by_nlp_model(max_length, nlp, _word_timestamps)
                )
            for i in new_elements.merge_by_max_length(max_length, detail=False):
                yield i

    def iter_sentence(
        self,
        detail: bool = False,
        rearrange: bool = False,
        word_timestamps: "Elements" = None,
    ):
        # detail=False, rearange=False: single sentence
        # detail=False, rearange=True: single sentence
        # detail=True, rearange=False: single sentence but multi line
        # detail=True, rearange=True: single sentence but multi line (every line if comma align by rearrange)
        if rearrange:
            _word_timestamps = self.get_word_timestamps(word_timestamps)
            sentences_list = _word_timestamps.merge_by_end_punctuation(detail=True)
            for sentence in sentences_list:
                elements_list = sentence.merge_by_mid_punctuation(detail=False)
                ret_sentence = Elements()
                for elements in elements_list:
                    ret_sentence.extend(elements)
                if detail:
                    yield ret_sentence
                else:
                    yield Elements(
                        Element(ret_sentence.start, ret_sentence.end, ret_sentence.text)
                    )
        else:
            split_elements = self.split_by_end_punctuation(word_timestamps)
            merge_elements = split_elements.merge_by_punctuation(detail)
            for elements in merge_elements:
                yield elements

    def iter_limit(
        self,
        max_length: int,
        max_line: int,
        word_timestamps: "Elements" = None,
        **kwargs,
    ):
        split_elements = self.split_by_max_length(
            max_length, max_line, word_timestamps, **kwargs
        )
        for element in split_elements:
            yield element

    def append(self, element: Element):
        self.elements.append(element)

    def extend(self, element: "Elements"):
        self.elements.extend(element.elements)

    def __getitem__(self, index) -> Union[Element, "Elements"]:
        if isinstance(index, slice):
            return Elements(self.elements[index])
        elif isinstance(index, int):
            return self.elements[index]
        else:
            raise TypeError("Invalid index type")

    def __setitem__(self, index, value: Element):
        if isinstance(index, slice):
            if isinstance(value, Elements):
                self.elements[index] = value.elements
            elif isinstance(value, Element):
                self.elements[index] = [value]
            elif value == []:
                self.elements[index] = value
            else:
                raise TypeError("Invalid value type")
        else:
            assert isinstance(
                value, Element
            ), f"value type should be 'Element', but got {type(value)}"
            self.elements[index] = value

    def __add__(self, other_elements: Union["Elements", list]):
        if isinstance(other_elements, list):
            return Elements(self.elements + other_elements)
        return Elements(self.elements + other_elements.elements)

    def __radd__(self, other_elements: Union["Elements", list]):
        if isinstance(other_elements, list):
            return Elements(other_elements + self.elements)
        return Elements(other_elements.elements + self.elements)

    def __bool__(self):
        return bool(self.elements)

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __eq__(self, other: Union["Elements", List[Element]]) -> bool:
        if not isinstance(other, (Elements, list)):
            return False
        if isinstance(other, Elements):
            other_elements = other.elements
        else:
            other_elements = other
        if len(self.elements) != len(other_elements):
            return False
        return all(e1 == e2 for e1, e2 in zip(self.elements, other_elements))

    def to_element(self):
        return Element(self.start, self.end, self.text)

    def subelements(self, start: float, end: float) -> "Elements":
        return Elements([e for e in self.elements if start <= e.start and e.end <= end])

    def replace(self, elements: "Elements"):
        left = bisect_left(self.elements, elements.start, key=lambda x: x.start)
        right = bisect_right(self.elements, elements.end, key=lambda x: x.end) - 1
        if (
            left != len(self.elements)
            and self.elements[left].start == elements.start
            and right != len(self.elements)
            and self.elements[right].end == elements.end
        ):
            self.elements[left : right + 1] = elements.elements
        else:
            assert False, "Elements.replace error"

    @property
    def start(self):
        return self.elements[0].start if self.elements else -1

    @property
    def end(self):
        return self.elements[-1].end if self.elements else -1

    @property
    def text(self):
        texts = [i.text for i in self.elements]
        return self.handler.join(texts)

    @property
    def texts(self):
        return [i.text for i in self.elements]

    def to_dict(self):
        return [i.to_dict() for i in self.elements]

    @classmethod
    def from_dict(cls, json_list: List[dict]):
        return cls([Element.from_dict(i) for i in json_list])

    def to_zip_dict(self):
        zip_elements = []
        for i in range(0, len(self.elements), 5):
            group = []
            for j in range(5):
                idx = i + j
                if idx < len(self.elements):
                    element = self.elements[idx]
                    group.extend([element.start, element.end, element.text])
            zip_elements.append(group)
        return zip_elements

    @classmethod
    def from_zip_dict(cls, json_list: List[dict]):
        unzip_elements = []
        for group in json_list:
            for i in range(0, len(group), 3):
                unzip_elements.append(Element(group[i], group[i + 1], group[i + 2]))
        return cls(unzip_elements)

    def fetch_subelements(
        self,
        word_num: int,
        overlap: int = 0,
        start_idx: int = 0,
        end_idx: int = sys.maxsize,
    ):
        # 左闭右开
        iter_elements = self.elements[start_idx:end_idx]
        ret_elements = Elements()
        overlap_elements = self.elements[max(0, start_idx - overlap) : start_idx]
        
        for idx, element in enumerate(iter_elements):
            ele_len = self.handler.length(element.text)
            ret_len = self.handler.length(ret_elements.text)
            if ret_len + ele_len > word_num:
                yield overlap_elements + ret_elements
                overlap_elements = iter_elements[max(0, idx - overlap) : idx]
                ret_elements = Elements(element)
            else:
                ret_elements.append(element)

        if ret_elements:
            yield overlap_elements + ret_elements

    def remove_overlap_element(self) -> "Elements":
        if not self.elements:
            return Elements()

        new_elements = Elements(copy.deepcopy(self.elements[0]))
        for i in self.elements[1:]:
            if float_equal(new_elements.end, i.start) or (new_elements.end < i.start):
                new_elements.append(copy.deepcopy(i))

        return new_elements

    def remove_repeat_element(self) -> "Elements":
        if not self.elements:
            return Elements()

        new_elements = Elements(copy.deepcopy(self.elements[0]))

        for current_element in self.elements[1:]:
            if current_element.text == new_elements[-1].text:
                new_elements[-1].end = current_element.end
            else:
                new_elements.append(copy.deepcopy(current_element))

        return new_elements


class SubIterator(Enum):
    ORIGIN = 1
    SENTENCE = 2
    LIMITLEN = 3


class Subtitle:

    def __init__(
        self,
        language: str,
        elements: Elements,
        filename: Union[str, Path],
        word_timestamps: Elements = [],
        **kwargs,
    ):
        self.lang = language
        with language_context(self.lang):
            self.elements = Elements(elements)
            self.filename = Path(filename)
            self.word_timestamps = Elements(word_timestamps)
            self.kwargs = kwargs

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other: "Subtitle") -> bool:
        if not isinstance(other, Subtitle):
            return False
        return (
            self.lang == other.lang
            and self.elements == other.elements
            and self.word_timestamps == other.word_timestamps
        )

    def elements_equal(self, other: Union["Subtitle", List[dict]]) -> bool:
        if isinstance(other, Subtitle):
            return self.elements == other.elements
        elif isinstance(other, list):
            other_segments = [Element(**seg) for seg in other]
            return self.elements == other_segments
        return False

    @property
    def text(self):
        return self.elements.text

    @property
    def start(self):
        return self.elements.start if self.elements else -1

    @property
    def end(self):
        return self.elements.end if self.elements else -1

    @property
    def valid_end(self):
        if self.elements:
            for element in reversed(self.elements):
                if element.text.strip():
                    return element.end
        else:
            return -1

    def __iter__(self):
        for element in self.elements:
            yield element

    def iter_align_sentence(self, max_length: int, nlp=None):
        with language_context(self.lang):
            elements_iterator = self.elements.iter_align_sentence(
                max_length, nlp, self.word_timestamps
            )
            while True:
                try:
                    yield next(elements_iterator)
                except StopIteration:
                    break

    def iter_sentence(self, detail: bool = False, rearrange: bool = False):
        with language_context(self.lang):
            elements_iterator = self.elements.iter_sentence(
                detail, rearrange, self.word_timestamps
            )
            while True:
                try:
                    yield next(elements_iterator)
                except StopIteration:
                    break

    def iter_limit(self, max_length: int, max_line: int, **kwargs):
        with language_context(self.lang):
            is_normal_course = os.getenv("COURSE_TYPE", "normal") != "short"
            with_watermark = kwargs.get("with_watermark", False)
            for element in self.elements.iter_limit(
                max_length, max_line, self.word_timestamps, **kwargs
            ):
                if (
                    with_watermark
                    and self.lang == "zh"
                    and (not re.match(r"^[\(\[].*[\]\)]$", element.text))
                    # and len(element.text) < 30
                    and is_normal_course
                ):
                    water_element = copy.deepcopy(element)
                    water_element.text = f"[B站:常青藤中英字幕课程] {water_element.text}"
                    with_watermark = False
                    yield Elements(water_element)
                else:
                    yield Elements(element)

    def get_fixed_punctuation_subtitle(self, threshold: int = 200) -> "Subtitle":
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024 * 1024 * 1024)

        max_threads = max(1, available_memory_gb // 2)
        max_threads = min(max_threads, 3)
        new_elements = []
        futures = []

        thread_local = threading.local()

        def get_model():
            if not hasattr(thread_local, "model"):
                thread_local.model = PunctuationModel()
            return thread_local.model

        def process_sentence(sentence: Elements):
            model = get_model()
            return sentence.restore_punctuation(
                model,
                threshold=threshold,
                line_threshold=threshold // 2,
                word_timestamps=self.word_timestamps,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            for i, sentence in enumerate(self.iter_sentence(True)):
                futures.append(
                    (
                        i,
                        executor.submit(process_sentence, sentence),
                    )
                )

            results = []
            for i, future in futures:
                results.append((i, future.result()))

            results.sort(key=lambda x: x[0])

            for _, result in results:
                new_elements.extend(result)

        return Subtitle(
            language=self.lang,
            elements=Elements(new_elements),
            filename=extend_filename(self.filename, "with_punc"),
            word_timestamps=self.word_timestamps,
        )

    def get_monotonic_subtitle(self) -> "Subtitle":
        return Subtitle(
            language=self.lang,
            elements=self._get_non_monotonic_timeline()[1],
            filename=extend_filename(self.filename, "_force_monotonic"),
        )

    def get_non_monotonic_timeline(self):
        return self._get_non_monotonic_timeline()[0]

    def _get_non_monotonic_timeline(self):
        non_monotonic_intervals = []
        previous_end_time = 0
        start_index = 0
        in_non_monotonic_interval = False
        monotonic_elements: List[Element] = []

        def push_to_monotonic_elements(start, end):
            while monotonic_elements and (
                monotonic_elements[-1].start >= self.elements[start].start
            ):
                monotonic_elements.pop()
            t = Elements([self.elements[i] for i in range(start, end)])
            monotonic_elements.append(Element(start=t.start, end=t.end, text=t.text))

        for i, sub in enumerate(self.elements):

            if sub.start > sub.end:
                if not in_non_monotonic_interval:
                    start_index = i
                    in_non_monotonic_interval = True
            elif sub.start < previous_end_time:
                if not in_non_monotonic_interval:
                    start_index = i - 1
                    in_non_monotonic_interval = True
            else:
                if in_non_monotonic_interval:
                    non_monotonic_intervals.append([start_index, i - 1])
                    in_non_monotonic_interval = False
                    push_to_monotonic_elements(start_index, i)
                monotonic_elements.append(sub)

            previous_end_time = max(previous_end_time, sub.start, sub.end)

        if in_non_monotonic_interval:
            non_monotonic_intervals.append([start_index, len(self.elements) - 1])
            push_to_monotonic_elements(start_index, len(self.elements))

        return non_monotonic_intervals, monotonic_elements

    @staticmethod
    def from_file(filename):
        filename = Path(filename)
        if filename.suffix == ".json":
            return Subtitle.from_json(filename)
        elif filename.suffix == ".srt":
            return Subtitle.from_srt(filename)
        else:
            raise ValueError(
                f"Invalid file {filename}, only support [.srt, .json] file"
            )

    @staticmethod
    def from_srt(filename):
        filename = Path(filename)
        with open(filename, encoding="utf-8") as f:
            contents = f.read()
            return Subtitle.from_str(contents, filename)

    @staticmethod
    def from_json(filename):
        with open(filename, encoding="utf-8") as f:
            content = json.loads(f.read())
            content["filename"] = filename
            return Subtitle.from_dict(content)

    @staticmethod
    def from_dict(json_dict: dict):
        # 兼容之前的 segments
        json_dict["elements"] = json_dict.pop("segments", json_dict.pop("elements", []))
        elements = Elements.from_dict(json_dict["elements"])
        filename = json_dict.get("filename", "anonymous.srt")
        language = json_dict.get("language", "en")
        word_timestamps = Elements.from_zip_dict((json_dict.get("word_timestamps", [])))
        return Subtitle(language, elements, filename, word_timestamps)

    @staticmethod
    def _clean_subtitle_text(text: str) -> str:
        text = CLEAN_H_PATTERN.sub("", text)
        text = text.replace("[? ", "[?")
        text = text.replace(" ?]", "?]")
        text = text.replace(" ]?", "]?")
        text = text.replace("?[ ", "?[")
        text = text.replace("\u200b", "")  # Zero-width space
        text = text.replace("Dr. ", "Dr.,").replace("Mr. ", "Mr.,").replace("Mrs. ", "Mrs.,").replace("Ms. ", "Ms.,")
        text = CLEAN_SPACE_PUNC_PATTERN.sub(r"\1", text)
        text = text.lstrip(",.?!;:-_”’]/ ")
        if "--" in text:
            text = "-- ".join(i.strip() for i in text.split("--"))
        return text

    @staticmethod
    def from_str(contents: str, filename="anonymous.srt"):
        # 分割符：\n\n1 \n\n2 \n\n3
        blocks = BLOCK_SPLIT_PATTERN.split(contents)
        elements = Elements()
        
        for idx, block in enumerate(blocks, 1):
            lines = block.strip("\n").split("\n")

            if len(lines) >= 2:
                if idx == 1:
                    lines = lines[1:]  # skip first seq_num
                timestamp = lines[0]
                text_line = lines[1:]

                # 处理空行
                if not text_line:
                    continue
                text_line = [j for j in text_line if j.strip()]
                if not text_line:
                    continue

                # 判断是否为 index，因为有可能在字幕显示区域添加了索引
                idx_srt = re.search(r"^\d*$", text_line[0])
                if idx_srt and int(idx_srt.group()) == idx:
                    text_line = text_line[1:]

                start, end = parse_srt_timestamp(timestamp)
                text = " ".join(text_line).strip()
                
                text = Subtitle._clean_subtitle_text(text)
                
                if not float_equal(start, end) and text.strip():
                    elements.append(Element(start, end, text))

        lang = LanguageHandler.guess_language(" ".join(segment.text for segment in elements[:20]))

        return Subtitle(language=lang, elements=elements, filename=filename)
    
    def check_timeline_monotonic(self):
        prev_time = 0
        for i in self.elements:
            if i.start > i.end or i.start < prev_time or i.end < prev_time:
                return False
            prev_time = i.end
        return True

    def to_dict(self):
        json_dict = {
            "filename": str(self.filename),
            "language": self.lang,
            "elements": self.elements.to_dict(),
        }
        if self.word_timestamps:
            json_dict["word_timestamps"] = self.word_timestamps.to_zip_dict()
        return json_dict

    def to_str(self, iterator_type: SubIterator = SubIterator.ORIGIN, **kwargs):
        result_str = []
        segments = self.elements
        if iterator_type == SubIterator.LIMITLEN:
            segments = [s for segment in self.iter_limit(**kwargs) for s in segment]
        elif iterator_type == SubIterator.SENTENCE:
            segments = [s for segment in self.iter_sentence(**kwargs) for s in segment]
        for index, segment in enumerate(segments, start=1):
            cleaned_text = re.sub(r"@@", "", segment.text)
            result_str.append(str(index))
            result_str.append(format_srt_timestamp(segment.start, segment.end))
            result_str.append(f"{cleaned_text}\n")
        return "\n".join(result_str).strip("\n ")

    def to_srt(
        self,
        output_path: Path = None,
        iter_type: SubIterator = SubIterator.ORIGIN,
        **kwargs,
    ):
        srt_path = self.filename
        if not output_path:
            srt_path = extend_filename(srt_path, f".[{self.lang}]")
        else:
            if not isinstance(output_path, Path):
                output_path = Path(output_path)
            if output_path.is_dir():
                srt_path = Path(output_path) / (Path(srt_path.name))
            else:
                srt_path = output_path

        srt_path = srt_path.with_suffix(".srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            print(self.to_str(iter_type, **kwargs), file=f, flush=True)

        logger.info(
            f"Subtitle.to_srt(iter_type={iter_type}, {kwargs}): {output_path} --> {srt_path}"
        )

        return srt_path
