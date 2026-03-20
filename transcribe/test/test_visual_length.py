import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subtitle_formatter import Subtitle

sub = Subtitle.from_whisperx_json('test/resource/DDS course_70s.m4a.aligned.json')

# Test cascaded grouping with a very small max_chars (e.g. 15 Chinese chars)
print("Testing smart grouping with visual max_chars=15...")
sub.group_smart(max_chars=15)

for i in range(10):
    if i < len(sub.segments):
        print(f"Seg {i+1}: {sub.segments[i]['text']} (visual length: {Subtitle._calculate_visual_length(sub.segments[i]['text'])})")

sub.to_srt('test/resource/test_visual_smart.srt')
