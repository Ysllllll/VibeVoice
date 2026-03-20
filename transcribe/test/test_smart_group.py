from subtitle_formatter import Subtitle

sub = Subtitle.from_whisperx_json('test/resource/DDS course_70s.m4a.aligned.json')

# Test cascaded grouping with a very small max_chars to force mid-punctuation and length splitting
print("Testing smart grouping with max_chars=15...")
sub.group_smart(max_chars=15)

for i in range(10):
    if i < len(sub.segments):
        print(f"Seg {i+1}: {sub.segments[i]['text']} ({sub.segments[i]['start']} -> {sub.segments[i]['end']})")

sub.to_srt('test/resource/test_smart.srt')
print(f"\nTotal segments generated: {len(sub.segments)}")
