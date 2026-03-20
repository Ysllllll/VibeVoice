from subtitle_formatter import Subtitle

# 1. Load the actual WhisperX JSON
sub = Subtitle.from_whisperx_json('test/resource/DDS course_70s.m4a.aligned.json')

# 2. Test Sentence grouping
sub.group_by_sentence()
print(f"Total sentence segments: {len(sub.segments)}")
for i in range(3):
    print(f"Seg {i+1}: {sub.segments[i]['text']} ({sub.segments[i]['start']} -> {sub.segments[i]['end']})")

print("\n----------------\n")

# 3. Test Length grouping (e.g., max 15 chars)
sub.group_by_length(max_chars=15)
print(f"Total length segments: {len(sub.segments)}")
for i in range(3):
    print(f"Seg {i+1}: {sub.segments[i]['text']} ({sub.segments[i]['start']} -> {sub.segments[i]['end']})")

# Export test
sub.to_srt('test/resource/test_output.srt')
print("\nExported test_output.srt")
