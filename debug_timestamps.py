import json

with open('data/audio/current_timestamps.json', 'r', encoding='utf-8') as f:
    ts = json.load(f)

with open('data/scripts/current.json', 'r', encoding='utf-8') as f:
    script = json.load(f)

words = ts.get('words', [])
assets = script['script']['narration']['visual_assets']

print('=== Image Time Ranges ===')
for asset in assets:
    asset_id = asset['visual_asset_id']
    start_idx = asset['start_word_index']
    end_idx = asset['end_word_index']
    
    start_time = None
    end_time = None
    for word in words:
        if word['index'] == start_idx:
            start_time = word['start']
        if word['index'] == end_idx:
            end_time = word['end']
    
    if start_time and end_time:
        print(f'{asset_id}: words {start_idx}-{end_idx} = time {start_time:.2f}s - {end_time:.2f}s')
    else:
        print(f'{asset_id}: words {start_idx}-{end_idx} = MISSING TIMESTAMPS')

print('\n=== Check for gaps ===')
prev_end = 0
for asset in assets:
    start_idx = asset['start_word_index']
    end_idx = asset['end_word_index']
    
    for word in words:
        if word['index'] == start_idx:
            start_time = word['start']
            if start_time > prev_end + 0.1:
                print(f"GAP before {asset['visual_asset_id']}: {prev_end:.2f}s to {start_time:.2f}s")
        if word['index'] == end_idx:
            prev_end = word['end']
