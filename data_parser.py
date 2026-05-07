import csv

def load_conversations(file_path):
    messages = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            block = row[0]
            lines = block.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Split speaker and message
                if ': ' in line:
                    speaker, text = line.split(': ', 1)
                    messages.append({'speaker': speaker.strip(), 'text': text.strip()})
                else:
                    # If it doesn't have a colon, it might be a continuation of the previous message
                    if messages:
                        messages[-1]['text'] += " " + line
    return messages

if __name__ == '__main__':
    msgs = load_conversations('conversations.csv')
    print(f"Loaded {len(msgs)} messages.")
    for i in range(5):
        print(msgs[i])
