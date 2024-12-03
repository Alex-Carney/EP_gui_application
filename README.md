First, create the `slack_notifier.py` file:

```python
from slack_sdk import WebClient

# Slack configuration
SLACK_TOKEN = "YOUR SLACK TOKEN HERE"
BOT_EVENTS_CHANNEL_ID = "FITZ LAB BOT CHANNEL ID"


class SlackNotifier:
    USERS = {
        "USER": 'USER SLACK ID',
        # Add other users as needed
    }

    def __init__(self, token=SLACK_TOKEN, channel_id=BOT_EVENTS_CHANNEL_ID):
        self.client = WebClient(token=token)
        self.channel_id = channel_id

    def send_message(self, user_names, experiment_status):
        user_ids = [self.USERS[name] for name in user_names if name in self.USERS]
        mentions = ' '.join([f"<@{user_id}>" for user_id in user_ids])
        message = f"Hello {mentions}, {experiment_status}"
        self.client.chat_postMessage(channel=self.channel_id, text=message)
```