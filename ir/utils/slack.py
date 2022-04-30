import time
from functools import wraps
from typing import Optional, Callable

import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.web import SlackResponse

from ir.config import constants


def send_until_success(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs) -> None:
        succeeded: bool = False
        while not succeeded:
            try:
                fn(*args, **kwargs)
                succeeded = True
            except SlackApiError as e:
                if e.response.status_code == 429:
                    # The `Retry-After` header will tell you how long to wait before retrying
                    delay = int(e.response.headers['Retry-After'])
                    print(f"Rate limited. Retrying in {delay} seconds")
                    time.sleep(delay)
                else:
                    raise e
            except requests.exceptions.ConnectionError:
                print("requests.exceptions.ConnectionError was raised! Couldn't connect to Slack.")
    return wrapper


class Slack:
    def __init__(self):
        self._client: WebClient = WebClient(token=constants.slack.TOKEN)
        self._thread_ts: Optional[str] = None

    def new_thread(self) -> None:
        self._thread_ts = None

    @send_until_success
    def send_text(self, text: str, mention: bool = False) -> None:
        if mention is True:
            text = f'@{constants.slack.USER_ID_TO_MENTION}, ' + text
        response: SlackResponse = self._client.chat_postMessage(
            channel=constants.slack.CHANNEL,
            text=text, link_names=mention,
            thread_ts=self._thread_ts
        )
        self._set_thread_ts(response)

    @send_until_success
    def send_file(self, filepath: str) -> None:
        response: SlackResponse = self._client.files_upload(
            channels=constants.slack.CHANNEL,
            file=filepath,
            thread_ts=self._thread_ts,
            title=f'{filepath}'
        )
        self._set_thread_ts(response)

    def _set_thread_ts(self, response: SlackResponse) -> None:
        if self._thread_ts is None:
            self._thread_ts = response['ts']


# knockknock

from typing import List
import os
import datetime
import traceback
import functools
import json
import socket
import requests

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def slack_sender(slack: Slack):
    def decorator_sender(func):
        @functools.wraps(func)
        def wrapper_sender(*args, **kwargs):
            start_time = datetime.datetime.now()
            host_name = socket.gethostname()
            func_name = func.__name__
            contents: List[str] = []
            # Handling distributed training edge case.
            # In PyTorch, the launch of `torch.distributed.launch` sets up a RANK environment variable for each process.
            # This can be used to detect the master process.
            # See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py#L211
            # Except for errors, only the master process will send notifications.
            if 'RANK' in os.environ:
                master_process = (int(os.environ['RANK']) == 0)
                host_name += ' - RANK: %s' % os.environ['RANK']
            else:
                master_process = True
            if master_process:
                contents = ['Your training has started 🎬',
                            'Machine name: %s' % host_name,
                            'Main call: %s' % func_name,
                            'Starting date: %s' % start_time.strftime(DATE_FORMAT)]
                slack.send_text('\n'.join(contents))
            try:
                value = func(*args, **kwargs)
                if master_process:
                    end_time = datetime.datetime.now()
                    elapsed_time = end_time - start_time
                    contents = ["Your training is complete 🎉",
                                'Machine name: %s' % host_name,
                                'Main call: %s' % func_name,
                                'Starting date: %s' % start_time.strftime(DATE_FORMAT),
                                'End date: %s' % end_time.strftime(DATE_FORMAT),
                                'Training duration: %s' % str(elapsed_time)]
                    try:
                        str_value = str(value)
                        contents.append('Main call returned value: %s'% str_value)
                    except:
                        contents.append('Main call returned value: %s'% "ERROR - Couldn't str the returned value.")
                    slack.send_text('\n'.join(contents))
                    slack.new_thread()
                    slack.send_text('\n'.join(contents))
                return value
            except Exception as ex:
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                contents = ["Your training has crashed ☠️",
                            'Machine name: %s' % host_name,
                            'Main call: %s' % func_name,
                            'Starting date: %s' % start_time.strftime(DATE_FORMAT),
                            'Crash date: %s' % end_time.strftime(DATE_FORMAT),
                            'Crashed training duration: %s\n\n' % str(elapsed_time),
                            "Here's the error:",
                            '%s\n\n' % ex,
                            "Traceback:",
                            '%s' % traceback.format_exc()]
                slack.send_text('\n'.join(contents))
                slack.new_thread()
                slack.send_text('\n'.join(contents))
                raise ex
        return wrapper_sender
    return decorator_sender
