import contextlib
import logging
from functools import wraps
from http.client import HTTPConnection

import requests

from .auth import get_credentials


def log_requests(func):
    """
    Debug utility to enable tracing
    """

    @wraps(func)
    @contextlib.contextmanager
    def wrapper_func(*args, **kwargs):
        HTTPConnection.debuglevel = 1
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        requests_log = logging.getLogger("requests.packages.urllib3")
        requests_log.setLevel(logging.DEBUG)
        requests_log.propagate = True
        res = func(*args, **kwargs)
        HTTPConnection.debuglevel = 0
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        root_logger.handlers = []
        requests_log = logging.getLogger("requests.packages.urllib3")
        requests_log.setLevel(logging.WARNING)
        requests_log.propagate = False
        return res

    return wrapper_func


class ApiOperator(requests.Session):
    api_url = "https://operator.forge.epita.fr/api"

    def __init__(self):
        super(ApiOperator, self).__init__()
        self.hooks["response"].append(self._reauth)
        self._get_token()

    def _get_token(self):
        self.headers["Authorization"] = f"Bearer {get_credentials().jwt}"
        self.headers["accept"] = "application/json"

    def _reauth(self, res: requests.Response, *args, **kwargs):
        if res.status_code == 401:
            if res.request.headers.get("REATTEMPT"):
                res.raise_for_status()

            self._get_token()
            req = res.request
            req.headers["REATTEMPT"] = "1"
            req.headers["Authorization"] = str(self.headers["Authorization"])
            res = self.send(req)
            return res

        res.raise_for_status()
        return res

    def activities(self) -> list[str]:
        res = self.get(f"{self.api_url}/activities")
        return [x["uri"] for x in res.json()]
