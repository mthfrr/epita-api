# Stolen from the pouply project
import base64
import hashlib
import json
import os
import re
import secrets
import sys
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Tuple

import requests
from requests import Response


@dataclass
class AuthConfiguration:
    # OAuth2 client id
    client_id: str
    # This is the port where the "temporary backend" defined in this file is hosted.
    # This is called back by the CRI and is used to retrieve the OAuth authcode.
    # (AHS = Ad-Hoc Server)
    # This cannot be changed freely because the CRI hardcodes the allowed hosts and
    # ports for redirection (this is standard security practice for avoiding OAuth
    # authcode leakage).
    temporary_server_port: int
    # File path where the tokens are stored.
    token_file_path: str
    # A URL where an authenticated request will be sent to validate
    # authentication.
    test_probe_url: str


# Python's built-in http server is very rudimentary and does not support proper
# inspection of query parameters, which is why we use this regex instead.
# Not elegant, but it works.
authcode_regex: re.Pattern = re.compile("/complete/epita/.*?[?&]code=(.+?)(&|$)")

__enable_debug = bool(os.environ.get("DEBUG"))

# region Utilities

productionAuthConfiguration = AuthConfiguration(
    client_id="825885",
    temporary_server_port=2097,
    token_file_path=os.path.expanduser("~/.cri-tokens"),
    test_probe_url="https://srvc-auth.api.forge.epita.fr",
)

stagingAuthConfiguration = AuthConfiguration(
    client_id="368197",
    temporary_server_port=2097,
    token_file_path=os.path.expanduser("~/.cri-tokens-staging"),
    test_probe_url="https://srvc-auth.api.forge.epita.app",
)


# NOTE: To use this profile, you'll need a Forge::dev service running on port
# 8081 with framework-auth enabled.
criTestAuthConfiguration = AuthConfiguration(
    client_id="125070",
    temporary_server_port=8080,
    token_file_path=os.path.expanduser("~/.cri-tokens-test"),
    test_probe_url="https://localhost:8081",
)


class DefaultProfiles(Enum):
    staging = stagingAuthConfiguration
    criTest = criTestAuthConfiguration
    prod = productionAuthConfiguration

    def __str__(self) -> str:
        return self.name


# All of these are sent to stderr so that we don't pollute stdout, which should
# only contain our token (if -p is provided).


def __dbg_print(message: str):
    if __enable_debug:
        print(f"DBG: {message}", file=sys.stderr)


def __success_print(message: str):
    print(f" OK: {message}", file=sys.stderr)


def __hl_print(message: str):
    print(f">>>: {message}", file=sys.stderr)


def __error_print(message: str):
    print(f"ERR: {message}", file=sys.stderr)


# endregion

# region Authentication & token management


# Dataclass that is stored in the file specified in token_file_path
@dataclass
class Credentials:
    jwt: str
    refresh_token: str


def __store_tokens(cfg: AuthConfiguration, tokens: Credentials):
    with open(cfg.token_file_path, "w") as file:
        file.write(json.dumps(tokens))
    os.chmod(cfg.token_file_path, 0o600)


token_url = "https://cri.epita.fr/token"


def __request_authcode(cfg: AuthConfiguration) -> Tuple[str, str]:
    # PKCE flow for authentication, see RFC 7636 for details
    code_verifier: str = secrets.token_urlsafe()
    code_challenge: str = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode("ascii")).digest()
    ).decode("ascii")[:-1]

    authorize_url: str = (
        "https://cri.epita.fr/authorize?"
        "response_type=code&"
        "scope=openid epita roles profile email&"
        f"client_id={cfg.client_id}&"
        f"code_challenge={code_challenge}&"
        "code_challenge_method=S256&"
        f"redirect_uri=http://localhost:{cfg.temporary_server_port}/complete/epita/"
    )

    redirect_uri: str = f"http://localhost:{cfg.temporary_server_port}/complete/epita/"

    # This class is responsible for handling the HTTP server requests
    class HttpHandler(BaseHTTPRequestHandler):
        # The following two properties are the only real way to transmit data
        # between the function and the server. Not exactly ideal, but functional.
        # This one tells the server to stop listening for requests when set to
        # False
        process_requests: bool = True
        # The authcode that retrieved
        authcode: Optional[str] = None

        def do_GET(self):
            match_authcode = authcode_regex.search(self.path)
            if self.path == "/":
                # This is the path we send people to to redirect them to the
                # back-end
                self.send_response(302)
                self.send_header("Location", f"{authorize_url}")
                self.end_headers()
            elif match_authcode:
                # We got an authcode!
                HttpHandler.authcode = match_authcode.group(1)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                message = "OK - You can now close this tab."
                self.wfile.write(bytes(message, "utf-8"))
                HttpHandler.process_requests = False
            else:
                # Bad or random request, reply with a 400.
                self.send_response(400)
                self.end_headers()

        # This override ensures the server does not log random messages on
        # stdout
        def log_message(self, format, *args):
            return

    __hl_print(
        f"Please go to this URL to log in: http://localhost:{cfg.temporary_server_port}"
    )
    # 127.0.0.1 so that it's only exposed to localhost
    server = HTTPServer(("127.0.0.1", int(cfg.temporary_server_port)), HttpHandler)
    while HttpHandler.process_requests:
        server.handle_request()

    __dbg_print("Received authentication code, processing...")
    token_payload: Dict[str, Optional[str]] = {
        "grant_type": "authorization_code",
        "code": HttpHandler.authcode,
        "redirect_uri": redirect_uri,
        "client_id": cfg.client_id,
        "code_verifier": code_verifier,
    }
    response: Response = requests.post(token_url, data=token_payload)

    if response.status_code != 200:
        __error_print("Failed to retrieve tokens, please try again.")
        print(response.text)
        exit(1)

    tokens: Any = response.json()

    if "id_token" not in tokens or "refresh_token" not in tokens:
        __error_print("Did not receive tokens, please try again.")
        exit(2)

    __dbg_print(f"Tokens received, storing in {cfg.token_file_path}")
    __store_tokens(cfg, tokens)

    __success_print("Successfully authenticated.")
    return tokens["id_token"], tokens["refresh_token"]


def read_stored_jwt_or_request_authcode(
    cfg: AuthConfiguration,
) -> Tuple[Credentials, bool]:
    if os.path.exists(cfg.token_file_path):
        with open(cfg.token_file_path, "r") as file:
            stored: Dict[str, Any] = json.loads(file.read().strip())
        __dbg_print("Using stored tokens")
        return Credentials(stored["id_token"], stored["refresh_token"]), False
    else:
        return Credentials(*__request_authcode(cfg)), True


def __request_new_token(cfg, refresh_token) -> Tuple[Optional[str], Optional[str]]:
    payload: Dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": cfg.client_id,
    }
    response: Response = requests.post(token_url, data=payload)
    if response.status_code != 200:
        __dbg_print(
            f"Failed to automatically refresh, err code {response.status_code}\n{response.text}"
        )
        return None, None
    tokens: Any = response.json()
    __store_tokens(cfg, tokens)
    return tokens["id_token"], tokens["refresh_token"]


# endregion

# region Back-end API utilities


def __send_protected_request(
    cfg: AuthConfiguration, credentials: Credentials, *args, **kwargs
):
    raise_for_status: bool = kwargs.pop("raise_for_status", True)

    kwargs.setdefault("method", "GET")
    kwargs.setdefault("headers", dict())
    kwargs["headers"]["Authorization"] = f"Bearer {credentials.jwt}"

    response: Response = requests.request(*args, **kwargs)
    if response.status_code == 401:
        __dbg_print("Request failed with 401, refreshing token...")

        new_jwt: Optional[str]
        new_refresh_token: Optional[str]
        new_jwt, new_refresh_token = __request_new_token(cfg, credentials.refresh_token)

        if new_jwt is None or new_refresh_token is None:
            __error_print(
                "Automatic token refresh failed, please try logging in again."
            )
            new_jwt, new_refresh_token = __request_authcode(cfg)

        credentials.jwt = new_jwt
        credentials.refresh_token = new_refresh_token

        kwargs["headers"]["Authorization"] = f"Bearer {credentials.jwt}"

        response = requests.request(*args, **kwargs)

    if raise_for_status:
        response.raise_for_status()
    return response


def get_credentials(
    cfg: AuthConfiguration = productionAuthConfiguration,
) -> Credentials:
    credentials: Credentials
    credentials, _ = read_stored_jwt_or_request_authcode(cfg)

    # As __send_protected_request handles authentication retries properly,
    # let it handle the entire pipeline and return the reprocessed credentials
    # Note that, yes, although / does not require authorization (and does not
    # even exist), Quarkus still checks authentication and throws a 401 if you
    # provide an invalid JWT.
    __send_protected_request(
        cfg,
        credentials,
        url="https://srvc-auth.api.forge.epita.fr/",
        raise_for_status=False,
    )

    return credentials
