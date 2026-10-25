"""Local checks never call a model service or open external connections."""

import ipaddress
import socket

import httpx
import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    original_connect = socket.socket.connect

    def blocked(*args, **kwargs):
        raise AssertionError("This offline suite must not open a network connection")

    def connect_local_pipe(sock, address):
        # Windows asyncio uses a loopback socket pair for its internal wakeup pipe.
        if isinstance(address, tuple) and ipaddress.ip_address(address[0]).is_loopback:
            return original_connect(sock, address)
        return blocked()

    monkeypatch.setattr(socket.socket, "connect", connect_local_pipe)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)
