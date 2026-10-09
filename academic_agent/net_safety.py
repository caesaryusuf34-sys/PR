"""URL and address validation (SSRF defence).

Every URL the agent is about to request - including every redirect hop - passes through
:func:`validate_url`. A URL is rejected unless its scheme is http(s), it carries no credentials,
it uses an ordinary web port and *every* address its host resolves to is a public, globally
routable address. The HTTP layer additionally checks the address it actually connected to
(see ``http_client``), which closes the DNS-rebinding window.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from typing import Callable, Iterable, Optional
from urllib.parse import urlsplit

ALLOWED_SCHEMES = ("http", "https")
ALLOWED_PORTS = (80, 443, 8080, 8443)
MAX_URL_LENGTH = 2048
_BLOCKED_HOSTS = {"localhost", "metadata", "metadata.google.internal", "instance-data", "ip6-localhost"}
_BLOCKED_SUFFIXES = (".local", ".localhost", ".internal", ".localdomain", ".lan", ".home", ".corp",
                     ".intranet", ".private", ".home.arpa", ".cluster.local")

_NUMERIC_HOST = re.compile(r"^(?:0x[0-9a-f]*|\d+)(?:\.(?:0x[0-9a-f]*|\d+))*$", re.I)

Resolver = Callable[[str, int], Iterable[str]]


class URLValidationError(ValueError):
    """Base class: the URL must not be requested."""


class UnsafeURL(URLValidationError):
    pass


class UnresolvableHost(URLValidationError):
    pass


def ip_is_public(ip) -> bool:
    """True only for globally routable unicast addresses (incl. IPv4 embedded in IPv6 forms)."""
    if isinstance(ip, str):
        ip = ipaddress.ip_address(ip.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip_is_public(ip.ipv4_mapped)
        if ip.sixtofour is not None and not ip_is_public(ip.sixtofour):
            return False
        if ip.teredo is not None:
            return False
        if ip in ipaddress.ip_network("64:ff9b::/96"):          # NAT64: judge the embedded IPv4 address
            return ip_is_public(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return bool(ip.is_global) and not ip.is_multicast


def default_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError) as exc:
        raise UnresolvableHost(f"cannot resolve host {host!r}: {exc}") from exc
    return [info[4][0] for info in infos]


def host_matches(host: str, domains: Iterable[str]) -> bool:
    host = (host or "").lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in domains)


def validate_url(url: str, resolver: Optional[Resolver] = None) -> str:
    """Return the URL if it is safe to request, otherwise raise :class:`URLValidationError`."""
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURL("empty URL")
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise UnsafeURL("URL too long")
    if any(ord(c) < 33 or ord(c) == 127 for c in url):
        raise UnsafeURL("URL contains control characters or whitespace")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise UnsafeURL(f"malformed URL: {exc}") from exc
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURL(f"scheme {parts.scheme!r} is not allowed")
    if parts.username or parts.password or "@" in (parts.netloc or ""):
        raise UnsafeURL("URLs with embedded credentials are not allowed")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise UnsafeURL("URL has no host")
    if port is None:
        port = 443 if parts.scheme.lower() == "https" else 80
    if port not in ALLOWED_PORTS:
        raise UnsafeURL(f"port {port} is not allowed")
    if host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_SUFFIXES):
        raise UnsafeURL(f"host {host!r} is internal")
    try:                                                         # literal IP address?
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
        if _NUMERIC_HOST.match(host):                            # 2130706433, 0x7f000001, 0177.0.0.1, 127.1 ...
            raise UnsafeURL("non-canonical IP address notation is not allowed")
    if literal is not None:
        if not ip_is_public(literal):
            raise UnsafeURL(f"address {host} is not a public address")
        return url
    try:
        host.encode("idna")
    except UnicodeError as exc:
        raise UnsafeURL("invalid host name") from exc
    addresses = list((resolver or default_resolver)(host, port))
    if not addresses:
        raise UnresolvableHost(f"host {host!r} did not resolve")
    for addr in addresses:
        try:
            if not ip_is_public(addr):
                raise UnsafeURL(f"host {host!r} resolves to non-public address {addr}")
        except ValueError as exc:
            raise UnsafeURL(f"host {host!r} resolved to an invalid address") from exc
    return url
