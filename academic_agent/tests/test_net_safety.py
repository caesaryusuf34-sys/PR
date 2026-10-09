"""SSRF protection: URL validation, unsafe redirects, blocked domains, connected-peer check."""
import socket

import pytest

from http_client import BlockedDomain, HttpClient, _check_peer
from net_safety import UnresolvableHost, UnsafeURL, ip_is_public, validate_url
from tests.conftest import FakeWeb, make_client

PUB = lambda host, port: ["93.184.216.34"]


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost/admin", "https://[::1]/", "http://10.0.0.5/", "http://192.168.1.1:8080/",
    "http://172.16.0.1/", "http://169.254.169.254/latest/meta-data/", "http://100.64.0.1/", "http://0.0.0.0/",
    "http://2130706433/", "http://0x7f000001/", "http://[::ffff:127.0.0.1]/", "http://[::ffff:10.0.0.1]/",
    "http://[fd00::1]/", "http://[fe80::1]/", "file:///etc/passwd", "ftp://example.org/x.pdf", "gopher://x/", "javascript:alert(1)",
    "http://user:pw@example.org/", "http://example.org:22/", "https://example.org:6379/", "http://metadata.google.internal/",
    "http://printer.local/", "http://db.internal/", "", "https://exa mple.org/", "http://" + "a" * 3000 + ".org/",
])
def test_unsafe_urls_are_rejected(url):
    with pytest.raises(UnsafeURL):
        validate_url(url, PUB)


@pytest.mark.parametrize("url", ["https://api.crossref.org/works", "http://example.org/a.pdf", "https://example.org:8443/x",
                                 "https://[2606:4700:4700::1111]/"])
def test_public_urls_pass(url):
    assert validate_url(url, PUB) == url


def test_hostname_resolving_to_private_address_is_rejected():
    with pytest.raises(UnsafeURL):
        validate_url("https://innocent.example.org/", lambda h, p: ["93.184.216.34", "10.0.0.7"])     # one bad record is enough


def test_unresolvable_host_is_distinct_error():
    def boom(host, port):
        raise UnresolvableHost("nx")
    with pytest.raises(UnresolvableHost):
        validate_url("https://nxdomain.example/", boom)


def test_ip_classification():
    assert ip_is_public("8.8.8.8") and ip_is_public("2606:4700:4700::1111")
    assert not ip_is_public("::ffff:192.168.0.1") and not ip_is_public("64:ff9b::7f00:1") and not ip_is_public("224.0.0.1")


def test_redirect_to_private_address_is_blocked(settings):
    web = FakeWeb().redirect("https://good.example/start", "http://169.254.169.254/latest/meta-data/")
    c = make_client(settings, web)
    with pytest.raises(UnsafeURL):
        c.request("GET", "https://good.example/start")
    assert not any("169.254" in call for call in web.calls)            # the private URL was never requested


def test_redirect_to_non_http_scheme_is_blocked(settings):
    web = FakeWeb().redirect("https://good.example/start", "file:///etc/passwd")
    with pytest.raises(UnsafeURL):
        make_client(settings, web).request("GET", "https://good.example/start")


def test_redirect_loop_is_bounded(settings):
    web = FakeWeb().redirect("https://a.example/", "https://b.example/").redirect("https://b.example/", "https://a.example/")
    from http_client import RequestFailed
    with pytest.raises(RequestFailed):
        make_client(settings, web).request("GET", "https://a.example/")
    assert len(web.calls) <= settings.max_redirects + 1


def test_blocked_domains_are_never_contacted(settings):
    web = FakeWeb().add("https://sci-hub.se/10.1/x", "x")
    c = make_client(settings, web)
    with pytest.raises(BlockedDomain):
        c.request("GET", "https://sci-hub.se/10.1/x")
    with pytest.raises(BlockedDomain):
        c.request("GET", "https://mirror.sci-hub.se/10.1/x")
    assert web.calls == []


def test_connected_peer_check_blocks_dns_rebinding():
    """Even if validation was fooled by DNS, the socket's real peer address is checked."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    cli = socket.create_connection(srv.getsockname())
    try:
        with pytest.raises(UnsafeURL):
            _check_peer(cli)
    finally:
        cli.close()
        srv.close()
