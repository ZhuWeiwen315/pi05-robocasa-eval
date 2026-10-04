"""List fixed RoboCasa assets; optionally make bounded metadata requests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit
from urllib import error, request

from robocasa_eval.assets import asset_specs


class _NoRedirect(request.HTTPRedirectHandler):
    """Prevent urllib from converting a redirected HEAD into a GET."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _RangeRedirect(request.HTTPRedirectHandler):
    """Keep the one-byte Range header on every HTTPS redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            return None
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.add_header("Range", "bytes=0-0")
        return redirected


@dataclass(frozen=True)
class ProbeResult:
    initial_url: str
    final_url: str
    status: str
    content_type: str
    content_disposition: str
    content_length: str
    content_range: str
    body_bytes: int
    archive_bytes: int | None


def _safe_url(url: str) -> str:
    """Do not print transient Box redirect tokens in reports or logs."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host == "public.boxcloud.com" or host.endswith(".boxcloud.com"):
        return f"{parts.scheme}://{parts.netloc}/[redacted]/download"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "[redacted]" if parts.query else "", ""))


def parse_content_range(value: str) -> int | None:
    """Accept only a valid one-byte partial response with a known total."""
    import re

    match = re.fullmatch(r"bytes 0-0/([1-9][0-9]*)", value.strip(), re.IGNORECASE)
    return int(match.group(1)) if match else None


def archive_size(status: int, content_type: str, disposition: str,
                 content_length: str, content_range: str) -> int | None:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in {"text/html", "application/xhtml+xml"}:
        return None
    if status == 206:
        return parse_content_range(content_range)
    if status != 200:
        return None
    is_zip = media_type in {"application/zip", "application/x-zip-compressed"}
    is_attachment_zip = "attachment" in disposition.lower() and ".zip" in disposition.lower()
    if not (is_zip or is_attachment_zip):
        return None
    return int(content_length) if content_length.isdecimal() else None


def range_probe(url: str, timeout: int = 15) -> ProbeResult:
    """Send Range: bytes=0-0, read at most one byte, and close immediately.

    The socket timeout applies to both connection and reads. Redirects remain
    HTTPS and carry the same Range header; no response is saved to disk.
    """
    if urlsplit(url).scheme != "https":
        raise ValueError("Only HTTPS asset URLs are allowed")
    probe = request.Request(url, headers={"Range": "bytes=0-0"}, method="GET")
    opener = request.build_opener(_RangeRedirect())
    response = None
    try:
        try:
            response = opener.open(probe, timeout=timeout)
        except error.HTTPError as exc:
            response = exc
        status = response.status if hasattr(response, "status") else response.code
        headers = response.headers
        fields = tuple(headers.get(key, "") for key in (
            "Content-Type", "Content-Disposition", "Content-Length", "Content-Range"
        ))
        # Error and redirect responses have no useful archive body metadata.
        body_bytes = len(response.read(1)) if status in (200, 206) else 0
        size = archive_size(status, *fields)
        return ProbeResult(_safe_url(url), _safe_url(response.geturl()), str(status),
                           *fields, body_bytes, size)
    except error.URLError:
        return ProbeResult(_safe_url(url), _safe_url(url), "unavailable", "", "", "", "", 0, None)
    finally:
        if response is not None:
            response.close()


def head_size(url: str, timeout: int = 20) -> tuple[str, int | None]:
    head = request.Request(url, method="HEAD")
    opener = request.build_opener(_NoRedirect())
    try:
        with opener.open(head, timeout=timeout) as response:
            raw = response.headers.get("Content-Length")
            content_type = response.headers.get("Content-Type", "").lower()
            attachment = "attachment" in response.headers.get("Content-Disposition", "").lower()
            is_archive = attachment or "zip" in content_type or "octet-stream" in content_type
            return str(response.status), int(raw) if is_archive and raw and raw.isdecimal() else None
    except error.HTTPError as exc:
        return str(exc.code), None
    except error.URLError:
        return "unavailable", None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--head", action="store_true", help="Query sizes with HEAD; never GET")
    mode.add_argument("--range-probe", action="store_true", help="Bounded one-byte Range GET metadata probe")
    args = parser.parse_args()
    if args.range_probe:
        print("name\texpected_subdir\tinitial_url\tfinal_url\tstatus\tcontent_type\tcontent_disposition\tcontent_length\tcontent_range\tbody_bytes\tarchive_bytes")
        for spec in asset_specs():
            result = range_probe(spec.direct_url)
            print("\t".join((spec.name, spec.subdir, result.initial_url, result.final_url,
                             result.status, result.content_type, result.content_disposition,
                             result.content_length, result.content_range, str(result.body_bytes),
                             str(result.archive_bytes) if result.archive_bytes is not None else "unknown")))
        return
    print("name\tshared_url\texpected_subdir\tstatus\tcontent_length_bytes")
    for spec in asset_specs():
        status, size = head_size(spec.direct_url) if args.head else ("not_queried", None)
        print(f"{spec.name}\t{spec.shared_url}\t{spec.subdir}\t{status}\t{size if size is not None else 'unknown'}")


if __name__ == "__main__":
    main()
