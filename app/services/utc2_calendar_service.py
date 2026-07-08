import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx
from unidecode import unidecode


class UTC2CalendarError(RuntimeError):
    """Raised when the public UTC2 calendar API cannot provide valid data."""


@dataclass(frozen=True)
class CalendarAnswer:
    answer: str
    source_url: str
    title: str
    post_id: str


@dataclass(frozen=True)
class CalendarContext:
    context: str
    source_url: str
    title: str
    post_id: str


class UTC2CalendarService:
    """Read-only client for UTC2 working-schedule posts."""

    POST_TYPE = "CALENDAR_ANNOUNCEMENT"
    DAY_NAMES = {
        "monday": "Thứ 2",
        "tuesday": "Thứ 3",
        "wednesday": "Thứ 4",
        "thursday": "Thứ 5",
        "friday": "Thứ 6",
        "saturday": "Thứ 7",
        "sunday": "Chủ nhật",
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        site_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        cache_ttl_seconds: Optional[int] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = (
            base_url
            or os.getenv("UTC2_API_BASE_URL")
            or "https://utc2.edu.vn/api/v1.0"
        ).rstrip("/")
        self.site_url = (
            site_url or os.getenv("UTC2_SITE_URL") or "https://utc2.edu.vn"
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("UTC2_API_TIMEOUT_SECONDS", "15")
        )
        self.cache_ttl_seconds = cache_ttl_seconds or int(
            os.getenv("UTC2_CALENDAR_CACHE_TTL_SECONDS", "600")
        )
        self._client = client
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_lock = asyncio.Lock()
        self._timezone = ZoneInfo(
            os.getenv("UTC2_TIMEZONE", "Asia/Ho_Chi_Minh")
        )

    def today(self) -> date:
        return datetime.now(self._timezone).date()

    @staticmethod
    def is_calendar_query(query: str) -> bool:
        normalized = unidecode(query or "").lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return False

        if "thoi khoa bieu" in normalized or "lich hoc" in normalized:
            return False

        explicit_terms = (
            "lich cong tac",
            "lich lam viec",
            "lich hop",
            "cong tac tuan",
        )
        if any(term in normalized for term in explicit_terms):
            return True

        temporal_terms = (
            "hom nay",
            "ngay mai",
            "tuan nay",
            "tuan sau",
            "tuan truoc",
            "thu 2",
            "thu 3",
            "thu 4",
            "thu 5",
            "thu 6",
            "thu 7",
            "chu nhat",
        )
        return "lich" in normalized and any(
            term in normalized for term in temporal_terms
        )

    async def answer_query(
        self, query: str, today: Optional[date] = None
    ) -> CalendarAnswer:
        calendar_context = await self.get_query_context(query, today=today)
        post = await self._get_post_detail(calendar_context.post_id)
        content = self._parse_content(post)
        current_date = today or datetime.now(self._timezone).date()
        target_date, day_only = self._resolve_target_date(query, current_date)
        answer = self._format_answer(
            post=post,
            content=content,
            source_url=calendar_context.source_url,
            target_date=target_date if day_only else None,
        )
        return CalendarAnswer(
            answer=answer,
            source_url=calendar_context.source_url,
            title=calendar_context.title,
            post_id=calendar_context.post_id,
        )

    async def get_query_context(
        self, query: str, today: Optional[date] = None
    ) -> CalendarContext:
        current_date = today or datetime.now(self._timezone).date()
        target_date, _ = self._resolve_target_date(query, current_date)
        return await self.get_context_for_date(
            target_date=target_date,
            current_date=current_date,
        )

    async def get_context_for_date(
        self,
        target_date: Optional[date],
        current_date: Optional[date] = None,
    ) -> CalendarContext:
        """Return normalized calendar JSON for the week containing target_date."""
        effective_current_date = current_date or datetime.now(self._timezone).date()
        post = await self.get_post_for_date(target_date)
        content = self._parse_content(post)
        source_url = self._build_source_url(post)
        normalized_schedule = self._normalize_schedule(post, content)
        context = json.dumps(
            {
                "current_date": effective_current_date.isoformat(),
                "calendar_title": post.get("title"),
                "calendar_url": source_url,
                "week_start": normalized_schedule["week_start"],
                "week_end": normalized_schedule["week_end"],
                "days": normalized_schedule["days"],
            },
            ensure_ascii=False,
            indent=2,
        )
        return CalendarContext(
            context=context,
            source_url=source_url,
            title=str(post.get("title") or "Lịch công tác UTC2"),
            post_id=str(post.get("id") or ""),
        )

    async def get_latest_post(self) -> Dict[str, Any]:
        posts = await self._list_posts(page_size=1)
        if not posts:
            raise UTC2CalendarError("UTC2 chưa có lịch công tác được công khai.")
        return await self._get_post_detail(str(posts[0]["id"]))

    async def get_post_for_date(
        self, target_date: Optional[date]
    ) -> Dict[str, Any]:
        if target_date is None:
            return await self.get_latest_post()

        posts = await self._list_posts(target_date=target_date, page_size=20)
        for post_summary in posts:
            post = await self._get_post_detail(str(post_summary["id"]))
            start_date = self._calendar_start_date(post)
            if start_date <= target_date <= start_date + timedelta(days=6):
                return post

        raise UTC2CalendarError(
            f"Không tìm thấy lịch công tác chứa ngày "
            f"{target_date.strftime('%d/%m/%Y')}."
        )

    async def _list_posts(
        self, target_date: Optional[date] = None, page_size: int = 20
    ) -> List[Dict[str, Any]]:
        filters = [f"type=={self.POST_TYPE}", "display==true"]
        if target_date:
            created_from = target_date - timedelta(days=14)
            created_to = target_date + timedelta(days=2)
            filters.extend(
                [
                    f"created_at>={created_from.strftime('%m/%d/%Y')}",
                    f"created_at<={created_to.strftime('%m/%d/%Y')}",
                ]
            )

        params = {
            "currentPage": 1,
            "pageSize": page_size,
            "sortField": "created_at",
            "sortOrder": "DESC",
            "filters": ", ".join(filters),
            "subCategorys": "",
        }
        cache_key = f"posts:{target_date}:{page_size}"
        payload = await self._get_json("/post", params=params, cache_key=cache_key)
        response_data = payload.get("responseData") or {}
        rows = response_data.get("rows")
        if not isinstance(rows, list):
            raise UTC2CalendarError("Danh sách lịch UTC2 không đúng định dạng.")
        return rows

    async def _get_post_detail(self, post_id: str) -> Dict[str, Any]:
        payload = await self._get_json(
            f"/post/{post_id}", cache_key=f"post:{post_id}"
        )
        post = payload.get("responseData")
        if not isinstance(post, dict):
            raise UTC2CalendarError("Chi tiết lịch UTC2 không đúng định dạng.")
        return post

    async def _get_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if cache_key:
            cached = await self._cache_get(cache_key)
            if cached is not None:
                return cached

        try:
            if self._client:
                response = await self._client.get(path, params=params)
            else:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                    follow_redirects=True,
                    headers={"Accept": "application/json"},
                ) as client:
                    response = await client.get(path, params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            raise UTC2CalendarError(
                "Không thể kết nối hoặc đọc dữ liệu lịch công tác UTC2."
            ) from exc

        if not isinstance(payload, dict) or payload.get("status") != "success":
            violations = payload.get("violations") if isinstance(payload, dict) else None
            raise UTC2CalendarError(
                self._violation_message(violations)
                or "API lịch công tác UTC2 trả về trạng thái không thành công."
            )

        if cache_key:
            await self._cache_set(cache_key, payload)
        return payload

    async def _cache_get(self, key: str) -> Optional[Any]:
        async with self._cache_lock:
            cached = self._cache.get(key)
            if not cached:
                return None
            expires_at, value = cached
            if expires_at <= time.monotonic():
                self._cache.pop(key, None)
                return None
            return value

    async def _cache_set(self, key: str, value: Any) -> None:
        async with self._cache_lock:
            self._cache[key] = (
                time.monotonic() + self.cache_ttl_seconds,
                value,
            )

    def _resolve_target_date(
        self, query: str, current_date: date
    ) -> Tuple[Optional[date], bool]:
        normalized = unidecode(query or "").lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()

        explicit_date = self._extract_explicit_date(normalized, current_date)
        if explicit_date:
            return explicit_date, True
        if "ngay mai" in normalized:
            return current_date + timedelta(days=1), True
        if "hom qua" in normalized:
            return current_date - timedelta(days=1), True
        if "hom nay" in normalized:
            return current_date, True

        week_start = current_date - timedelta(days=current_date.weekday())
        if "tuan sau" in normalized:
            return week_start + timedelta(days=7), False
        if "tuan truoc" in normalized:
            return week_start - timedelta(days=7), False

        weekday_offset = self._extract_weekday_offset(normalized)
        if weekday_offset is not None:
            return week_start + timedelta(days=weekday_offset), True
        if "tuan nay" in normalized:
            return current_date, False

        return None, False

    @staticmethod
    def _extract_explicit_date(
        normalized_query: str, current_date: date
    ) -> Optional[date]:
        match = re.search(
            r"(?<!\d)(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?(?!\d)",
            normalized_query,
        )
        if not match:
            return None
        day_value = int(match.group(1))
        month_value = int(match.group(2))
        year_text = match.group(3)
        year_value = current_date.year
        if year_text:
            year_value = int(year_text)
            if year_value < 100:
                year_value += 2000
        try:
            return date(year_value, month_value, day_value)
        except ValueError:
            return None

    @staticmethod
    def _extract_weekday_offset(normalized_query: str) -> Optional[int]:
        weekday_patterns = (
            (r"\bthu\s*2\b", 0),
            (r"\bthu\s*3\b", 1),
            (r"\bthu\s*4\b", 2),
            (r"\bthu\s*5\b", 3),
            (r"\bthu\s*6\b", 4),
            (r"\bthu\s*7\b", 5),
            (r"\bchu nhat\b", 6),
        )
        for pattern, offset in weekday_patterns:
            if re.search(pattern, normalized_query):
                return offset
        return None

    def _parse_content(self, post: Dict[str, Any]) -> Dict[str, Any]:
        raw_content = post.get("content")
        if isinstance(raw_content, dict):
            content = raw_content
        elif isinstance(raw_content, str):
            try:
                content = json.loads(raw_content)
            except json.JSONDecodeError as exc:
                raise UTC2CalendarError(
                    "Nội dung lịch UTC2 không phải JSON hợp lệ."
                ) from exc
        else:
            raise UTC2CalendarError("Bài lịch UTC2 không có nội dung.")

        if not isinstance(content.get("days"), list):
            raise UTC2CalendarError("Nội dung lịch UTC2 thiếu danh sách ngày.")
        return content

    def _calendar_start_date(self, post: Dict[str, Any]) -> date:
        content = self._parse_content(post)
        start_date = content.get("startDate")
        if not isinstance(start_date, str):
            raise UTC2CalendarError("Nội dung lịch UTC2 thiếu ngày bắt đầu.")
        try:
            parsed = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(self._timezone).date()
        except ValueError as exc:
            raise UTC2CalendarError("Ngày bắt đầu của lịch UTC2 không hợp lệ.") from exc

    def _format_answer(
        self,
        post: Dict[str, Any],
        content: Dict[str, Any],
        source_url: str,
        target_date: Optional[date],
    ) -> str:
        start_date = self._calendar_start_date(post)
        sections: List[str] = [f"## {post.get('title', 'Lịch công tác UTC2')}"]
        matched_day = False

        for index, day_data in enumerate(content["days"]):
            day_date = start_date + timedelta(days=index)
            if target_date and day_date != target_date:
                continue
            matched_day = True
            day_key = str(day_data.get("date") or "").lower()
            day_name = self.DAY_NAMES.get(day_key, day_key or f"Ngày {index + 1}")
            schedules = [
                item
                for item in day_data.get("schedules", [])
                if isinstance(item, dict) and self._has_schedule_content(item)
            ]
            sections.append(f"### {day_name}, {day_date.strftime('%d/%m/%Y')}")
            if not schedules:
                sections.append("- Không có lịch công tác được công bố.")
                continue
            schedules.sort(key=lambda item: str(item.get("startTime") or "99:99"))
            sections.extend(self._format_schedule(item) for item in schedules)

        if target_date and not matched_day:
            sections.append(
                f"Không có dữ liệu lịch cho ngày {target_date.strftime('%d/%m/%Y')}."
            )

        sections.append(f"[Xem lịch công tác trên website UTC2]({source_url})")
        return "\n\n".join(sections)

    def _normalize_schedule(
        self, post: Dict[str, Any], content: Dict[str, Any]
    ) -> Dict[str, Any]:
        start_date = self._calendar_start_date(post)
        days = []
        for index, day_data in enumerate(content["days"]):
            day_date = start_date + timedelta(days=index)
            day_key = str(day_data.get("date") or "").lower()
            schedules = [
                {
                    "start_time": str(item.get("startTime") or "").strip(),
                    "end_time": str(item.get("endTime") or "").strip(),
                    "content": str(item.get("content") or "").strip(),
                    "address": str(item.get("address") or "").strip(),
                    "participant": str(item.get("participant") or "").strip(),
                    "implementer": str(item.get("implementer") or "").strip(),
                    "requirement": str(item.get("requirement") or "").strip(),
                }
                for item in day_data.get("schedules", [])
                if isinstance(item, dict) and self._has_schedule_content(item)
            ]
            schedules.sort(key=lambda item: item["start_time"] or "99:99")
            days.append(
                {
                    "date": day_date.isoformat(),
                    "date_display": day_date.strftime("%d/%m/%Y"),
                    "day_name": self.DAY_NAMES.get(
                        day_key, day_key or f"Ngày {index + 1}"
                    ),
                    "schedules": schedules,
                }
            )
        return {
            "week_start": start_date.isoformat(),
            "week_end": (start_date + timedelta(days=6)).isoformat(),
            "days": days,
        }

    @staticmethod
    def _has_schedule_content(schedule: Dict[str, Any]) -> bool:
        return any(
            str(schedule.get(field) or "").strip()
            for field in ("content", "startTime", "endTime", "address")
        )

    @staticmethod
    def _format_schedule(schedule: Dict[str, Any]) -> str:
        start_time = str(schedule.get("startTime") or "").strip()
        end_time = str(schedule.get("endTime") or "").strip()
        time_range = "–".join(value for value in (start_time, end_time) if value)
        title = str(schedule.get("content") or "Nội dung chưa cập nhật").strip()
        lines = [f"- **{time_range or 'Chưa có giờ'}** — {title}"]
        details = (
            ("Địa điểm", schedule.get("address")),
            ("Thành phần", schedule.get("participant")),
            ("Chủ trì", schedule.get("implementer")),
            ("Yêu cầu", schedule.get("requirement")),
        )
        for label, value in details:
            normalized_value = str(value or "").strip()
            if normalized_value:
                lines.append(f"  - {label}: {normalized_value}")
        return "\n".join(lines)

    def _build_source_url(self, post: Dict[str, Any]) -> str:
        seo_text = str(post.get("seo_text") or "").strip().strip("/")
        if not seo_text:
            return self.site_url
        return f"{self.site_url}/noi-bo/lich-cong-tac/{seo_text}"

    @staticmethod
    def _violation_message(violations: Any) -> Optional[str]:
        if not isinstance(violations, list):
            return None
        messages = [
            str(item.get("message")).strip()
            for item in violations
            if isinstance(item, dict) and item.get("message")
        ]
        return "; ".join(messages) or None
