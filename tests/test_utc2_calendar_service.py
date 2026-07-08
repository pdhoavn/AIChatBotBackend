import json
import unittest
from datetime import date
from urllib.parse import parse_qs

import httpx

from app.services.utc2_calendar_service import (
    UTC2CalendarError,
    UTC2CalendarService,
)


POST_ID = "05e3341b-23df-4b28-bbb8-844fd1f76b9d"
SEO_TEXT = "lich-cong-tac-tuan-tu-2262026-den-2862026"


def build_post():
    return {
        "id": POST_ID,
        "title": "LỊCH CÔNG TÁC TUẦN TỪ 22/6/2026 ĐẾN 28/6/2026",
        "seo_text": SEO_TEXT,
        "type": "CALENDAR_ANNOUNCEMENT",
        "display": True,
        "content": json.dumps(
            {
                "startDate": "2026-06-21T17:00:00.000Z",
                "days": [
                    {
                        "date": "monday",
                        "schedules": [
                            {
                                "startTime": "08:30",
                                "endTime": "11:00",
                                "content": "Hội nghị giao ban đào tạo",
                                "address": "Phòng P.601 - A1",
                                "participant": "Ban Giám hiệu",
                                "implementer": "Hiệu trưởng",
                                "requirement": "",
                            }
                        ],
                    },
                    {
                        "date": "tuesday",
                        "schedules": [
                            {
                                "startTime": "10:00",
                                "endTime": "11:30",
                                "content": "Họp Chi bộ Hành chính",
                                "address": "Phòng họp D3",
                                "participant": "Toàn thể đảng viên",
                                "implementer": "Bí thư chi bộ",
                                "requirement": "",
                            }
                        ],
                    },
                    {"date": "wednesday", "schedules": []},
                    {"date": "thursday", "schedules": []},
                    {"date": "friday", "schedules": []},
                    {"date": "saturday", "schedules": []},
                    {"date": "sunday", "schedules": []},
                ],
            },
            ensure_ascii=False,
        ),
    }


class UTC2CalendarServiceTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url.path.endswith(f"/post/{POST_ID}"):
                return httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "message": "ok",
                        "responseData": build_post(),
                    },
                )
            if request.url.path.endswith("/post"):
                return httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "message": "ok",
                        "responseData": {
                            "count": 1,
                            "rows": [
                                {
                                    "id": POST_ID,
                                    "title": build_post()["title"],
                                    "seo_text": SEO_TEXT,
                                }
                            ],
                        },
                    },
                )
            return httpx.Response(404)

        self.client = httpx.AsyncClient(
            base_url="https://utc2.edu.vn/api/v1.0",
            transport=httpx.MockTransport(handler),
        )
        self.service = UTC2CalendarService(
            client=self.client,
            cache_ttl_seconds=600,
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    def test_detects_calendar_queries_without_matching_timetables(self):
        self.assertTrue(
            self.service.is_calendar_query("Cho mình xem lịch công tác tuần này")
        )
        self.assertTrue(self.service.is_calendar_query("Lịch họp thứ 3 có gì?"))
        self.assertFalse(self.service.is_calendar_query("Thời khóa biểu tuần này"))
        self.assertFalse(self.service.is_calendar_query("Lịch học môn Toán"))

    async def test_answers_only_requested_day(self):
        result = await self.service.answer_query(
            "Hôm nay có lịch công tác gì?", today=date(2026, 6, 23)
        )

        self.assertIn("Thứ 3, 23/06/2026", result.answer)
        self.assertIn("Họp Chi bộ Hành chính", result.answer)
        self.assertNotIn("Hội nghị giao ban đào tạo", result.answer)
        self.assertIn(
            f"https://utc2.edu.vn/noi-bo/lich-cong-tac/{SEO_TEXT}",
            result.answer,
        )

        list_request = self.requests[0]
        query = parse_qs(list_request.url.query.decode())
        self.assertEqual(query["pageSize"], ["20"])
        self.assertIn("type==CALENDAR_ANNOUNCEMENT", query["filters"][0])
        self.assertIn("display==true", query["filters"][0])

    async def test_answers_full_current_week(self):
        result = await self.service.answer_query(
            "Lịch công tác tuần này", today=date(2026, 6, 23)
        )

        self.assertIn("Thứ 2, 22/06/2026", result.answer)
        self.assertIn("Thứ 3, 23/06/2026", result.answer)
        self.assertIn("Hội nghị giao ban đào tạo", result.answer)
        self.assertIn("Họp Chi bộ Hành chính", result.answer)

    async def test_builds_structured_context_for_llm(self):
        result = await self.service.get_query_context(
            "Ai họp ở phòng D3?", today=date(2026, 6, 23)
        )
        context = json.loads(result.context)

        self.assertEqual(context["current_date"], "2026-06-23")
        self.assertEqual(context["week_start"], "2026-06-22")
        self.assertEqual(context["week_end"], "2026-06-28")
        tuesday = context["days"][1]
        self.assertEqual(tuesday["day_name"], "Thứ 3")
        self.assertEqual(
            tuesday["schedules"][0]["address"], "Phòng họp D3"
        )
        self.assertEqual(
            tuesday["schedules"][0]["participant"], "Toàn thể đảng viên"
        )

    async def test_reuses_cached_list_and_detail(self):
        await self.service.answer_query(
            "Lịch công tác hôm nay", today=date(2026, 6, 23)
        )
        await self.service.answer_query(
            "Lịch công tác hôm nay", today=date(2026, 6, 23)
        )

        self.assertEqual(len(self.requests), 2)

    async def test_rejects_successful_http_with_failed_api_status(self):
        async def failed_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "status": "fail",
                    "message": "error",
                    "responseData": None,
                    "violations": [{"message": "API failure"}],
                },
            )

        async with httpx.AsyncClient(
            base_url="https://utc2.edu.vn/api/v1.0",
            transport=httpx.MockTransport(failed_handler),
        ) as client:
            service = UTC2CalendarService(client=client)
            with self.assertRaisesRegex(UTC2CalendarError, "API failure"):
                await service.get_latest_post()


if __name__ == "__main__":
    unittest.main()
