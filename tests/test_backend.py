import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import create_app
from backend.jobs import JobService
from backend.pipeline import PipelineRequest, build_pipeline_plan, video_id_for_path
from backend.settings import BackendSettings


class BackendTests(unittest.TestCase):
    def test_build_pipeline_plan_starts_from_zero_and_uses_windows_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "padel_start.mov"
            video.write_bytes(b"sample-video")
            settings = BackendSettings(
                repo_root=root,
                jobs_dir=root / "backend_jobs",
                run_script=root / "run.ps1",
            )
            request = PipelineRequest(
                video_path=video,
                initial_points="0-0",
                initial_games="0-0",
                proposal_min_time_seconds=0.0,
            )

            plan = build_pipeline_plan(request, settings)

            self.assertEqual(plan.video_id, video_id_for_path(video))
            self.assertEqual(plan.env["PADEL_SCORE_INITIAL_POINTS"], "0-0")
            self.assertEqual(plan.env["PADEL_SCORE_INITIAL_GAMES"], "0-0")
            self.assertEqual(plan.env["PADEL_PROPOSAL_MIN_TIME_SECONDS"], "0.0")
            self.assertEqual(plan.env["PADEL_SCORE_SOURCE"], "proposal")
            self.assertIn("-File", plan.command)
            self.assertIn(str(settings.run_script), plan.command)
            self.assertIn("-VideoPath", plan.command)
            self.assertIn(str(video), plan.command)

    def test_api_submits_job_and_persists_successful_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "padel_start.mov"
            video.write_bytes(b"sample-video")
            settings = BackendSettings(
                repo_root=root,
                jobs_dir=root / "backend_jobs",
                run_script=root / "run.ps1",
            )

            def fake_runner(plan, log_path):
                log_path.write_text("ok", encoding="utf-8")
                return 0

            service = JobService(settings=settings, runner=fake_runner, run_inline=True)
            app = create_app(service)
            client = TestClient(app)

            response = client.post(
                "/v1/jobs",
                json={
                    "video_path": str(video),
                    "initial_points": "0-0",
                    "initial_games": "0-0",
                    "proposal_min_time_seconds": 0.0,
                },
            )

            self.assertEqual(response.status_code, 202)
            body = response.json()
            self.assertEqual(body["status"], "succeeded")
            self.assertEqual(body["request"]["initial_points"], "0-0")

            status_response = client.get(f"/v1/jobs/{body['job_id']}")
            self.assertEqual(status_response.status_code, 200)
            self.assertEqual(status_response.json()["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
