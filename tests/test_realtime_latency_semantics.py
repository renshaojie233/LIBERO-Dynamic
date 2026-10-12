from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.realtime_eval.control import ActionChunkPacket, ActionRuntime, LatencySampler, ObservationPacket
from scripts.realtime_eval.strategies.sync_hold import SyncHoldController
from scripts.realtime_eval.strategies.async_queue import AsyncQueueController
from scripts.realtime_eval.strategies.rtc_queue import RtcQueueController
from scripts.realtime_eval.strategies.vlash_future_state import VLASHFutureStateController
from realtime_eval.runner import build_parser, write_summary
from scripts.render_dynamic_config_previews import advance_physical_dynamic_time


class _Policy:
    def get_action_chunk(self, obs, language):
        del obs, language
        return np.zeros((10, 7), dtype=np.float32)


class _RtcPolicy(_Policy):
    supports_rtc_conditioning = True

    def get_action_chunk_rtc(
        self,
        obs,
        language,
        *,
        prev_chunk_left_over,
        inference_delay_steps,
    ):
        del obs, language, prev_chunk_left_over, inference_delay_steps
        return np.zeros((10, 7), dtype=np.float32)


class _HistoryPolicy(_Policy):
    required_observation_history_frames = 5
    required_action_history_steps = 16

    def __init__(self):
        self.requests = []

    def get_action_chunk(self, obs, language):
        del language
        self.requests.append(obs)
        return np.zeros((4, 7), dtype=np.float32)


class _BlockingPolicy(_Policy):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def get_action_chunk(self, obs, language):
        del obs, language
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test policy was not released")
        return np.zeros((10, 7), dtype=np.float32)


class _VLASHPolicy(_Policy):
    def __init__(self) -> None:
        self.requests = []

    def get_action_chunk(self, obs, language):
        del language
        self.requests.append(obs)
        return np.zeros((5, 7), dtype=np.float32)


def _packet() -> ObservationPacket:
    return ObservationPacket(
        episode_id="test",
        step_idx=0,
        sim_time_s=0.0,
        language="test",
        obs={},
        queue_len=0,
        last_action=np.zeros(7, dtype=np.float32),
    )


def _image_packet(step_idx: int) -> ObservationPacket:
    value = np.uint8(step_idx)
    return ObservationPacket(
        episode_id="test",
        step_idx=step_idx,
        sim_time_s=step_idx / 20.0,
        language="test",
        obs={
            "agentview_image": np.full((2, 2, 3), value, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.full((2, 2, 3), value, dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        },
        queue_len=0,
        last_action=np.zeros(7, dtype=np.float32),
    )


def test_history_policy_receives_observations_and_executed_actions() -> None:
    policy = _HistoryPolicy()
    controller = SyncHoldController(
        policy=policy,
        language="test",
        control_freq=20,
        mode="sync_hold",
        prefetch_threshold=0,
        action_chunk_size=4,
        chunk_size_threshold=None,
        latency_sampler=LatencySampler(source="fixed", fixed_seconds=0.0),
    )
    actions = ActionRuntime(7, empty_action="zero")
    try:
        for step_idx in range(5):
            if step_idx < 4:
                actions.executed_action_history.append(
                    np.full(7, float(step_idx), dtype=np.float32)
                )
            controller.tick(_image_packet(step_idx), actions)
            actions._queue = []
        request = policy.requests[-1]
        assert len(request["__observation_history"]) == 5
        assert int(request["__observation_history"][0]["agentview_image"][0, 0, 0]) == 0
        assert int(request["__observation_history"][-1]["agentview_image"][0, 0, 0]) == 4
        assert request["__executed_action_history"].shape == (4, 7)
    finally:
        controller.close()


def test_fixed_latency_can_model_first_request_warm_start() -> None:
    sampler = LatencySampler(source="fixed", fixed_seconds=0.08, first_seconds=0.31)
    assert sampler.sample() == pytest.approx(0.31)
    assert sampler.sample() == pytest.approx(0.08)
    assert sampler.sample() == pytest.approx(0.08)
    summary = sampler.summary()
    assert summary["calibrated_first_inference_seconds"] == pytest.approx(0.31)
    assert summary["calibrated_latency_seconds"] == pytest.approx(0.08)


def test_vlash_measured_latency_maps_to_delayed_image_without_second_queue_delay() -> None:
    measured_latency_s = 0.10398730050292215
    policy = _VLASHPolicy()
    controller = VLASHFutureStateController(
        policy=policy,
        language="test",
        control_freq=20,
        mode="vlash_future_state",
        prefetch_threshold=0,
        action_chunk_size=5,
        chunk_size_threshold=None,
        latency_sampler=LatencySampler(source="fixed", fixed_seconds=0.0),
        vlash_visual_delay_steps=3,
        vlash_measured_latency_seconds=measured_latency_s,
    )
    actions = ActionRuntime(7, empty_action="zero")
    try:
        ready = controller.tick(_image_packet(0), actions)
        assert len(ready) == 1
        actions.next_action()
        for step_idx in range(1, 5):
            assert controller.tick(_image_packet(step_idx), actions) == []
            actions.next_action()
        assert actions.queue_len == 0

        packet = _image_packet(5)
        packet.obs["robot0_eef_pos"][:] = 5.0
        ready = controller.tick(packet, actions)
        assert len(ready) == 1
        assert int(policy.requests[-1]["agentview_image"][0, 0, 0]) == 2
        assert np.all(policy.requests[-1]["robot0_eef_pos"] == 5.0)
        assert ready[0].available_step_idx == 5
        assert ready[0].metadata["delay_steps"] == 0
        assert ready[0].metadata["vlash_visual_observation_step_idx"] == 2
        assert ready[0].metadata["vlash_state_observation_step_idx"] == 5

        summary = controller.summary()
        assert summary["vlash_latency_steps_from_measurement"] == 3
        assert summary["vlash_represented_latency_seconds"] == pytest.approx(0.15)
        assert summary["vlash_latency_quantization_error_seconds"] == pytest.approx(
            0.15 - measured_latency_s
        )
        assert summary["vlash_latency_double_counted"] is False
    finally:
        controller.close()


def test_vlash_rejects_inconsistent_measured_latency_mapping() -> None:
    with pytest.raises(ValueError, match="mapping mismatch"):
        VLASHFutureStateController(
            policy=_VLASHPolicy(),
            language="test",
            control_freq=20,
            mode="vlash_future_state",
            prefetch_threshold=0,
            action_chunk_size=5,
            chunk_size_threshold=None,
            latency_sampler=LatencySampler(source="fixed", fixed_seconds=0.0),
            vlash_visual_delay_steps=2,
            vlash_measured_latency_seconds=0.10398730050292215,
        )


@pytest.mark.parametrize("latency", [0.0, 0.03, 0.05, 0.075])
def test_sync_latency_does_not_insert_control_underflow(latency: float) -> None:
    controller = SyncHoldController(
        policy=_Policy(),
        language="test",
        control_freq=20,
        mode="sync_hold",
        prefetch_threshold=0,
        action_chunk_size=10,
        chunk_size_threshold=None,
        latency_sampler=LatencySampler(source="fixed", fixed_seconds=latency),
    )
    actions = ActionRuntime(7, empty_action="zero")
    try:
        ready = controller.tick(_packet(), actions)
        assert len(ready) == 1
        assert actions.queue_len == 10
        assert ready[0].available_step_idx == 0
        assert ready[0].metadata["delay_steps"] == 0
        assert ready[0].metadata["latency_application"] == "continuous_physics"
        assert ready[0].metadata["sim_latency_s"] == pytest.approx(latency)

        _action, _original, source, underflow, _chunk, _before, _after = actions.next_action()
        assert source == "queue"
        assert underflow is False
        assert actions.underflow_steps == 0
    finally:
        controller.close()


@pytest.mark.parametrize("controller_cls", [AsyncQueueController, RtcQueueController])
def test_async_queue_latency_remains_control_slot_indexed(controller_cls) -> None:
    controller = controller_cls(
        policy=_Policy(),
        language="test",
        control_freq=20,
        mode="async_queue" if controller_cls is AsyncQueueController else "rtc_queue",
        prefetch_threshold=0,
        action_chunk_size=10,
        chunk_size_threshold=None,
        latency_sampler=LatencySampler(source="fixed", fixed_seconds=0.05),
    )
    actions = ActionRuntime(7, empty_action="zero")
    try:
        assert controller.tick(_packet(), actions) == []
        packet = _packet()
        packet = ObservationPacket(**{**packet.__dict__, "step_idx": 1, "sim_time_s": 0.05})
        ready = controller.tick(packet, actions)
        assert len(ready) == 1
        assert ready[0].available_step_idx == 1
        assert ready[0].metadata["delay_steps"] == 1
        assert ready[0].metadata["executed_queue_actions_since_submit"] == 0
        if controller_cls is RtcQueueController:
            assert ready[0].metadata["merge_dropped_prefix"] == 1
            assert ready[0].metadata["rtc_conditioning_requested"] is True
            assert ready[0].metadata["rtc_conditioning_supported"] is False
            assert ready[0].metadata["rtc_conditioning_used"] is False
            summary = controller.summary()
            assert summary["rtc_conditioning_requested"] is True
            assert summary["rtc_conditioning_supported"] is False
            assert summary["rtc_conditioning_used"] is False
        else:
            assert ready[0].metadata["merge_dropped_prefix"] == 0
    finally:
        controller.close()


@pytest.mark.parametrize("controller_cls", [AsyncQueueController, RtcQueueController])
def test_zero_latency_async_startup_has_no_underflow(controller_cls) -> None:
    controller = controller_cls(
        policy=_Policy(),
        language="test",
        control_freq=20,
        mode="async_queue" if controller_cls is AsyncQueueController else "rtc_queue",
        prefetch_threshold=0,
        action_chunk_size=10,
        chunk_size_threshold=None,
        latency_sampler=LatencySampler(source="fixed", fixed_seconds=0.0),
    )
    actions = ActionRuntime(7, empty_action="zero")
    try:
        ready = controller.tick(_packet(), actions)
        assert len(ready) == 1
        assert ready[0].metadata["delay_steps"] == 0
        assert actions.queue_len > 0
        _action, _original, _source, underflow, _chunk, _before, _after = actions.next_action()
        assert underflow is False
    finally:
        controller.close()


@pytest.mark.parametrize("controller_cls", [AsyncQueueController, RtcQueueController])
def test_fixed_latency_blocks_wallclock_at_due_step_even_with_queued_actions(controller_cls) -> None:
    policy = _BlockingPolicy()
    controller = controller_cls(
        policy=policy,
        language="test",
        control_freq=20,
        mode="async_queue" if controller_cls is AsyncQueueController else "rtc_queue",
        prefetch_threshold=0,
        action_chunk_size=10,
        chunk_size_threshold=None,
        latency_sampler=LatencySampler(source="fixed", fixed_seconds=0.05),
    )
    actions = ActionRuntime(7, empty_action="hold")
    timer = None
    try:
        assert controller.tick(_packet(), actions) == []
        assert policy.started.wait(timeout=1.0)
        actions.enqueue(
            ActionChunkPacket(
                episode_id="test",
                chunk_id=1,
                obs_step_idx=0,
                submit_step_idx=0,
                available_step_idx=0,
                policy_elapsed_s=0.0,
                actions=np.zeros((3, 7), dtype=np.float32),
                controller="seed",
            )
        )
        timer = threading.Timer(0.05, policy.release.set)
        timer.start()
        packet = ObservationPacket(**{**_packet().__dict__, "step_idx": 1, "sim_time_s": 0.05})
        started = time.perf_counter()
        ready = controller.tick(packet, actions)
        blocked = time.perf_counter() - started

        assert len(ready) == 1
        assert blocked >= 0.03
        assert controller.summary()["wallclock_wait_blocked_s"] >= 0.03
    finally:
        policy.release.set()
        if timer is not None:
            timer.cancel()
        controller.close()


def test_rtc_summary_records_true_model_conditioning() -> None:
    controller = RtcQueueController(
        policy=_RtcPolicy(),
        language="test",
        control_freq=20,
        mode="rtc_queue",
        prefetch_threshold=10,
        action_chunk_size=10,
        chunk_size_threshold=None,
        latency_sampler=LatencySampler(source="fixed", fixed_seconds=0.0),
    )
    actions = ActionRuntime(7, empty_action="zero")
    try:
        ready = controller.tick(_packet(), actions)
        assert len(ready) == 1
        assert ready[0].metadata["rtc_conditioning_requested"] is True
        assert ready[0].metadata["rtc_conditioning_supported"] is True
        assert ready[0].metadata["rtc_conditioning_used"] is True
        summary = controller.summary()
        assert summary["rtc_conditioning_requested"] is True
        assert summary["rtc_conditioning_supported"] is True
        assert summary["rtc_conditioning_used"] is True
    finally:
        controller.close()


@pytest.mark.parametrize(
    ("controller_cls", "latency", "prefetch_threshold", "chunk_size_threshold", "expected_underflows", "expected_effective"),
    [
        (AsyncQueueController, 0.1478, 2, 0.5, 3, 5),
        (RtcQueueController, 0.1718, 10, None, 4, 10),
    ],
)
def test_configured_prefetch_threshold_runtime_behavior(
    controller_cls,
    latency: float,
    prefetch_threshold: int,
    chunk_size_threshold: float | None,
    expected_underflows: int,
    expected_effective: int,
) -> None:
    controller = controller_cls(
        policy=_Policy(),
        language="test",
        control_freq=20,
        mode="async_queue" if controller_cls is AsyncQueueController else "rtc_queue",
        prefetch_threshold=prefetch_threshold,
        action_chunk_size=10,
        chunk_size_threshold=chunk_size_threshold,
        latency_sampler=LatencySampler(source="fixed", fixed_seconds=latency),
    )
    actions = ActionRuntime(7, empty_action="hold")
    try:
        for step_idx in range(40):
            packet = _packet()
            packet = ObservationPacket(
                **{**packet.__dict__, "step_idx": step_idx, "sim_time_s": step_idx / 20.0}
            )
            controller.tick(packet, actions)
            actions.next_action()

        assert controller.latency_budget_steps() == int(np.ceil(latency * 20))
        assert controller.effective_prefetch_threshold_steps() == expected_effective
        assert actions.underflow_steps == expected_underflows
    finally:
        controller.close()


def test_official_async_threshold_is_not_inflated_by_latency_budget() -> None:
    controller = AsyncQueueController(
        policy=_Policy(),
        language="test",
        control_freq=20,
        mode="async_queue",
        prefetch_threshold=0,
        action_chunk_size=10,
        chunk_size_threshold=0.1,
        latency_sampler=LatencySampler(source="fixed", fixed_seconds=0.2),
    )
    actions = ActionRuntime(7, empty_action="hold")
    try:
        actions.enqueue(
            ActionChunkPacket(
                episode_id="test",
                chunk_id=1,
                obs_step_idx=0,
                submit_step_idx=0,
                available_step_idx=0,
                policy_elapsed_s=0.0,
                actions=np.zeros((2, 7), dtype=np.float32),
                controller="async_queue",
            )
        )
        assert controller.latency_budget_steps() == 4
        assert controller.effective_prefetch_threshold_steps() == 1
        assert controller.official_threshold_ready(actions) is False
        actions.next_action()
        assert controller.official_threshold_ready(actions) is True
    finally:
        controller.close()


def test_official_async_merge_drops_stale_prefix_and_replaces_future_queue() -> None:
    actions = ActionRuntime(2, empty_action="hold")
    actions.enqueue(
        ActionChunkPacket(
            episode_id="test",
            chunk_id=1,
            obs_step_idx=0,
            submit_step_idx=0,
            available_step_idx=0,
            policy_elapsed_s=0.0,
            actions=np.ones((4, 2), dtype=np.float32),
            controller="async_queue",
        )
    )
    incoming = np.asarray([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]], dtype=np.float32)
    stats = actions.merge_weighted(
        ActionChunkPacket(
            episode_id="test",
            chunk_id=2,
            obs_step_idx=0,
            submit_step_idx=5,
            available_step_idx=7,
            policy_elapsed_s=0.1,
            actions=incoming,
            controller="async_queue",
            metadata={"executed_queue_actions_since_submit": 2},
        )
    )

    assert stats == {"overlap": 3, "appended": 0, "dropped_prefix": 2}
    np.testing.assert_allclose(
        actions.snapshot_actions(),
        np.asarray([[21.3, 21.3], [28.3, 28.3], [35.3, 35.3]], dtype=np.float32),
    )


def test_official_async_underflow_holds_do_not_age_returned_actions() -> None:
    actions = ActionRuntime(2, empty_action="hold")
    for _ in range(3):
        _action, _original, _source, underflow, _chunk, _before, _after = actions.next_action()
        assert underflow is True

    stats = actions.merge_weighted(
        ActionChunkPacket(
            episode_id="test",
            chunk_id=1,
            obs_step_idx=0,
            submit_step_idx=0,
            available_step_idx=3,
            policy_elapsed_s=0.15,
            actions=np.asarray([[1, 1], [2, 2], [3, 3]], dtype=np.float32),
            controller="async_queue",
            metadata={"executed_queue_actions_since_submit": 0},
        )
    )

    assert actions.executed_actions == 3
    assert actions.executed_queue_actions == 0
    assert stats["dropped_prefix"] == 0
    np.testing.assert_allclose(
        actions.snapshot_actions(),
        np.asarray([[1, 1], [2, 2], [3, 3]], dtype=np.float32),
    )


def test_official_async_only_executed_queue_actions_age_returned_chunk() -> None:
    actions = ActionRuntime(2, empty_action="hold")
    actions.enqueue(
        ActionChunkPacket(
            episode_id="test",
            chunk_id=1,
            obs_step_idx=0,
            submit_step_idx=0,
            available_step_idx=0,
            policy_elapsed_s=0.0,
            actions=np.ones((2, 2), dtype=np.float32),
            controller="async_queue",
        )
    )
    actions.next_action()
    actions.next_action()
    actions.next_action()

    stats = actions.merge_weighted(
        ActionChunkPacket(
            episode_id="test",
            chunk_id=2,
            obs_step_idx=0,
            submit_step_idx=0,
            available_step_idx=3,
            policy_elapsed_s=0.15,
            actions=np.asarray([[10, 10], [20, 20], [30, 30]], dtype=np.float32),
            controller="async_queue",
            metadata={"executed_queue_actions_since_submit": 2},
        )
    )

    assert actions.executed_actions == 3
    assert actions.executed_queue_actions == 2
    assert stats == {"overlap": 0, "appended": 1, "dropped_prefix": 2}
    np.testing.assert_allclose(actions.snapshot_actions(), np.asarray([[30, 30]], dtype=np.float32))


class _FakeSim:
    def __init__(self) -> None:
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.step_durations: list[float] = []

    def step(self) -> None:
        self.step_durations.append(float(self.model.opt.timestep))

    def forward(self) -> None:
        pass


@pytest.mark.parametrize(
    ("duration", "expected_steps"),
    [(0.0, 0), (0.03, 15), (0.05, 25), (0.075, 38)],
)
def test_continuous_latency_uses_physics_substeps(duration: float, expected_steps: int) -> None:
    env = SimpleNamespace(sim=_FakeSim())
    advanced = advance_physical_dynamic_time(env, [], elapsed=1.0, duration=duration)

    assert advanced == pytest.approx(duration, abs=1e-12)
    assert len(env.sim.step_durations) == expected_steps
    assert sum(env.sim.step_durations) == pytest.approx(duration, abs=1e-12)
    assert env.sim.model.opt.timestep == pytest.approx(0.002)


def test_default_video_is_time_faithful_compact_dual_view() -> None:
    args = build_parser().parse_args(["--output-dir", "/tmp/realtime-video-default-test"])

    assert args.save_videos is True
    assert args.save_review_video is False
    assert args.save_model_input_videos is True
    assert args.save_wrist_input_videos is False
    assert args.model_input_video_layout == "dual"
    assert args.model_input_video_size == 128
    assert args.video_fps == pytest.approx(60.0)
    assert args.record_latency_frames is True


def test_video_fps_is_runtime_configurable() -> None:
    args = build_parser().parse_args(
        ["--output-dir", "/tmp/realtime-video-override-test", "--video-fps", "100"]
    )

    assert args.video_fps == pytest.approx(100.0)
    assert args.control_freq == 20


def test_half_episode_audit_can_disable_early_success_stop() -> None:
    args = build_parser().parse_args(
        ["--output-dir", "/tmp/realtime-half-audit-test", "--max-steps", "300", "--no-stop-on-success"]
    )

    assert args.max_steps == 300
    assert args.stop_on_success is False


def test_stratified_item_file_and_worker_count_are_configurable() -> None:
    args = build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/realtime-stratified-test",
            "--item-ids-file",
            "/tmp/items.json",
            "--num-workers",
            "4",
        ]
    )

    assert args.item_ids_file == Path("/tmp/items.json")
    assert args.num_workers == 4


def test_summary_uses_main_repository_action_metric_definitions(tmp_path: Path) -> None:
    result = {
        "item_id": "item-0",
        "status": "ok",
        "success": True,
        "diagnostics": {
            "ok": True,
            "flags": [],
            "targets": {
                "bowl": {
                    "max_platform_path_error_m": 0.001,
                    "max_follow_error_m": 0.002,
                    "max_tilt_deg": 1.0,
                    "min_relative_bottom_z_m": -0.001,
                    "birdview_offscreen_ratio": 0.0,
                    "agentview_offscreen_ratio": 0.0,
                }
            },
        },
        "control_mode": "sync_hold",
        "policy": "zero",
        "latency_source": "fixed",
        "sim_latency_decoupled_from_wallclock": True,
        "shared_policy_latency_contaminated": False,
        "action_runtime": {
            "queue_underflow_steps": 1,
            "enqueued_chunks": 2,
            "executed_actions": 10,
        },
        "action_metrics": {
            "action_l2": {"count": 10, "mean": 0.2, "p95": 0.4, "max": 0.5},
            "action_delta_l2": {"count": 10, "mean": 0.03, "p95": 0.08, "max": 0.1},
        },
        "controller_summary": {
            "policy_requests": 2,
            "policy_latency_max_s": 0.01,
            "simulated_policy_latency_max_s": 0.06,
            "rtc_conditioning_requested": False,
            "rtc_conditioning_supported": False,
            "rtc_conditioning_used": False,
        },
        "sync_inference_latency_elapsed_s": 0.12,
    }

    write_summary(tmp_path, [result])
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["metric_schema_version"] == "libero_dynamic_main_v1"
    assert summary["success_count"] == 1
    assert summary["runtime_metrics"]["queue_underflow_rate"] == pytest.approx(0.1)
    assert summary["action_metrics"]["action_l2"]["definition"] == "||a_t||_2"
    assert summary["action_metrics"]["action_delta_l2"]["definition"] == "||a_t - a_(t-1)||_2"
    assert summary["action_metrics"]["action_delta_l2"]["max"] == pytest.approx(0.1)
    assert summary["physical_diagnostics"]["max_follow_error_m"] == pytest.approx(0.002)
