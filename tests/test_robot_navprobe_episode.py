"""Exercise the public robot runner with real RPC decoding and RGB-D mapping."""
from copy import deepcopy
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

from navprobe.agent import vln_runner
from navprobe.config.vln_runtime import VlnRuntimeConfig
from navprobe.env.robot_rpc_env import RobotRPCEnv
from navprobe.env.rpc_protocol import encode_array, encode_depth_meters, encode_rgb_image
from navprobe.llm.client import LLMClient


class RobotService:
    def __init__(self):
        self.pose = dict(x=0.0, y=0.0, z=0.0, yaw=0.0)
        self.calls = []
        self.turn_yaws = []

    def observation(self):
        yaw = math.radians(self.pose['yaw'])
        c, s = math.cos(yaw), math.sin(yaw)
        base = np.eye(4, dtype=np.float32)
        base[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        base[:3, 3] = [self.pose[k] for k in ('x', 'y', 'z')]
        camera = base.copy()
        camera[2, 3] += 1.0
        optical = np.array([[0, 0, 1, 0], [-1, 0, 0, 0],
                            [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        rows = np.arange(64, dtype=np.float32)[:, None]
        depth = np.broadcast_to(np.minimum(5.0, 45.0 / np.maximum(rows - 32, 0.1)), (64, 64))
        return dict(
            pose=dict(self.pose), rgb=encode_rgb_image(np.zeros((64, 64, 3), dtype=np.uint8)),
            depth=encode_depth_meters(depth),
            intrinsics=encode_array(np.array([[45, 0, 32], [0, 45, 32], [0, 0, 1]], dtype=np.float32)),
            T_cam_odom=encode_array(np.linalg.inv(camera @ optical)), T_odom_base=encode_array(base),
        )

    def request(self, url, json=None, timeout=None):
        endpoint = '/' + url.rsplit('/', 1)[-1]
        self.calls.append((endpoint, deepcopy(json)))
        if endpoint == '/health':
            payload = {'ok': True, 'supports_path_move': False}
        elif endpoint in ('/reset', '/get_obs'):
            payload = self.observation()
        elif endpoint == '/current_episode':
            payload = {'action_step_total': len(self.turn_yaws) + sum(c[0] == '/move' for c in self.calls)}
        elif endpoint == '/turn':
            self.pose['yaw'] = (self.pose['yaw'] + (90 if json['direction'] == 'left' else -90)) % 360
            self.turn_yaws.append(self.pose['yaw'])
            payload = {'pose': dict(self.pose)}
        elif endpoint == '/move':
            self.pose.update({k: json[k] for k in self.pose if k in json})
            payload = {'pose': dict(self.pose), 'intermediate_observations': [self.observation()]}
        elif endpoint == '/stop':
            payload = {'pose': dict(self.pose)}
        elif endpoint == '/finalize_run':
            payload = {'reason': json['reason']}
        else:
            raise AssertionError(endpoint)
        return SimpleNamespace(ok=True, json=lambda: payload)


@pytest.fixture
def robot(monkeypatch):
    service = RobotService()
    monkeypatch.setattr('navprobe.env.robot_rpc_env.requests.get', service.request)
    monkeypatch.setattr('navprobe.env.robot_rpc_env.requests.post', service.request)
    monkeypatch.setattr(vln_runner, 'YOLOWorldLocalDetector', lambda **kwargs: SimpleNamespace())
    original = VlnRuntimeConfig.global_bev_kwargs
    monkeypatch.setattr(VlnRuntimeConfig, 'global_bev_kwargs', lambda self: {**original(self), 'size': 240})
    states = []
    build_state = vln_runner._build_state

    def capture_state(**kwargs):
        state = build_state(**kwargs)
        states.append(state)
        return state

    monkeypatch.setattr(vln_runner, '_build_state', capture_state)
    return service, states


def scripted_model(monkeypatch, *, move=False, fail=False):
    calls = []
    executive_round = 0
    navigation_round = 0

    def completion(self, **kwargs):
        nonlocal executive_round, navigation_round
        name = kwargs['call_name']
        calls.append(kwargs)
        if name == 'visual_task_progress_generator':
            return {'agenda': [{'content': 'Stop at the endpoint.', 'status': 'active', 'result': ''}]}
        if name == 'vln_landmark_detection_list':
            return {'landmark_categories': []}
        if name == 'visual_node_summary':
            return {'node_summary': 'An open indoor area.', 'direction_summaries': {
                direction: 'Open floor.' for direction in ('front', 'back', 'left', 'right')}}
        if name == 'vln_task_progress_updater':
            if fail:
                raise RuntimeError('model unavailable')
            executive_round += 1
            response = {'task_state_assessment': 'Check the original endpoint.', 'tool_calls': []}
            if executive_round == (3 if move else 2):
                response['tool_calls'] = [{'name': 'update_task_state', 'arguments': {
                    'agenda_updates': [{'op': 'complete', 'subgoal_id': 'sg0', 'result': 'At the endpoint.'}],
                    'terminal_check': {'decision': 'done', 'missing_constraints': []},
                }}]
            return response
        if name == 'vln_progress_conditioned_navigation_planner':
            navigation_round += 1
            if move and navigation_round == 1:
                return {'go_to_waypoint': {'subgoal_id': 'sg0', 'direction': 'front', 'reason': 'Reach open floor ahead.'}}
            return {'approach_to_stop': {'subgoal_id': 'sg0', 'movement': 'stay',
                'stop_objective': 'Stop at the endpoint.', 'reason': 'The current pose is the endpoint.'}}
        if name == 'vln_waypoint_planner':
            text = '\n'.join(block.get('text', '') for block in kwargs['user_prompt'])
            match = re.search(r'Available waypoint labels: (\d+)', text)
            assert match, text
            return {'reasoning': 'Use the first reachable floor candidate.',
                    'selected_direction': 'front', 'candidate_label': int(match[1]), 'failure_reason': ''}
        raise AssertionError(name)

    monkeypatch.setattr(LLMClient, '_create_visual_json_completion', completion)
    return calls


@pytest.mark.parametrize('move', [False, True])
def test_public_runner_reaches_stop_through_navprobe_and_robot_rpc(robot, monkeypatch, move):
    service, states = robot
    calls = scripted_model(monkeypatch, move=move)
    result = vln_runner.run_vln_episode(env=RobotRPCEnv('http://robot'), instruction='Stop at the endpoint.',
        model='robot-model', api_key='test', config=VlnRuntimeConfig(max_place_steps=4))
    assert result['declared_done'], result
    assert result['place_steps'] == (3 if move else 2)
    assert service.turn_yaws[:4] == [90, 180, 270, 0]
    assert sum(path == '/move' for path, _ in service.calls) == int(move)
    assert ('/finalize_run', {'reason': 'done'}) in service.calls
    memory = states[0].system.memory.task_progress
    assert memory.items == []
    assert memory.history[0]['subgoal_id'] == 'sg0'
    if move:
        edges = list(states[0].graph.iter_edges())
        assert edges and edges[0].rgb_history_obs_ids
        assert memory.history[0]['start_node_id']
    names = [c['call_name'] for c in calls]
    assert names.count('vln_task_progress_updater') == result['place_steps']
    assert names.count('vln_progress_conditioned_navigation_planner') == result['place_steps'] - 1


def test_public_runner_stops_robot_when_executive_fails(robot, monkeypatch):
    service, _ = robot
    scripted_model(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match='model unavailable'):
        vln_runner.run_vln_episode(env=RobotRPCEnv('http://robot'), instruction='Stop at the endpoint.',
            model='robot-model', api_key='test')
    assert service.calls[-2:] == [('/stop', {}), ('/finalize_run', {'reason': 'exception'})]


def test_stop_proposal_without_next_step_verification_is_not_done(robot, monkeypatch):
    service, _ = robot
    scripted_model(monkeypatch)
    result = vln_runner.run_vln_episode(env=RobotRPCEnv('http://robot'), instruction='Stop at the endpoint.',
        model='robot-model', api_key='test', config=VlnRuntimeConfig(max_place_steps=1))
    assert result['termination_reason'] == 'max_place_steps'
    assert result['declared_done'] is False
    assert ('/stop', {}) in service.calls
    assert ('/finalize_run', {'reason': 'max_place_steps'}) in service.calls
