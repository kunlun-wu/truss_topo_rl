import numpy as np
import os
import json
import collections
from anastruct import SystemElements

PhysicsResult = collections.namedtuple(
    'PhysicsResult', ['max_displacement', 'weight', 'valid']
)

class TrussPhysics:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(
                base_dir, 'configs', 'default_config.json'
            )
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.grid_width = self.config['grid_width']
        self.grid_height = self.config['grid_height']
        self.nodes = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                self.nodes.append([x, y])
        self.all_possible_bars = []
        max_dist = self.config['max_connection_dist']
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                node1 = np.array(self.nodes[i])
                node2 = np.array(self.nodes[j])
                dist = np.linalg.norm(node1 - node2)
                if dist <= max_dist:
                    self.all_possible_bars.append((i, j))
        self.support_indices = []
        fixed_locs = self.config.get('fixed_supports', [])
        for supp_loc in fixed_locs:
            for i, n in enumerate(self.nodes):
                if n == supp_loc:
                    self.support_indices.append(i)
        rolling_locs = [item['location'] for item in
                        self.config.get('rolling_supports', [])]
        for supp_loc in rolling_locs:
            for i, n in enumerate(self.nodes):
                if n == supp_loc:
                    self.support_indices.append(i)
        print(
            f"[Physics] Initialized: {len(self.nodes)} nodes, {len(self.all_possible_bars)} bars."
        )

    def _get_cleaned_structure(self, active_indices):
        """
        Applies two passes of cleaning to the structure:
        1. BFS Connectivity: Removes 'floating islands' not attached to supports.
        2. Iterative Pruning: Removes 'dangling hairs' (nodes with < 2 connections).
        """
        current_indices = active_indices.copy()
        adj = collections.defaultdict(list)
        for bar_idx, is_active in enumerate(current_indices):
            if is_active == 1:
                n1, n2 = self.all_possible_bars[bar_idx]
                adj[n1].append(n2)
                adj[n2].append(n1)
        reachable = set()
        queue = collections.deque(self.support_indices)
        for idx in self.support_indices:
            reachable.add(idx)
        while queue:
            curr = queue.popleft()
            for neighbor in adj[curr]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        for bar_idx, is_active in enumerate(current_indices):
            if is_active == 1:
                n1, n2 = self.all_possible_bars[bar_idx]
                if (n1 not in reachable) or (n2 not in reachable):
                    current_indices[bar_idx] = 0
        while True:
            node_counts = collections.defaultdict(int)
            for bar_idx, is_active in enumerate(current_indices):
                if is_active == 1:
                    n1, n2 = self.all_possible_bars[bar_idx]
                    node_counts[n1] += 1
                    node_counts[n2] += 1
            pruned_something = False
            for bar_idx, is_active in enumerate(current_indices):
                if is_active == 1:
                    n1, n2 = self.all_possible_bars[bar_idx]
                    n1_unstable = (node_counts[n1] < 2) and (
                            n1 not in self.support_indices)
                    n2_unstable = (node_counts[n2] < 2) and (
                            n2 not in self.support_indices)
                    if n1_unstable or n2_unstable:
                        current_indices[bar_idx] = 0
                        pruned_something = True
            if not pruned_something:
                break
        return current_indices

    def solve(self, active_bars_indices):
        if np.sum(active_bars_indices) == 0:
            return PhysicsResult(999.0, 0, False)
        effective_indices = self._get_cleaned_structure(active_bars_indices)
        weight = np.sum(effective_indices)
        if weight == 0:
            return PhysicsResult(999.0, 0, False)
        def run_simulation(fx, fy):
            ss = SystemElements()
            for i, exists in enumerate(effective_indices):
                if exists == 1:
                    n1, n2 = self.all_possible_bars[i]
                    ss.add_element(
                        location=[self.nodes[n1], self.nodes[n2]],
                        spring={1: 0, 2: 0}
                    )

            fixed_locs = self.config.get('fixed_supports', [])
            for supp_loc in fixed_locs:
                nid = ss.find_node_id(vertex=supp_loc)
                if nid:
                    ss.add_support_hinged(node_id=nid)

            rolling_data = self.config.get('rolling_supports', [])
            for item in rolling_data:
                loc = item['location']
                direction = item[
                    'direction']  # 2 = rolls X, 1 = rolls Y
                nid = ss.find_node_id(vertex=loc)
                if nid:
                    ss.add_support_roll(node_id=nid, direction=direction)

            load_loc = self.config['load_location']
            nid = ss.find_node_id(vertex=load_loc)
            if nid: ss.point_load(node_id=nid, Fx=fx, Fy=fy)

            # solve
            try:
                ss.solve(force_linear=True, verbose=False)
                displacements = ss.get_node_result_range('uy')
                displacements_x = ss.get_node_result_range('ux')
                if not displacements: return 999.0
                max_d = 0
                for d_y, d_x in zip(displacements, displacements_x):
                    total_disp = np.sqrt(d_y ** 2 + d_x ** 2)
                    if total_disp > max_d:
                        max_d = total_disp
                return max_d
            except:
                return 999.0

        # load case 1: gravity
        disp_gravity = run_simulation(fx=0.0, fy=self.config['load_force_y'])

        # load case 2: wind
        wind_force = abs(self.config['load_force_y']) * 0.20
        disp_wind = run_simulation(fx=wind_force, fy=0.0)

        worst_disp = max(disp_gravity, disp_wind)
        if worst_disp > 100.0:
            return PhysicsResult(100.0, weight, False)
        return PhysicsResult(worst_disp, weight, True)