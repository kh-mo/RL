import numpy as np
from collections import defaultdict

class environment:
    def __init__(self):
        self.width = 5
        self.height = 5
        self.state = [(i,j) for i in range(self.height) for j in range(self.width)]
        self.reward = [[0.]*self.width for _ in range(self.height)]
        self.reward[2][2] = 100.
        self.reward[1][2] = -100.
        self.reward[2][1] = -100.
        self.discount_factor = 0.9
        self.move_step = [(-1,0),(1,0),(0,-1),(0,1)] #상,하,좌,우

    def _step(self, state, action):
        # 동작 수행
        # 다음 상태와 리워드 반환
        next_state = self.state_after_action(state, action)
        immediate_reward = self.reward[next_state[0]][next_state[1]]
        done = False
        if immediate_reward == 100 or immediate_reward == -100:
            done = True
        return next_state, immediate_reward, done

    def _reset(self):
        # 환경 초기화
        return NotImplementedError

    def _render(self):
        # 화면으로 보여주기
        return NotImplementedError

    def state_after_action(self, state, action):
        step = self.move_step[action]
        return self.check_boundary([state[0] + step[0], state[1] + step[1]])

    def check_boundary(self, state):
        state[0] = (0 if state[0] < 0 else self.width - 1
                    if state[0] > self.width - 1 else state[0])
        state[1] = (0 if state[1] < 0 else self.height - 1
                    if state[1] > self.height - 1 else state[1])
        return state

class MC_agent:
    def __init__(self):
        self.width = 5
        self.height = 5
        self.actions = [0,1,2,3] # 상, 하, 좌, 우
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.samples = []
        self.value_table = defaultdict(float)

    def save_sample(self, next_state, reward, done):
        self.samples.append([next_state, reward, done])

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            # 랜덤 행동
            action = np.random.choice(self.actions)
        else:
            # 큐 함수에 따른 행동
            next_state = self.possible_next_state(state)
            action = self.arg_max(next_state)
        return int(action)

    def possible_next_state(self, state):
        next_state = [0.0] * 4

        for action in self.actions:
            if action == 0:
                move_step = [-1,0]
                next_state[0] = self.value_table[str(self.check_boundary([state[0] + move_step[0], state[1] + move_step[1]]))]
            elif action == 1:
                move_step = [1, 0]
                next_state[1] = self.value_table[
                    str(self.check_boundary([state[0] + move_step[0], state[1] + move_step[1]]))]
            elif action == 2:
                move_step = [0, -1]
                next_state[2] = self.value_table[
                    str(self.check_boundary([state[0] + move_step[0], state[1] + move_step[1]]))]
            else:
                move_step = [0, 1]
                next_state[3] = self.value_table[
                    str(self.check_boundary([state[0] + move_step[0], state[1] + move_step[1]]))]
        return next_state

    def arg_max(self, state):
        max_index_list = []
        max_value = state[0]
        for index, value in enumerate(state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return np.random.choice(max_index_list)

    def check_boundary(self, state):
        state[0] = (0 if state[0] < 0 else self.width - 1
                    if state[0] > self.width - 1 else state[0])
        state[1] = (0 if state[1] < 0 else self.height - 1
                    if state[1] > self.height - 1 else state[1])
        return state

    def update(self):
        G_t = 0
        visit_state = []
        for reward in reversed(self.samples):
            state = str(reward[0])
            if state not in visit_state:
                visit_state.append(state)
                G_t = reward[1] + self.discount_factor * G_t
                self.value_table[state] += self.learning_rate * (G_t - self.value_table[state])

if __name__ == "__main__":
    env = environment()
    agent = MC_agent()

    for episode in range(1000):
        print(episode)
        # start state 정의
        # state = env.reset()
        state = (0, 0)

        while True:
            # env.render()
            # 행동 후 환경에서 현재 정책을 따라 다음 행동 수행
            action = agent.get_action(state)
            next_state, reward, done = env._step(state, action)
            # next_state, reward, done = env.step(action)
            # 다음 상태와 보상을 지속적으로 저장
            agent.save_sample(next_state, reward, done)
            state = next_state
            # print(done)
            # 터미널 스테이트에 도달했을 때
            if done:
                # 에이전트 상태 업데이트
                agent.update()
                agent.samples.clear()
                break
