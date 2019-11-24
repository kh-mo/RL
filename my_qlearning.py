import numpy as np
from collections import defaultdict

class environment:
    def __init__(self):
        self.width = 5
        self.height = 5

    def step(self, state, action):
        next_state = None
        if action == 0:  # 상
            next_state = self.check_boundary([state[0]-1, state[1]])
        elif action == 1:  # 하
            next_state = self.check_boundary([state[0]+1, state[1]])
        elif action == 2:  # 좌
            next_state = self.check_boundary([state[0], state[1]-1])
        elif action == 3:  # 우
            next_state = self.check_boundary([state[0], state[1]+1])

        # 보상 함수
        if next_state == (2,2):
            reward = 100
            done = True
        elif next_state in [(2,1),(1,2)]:
            reward = -100
            done = True
        else:
            reward = 0
            done = False
        return next_state, reward, done

    def check_boundary(self, state):
        state[0] = (0 if state[0] < 0 else self.width - 1
                    if state[0] > self.width - 1 else state[0])
        state[1] = (0 if state[1] < 0 else self.height - 1
                    if state[1] > self.height - 1 else state[1])
        return state

class q_learning_agent:
    def __init__(self):
        self.actions = [0,1,2,3] #상하좌우
        self.epsilon = 0.1
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.q_table = defaultdict(lambda : [0., 0., 0., 0.])

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table[state]
            action = self.argmax(state_action)
        return action

    def argmax(self, state):
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

    def learn(self, state, action, reward, n_state):
        current_q = self.q_table[state][action]
        n_state_q = max(self.q_table[n_state])
        current_q += self.learning_rate * (reward + self.discount_factor * n_state_q - current_q)

if __name__ == "__main__":
    env = environment()
    agent = q_learning_agent()

    for episode in range(200):
        print(episode)
        state = (0,0)
        # 현재 상태에 대한 행동을 선택
        action = agent.get_action(str(state))

        while True:
            # 행동을 위한 후 다음상태 보상 에피소드의 종료 여부를 받아옴
            next_state, reward, done = env.step(state, action)
            # 다음 상태에서의 다음 행동 선택

            # <s,a,r,s'>로 큐함수를 업데이트
            agent.learn(str(state), action, reward, str(next_state))

            state = next_state

            if done:
                break
