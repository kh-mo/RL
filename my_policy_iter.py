
class environment:
    def __init__(self):
        self.width = 5
        self.height = 5
        self.state = [(i,j) for i in range(self.height) for j in range(self.width)]
        self.reward = [[0.]*self.width for _ in range(self.height)]
        self.reward[2][2] = 1.
        self.reward[1][2] = -1.
        self.reward[2][1] = -1.
        self.discount_factor = 0.9
        self.move_step = [(-1,0),(1,0),(0,-1),(0,1)] #상,하,좌,우

    def _step(self, state, action):
        # 동작 수행
        # 다음 상태와 리워드 반환
        next_state = self.state_after_action(state, action)
        immediate_reward = self.reward[next_state[0]][next_state[1]]
        return next_state, immediate_reward

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

class policy_iter_agent:
    def __init__(self, env):
        self.env = env
        self.possible_actions = [0, 1, 2, 3] # 상, 하, 좌, 우
        self.value_table = [[0.] * self.env.width for _ in range(self.env.height)]
        self.policy_table = [[[0.25,0.25,0.25,0.25]] * self.env.width for _ in range(self.env.height)]
        self.policy_table[2][2] = []

    def evaluate(self):
        # k+1 가치함수 테이블 선언
        new_value_table = [[0.] * self.env.width for _ in range(self.env.height)]

        # k 가치함수 테이블에서 k+1 가치함수 테이블 값 구하기
        for state in self.env.state:
            if state == (2,2):
                ## Q. 해당 터미널 스테이트의 가치는 높을수록 좋은 것이 아닌가?
                new_value_table[2][2] = 0.0
                continue
            # 벨만 기대 방정식
            for action in self.possible_actions:
                next_state, reward = self.env._step(state, action)
                new_value = self.policy_table[state[0]][state[1]][action] * \
                            (reward + self.env.discount_factor * self.value_table[next_state[0]][next_state[1]])
                new_value_table[state[0]][state[1]] += round(new_value, 2)
        self.value_table = new_value_table

    def improve(self):
        new_policy_table = self.policy_table
        for state in self.env.state:
            value = -99999
            max_index = []
            result = [0., 0., 0., 0.]
            if state == (2,2):
                continue
            for index, action in enumerate(self.possible_actions):
                next_state, reward = self.env._step(state, action)
                tmp = reward + self.env.discount_factor * self.value_table[next_state[0]][next_state[1]]

                if tmp == value:
                    max_index.append(index)
                elif tmp > value:
                    value = tmp
                    max_index.clear()
                    max_index.append(index)

            prob = 1 / len(max_index)
            for index in max_index:
                result[index] = prob

            new_policy_table[state[0]][state[1]] = result
        self.policy_table = new_policy_table

if __name__ == "__main__":
    env = environment()
    agent = policy_iter_agent(env)

    agent.value_table
    agent.policy_table

    for i in range(10):
        agent.evaluate()
        agent.improve()
