class Wrapper:
    def __init__(self, env):
        self._env = env

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, action):
        return self.step(action)

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def seed(self, seed):
        self._env.seed(seed)

    def __getattr__(self, attr):
        return getattr(self._env, attr)
