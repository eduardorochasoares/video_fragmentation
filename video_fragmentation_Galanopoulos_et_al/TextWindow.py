class TextWindow:
    def __init__(self, cues, init_time, id):
        self.cues = cues  # cue vector
        self.init_time = init_time # the initial time of the time window
        self.depth = 0
        self.id = id