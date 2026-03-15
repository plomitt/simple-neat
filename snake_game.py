import random

class SnakeGame:
    def __init__(self, grid_size=15, seed=0):
        self.size = grid_size
        self.rng = random.Random(seed) # Seeded to guarantee fairness per generation
        
        self.snake =[(grid_size // 2, grid_size // 2)]
        self.direction = (0, -1) # Start facing UP
        
        self.score = 0
        self.steps_since_eat = 0
        self.dead = False
        self.apple = self._spawn_apple()

    def _spawn_apple(self):
        empty_cells =[(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in self.snake]
        return self.rng.choice(empty_cells) if empty_cells else None

    def get_vision(self):
        """Calculates 6 Egocentric Binary Inputs (Forward, Left, Right)"""
        if self.dead: return [0]*6
        
        dx, dy = self.direction
        left_dir = (dy, -dx)
        right_dir = (-dy, dx)
        head_x, head_y = self.snake[0]

        def is_clear(d_x, d_y):
            nx, ny = head_x + d_x, head_y + d_y
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size: return 0.0 # Wall
            if (nx, ny) in self.snake: return 0.0 # Self
            return 1.0 # Clear

        # 1. Obstacle Detection
        clear_f = is_clear(dx, dy)
        clear_l = is_clear(*left_dir)
        clear_r = is_clear(*right_dir)

        # 2. Apple Detection (Using Vector Dot Products)
        apple_f = apple_l = apple_r = 0.0
        if self.apple:
            vx, vy = self.apple[0] - head_x, self.apple[1] - head_y
            if vx * dx + vy * dy > 0: apple_f = 1.0
            if vx * left_dir[0] + vy * left_dir[1] > 0: apple_l = 1.0
            if vx * right_dir[0] + vy * right_dir[1] > 0: apple_r = 1.0

        return[clear_f, clear_l, clear_r, apple_f, apple_l, apple_r]

    def step(self, turn_left, turn_right):
        if self.dead: return

        self.steps_since_eat += 1
        
        # Starvation mechanic prevents infinite looping cheats
        if self.steps_since_eat > 100:
            self.dead = True
            return

        # Execute Actions
        dx, dy = self.direction
        if turn_left > 0.5 and turn_right <= 0.5:
            self.direction = (dy, -dx) # Rotate Left
        elif turn_right > 0.5 and turn_left <= 0.5:
            self.direction = (-dy, dx) # Rotate Right

        dx, dy = self.direction
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        # Collision Check
        if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or (nx, ny) in self.snake:
            self.dead = True
            return

        # Move forward
        self.snake.insert(0, (nx, ny))

        # Check Apple
        if (nx, ny) == self.apple:
            self.score += 1
            self.steps_since_eat = 0
            self.apple = self._spawn_apple()
            if not self.apple: # Victory (Filled the board)
                self.dead = True 
        else:
            self.snake.pop() # Remove tail if no apple eaten