"""Global configuration constants for F1 Mars Simulator."""

# Display settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60

# Physics settings
PHYSICS_STEPS_PER_FRAME = 4
PHYSICS_DT = 1.0 / (FPS * PHYSICS_STEPS_PER_FRAME)

# Car physics constants
CAR_MAX_SPEED = 300  # units/second
CAR_MAX_STEERING = 0.6  # radians
CAR_ACCELERATION = 150  # units/second^2
CAR_BRAKE_FORCE = 250  # units/second^2
CAR_DRAG_COEFFICIENT = 0.98
CAR_MASS = 800  # kg
CAR_LENGTH = 30  # pixels
CAR_WIDTH = 15  # pixels

# Tire degradation settings
TIRE_DEGRADATION_RATE = 0.0001  # per frame
TIRE_COMPOUNDS = {
    "soft": {"grip": 1.0, "degradation": 1.5},
    "medium": {"grip": 0.9, "degradation": 1.0},
    "hard": {"grip": 0.8, "degradation": 0.7},
}

# Track settings
TRACK_WIDTH = 80  # pixels
CHECKPOINT_TOLERANCE = 30  # pixels

# Rendering colors (RGB)
COLOR_TRACK = (50, 50, 50)
COLOR_GRASS = (34, 139, 34)
COLOR_CAR_PLAYER = (255, 0, 0)
COLOR_CAR_AI = (0, 0, 255)
COLOR_CHECKPOINT = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_HUD_BG = (30, 30, 30, 200)

# Training settings
MAX_EPISODE_STEPS = 10000
REWARD_CHECKPOINT = 100
REWARD_LAP_COMPLETE = 1000
REWARD_COLLISION = -100
REWARD_REVERSE = -10
REWARD_SPEED_BONUS = 0.1

# Agent observation space
LIDAR_RAYS = 16  # Number of distance sensors
LIDAR_RANGE = 200  # Maximum distance in pixels

# Paths
TRACKS_DIR = "tracks"
MODELS_DIR = "trained_models"
LOGS_DIR = "logs"
