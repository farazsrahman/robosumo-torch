"""
Modern MuJoCo environment base class.

This version uses only modern mujoco and gymnasium APIs, without any
legacy mujoco_py or old gym compatibility code.
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class MjStateWrapper:
    """Wrapper for MuJoCo state that mimics mujoco_py's get_state() return value."""
    
    def __init__(self, data):
        """
        Create state wrapper.
        
        Args:
            data: MjData instance
        """
        self.qpos = data.qpos.copy()
        self.qvel = data.qvel.copy()


class MjSimWrapper:
    """Wraps modern mujoco to match mujoco_py MjSim API."""
    
    def __init__(self, model):
        """
        Create simulation wrapper.
        
        Args:
            model: MjModel instance or MjModelWrapper
        """
        self.model = model
        # If model is wrapped, unwrap it for MjData
        if isinstance(model, MjModelWrapper):
            self.data = mujoco.MjData(model._model)
        else:
            self.data = mujoco.MjData(model)
    
    def _get_model(self):
        """Get unwrapped model for mujoco functions."""
        if isinstance(self.model, MjModelWrapper):
            return self.model._model
        return self.model
    
    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self._get_model(), self.data)
    
    def forward(self):
        """Compute forward dynamics (kinematics and derived quantities)."""
        mujoco.mj_forward(self._get_model(), self.data)
    
    def step(self):
        """Advance simulation by one timestep."""
        mujoco.mj_step(self._get_model(), self.data)
    
    def get_state(self):
        """
        Get current simulation state.
        
        Returns:
            MjStateWrapper: Object with qpos and qvel arrays that can be modified
        """
        return MjStateWrapper(self.data)
    
    def set_state(self, state):
        """
        Set simulation state.
        
        Args:
            state: MjStateWrapper or object with qpos and qvel attributes
        """
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        # Run forward kinematics after setting state
        mujoco.mj_forward(self._get_model(), self.data)
    
    def render(self, width, height, camera_name=None, depth=False):
        """
        Render scene to numpy array (offscreen).
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            camera_name: Name of camera to render from (optional)
            depth: Whether to return depth instead of RGB (not yet implemented)
        
        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3)
        """
        # Get unwrapped model for renderer
        model = self._get_model()
        
        # Create or recreate renderer if size changed
        if not hasattr(self, '_renderer') or self._renderer_size != (width, height):
            self._renderer = mujoco.Renderer(model, height=height, width=width)
            self._renderer_size = (width, height)
        
        # Update scene with current state
        # Note: camera_name can be None, which means use default camera (-1)
        if camera_name is not None:
            self._renderer.update_scene(self.data, camera=camera_name)
        else:
            self._renderer.update_scene(self.data)
        
        # Render
        img = self._renderer.render()
        
        # Modern mujoco renders correctly (not upside-down like mujoco_py)
        return img


class MjModelWrapper:
    """Wrapper for MjModel to add mujoco_py compatibility attributes."""
    
    def __init__(self, model):
        self._model = model
        
    def __getattr__(self, name):
        """Forward all attributes to the wrapped model."""
        return getattr(self._model, name)
    
    @property
    def body_names(self):
        """Get list of body names (mujoco_py compatibility)."""
        # In modern mujoco, use mujoco.mj_id2name
        names = []
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            names.append(name if name else '')
        return names
    
    @property
    def geom_names(self):
        """Get list of geom names (mujoco_py compatibility)."""
        names = []
        for i in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, i)
            names.append(name if name else '')
        return names
    
    @property
    def joint_names(self):
        """Get list of joint names (mujoco_py compatibility)."""
        names = []
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            names.append(name if name else '')
        return names


def load_model_from_path(xml_path):
    """
    Load MuJoCo model from XML path (mujoco_py compatible).
    
    Args:
        xml_path: Path to XML model file
    
    Returns:
        MjModelWrapper: Wrapped model with compatibility attributes
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    return MjModelWrapper(model)


def _read_pixels(sim, width=None, height=None, camera_name=None):
    """Reads pixels w/o markers and overlay from the same camera as screen."""
    if width is None or height is None:
        width, height = 1600, 1280  # Default resolution
    img = sim.render(width, height, camera_name=camera_name)
    return img


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments using modern mujoco and gymnasium."""
    
    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        
        self.frame_skip = frame_skip
        self.model = load_model_from_path(fullpath)
        self.sim = MjSimWrapper(self.model)
        self.data = self.sim.data
        self.viewer = None
        # Use 640x480 for headless rendering compatibility (fits MuJoCo default framebuffer)
        self.buffer_size = (640, 480)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 60,
        }

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        
        # Get observation dimension by taking a dummy step
        # Note: _step_impl always returns 4 values (old API)
        observation, _reward, done, _info = self._step_impl(np.zeros(self.model.nu))
        
        # Handle done for both single and multi-agent
        if isinstance(done, (tuple, list)):
            assert not any(done)
        else:
            assert not done
        self.obs_dim = np.sum([o.size for o in observation]) if (
            type(observation) is tuple) else observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds[:, 0], bounds[:, 1]
        
        # Use gymnasium spaces with proper dtype
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        high = np.inf * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Initialize random number generator
        self.seed()

    def seed(self, seed=None):
        """Gymnasium API for seeding."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ------------------------------------------------------------------------

    def reset_model(self):
        """Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """Called when the viewer is initialized and after every reset.
        Optionally implement this method, if you need to tinker with camera
        position and so forth.
        """
        pass

    # ------------------------------------------------------------------------
    
    def _reset_impl(self):
        """Internal reset implementation."""
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob

    def reset(self, seed=None, options=None):
        """Gymnasium API: returns (observation, info)."""
        if seed is not None:
            self.seed(seed)
        ob = self._reset_impl()
        return ob, {}

    def _step_impl(self, action):
        """
        Internal step implementation that subclasses override.
        
        Subclasses should override this method (or _step for old API compatibility).
        Must return (obs, reward, done, info) for old API.
        
        For gymnasium, this will be wrapped to return 5 values.
        """
        # If subclass has _step, call it
        if hasattr(self.__class__, '_step') and self.__class__._step is not MujocoEnv._step:
            return self._step(action)
        else:
            raise NotImplementedError("Subclass must implement _step() or step()")
    
    def _step(self, action):
        """
        Old gym API step (returns 4 values).
        
        Subclasses should override this. Base implementation just raises NotImplementedError.
        """
        raise NotImplementedError("Subclass must implement _step()")
    
    def step(self, action):
        """
        Gymnasium API step (returns 5 values: obs, reward, terminated, truncated, info).
        
        Wraps _step() to provide gymnasium-compatible API.
        """
        # Call _step (old API) and convert to gymnasium API
        obs, reward, done, info = self._step_impl(action)
        
        # For gymnasium, split "done" into terminated and truncated
        # Default: treat all "done" as terminated, truncated=False
        # Subclasses can override to provide more specific termination info
        if isinstance(done, (list, tuple)):
            # Multi-agent: convert each done
            terminated = list(done)
            truncated = [False] * len(done)
        else:
            terminated = done
            truncated = False
        
        return obs, reward, terminated, truncated, info

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,)
        assert qvel.shape == (self.model.nv,)
        state = self.sim.get_state()
        for i in range(self.model.nq):
            state.qpos[i] = qpos[i]
        for i in range(self.model.nv):
            state.qvel[i] = qvel[i]
        self.sim.set_state(state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = ctrl[i]
        for _ in range(n_frames):
            self.sim.step()

    def render(self):
        """
        Gymnasium API render (no mode parameter).
        
        Returns RGB array for offscreen rendering.
        For 'human' mode, just returns None (headless).
        """
        try:
            # Gymnasium API: always return RGB array for headless
            self.viewer_setup()
            return _read_pixels(self.sim, *self.buffer_size)
        except Exception as e:
            # In headless environments, rendering may fail
            # Return a dummy image or None
            print(f"Warning: Rendering failed in headless environment: {e}")
            return None

    def state_vector(self):
        state = self.sim.get_state()
        return np.concatenate([state.qpos.flat, state.qvel.flat])


# Alias for backwards compatibility
MjSim = MjSimWrapper
MjViewer = None  # Viewer not implemented for headless mode
