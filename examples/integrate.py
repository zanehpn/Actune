"""Connect externally loaded policy/checkpoint/calibration to ActTune.

No model or data is downloaded by this module. The caller owns preprocessing,
postprocessing and episode observation delivery.
"""
import torch

from actune import Controller, Prediction
from actune.layers import install_bank


def make_controller(model, bank, clips, tree, predict_postprocessed, *,
                    channel_scales=None, device=None, hardware_policy=None,
                    training_state_bound=None, tune_kernels=False):
    """predict_postprocessed(observation) -> (actions[H,7], gripper_margin).

    model is the calibrated backbone; bank layer paths are relative to model.
    predict_postprocessed may call a larger policy containing that backbone.
    It must perform exactly one policy inference, with no reference/shadow pass.
    """
    runtime = install_bank(model, bank, clips, channel_scales=channel_scales, tune=tune_kernels)

    @torch.inference_mode()
    def policy(observation, configuration):
        runtime.before()
        actions, margin = predict_postprocessed(observation)
        return Prediction(actions, margin)

    return Controller(tree, policy, device=device, hardware_policy=hardware_policy,
        state_bound=training_state_bound, synchronize=torch.cuda.synchronize,
        audit=runtime.audit)


def run_episode(controller, observations, execute_actions):
    controller.start_episode()
    try:
        for observation in observations:
            result = controller.predict(observation)
            execute_actions(result.actions)
    finally:
        controller.end_episode()
