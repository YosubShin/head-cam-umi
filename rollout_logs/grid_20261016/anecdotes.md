# Rollout anecdotes, grid_20261016 (operator notes, transcribed)

- Scene 5, budget pair: head_Hgt48 (teleop only, 48 eps) could not close in on the block at all; mix_gHs3_48 first failed
  to close the gripper fully and missed, then came back, re-grasped and succeeded. Recovery behaviour present only in the mix.
- Scene 5, mix_gHA (head labels, wrist obs): dropped the block mid-transport; looked like the policy opened the gripper
  without the block in view (as if it did not register the block in hand).
- Wrist-cam arms (mix_FA, mix_gHA) drop the block mid-transport when the wrist view is looking AWAY from the bin; never
  observed the converse (drop while the bin was visible). Head-cam arms keep gripper and bin in view during transport.

- Scene 15 (16 Oct, ~18:45): the robot arm pushed one workspace tile during a rollout. Tracker park verification (spot_1016b, 18:30) predates it; tiles are not used by the evaluator or the policies at rollout time. Tile left as is for the remaining scenes (same for all arms). Return or refit before any tracker restart with --hand-home or any new collection.
- Scene 16 reference (18:5x): tile 5 (the pushed one) re-taped by ghost alignment; fixed-cam detection puts it 0.4 mm from its scene-1 position, tiles 3/4 unmoved (<=1 mm) throughout. No action needed before a tracker restart or new collection.
