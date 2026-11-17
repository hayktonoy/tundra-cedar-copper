# Technical document

This document describes our method for ball possession in the match BAR–ATH. The output has one
row for each frame. The `label` column contains `H` (home), `A` (away) or `D` (dead ball).
Frames outside the two halves use `D`. One command makes the submitted file from the raw match data.

## Approach

For each frame, the pipeline answers two questions:

- Is the ball in play?
- Which team has the ball?

The pipeline uses many independent sources of evidence. A decoder puts the evidence into one
timeline. Rules correct the timeline where an event shows an error. Then a touch stage gives the
ball to the other team when one of its players touches the ball.

Each source of evidence has different errors:

- The provided ball track has gaps.
- The ball track from the video loses the ball in duels.
- The Bundesliga models learn from cleaner tracking data.

The pipeline combines these sources.

## The steps

1. **Sensors.**
   - A public football detector finds the ball and the players in the two videos.
   - A tracker connects the ball detections. It uses physical limits for speed and acceleration.
   - The tracker gives a penalty to positions outside the pitch and to static objects far from
     all players. It fills short gaps.
   - The two camera views together give the height of the ball.
   - The colour of the jersey gives the team of each detected player.
   - A public action spotter finds passes, throw-ins, free kicks, shots, ball-outs and fouls in
     the two videos. It gives a team for each action.
   - The audio of the broadcast video gives a whistle signal.
2. **Dead or live.** Three sources each give the probability of a dead ball:
   - A two-state model that learns from physical signals in this match.
   - Gradient-boosted trees on the provided ball. The trees learn from seven Bundesliga matches.
   - The same trees on the ball from the video.

   The pipeline uses fixed weights chosen from the agreement between the three sources
   (Dawid–Skene method).
3. **Which team.** Five sources give evidence:
   - A model that reads the shape of the teams and the pressure on the ball. An algorithm makes
     its targets: frames where a slow ball is at the feet of one team.
   - Direct contact between the ball and the feet of a player.
   - A team model that learns from the Bundesliga matches.
   - The team of each spotted action.
   - An ownership grammar. In this grammar, a ball in flight stays with the team that played it.
4. **Decoder.** A sequence decoder with three states (home, away, dead) changes the evidence
   into one timeline. A change between live and dead has a cost. These events can decrease the
   cost: the ball goes out of the pitch, the players stop together, or a player kicks a ball at
   rest. A second decoder step adds the ownership grammar to the timeline.
5. **Rule stages.** Eleven rules correct stoppages and short possessions where an event shows an
   error. The pipeline applies the rules in this sequence:
   1. Confirm or remove short islands with the action spotter.
   2. Start a stoppage at a spotted ball-out.
   3. Trim the edges of stoppages.
   4. Keep the ball live when a goalkeeper holds it.
   5. Set a dead period for a ball-out followed by a goal kick.
   6. Correct short throw-ins.
   7. Correct short periods of control.
   8. Correct restarts in the penalty area.
   9. Move the start of a stoppage to the whistle.
   10. Permit no live ball before the kick-off.
   11. Permit no new possession at a spotted foul.
6. **Stoppage start.** The pipeline looks for touches by the team in possession within 60
   frames of the start of a stoppage. A touch can show that play continued. The pipeline starts
   the stoppage 5 frames after the last qualifying touch and keeps at least 50 dead frames.
7. **Lag.** The pipeline moves each team change up to 10 frames later, each stoppage start up to
   15 frames later, and each restart up to 20 frames later. The move preserves at least 15
   frames of the next run, or its full length if it is already shorter.
8. **Touch stage.** In each possession, the pipeline looks for a touch by the other team. A
   touch is a point where the path of the ball bends near a player. A touch with high confidence
   can start a short possession of the other team. A second model must agree. This carrier
   model is a hidden Markov model of the 46 track slots. It learns from the tracking and touch
   signals. The proposed possession starts 10 frames after the touch. The pipeline makes no
   possession shorter than 15 frames.

The build runs the decoder and the rule stages with fixed settings files
(`pipeline/fm-rebuild/recipe/*.json`).

## Possession rules

- **Dead ball.** The pipeline first finds the dead frames. A dead ball has no team.
- **Loose balls and duels.** The ball stays with the last team that had it.
- **Change of possession.** The possession changes when the other team has the ball for a
  continuous period. It also changes at one confirmed touch by an opponent. A tackle or a block
  without control is such a touch.
- **Ball in flight.** The ball stays with the team that played it until an opponent touches it.
- **Restarts.** The possession starts when a player puts the ball back into play. A goalkeeper
  who holds the ball is in possession.
- **Stoppages.** A stoppage starts when the teams stop play. The teams can continue play after
  a whistle.
- **Minimum length.** No possession is shorter than 15 frames (0.6 s).

## Data and models

| Asset | Use |
|---|---|
| Tactical video | Ball detection, player detection, the primary ball track, action spotter |
| Broadcast video | Second view of the ball (height, agreement between views), player boxes, action spotter, audio (whistle signal) |
| Provided tracking data | Player positions and teams. The provided ball is one source for dead or live |
| Homography | Projection of detections onto the pitch |
| Smart Tagging | Set-piece tags as input to the two-state model and the team model. Goal-kick tags in one rule |
| Code of the organisers | The build imports `games.py`, the Smart Tagging loader and `possession_metrics.py` |
| External: IDSSE/DFL Bundesliga open data, seven matches | Training of the models for dead or live and for the team |
| External: public football detector (`Adit-jain/soccana` on Hugging Face, fixed revision) | Ball detection and player detection in the two videos |
| External: public T-DEED action spotter (`SoccerNet/sn-teamspotting`, commit `091fed2`) | Spotted actions with a team, in the two videos |
| External: Python packages, with the PyTorch CUDA wheels | The build downloads them |

The pipeline uses no hosted model.

## Training data

| Dataset | Source | Training targets and source | Quantity | Use |
|---|---|---|---|---|
| IDSSE/DFL open tracking data and event data (Bassek et al., 2025, doi:10.6084/m9.figshare.28196177) | Public download. `pipeline/dfl-owner/sources.json` records the file identifiers, sizes and MD5 sums | The league's data provider gives the ball status and the team with the ball for each frame | Seven matches, 1,002,644 frames at 25 Hz | Training of the trees for dead or live and of the team model |
| Detector weights (`Adit-jain/soccana`, YOLO11n) | Public. `pipeline/football-ball-detector/download.py` records the revision | The author trained the detector on the public dataset `Adit-jain/Soccana_player_ball_detection_v1`. The dataset has boxes for players, ball and referee. The images come from SoccerNet, video-game renders, public match video and eight Roboflow Universe datasets. The dataset card does not tell who made the boxes | Approximately 23,000 to 25,000 images (dataset card) | Ball detection and player detection. We use the published weights |
| T-DEED action spotter weights (`teamball_baseline.pt`, `ball_challenge1.pt`) | Public download from `SoccerNet/sn-teamspotting`. `pipeline/from-raw/build.py` records the download file identifiers | The authors trained the spotter on the SoccerNet ball-action data. SoccerNet made the action times and the teams | As the authors published | Spotted actions in the two videos. We use the published weights |

The build trains the trees for dead or live and the Bundesliga team model.

The pipeline learns these models from the inputs of the provided match:

- A two-state model for dead or live (expectation maximisation on physical signals).
- A classifier for jersey colour. It learns from the pixels of players whose team the provided
  tracking data gives.
- A team model with targets from an algorithm (a slow ball at the feet of one team). A second
  model, local to the ball, learns together with it.
- A prior for the ownership grammar (Baum–Welch method).
- The carrier model (expectation maximisation).

## Source of each setting

| Setting | Source or use |
|---|---|
| Other constants of the decoder, the grammar and the rules (`pipeline/fm-rebuild/recipe/*.json`) | Each constant has a list of possible values. The setting is the lower median of the list |
| Weights of the three sources for dead or live | Dawid–Skene method on the agreement between the sources, with the weights recorded in the decoder recipe |
| Smoothing for dead or live: scale 0.2, bias 2 | Development compared four statistics from the seven Bundesliga matches: stoppages per hour, share of time when the ball is dead, share of stoppages shorter than five seconds, and share of live periods shorter than five seconds |
| Switch cost 80 | The decoder charges this cost for a switch between live and dead. The cost is large compared with the evidence in one frame. Evidence can build up over several frames, and events can lower the cost |
| Stop-together bonus 80 | This bonus multiplies the signal for players slowing down together. A strong signal can offset the cost of starting a stoppage |
| Event weight 1 | The decoder applies the event bonuses at full strength when it sets the costs of starting and ending a stoppage |
| Grammar weight 0.1 in the team decision | The decoder gives the grammar one tenth of the weight when it mixes the grammar with the other team evidence. This gives the grammar a small influence |
| Dead-period fringe 50 frames (`goal_clean.fringe`) | `goal_clean` marks the dead period at a goal. After a rule inserts a dead period, a live fragment of at most 50 frames (2 s at 25 Hz) that touches the new dead period is merged into the previous live run. This removes a brief ownership guess for the next restart |
| Stoppage-start rule: 60 frames, touch confidence 0.7, 5 frames after the touch | The rule looks for a confident touch by the team in possession. It leaves a short gap after the last qualifying touch and keeps at least 50 dead frames |
| Lag: team change +10 frames, stoppage start +15, restart +20 | Fixed delays by boundary type. The pipeline limits each delay to preserve the next run |
| Touch stage: carrier temperature, carrier agreement, merge, touch confidence 0.7 | Development used seven Bundesliga matches, with one match left out each time. The tests added noise to the ball track to represent the provided tracking |
| Touch stage: delay of 10 frames | The touch stage places a proposed change of team 10 frames after the touch. It uses the same delay for a return touch |
| Settings of the ball tracker | Agreement between the two cameras |
| Physical limits (tracker limits, contact distances, minimum durations) and the weights in the touch and grammar evidence | Fixed constants |
| Camera synchronisation | One line for each half. The line fits the time difference between the two provided ball tracks (`calibration/broadcast_sync.json`) |
| Camera positions | Calculated from the provided homographies (`pipeline/hc-ball/common.py`) |
| Half limits, frame count, pitch size, the goal of each team | Match data from the documents of the organisers and the provided tracking data |

## Limits

- **One target match.** We built the full pipeline for BAR–ATH.
- **Gaps in the ball tracks.** The provided tracking has missing ball positions. The video track
  also loses the ball in duels, where touches are hard to see.
- **Touch uncertainty.** A bend in the ball path can come from a touch or from tracking noise.
  The carrier model must agree before the pipeline adds a short possession.
- **Fixed lag.** Each type of boundary has one delay. The delay can be smaller to preserve the
  next run.
- **Some events have no signal.** The pipeline does not find touches in aerial duels or
  contacts that players hide. It does not find a stoppage without a physical signal, unless the
  whistle signal or the action spotter finds it.
- **The Bundesliga matches have different tracking data.** They have clean tracking data and
  stable player identities.
- **We made the file on one computer.** On our computer (RTX 4060 laptop), the build makes the
  same file each time, byte for byte. The detectors and the action spotter use the GPU. On
  different hardware, their results can be different by some frames. The build script uses
  Windows process tools.

## With more time

These are the next steps for this method:

- **A ball detector for this match.** Train the detector on ball positions where the two camera
  views agree.
- **Better touch detection.** Adjust the touch stage to use the newer ball track.
- **A touch model from event data.** Train a touch model on the events of the Bundesliga
  matches.

## Reproduction

1. Copy the starter tree of the organisers into `official/` (revision `f8f2629`).
2. Put the data bundle in `official/data/`. Obey the instructions of the organisers.
3. Run this command from the root of the repository:

```powershell
uv run --no-project python pipeline/last-touch/build.py
```

The build needs Windows, Git, uv, ffmpeg with CUDA and a CUDA GPU with 8 GB. We used an RTX 4060
laptop. The build downloads the detector weights, the action spotter and the Bundesliga data
(approximately 2.5 GB). It does all steps from the raw videos and the tracking data in
approximately four and a half hours. It writes `pipeline/last-touch/.local/last_touch.csv`.
