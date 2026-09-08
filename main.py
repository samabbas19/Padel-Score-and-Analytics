import timeit
import json
from pathlib import Path
import cv2
import numpy as np
import supervision as sv

from trackers import (
    PlayerTracker, 
    BallTracker, 
    KeypointsTracker, 
    Keypoint,
    Keypoints,
    PlayerKeypointsTracker,
    TrackingRunner,
)
from config import *
from analytics.audio_visual_events import (
    AudioVisualEventConfig,
    detect_audio_visual_events,
)
from analytics.scoring import ScoreConfig, calculate_scores
from analytics.scoreboard_ground_truth import (
    ScoreboardExtractorConfig,
    build_scoreboard_prediction_timeline,
    extract_scoreboard_ground_truth,
    load_scoreboard_template_seeds,
    validate_score_timeline,
)
from analytics.gameplay_alignment import align_event_candidates_to_scoreboard
from analytics.gameplay_alignment import build_gameplay_point_label_dataset
from analytics.gameplay_proposals import (
    GameplayProposalConfig,
    build_event_gap_point_proposals,
    build_label_replay_score_timeline,
    build_proposal_score_timeline,
    build_scoreboard_transition_expansion_report,
    evaluate_score_checkpoints_against_scoreboard,
    evaluate_score_sequence_against_scoreboard,
    evaluate_proposals_against_labels,
    evaluate_winners_against_labels,
)
from visualizations.score_overlay import render_score_overlay_video


SELECTED_KEYPOINTS = []

"""
PADEL COURT KEYPOINTS 

-> To be selected using the image pop-up

        k11--------------------k12
        |                       |
        k8-----------k9--------k10
        |            |          |
        |            |          |
        |            |          |
        k6----------------------k7
        |            |          |
        |            |          |
        |            |          |
        k3-----------k4---------k5
        |                       |
        k1----------------------k2
        
"""

def click_event(event, x, y, flags, params): 
  
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        SELECTED_KEYPOINTS.append((x, y))
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('frame', img) 


def make_polygon_zone(polygon: np.ndarray, video_info: sv.VideoInfo) -> sv.PolygonZone:
    try:
        return sv.PolygonZone(
            polygon,
            frame_resolution_wh=video_info.resolution_wh,
        )
    except TypeError as exc:
        if "frame_resolution_wh" not in str(exc):
            raise
        return sv.PolygonZone(polygon)


if __name__ == "__main__":
    
    t1 = timeit.default_timer()

    video_info = sv.VideoInfo.from_video_path(video_path=INPUT_VIDEO_PATH)
    fps, w, h, total_frames = (
        video_info.fps, 
        video_info.width,
        video_info.height,
        video_info.total_frames,
    )

    first_frame_generator = sv.get_video_frames_generator(
        INPUT_VIDEO_PATH,
        start=0,
        stride=1,
        end=1,
    )

    img = next(first_frame_generator)

    fixed_keypoints_load_path = (
        Path(FIXED_COURT_KEYPOINTS_LOAD_PATH)
        if FIXED_COURT_KEYPOINTS_LOAD_PATH is not None
        else None
    )

    if fixed_keypoints_load_path is not None and fixed_keypoints_load_path.exists():
        print(f"main: Loading court keypoints from {fixed_keypoints_load_path}")
        with open(fixed_keypoints_load_path, "r") as f:
            SELECTED_KEYPOINTS = json.load(f)
    else:
        if fixed_keypoints_load_path is not None and FIXED_COURT_KEYPOINTS_SAVE_PATH is None:
            FIXED_COURT_KEYPOINTS_SAVE_PATH = str(fixed_keypoints_load_path)
        print("main: Select 12 court keypoints in the OpenCV window, then press any key.")
        cv2.imshow('frame', img)
        cv2.setMouseCallback('frame', click_event) 
        # wait for a key to be pressed to exit 
        cv2.waitKey(0) 
        # close the window 
        cv2.destroyAllWindows() 

    if len(SELECTED_KEYPOINTS) != 12:
        raise ValueError(
            f"Expected 12 selected court keypoints, got {len(SELECTED_KEYPOINTS)}"
        )

    if FIXED_COURT_KEYPOINTS_SAVE_PATH is not None:
        fixed_keypoints_save_path = Path(FIXED_COURT_KEYPOINTS_SAVE_PATH)
        fixed_keypoints_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fixed_keypoints_save_path, "w") as f:
            json.dump(SELECTED_KEYPOINTS, f)
        print(f"main: Saved court keypoints to {fixed_keypoints_save_path}")

    fixed_keypoints_detection = Keypoints(
        [
            Keypoint(
                id=i,
                xy=tuple(float(x) for x in v)
            )
            for i, v in enumerate(SELECTED_KEYPOINTS)
        ]
    )

    keypoints_array = np.array(SELECTED_KEYPOINTS)
    # Polygon to filter person detections inside padel court
    polygon_zone = make_polygon_zone(
        np.concatenate(
            (
                np.expand_dims(keypoints_array[0], axis=0),
                np.expand_dims(keypoints_array[1], axis=0),
                np.expand_dims(keypoints_array[-1], axis=0),
                np.expand_dims(keypoints_array[-2], axis=0),
            ),
            axis=0,
        ).astype(np.int64),
        video_info,
    )


    # FILTER FRAMES OF INTEREST (TODO)


    # Instantiate trackers
    players_tracker = PlayerTracker(
        PLAYERS_TRACKER_MODEL,
        polygon_zone,
        batch_size=PLAYERS_TRACKER_BATCH_SIZE,
        annotator=PLAYERS_TRACKER_ANNOTATOR,
        show_confidence=True,
        load_path=PLAYERS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_TRACKER_SAVE_PATH,
    )

    player_keypoints_tracker = PlayerKeypointsTracker(
        PLAYERS_KEYPOINTS_TRACKER_MODEL,
        train_image_size=PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE,
        batch_size=PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE,
        load_path=PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH,
    )
  
    ball_tracker = BallTracker(
        BALL_TRACKER_MODEL,
        BALL_TRACKER_INPAINT_MODEL,
        batch_size=BALL_TRACKER_BATCH_SIZE,
        median_max_sample_num=BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
        median=None,
        load_path=BALL_TRACKER_LOAD_PATH,
        save_path=BALL_TRACKER_SAVE_PATH,
    )

    keypoints_tracker = KeypointsTracker(
        model_path=KEYPOINTS_TRACKER_MODEL,
        batch_size=KEYPOINTS_TRACKER_BATCH_SIZE,
        model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
        fixed_keypoints_detection=fixed_keypoints_detection,
        load_path=KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=KEYPOINTS_TRACKER_SAVE_PATH,
    )

    runner = TrackingRunner(
        trackers=[
            players_tracker, 
            player_keypoints_tracker, 
            ball_tracker,
            keypoints_tracker,    
        ],
        video_path=INPUT_VIDEO_PATH,
        inference_path=OUTPUT_VIDEO_PATH,
        start=0,
        end=MAX_FRAMES,
        collect_data=COLLECT_DATA,
    )

    runner.run()

    if COLLECT_DATA:
        data = runner.data_analytics.into_dataframe(runner.video_info.fps)
        data.to_csv(COLLECT_DATA_PATH, index=False)
        print(f"main: Saved trajectory data to {COLLECT_DATA_PATH}")

        if DETECT_AUDIO_VISUAL_EVENTS:
            print("main: Detecting no-training audio + vision event candidates ...")
            event_summary = detect_audio_visual_events(
                video_path=INPUT_VIDEO_PATH,
                data_path=COLLECT_DATA_PATH,
                output_path=EVENT_CANDIDATES_PATH,
                summary_path=EVENT_SUMMARY_PATH,
                fps=runner.video_info.fps,
                ball_detections_path=BALL_TRACKER_LOAD_PATH or BALL_TRACKER_SAVE_PATH,
                players_detections_path=PLAYERS_TRACKER_LOAD_PATH
                or PLAYERS_TRACKER_SAVE_PATH,
                config=AudioVisualEventConfig(
                    audio_peak_z=AUDIO_EVENT_PEAK_Z,
                    candidate_min_confidence=AUDIO_EVENT_MIN_CONFIDENCE,
                    auto_confidence=AUDIO_EVENT_AUTO_CONFIDENCE,
                ),
            )
            print(
                "main: Event candidates saved "
                f"{event_summary['candidates_kept']} candidate(s) to {EVENT_CANDIDATES_PATH}"
            )
            print(f"main: Event summary saved to {EVENT_SUMMARY_PATH}")

        if AUTO_SCORE:
            gameplay_events_path = (
                SCORE_EVENTS_PATH if SCORE_SOURCE == "gameplay" else GAMEPLAY_SCORE_EVENTS_PATH
            )
            gameplay_summary_path = (
                SCORE_SUMMARY_PATH if SCORE_SOURCE == "gameplay" else GAMEPLAY_SCORE_SUMMARY_PATH
            )
            print("main: Calculating gameplay trajectory score without OCR ...")
            gameplay_summary = calculate_scores(
                data_path=COLLECT_DATA_PATH,
                events_path=gameplay_events_path,
                summary_path=gameplay_summary_path,
                fps=runner.video_info.fps,
                config=ScoreConfig(
                    golden_point=SCORE_GOLDEN_POINT,
                    point_break_seconds=SCORE_POINT_BREAK_SECONDS,
                    auto_apply_confidence=SCORE_AUTO_APPLY_CONFIDENCE,
                    provisional_confidence=SCORE_PROVISIONAL_CONFIDENCE,
                ),
            )
            print(
                "main: Gameplay score calculation saved "
                f"{gameplay_summary['points_detected']} point event(s) to {gameplay_events_path}"
            )
            print(f"main: Gameplay score summary saved to {gameplay_summary_path}")

            if DETECT_AUDIO_VISUAL_EVENTS:
                proposal_events_path = (
                    SCORE_EVENTS_PATH
                    if SCORE_SOURCE == "proposal"
                    else GAMEPLAY_PROPOSAL_SCORE_EVENTS_PATH
                )
                proposal_summary_path = (
                    SCORE_SUMMARY_PATH
                    if SCORE_SOURCE == "proposal"
                    else GAMEPLAY_PROPOSAL_SCORE_SUMMARY_PATH
                )
                proposal_summary = build_event_gap_point_proposals(
                    event_candidates_path=EVENT_CANDIDATES_PATH,
                    proposals_path=GAMEPLAY_POINT_PROPOSALS_PATH,
                    summary_path=GAMEPLAY_POINT_PROPOSALS_SUMMARY_PATH,
                    config=GameplayProposalConfig(
                        golden_point=SCORE_GOLDEN_POINT,
                        min_time_seconds=PROPOSAL_MIN_TIME_SECONDS,
                        initial_points=SCORE_INITIAL_POINTS,
                        initial_games=SCORE_INITIAL_GAMES,
                        auto_apply_confidence=SCORE_AUTO_APPLY_CONFIDENCE,
                        provisional_confidence=SCORE_PROVISIONAL_CONFIDENCE,
                    ),
                )
                print(
                    "main: Gameplay point proposals saved "
                    f"{proposal_summary['point_proposals']} proposal(s)"
                )

                proposal_score_summary = build_proposal_score_timeline(
                    proposals_path=GAMEPLAY_POINT_PROPOSALS_PATH,
                    events_path=proposal_events_path,
                    summary_path=proposal_summary_path,
                    fps=runner.video_info.fps,
                    config=GameplayProposalConfig(
                        golden_point=SCORE_GOLDEN_POINT,
                        min_time_seconds=PROPOSAL_MIN_TIME_SECONDS,
                        initial_points=SCORE_INITIAL_POINTS,
                        initial_games=SCORE_INITIAL_GAMES,
                        auto_apply_confidence=SCORE_AUTO_APPLY_CONFIDENCE,
                        provisional_confidence=SCORE_PROVISIONAL_CONFIDENCE,
                    ),
                )
                print(
                    "main: Gameplay proposal score saved "
                    f"{proposal_score_summary['points_scored']} scored point(s) "
                    f"to {proposal_events_path}"
                )
            else:
                proposal_events_path = GAMEPLAY_PROPOSAL_SCORE_EVENTS_PATH

            if EXTRACT_SCOREBOARD_GROUND_TRUTH:
                print("main: Extracting scoreboard ground truth for validation only ...")
                scoreboard_config = ScoreboardExtractorConfig(
                    sample_stride_frames=SCOREBOARD_SAMPLE_STRIDE_FRAMES,
                )
                if SCOREBOARD_TEMPLATE_SEEDS_PATH is not None:
                    scoreboard_config = load_scoreboard_template_seeds(
                        SCOREBOARD_TEMPLATE_SEEDS_PATH,
                        scoreboard_config,
                    )
                    print(
                        "main: Loaded scoreboard template seeds from "
                        f"{SCOREBOARD_TEMPLATE_SEEDS_PATH}"
                    )
                scoreboard_summary = extract_scoreboard_ground_truth(
                    video_path=INPUT_VIDEO_PATH,
                    states_path=SCOREBOARD_STATES_PATH,
                    events_path=SCOREBOARD_EVENTS_PATH,
                    summary_path=SCOREBOARD_SUMMARY_PATH,
                    config=scoreboard_config,
                )
                print(
                    "main: Scoreboard ground truth saved "
                    f"{scoreboard_summary['visible_state_samples']} sample(s) and "
                    f"{scoreboard_summary['score_change_events']} event(s)."
                )

                transition_expansion_report = build_scoreboard_transition_expansion_report(
                    scoreboard_events_path=SCOREBOARD_EVENTS_PATH,
                    report_path=SCOREBOARD_TRANSITION_EXPANSION_REPORT_PATH,
                    details_path=SCOREBOARD_TRANSITION_EXPANSION_DETAILS_PATH,
                    initial_points=SCORE_INITIAL_POINTS,
                    initial_games=SCORE_INITIAL_GAMES,
                    golden_point=SCORE_GOLDEN_POINT,
                )
                print(
                    "main: Scoreboard transition expansion saved "
                    f"minimum_points={transition_expansion_report['minimum_point_events']}, "
                    f"hidden_points={transition_expansion_report['hidden_point_events']}"
                )

                baseline_summary = build_scoreboard_prediction_timeline(
                    scoreboard_events_path=SCOREBOARD_EVENTS_PATH,
                    output_path=SCOREBOARD_BASELINE_EVENTS_PATH,
                    summary_path=SCOREBOARD_BASELINE_SUMMARY_PATH,
                    fps=runner.video_info.fps,
                )
                print(
                    "main: Scoreboard baseline saved "
                    f"{baseline_summary['points_detected']} transition(s)."
                )

                if SCORE_SOURCE == "scoreboard":
                    selected_summary = build_scoreboard_prediction_timeline(
                        scoreboard_events_path=SCOREBOARD_EVENTS_PATH,
                        output_path=SCORE_EVENTS_PATH,
                        summary_path=SCORE_SUMMARY_PATH,
                        fps=runner.video_info.fps,
                    )
                    print(
                        "main: Selected scoreboard-recognition score source "
                        f"saved {selected_summary['points_detected']} transition(s) to "
                        f"{SCORE_EVENTS_PATH}"
                    )

                if VALIDATE_SCORE_TIMELINE:
                    print("main: Validating predicted score timeline against scoreboard ...")
                    validation_report = validate_score_timeline(
                        predicted_events_path=SCORE_EVENTS_PATH,
                        ground_truth_states_path=SCOREBOARD_STATES_PATH,
                        ground_truth_events_path=SCOREBOARD_EVENTS_PATH,
                        report_path=SCORE_VALIDATION_REPORT_PATH,
                        mismatches_path=SCORE_VALIDATION_MISMATCHES_PATH,
                        fps=runner.video_info.fps,
                    )
                    print(
                        "main: Score validation saved "
                        f"frame_accuracy={validation_report['frame_accuracy']:.3f}, "
                        f"event_accuracy={validation_report['event_accuracy']:.3f}"
                    )

                    gameplay_report = validate_score_timeline(
                        predicted_events_path=gameplay_events_path,
                        ground_truth_states_path=SCOREBOARD_STATES_PATH,
                        ground_truth_events_path=SCOREBOARD_EVENTS_PATH,
                        report_path=GAMEPLAY_SCORE_VALIDATION_REPORT_PATH,
                        mismatches_path=GAMEPLAY_SCORE_VALIDATION_MISMATCHES_PATH,
                        fps=runner.video_info.fps,
                    )
                    print(
                        "main: Gameplay score validation saved "
                        f"frame_accuracy={gameplay_report['frame_accuracy']:.3f}, "
                        f"event_accuracy={gameplay_report['event_accuracy']:.3f}"
                    )

                    if DETECT_AUDIO_VISUAL_EVENTS:
                        proposal_report = validate_score_timeline(
                            predicted_events_path=proposal_events_path,
                            ground_truth_states_path=SCOREBOARD_STATES_PATH,
                            ground_truth_events_path=SCOREBOARD_EVENTS_PATH,
                            report_path=GAMEPLAY_PROPOSAL_VALIDATION_REPORT_PATH,
                            mismatches_path=GAMEPLAY_PROPOSAL_VALIDATION_MISMATCHES_PATH,
                            fps=runner.video_info.fps,
                        )
                        print(
                            "main: Gameplay proposal validation saved "
                            f"frame_accuracy={proposal_report['frame_accuracy']:.3f}, "
                            f"event_accuracy={proposal_report['event_accuracy']:.3f}"
                        )

                        sequence_report = evaluate_score_sequence_against_scoreboard(
                            predicted_events_path=proposal_events_path,
                            scoreboard_events_path=SCOREBOARD_EVENTS_PATH,
                            report_path=GAMEPLAY_SCORE_SEQUENCE_REPORT_PATH,
                            mismatches_path=GAMEPLAY_SCORE_SEQUENCE_MISMATCHES_PATH,
                        )
                        print(
                            "main: Gameplay score sequence validation saved "
                            f"accuracy={sequence_report['score_sequence_accuracy']:.3f}"
                        )

                        checkpoint_report = evaluate_score_checkpoints_against_scoreboard(
                            predicted_events_path=proposal_events_path,
                            scoreboard_events_path=SCOREBOARD_EVENTS_PATH,
                            report_path=GAMEPLAY_SCORE_CHECKPOINT_REPORT_PATH,
                            mismatches_path=GAMEPLAY_SCORE_CHECKPOINT_MISMATCHES_PATH,
                        )
                        print(
                            "main: Gameplay score checkpoint validation saved "
                            f"accuracy={checkpoint_report['checkpoint_accuracy']:.3f}"
                        )

                        alignment_summary = align_event_candidates_to_scoreboard(
                            scoreboard_events_path=SCOREBOARD_EVENTS_PATH,
                            event_candidates_path=EVENT_CANDIDATES_PATH,
                            output_path=GAMEPLAY_ALIGNMENT_PATH,
                            summary_path=GAMEPLAY_ALIGNMENT_SUMMARY_PATH,
                        )
                        print(
                            "main: Gameplay event alignment saved "
                            f"{alignment_summary['nearest_within_tolerance']}/"
                            f"{alignment_summary['scoreboard_transitions']} nearest "
                            "candidate(s) within tolerance"
                        )
                        label_summary = build_gameplay_point_label_dataset(
                            alignment_path=GAMEPLAY_ALIGNMENT_PATH,
                            output_path=GAMEPLAY_POINT_LABELS_PATH,
                            summary_path=GAMEPLAY_POINT_LABELS_SUMMARY_PATH,
                        )
                        print(
                            "main: Gameplay point labels saved "
                            f"{label_summary['usable_labels']}/"
                            f"{label_summary['labels']} usable label(s)"
                        )
                        proposal_label_report = evaluate_proposals_against_labels(
                            proposals_path=GAMEPLAY_POINT_PROPOSALS_PATH,
                            labels_path=GAMEPLAY_POINT_LABELS_PATH,
                            report_path=GAMEPLAY_POINT_PROPOSAL_LABEL_REPORT_PATH,
                            mismatches_path=GAMEPLAY_POINT_PROPOSAL_LABEL_MISMATCHES_PATH,
                        )
                        print(
                            "main: Point proposal label validation saved "
                            f"precision={proposal_label_report['proposal_precision']:.3f}, "
                            f"recall={proposal_label_report['proposal_recall']:.3f}"
                        )
                        winner_report = evaluate_winners_against_labels(
                            predicted_events_path=proposal_events_path,
                            labels_path=GAMEPLAY_POINT_LABELS_PATH,
                            report_path=GAMEPLAY_WINNER_ATTRIBUTION_REPORT_PATH,
                            mismatches_path=GAMEPLAY_WINNER_ATTRIBUTION_MISMATCHES_PATH,
                        )
                        print(
                            "main: Winner attribution validation saved "
                            f"accuracy={winner_report['winner_accuracy']:.3f}"
                        )
                        label_replay_summary = build_label_replay_score_timeline(
                            labels_path=GAMEPLAY_POINT_LABELS_PATH,
                            events_path=GAMEPLAY_LABEL_REPLAY_EVENTS_PATH,
                            summary_path=GAMEPLAY_LABEL_REPLAY_SUMMARY_PATH,
                            fps=runner.video_info.fps,
                            config=GameplayProposalConfig(
                                golden_point=SCORE_GOLDEN_POINT,
                                auto_apply_confidence=SCORE_AUTO_APPLY_CONFIDENCE,
                                provisional_confidence=SCORE_PROVISIONAL_CONFIDENCE,
                            ),
                        )
                        print(
                            "main: Gameplay label replay saved "
                            f"{label_replay_summary['points_detected']} transition(s)"
                        )
                        label_replay_report = validate_score_timeline(
                            predicted_events_path=GAMEPLAY_LABEL_REPLAY_EVENTS_PATH,
                            ground_truth_states_path=SCOREBOARD_STATES_PATH,
                            ground_truth_events_path=SCOREBOARD_EVENTS_PATH,
                            report_path=GAMEPLAY_LABEL_REPLAY_VALIDATION_REPORT_PATH,
                            mismatches_path=GAMEPLAY_LABEL_REPLAY_VALIDATION_MISMATCHES_PATH,
                            fps=runner.video_info.fps,
                        )
                        print(
                            "main: Gameplay label replay validation saved "
                            f"frame_accuracy={label_replay_report['frame_accuracy']:.3f}, "
                            f"event_accuracy={label_replay_report['event_accuracy']:.3f}"
                        )

                    baseline_report = validate_score_timeline(
                        predicted_events_path=SCOREBOARD_BASELINE_EVENTS_PATH,
                        ground_truth_states_path=SCOREBOARD_STATES_PATH,
                        ground_truth_events_path=SCOREBOARD_EVENTS_PATH,
                        report_path=SCOREBOARD_BASELINE_VALIDATION_REPORT_PATH,
                        mismatches_path=SCOREBOARD_BASELINE_VALIDATION_MISMATCHES_PATH,
                        fps=runner.video_info.fps,
                    )
                    print(
                        "main: Scoreboard baseline validation saved "
                        f"frame_accuracy={baseline_report['frame_accuracy']:.3f}, "
                        f"event_accuracy={baseline_report['event_accuracy']:.3f}"
                    )

            if RENDER_SCORE_OVERLAY and SCORE_OVERLAY_VIDEO_PATH is not None:
                print("main: Rendering lightweight score overlay video ...")
                render_score_overlay_video(
                    input_video_path=OUTPUT_VIDEO_PATH,
                    output_video_path=SCORE_OVERLAY_VIDEO_PATH,
                    events_path=SCORE_EVENTS_PATH,
                    summary_path=SCORE_SUMMARY_PATH,
                )
                print(f"main: Score overlay video saved to {SCORE_OVERLAY_VIDEO_PATH}")

    t2 = timeit.default_timer()

    print("Duration (min): ", (t2 - t1) / 60)
