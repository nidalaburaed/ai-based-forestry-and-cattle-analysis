from flask import Flask, Response, render_template, request
import cv2
import time

# Import pipeline classes
try:
    from src.pipeline import EnhancedForestAnalysisPipeline, PipelineConfig, AnalysisMode
    from src.cattle_pipeline import CattleAnalysisPipeline
except Exception as e:
    print(f"Warning: Failed to import pipelines: {e}")
    EnhancedForestAnalysisPipeline = None
    CattleAnalysisPipeline = None
    PipelineConfig = None
    AnalysisMode = None

app = Flask(__name__)


def generate_frames(src: str):
    """Yield MJPEG frames from a video source (file path or camera index)."""
    # Allow camera index passed as integer string
    try:
        src_val = int(src) if str(src).isdigit() else src
    except Exception:
        src_val = src

    # Try to open with multiple backends for better camera compatibility
    cap = None
    if isinstance(src_val, int):
        # Try DirectShow first (usually most reliable on Windows)
        cap = cv2.VideoCapture(src_val, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            # Try Media Foundation
            cap = cv2.VideoCapture(src_val, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap.release()
                # Try default backend
                cap = cv2.VideoCapture(src_val)
    else:
        cap = cv2.VideoCapture(src_val)

    if not cap.isOpened():
        # yield a single blank frame with an error message encoded as JPEG
        import numpy as np
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(blank, "Cannot open source", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        ret, buf = cv2.imencode('.jpg', blank)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # small delay before retrying or ending
                time.sleep(0.05)
                continue

            # Encode frame as JPEG
            ret2, buffer = cv2.imencode('.jpg', frame)
            if not ret2:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # throttle a bit
            time.sleep(0.03)
    finally:
        cap.release()


@app.route('/')
def index():
    # default source is 0 (webcam). Can be camera index, file path, or stream URL.
    src = request.args.get('src', '0')
    return render_template('index.html', src=src)


@app.route('/video_feed')
def video_feed():
    src = request.args.get('src', '0')
    return Response(generate_frames(src), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Run the pipeline on the provided source and show results.

    This runs synchronously with user-configured parameters.
    """
    src = request.form.get('src', '0')

    # Convert camera index strings to integers (e.g., "0" -> 0)
    try:
        src = int(src) if str(src).isdigit() else src
    except Exception:
        pass  # Keep as string if conversion fails

    # Get pipeline selection
    pipeline_type = request.form.get('pipeline', 'forest')

    # Get all configuration parameters from form
    try:
        max_frames = int(request.form.get('max_frames', '10'))
    except Exception:
        max_frames = 10

    try:
        frame_interval = int(request.form.get('frame_interval', '2'))
    except Exception:
        frame_interval = 2

    # Analysis mode
    mode_str = request.form.get('mode', 'standard')

    # Model selection
    use_sam2 = request.form.get('use_sam2') == 'on'
    use_deepforest = request.form.get('use_deepforest') == 'on'
    use_vit = request.form.get('use_vit') == 'on'

    # Debug options
    save_debug = request.form.get('save_debug') == 'on'

    # Report generation
    generate_reports = request.form.get('generate_reports') == 'on'

    if PipelineConfig is None:
        error = "Pipeline not available (import failed)."
        return render_template('result.html', error=error, src=src, stats=None, plan=None,
                             report_paths=None, pipeline_type=pipeline_type)

    # Configure pipeline with user settings
    try:
        config = PipelineConfig(
            mode=AnalysisMode(mode_str),
            frame_interval=frame_interval,
            max_frames=max_frames,
            use_sam2=use_sam2,
            use_deepforest=use_deepforest,
            use_vit=use_vit,
            save_debug_frames=save_debug
        )
    except Exception as e:
        error = f"Configuration error: {e}"
        return render_template('result.html', error=error, src=src, stats=None, plan=None,
                             report_paths=None, pipeline_type=pipeline_type)

    # Select and instantiate the appropriate pipeline
    try:
        if pipeline_type == 'cattle':
            if CattleAnalysisPipeline is None:
                error = "Cattle pipeline not available (import failed)."
                return render_template('result.html', error=error, src=src, stats=None, plan=None,
                                     report_paths=None, pipeline_type=pipeline_type)
            pipeline = CattleAnalysisPipeline(config=config)
        else:  # forest
            if EnhancedForestAnalysisPipeline is None:
                error = "Forest pipeline not available (import failed)."
                return render_template('result.html', error=error, src=src, stats=None, plan=None,
                                     report_paths=None, pipeline_type=pipeline_type)
            pipeline = EnhancedForestAnalysisPipeline(config=config)
    except Exception as e:
        error = f"Pipeline initialization error: {e}"
        return render_template('result.html', error=error, src=src, stats=None, plan=None,
                             report_paths=None, pipeline_type=pipeline_type)

    try:
        plan = pipeline.process_video(src)
    except Exception as e:
        # Render error message
        import traceback
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        return render_template('result.html', error=error_detail, src=src, stats=None, plan=None,
                             report_paths=None, pipeline_type=pipeline_type)

    # Try to serialize the plan and aggregated stats
    stats = getattr(pipeline, 'aggregated_stats', None)

    # Convert plan to JSON-friendly structure if possible
    plan_obj = None
    try:
        import dataclasses
        plan_obj = dataclasses.asdict(plan)
    except Exception:
        try:
            plan_obj = plan.__dict__
        except Exception:
            plan_obj = str(plan)

    # Generate reports if requested
    report_paths = None
    if generate_reports and stats:
        try:
            if pipeline_type == 'cattle':
                # Use cattle-specific report generator
                from src.cattle_report_generator import CattleReportGenerator
                report_gen = CattleReportGenerator(output_dir=config.output_dir)
                report_paths = report_gen.generate_all_reports(
                    cattle_plan=plan,
                    aggregated_stats=stats,
                    video_path=src
                )
            else:
                # Use forest report generator
                from src import ReportGenerator
                report_gen = ReportGenerator(output_dir=config.output_dir)
                report_paths = report_gen.generate_all_reports(
                    forest_plan=plan,
                    aggregated_stats=stats,
                    video_path=src
                )
            # Convert Path objects to strings for JSON serialization
            report_paths = {k: str(v) for k, v in report_paths.items()}
        except Exception as e:
            print(f"Warning: Report generation failed: {e}")
            import traceback
            traceback.print_exc()

    return render_template('result.html', error=None, src=src, stats=stats, plan=plan_obj,
                         report_paths=report_paths, pipeline_type=pipeline_type)


if __name__ == '__main__':
    # For development only. Use a proper WSGI server for production.
    # Disable reloader on Windows to avoid socket errors
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
