
import sys, json, time, threading, subprocess

def vram_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL)
        return int(out.decode(errors="replace").strip().splitlines()[0])
    except Exception:
        return -1

def main():
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
    res = {"model": cfg["model"], "compute_type": cfg["compute_type"], "ok": False}
    idle = vram_used_mb()
    peak = [idle]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            v = vram_used_mb()
            if v > peak[0]:
                peak[0] = v
            time.sleep(0.15)

    threading.Thread(target=poll, daemon=True).start()
    try:
        from faster_whisper import WhisperModel
        t0 = time.time()
        model = WhisperModel(cfg["model"], device="cuda", compute_type=cfg["compute_type"])
        load_s = time.time() - t0
        def one_pass():
            t = time.time()
            segments, info = model.transcribe(
                cfg["audio"], beam_size=cfg["beam_size"], language=cfg["language"])
            n = 0
            for _ in segments:
                n += 1
            return time.time() - t, float(info.duration), n

        one_pass()  # pemanasan, hasilnya dibuang
        times = []
        dur, nseg = 0.0, 0
        for _ in range(int(cfg.get("repeats", 3))):
            dt, dur, nseg = one_pass()
            times.append(dt)
        times.sort()
        proc_s = times[len(times) // 2]
        res.update(ok=True,
                   load_s=round(load_s, 2),
                   proc_s=round(proc_s, 2),
                   audio_s=round(dur, 2),
                   rtf=round(proc_s / dur, 4),
                   rtf_min=round(times[0] / dur, 4),
                   rtf_max=round(times[-1] / dur, 4),
                   n_repeats=len(times),
                   n_segments=nseg)
    except Exception as e:
        res.update(ok=False, error=(type(e).__name__ + ": " + str(e))[:400])
    finally:
        stop.set()
        time.sleep(0.4)
        res["vram_idle_mb"] = idle
        res["vram_peak_mb"] = peak[0]
        res["vram_delta_mb"] = (peak[0] - idle) if idle >= 0 else -1
    sys.stdout.write("###RESULT###" + json.dumps(res))

main()
