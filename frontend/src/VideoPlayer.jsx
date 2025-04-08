import { useEffect, useRef, useState } from "react";
import videojs from "video.js";
import "@videojs/http-streaming";
import "video.js/dist/video-js.css";

const VideoPlayer = ({ src }) => {
  const videoRef = useRef(null);
  const playerRef = useRef(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    console.log("VideoPlayer: Loading source:", src);
    setErrorMessage("");

    // Always recreate the player for better performance with HLS
    if (playerRef.current) {
      // Dispose old player if reinitializing
      playerRef.current.dispose();
      playerRef.current = null;
    }

    const videoElement = videoRef.current;
    if (!videoElement) return;

    // Configure videojs with optimized settings for HLS
    const options = {
      controls: true,
      autoplay: false,
      preload: "auto",
      fluid: true,
      responsive: true,
      liveui: true, // Better UI for live streams
      html5: {
        vhs: {
          overrideNative: true,
          useNetworkInformationApi: true,
          withCredentials: false,
          handlePartialData: true,
          useBandwidthFromLocalStorage: true,
          // Low latency mode settings
          lowLatencyMode: true,
          // Buffering tuning
          backBufferLength: 30, // in seconds
          liveSyncDuration: 2, // sync window for live streams
          liveMaxLatencyDuration: 6, // maximum buffer in live streams
          maxBufferLength: 10, // maximum buffer length in seconds
          bufferWhilePaused: false, // don't buffer when paused
        },
        nativeVideoTracks: false,
        nativeAudioTracks: false,
        nativeTextTracks: false,
      },
      liveTracker: {
        trackingThreshold: 0.5, // in seconds
        liveTolerance: 10, // tolerance before we consider ourselves behind live
      },
      sources: [
        {
          src: src,
          type: "application/x-mpegURL", // m3u8 content type
        },
      ],
    };

    // Initialize player
    const player = videojs(videoElement, options);
    playerRef.current = player;

    // Add restart button for better UX
    const Button = videojs.getComponent("Button");
    const RestartButton = videojs.extend(Button, {
      constructor: function () {
        Button.apply(this, arguments);
        this.controlText("Restart Stream");
      },
      handleClick: function () {
        console.log("Restarting stream");
        setRetryCount((c) => c + 1);
        player.src({
          src: src + "?t=" + new Date().getTime(), // Force reload
          type: "application/x-mpegURL",
        });
        player.play();
      },
    });

    videojs.registerComponent("RestartButton", RestartButton);
    player.controlBar.addChild(
      "RestartButton",
      {},
      player.controlBar.children_.length - 1
    );

    // Improve error handling
    player.on("error", () => {
      const error = player.error();
      const errorMsg = error
        ? `Error ${error.code}: ${error.message}`
        : "Unknown video error";
      console.error("VideoJS Error:", errorMsg);
      setErrorMessage(errorMsg);

      // Auto-recover from non-fatal errors
      if (error && error.code !== 4) {
        console.log("Attempting to recover from error...");
        setTimeout(() => {
          player.src({
            src: src + "?t=" + new Date().getTime(), // Force reload
            type: "application/x-mpegURL",
          });
          player.play().catch((e) => console.error("Recovery play failed:", e));
        }, 2000);
      }
    });

    // Add debug event listeners
    player.on("loadstart", () => console.log("Video loadstart event fired"));
    player.on("loadedmetadata", () =>
      console.log("Video loadedmetadata event fired")
    );
    player.on("loadeddata", () => console.log("Video loadeddata event fired"));
    player.on("canplay", () => console.log("Video canplay event fired"));

    // Handle buffering events
    player.on("waiting", () => {
      console.log("Video waiting for data");
      // Show loading spinner
      player.addClass("vjs-waiting");
    });

    player.on("playing", () => {
      // Hide loading spinner
      player.removeClass("vjs-waiting");
      console.log("Video playing");
    });

    player.on("stalled", () => {
      console.log("Video playback stalled");
    });

    // Handle HLS specific events
    const tech = player.tech({ IWillNotUseThisInPlugins: true });
    if (tech && tech.vhs) {
      tech.vhs.on("mediaupdatetimeout", () => {
        console.log("Media update timeout - reloading manifest");
        tech.vhs.playlistController_.mainPlaylistLoader_.load();
      });
    }

    return () => {
      // Cleanup
      if (playerRef.current) {
        playerRef.current.dispose();
        playerRef.current = null;
      }
    };
  }, [src, retryCount]);

  return (
    <div data-vjs-player className="video-player-wrapper">
      <video
        ref={videoRef}
        className="video-js vjs-big-play-centered vjs-default-skin"
        playsInline
      />
      <div className="player-controls">
        {errorMessage && (
          <div className="error-message-container">
            <p className="error-message">Error: {errorMessage}</p>
            <button
              className="restart-btn"
              onClick={() => setRetryCount((c) => c + 1)}
            >
              Restart Stream
            </button>
          </div>
        )}
      </div>
      <div className="debug-info">
        <p>Current source: {src}</p>
      </div>
    </div>
  );
};

export default VideoPlayer;
