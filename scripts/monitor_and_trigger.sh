#!/usr/bin/env bash
# Polls squeue every 60 minutes; when a queue (regrid or coarsen) is empty,
# runs the corresponding um-slurm-control command.
#
# Usage (run from the repo root, NOT from scripts/):
#   cd /path/to/um_to_healpix
#   bash scripts/monitor_and_trigger.sh CTC_km4p4_CoMA9_TBv1.n2560_CoMA9_hier_v2
#
# Must be run from the repo root so that um-slurm-control can locate config/.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <model_name> [poll_interval_seconds]" >&2
    echo "  e.g. $0 CTC_km4p4_CoMA9_TBv1.n2560_CoMA9_hier_v2" >&2
    echo "  e.g. $0 CTC_km4p4_CoMA9_TBv1.n2560_CoMA9_hier_v2 1800" >&2
    exit 1
fi

MODEL_NAME="$1"
POLL_INTERVAL="${2:-3600}"  # default 60 minutes
REGRID_CMD="um-slurm-control process $MODEL_NAME"
COARSEN_CMD="um-slurm-control coarsen $MODEL_NAME"
LOG_FILE="${LOG_FILE:-${HOME}/monitor_and_trigger.log}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

queue_is_empty() {
    local name="$1"
    local job_count
    job_count=$(squeue --name="$name" --user="$USER" --noheader 2>/dev/null | wc -l)
    [[ "$job_count" -eq 0 ]]
}

check_and_trigger() {
    local queue_name="$1"
    local trigger_cmd="$2"
    if queue_is_empty "$queue_name"; then
        log "[$queue_name] Queue is empty — firing trigger."
        if $trigger_cmd >> "$LOG_FILE" 2>&1; then
            log "[$queue_name] Trigger command completed successfully."
        else
            log "[$queue_name] WARNING: Trigger command exited with status $?."
        fi
    else
        log "[$queue_name] Jobs present in queue — no action."
    fi
}

log "Starting monitor (PID $$, user=$USER, model=$MODEL_NAME, interval=${POLL_INTERVAL}s)"
log "Regrid trigger command: $REGRID_CMD"
log "Coarsen trigger command: $COARSEN_CMD"
log "Log file: $LOG_FILE"

while true; do
    check_and_trigger "regrid"  "$REGRID_CMD"
    check_and_trigger "coarsen" "$COARSEN_CMD"

    log "Sleeping ${POLL_INTERVAL}s until next check."
    sleep "$POLL_INTERVAL"
done
