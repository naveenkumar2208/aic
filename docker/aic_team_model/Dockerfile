# Submission image: ACT policy (SfpInsertACT) + bundled checkpoint.
# Build from repo root (parent of docker/):
#   docker compose -f docker/docker-compose.yaml build team_model
#
# Push as v2 (ECR tags are immutable):
#   docker tag my-aic-act:v2 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/kd_robotics:v2
#
# Override checkpoint at build time:
#   docker compose -f docker/docker-compose.yaml build team_model \
#     --build-arg CHECKPOINT_DIR=outputs/train/act_aic_sfp_127ep/checkpoints/040000/pretrained_model

FROM docker.io/library/ros:kilted-ros-core AS build

RUN apt update && apt install -y git && curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=0.67.2 bash
ENV PATH="/root/.pixi/bin:$PATH"

COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_team_policy /ws_aic/src/aic/aic_team_policy
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock robostack_mapping.json /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/
COPY docker/aic_eval/aic_zenoh_config.json5 /aic_zenoh_config.json5

ARG CHECKPOINT_DIR=outputs/train/act_aic_sfp_127ep/checkpoints/040000/pretrained_model
COPY ${CHECKPOINT_DIR} /ws_aic/policy/pretrained_model

SHELL ["/bin/bash", "-c"]
# Host pixi.lock covers the full repo; this image only COPYs a subset of path packages.
RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi lock && pixi install --locked
# Pixi caches the team-policy wheel; overwrite with the sources we just COPY'd.
RUN cd /ws_aic/src/aic && \
    SITE_PKG="$(find .pixi/envs/default/lib -type d -path '*/site-packages/aic_team_policy' | head -1)" && \
    test -n "${SITE_PKG}" && \
    cp -a aic_team_policy/aic_team_policy/. "${SITE_PKG}/"
# ACT vision backbone (resnet18) — eval cluster has no outbound DNS at runtime.
RUN cd /ws_aic/src/aic && pixi run --as-is python -c "\
from torchvision.models import ResNet18_Weights, resnet18; \
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); \
print('Cached ResNet18 weights')"
# Sanity-check bundled checkpoint is present.
RUN test -f /ws_aic/policy/pretrained_model/model.safetensors && \
    test -f /ws_aic/policy/pretrained_model/config.json

WORKDIR /ws_aic/src/aic
ENV AIC_ACT_POLICY_PRETRAINED_PATH=/ws_aic/policy/pretrained_model
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_SESSION_CONFIG_URI=/aic_zenoh_config.json5

# Production (see docs/custom_dockerfile.md) sets AIC_MODEL_ROUTER_ADDR.
ROUTER_ADDR="${AIC_MODEL_ROUTER_ADDR:-${AIC_ROUTER_ADDR:-}}"
if [[ -z "$ROUTER_ADDR" ]]; then
  echo "AIC_MODEL_ROUTER_ADDR or AIC_ROUTER_ADDR must be provided"
  exit 1
fi

should_use_zenoh_auth() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]] && \
    [[ -n "${AIC_MODEL_PASSWD:-}" ]]
}

# Production portal sets AIC_MODEL_PASSWD; eval router uses ACL even if AIC_ENABLE_ACL
# is unset on the model container. Set AIC_ZENOH_AUTH=off only for no-auth local tests.
if should_use_zenoh_auth || \
   { [[ -n "${AIC_MODEL_PASSWD:-}" && "${AIC_ZENOH_AUTH:-}" != "off" ]]; }; then
  echo "model:${AIC_MODEL_PASSWD}" > /credentials.txt
fi

# Client mode: connect only to the eval router (separate container from eval peers).
ZENOH_CONFIG_OVERRIDE='mode="client"'
ZENOH_CONFIG_OVERRIDE+=';connect/endpoints=["tcp/'"$ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';scouting/gossip/enabled=false'
ZENOH_CONFIG_OVERRIDE+=';scouting/multicast/enabled=false'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if [[ -f /credentials.txt ]]; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"

wait_for_router() {
  local host="${ROUTER_ADDR%%:*}"
  local port="${ROUTER_ADDR##*:}"
  local attempts="${ZENOH_ROUTER_CHECK_ATTEMPTS:-180}"
  echo "Waiting for Zenoh router at ${host}:${port} (attempts=${attempts}) ..."
  local n=1
  while true; do
    if timeout 1 bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null; then
      echo "Zenoh router is reachable."
      sleep 2
      return 0
    fi
    if [[ "$attempts" != "-1" && "$n" -ge "$attempts" ]]; then
      echo "ERROR: Zenoh router not reachable at ${ROUTER_ADDR}" >&2
      return 1
    fi
    sleep 1
    n=$((n + 1))
  done
}
wait_for_router

# Warm policy import before rclpy init (see docs/troubleshooting.md).
pixi run --as-is python -c "import aic_team_policy.ros.SfpInsertACT" >/dev/null 2>&1 || true

exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_team_policy.ros.SfpInsertACT", "-p", "use_sim_time:=true"]
