FROM atm26-nnunetv2:2.6.4-cuda11.8

LABEL org.atm26.batch="1"
LABEL org.grand-challenge.api-method="exec"

ENV PYTHONUNBUFFERED=1
ENV PATH="/home/user/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH=/opt/app
ENV MPLCONFIGDIR=/tmp/mplconfig
ENV XDG_CACHE_HOME=/tmp/cache
ENV HOME=/tmp/home
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

USER root
RUN rm -rf /opt/app
WORKDIR /opt/app

COPY --chown=user:user requirements.txt constraints.txt /opt/app/
RUN python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --constraint /opt/app/constraints.txt \
    -r /opt/app/requirements.txt

COPY --chown=user:user nnunet_trainers/airway_trainers.py /tmp/airway_trainers.py
COPY --chown=user:user nnunet_trainers/track2_trainers.py /tmp/track2_trainers.py
RUN TDIR="$(python -c 'import os,nnunetv2.training.nnUNetTrainer as m; print(os.path.dirname(m.__file__))')" \
    && echo "installing custom trainers into ${TDIR}" \
    && cp /tmp/airway_trainers.py "${TDIR}/airway_trainers.py" \
    && cp /tmp/track2_trainers.py "${TDIR}/track2_trainers.py" \
    && chown user:user "${TDIR}/airway_trainers.py" "${TDIR}/track2_trainers.py" \
    && rm /tmp/airway_trainers.py /tmp/track2_trainers.py

COPY --chown=user:user resources /opt/app/resources

COPY --chown=user:user airway /opt/app/airway
COPY --chown=user:user inference.py /opt/app/inference.py

RUN test -n "$(ls -d /opt/app/resources/nnUNet_ckpts/fold_* 2>/dev/null)" \
    && test -n "$(ls -d /opt/app/resources/nnUNet_t2warm/fold_* 2>/dev/null)" \
    && python -c "from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name as f; \
assert f('nnUNetTrainerAirwayCB') is not None and f('nnUNetTrainerT2Warm') is not None"

RUN mkdir -p /tmp/mplconfig /tmp/cache /tmp/home && chown -R user:user /tmp/mplconfig /tmp/cache /tmp/home

USER user
WORKDIR /opt/app

ENTRYPOINT ["python", "/opt/app/inference.py"]
