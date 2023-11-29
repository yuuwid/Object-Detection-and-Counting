

# ENUM STATE TRACK
class TrackState:
    TRIAL = 1
    CONFIRMED = 2
    DELETED = 3


class Track:

    def __init__(self, mean, covariance,
                 track_id, class_id, n_init,
                 max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.class_id = class_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.yolo_bbox = [0, 0, 0, 0]

        self.state = TrackState.TRIAL
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    # tlwh = Top-Left Width-Height
    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    # tlbr = Top-Left Bottom-Right

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def get_yolo_pred(self):
        return self.yolo_bbox

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment_age()

    def update(self, kf, detection, class_id):
        self.yolo_bbox = detection
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.class_id = class_id

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TRIAL and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        if self.state == TrackState.TRIAL:
            self.state = TrackState.DELETED
        elif self.time_since_update > self._max_age:
            self.state = TrackState.DELETED

    def is_trial(self):
        return self.state == TrackState.TRIAL

    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED

    def is_deleted(self):
        return self.state == TrackState.DELETED
