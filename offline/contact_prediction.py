import numpy as np

class ContactPrediction:
    @classmethod
    def from_result(cls, result):
        value = cls.__new__(cls)
        value.links = result["target_links"]
        value.mode = result["mode"]
        value.model_sha256 = result["model_sha256"]
        value.field = result["field"]
        value.torque_field = result["torque_field"]
        if value.torque_field.shape != value.field.shape:
            raise ValueError("Force and torque fields must share [frames,30,3] layout")
        value.event_valid = result["observation_valid"]
        value.valid = result["force_observation_valid"]
        value.force_latency_s = result["force_latency_s"]
        value.contact_active = result["contact_active"]
        value.localization_link_idx = result["localization_link_idx"]
        value.link_idx = np.where(value.valid, value.localization_link_idx, -1)
        value.force = value.field[np.arange(len(value.field)), value.link_idx.clip(0)].copy()
        value.force[value.link_idx < 0] = 0
        value.torque = value.torque_field[np.arange(len(value.field)), value.link_idx.clip(0)].copy()
        value.torque[value.link_idx < 0] = 0
        value.magnitude = np.linalg.norm(value.force, axis=-1)
        return value

    def payload(self):
        return dict(
            force_pelvis_N=self.force,
            torque_pelvis_Nm=self.torque,
            magnitude_N=self.magnitude,
            event_valid=self.event_valid,
            force_latency_s=self.force_latency_s,
            link_idx=self.link_idx,
            target_links=self.links,
            valid=self.valid,
            localization_link_idx=self.localization_link_idx,
            mode=self.mode,
            model_sha256=self.model_sha256,
            anchor="predicted link body origin; contact point is not predicted",
            moment_origin="link body origin",
            wrench_frame="pelvis",
            direct_output=True,
            confidence_output=False,
            contact_active=self.contact_active,
            dense_output=True,
            force_field_pelvis_N=self.field,
            torque_field_pelvis_Nm=self.torque_field,
            wrench_pelvis_N_Nm=np.concatenate([self.force, self.torque], axis=-1),
            wrench_field_pelvis_N_Nm=np.concatenate([self.field, self.torque_field], axis=-1),
            torque_source="learned conditional wrench flow",
            link_contact_active=np.linalg.norm(self.field, axis=-1) > 0,
        )
