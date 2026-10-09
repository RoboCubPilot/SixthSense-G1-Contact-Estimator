import numpy as np

def window_noise(seed, generation, end, candidate, shape):
    return np.random.default_rng(np.random.SeedSequence(
        [int(seed), int(generation), int(end), int(candidate)]
    )).standard_normal(shape, dtype=np.float32)

class CandidateSelector:
    def __init__(self):
        self.previous = None

    def prepare(self, generation, end):
        if self.previous is not None:
            g, old_end, _ = self.previous
            if g != generation or end <= old_end or end - old_end >= 50:
                self.previous = None

    @staticmethod
    def distance(a, b, ia, ib):

        ma, mb = a['link_contact_active'][ia], b['link_contact_active'][ib]
        union = (ma | mb).sum(-1)
        mask = np.divide((ma != mb).sum(-1), union,
                         out=np.zeros_like(union, dtype=float), where=union > 0)
        df = np.linalg.norm(a['force_field'][ia] - b['force_field'][ib], axis=-1).sum(-1)
        dt = np.linalg.norm(a['torque_field'][ia] - b['torque_field'][ib], axis=-1).sum(-1)
        return float(np.mean(mask + .25*np.minimum(df/40., 2.) + .10*np.minimum(dt/4., 2.)))

    def overlap(self, end):
        if self.previous is None:
            return None
        _, old_end, old = self.previous
        shift = end - old_end

        ia = np.arange(10, min(40, 50-shift))
        return (old, ia, ia+shift) if len(ia) else None

    def trigger(self, candidate, end):
        overlap = self.overlap(end)
        if overlap is None:
            return False, 'no_history'
        old, ia, ib = overlap
        offset = 24
        shift = end-self.previous[1]
        if offset+shift >= 50:
            return True, 'history_gap'
        if not np.array_equal(candidate['link_contact_active'][offset],
                              old['link_contact_active'][offset+shift]):
            return True, 'contact_set_disagreement'
        jump = np.linalg.norm(candidate['force_field'][offset]-old['force_field'][offset+shift],axis=-1).sum()
        if jump > 20.:
            return True, 'force_disagreement_gt20N'
        return False, 'consistent'

    def choose(self, candidates, generation, end):
        self.prepare(generation,end)
        n=len(candidates);scores=np.zeros(n)
        center=np.arange(10,40)
        if n > 1:
            for i in range(n):
                scores[i]=np.mean([self.distance(candidates[i],candidates[j],center,center)
                                   for j in range(n) if j != i])
        overlap=self.overlap(end)
        if overlap is not None:
            old,ia,ib=overlap
            scores += .6*np.array([self.distance(c,old,ia,ib) for c in candidates])
        index=int(np.argmin(scores))
        self.previous=(generation,end,candidates[index])
        return candidates[index],dict(selected_candidate=index,candidate_scores=scores.tolist())
