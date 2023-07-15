import numpy as np

from utils.ReplayBuffer import PERBuffer



if __name__ == '__main__':
    buffer = PERBuffer((1,), (1,), (1,), (1,), 64)
    buffer.beta = buffer.alpha = 1

    probs = np.array([0.4, 0.1, 0.3, 0.2])

    for p in probs:
        buffer.store(0, 0, 0, 0, 0)

    buffer.update_batch(np.arange(len(probs)), probs)
    sample_times = 100000

    def test_hits(sample_times, batch_size=25):
        hits = {}
        for i in range(sample_times):
            expr, is_weights, batch_idx = buffer.sample(batch_size=batch_size)
            for sample in batch_idx:
                hits[sample] = hits.get(sample, 0) + 1

        for i, (k, v) in enumerate(hits.items()):
            print(f'{k=} with designed prob {probs[k]}, hits {v/(sample_times*batch_size)}')


    # before
    test_hits(sample_times)

    probs = np.array([0.2, 0.5, 0.25, 0.05])
    buffer.update_batch(np.arange(len(probs)), probs)

    print(f'Updating batch to {probs}')
    test_hits(sample_times)


    pass
