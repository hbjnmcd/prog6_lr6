import timeit
import time
import matplotlib.pyplot as plt
import numpy as np
from ferma_fact import fermat_factorization as py_ferma
from ferma_fact_pyx import fermat_factorization as cy_ferma
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


def timed_run(func, data, mode="process", workers=4):
    start = time.time()
    if mode == "process":
        with Pool(workers) as pool:
            results = pool.map(func, data)
    elif mode == "thread":
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(func, data))
    else:
        raise ValueError("Mode must be 'process' or 'thread'")
    elapsed = time.time() - start
    return elapsed, results


TEST_LST = [101, 9973, 104729, 101909, 609133, 1300039]
MODES = ['thread', 'process']
FUNCS = [('Python', py_ferma), ('Cython', cy_ferma)]

if __name__ == '__main__':
    results = {}

    py_times = []
    for n in TEST_LST:
        t = timeit.timeit(lambda: py_ferma(n), number=10)
        py_times.append(t)
        print(f"Python: {n} - {t:.5f} сек")
    py_sum_time = sum(py_times)
    print(f"Суммарное время: {py_sum_time:.5f}\n")

    cy_times = []
    for n in TEST_LST:
        t = timeit.timeit(lambda: cy_ferma(n), number=10)
        cy_times.append(t)
        print(f"Cython: {n} - {t:.5f} сек")
    cy_sum_time = sum((cy_times))
    print(f"Суммарное время: {cy_sum_time:.5f}\n")

    for name, func in FUNCS:
        for mode in MODES:
            key = f"{name}_{mode}"
            t, _ = timed_run(func, TEST_LST, mode=mode, workers=4)
            results[key] = t
            print(f"{key}: {t:.5f} сек\n")


    # График для сравнени py и c
    x = np.arange(len(TEST_LST))  # позиции по X
    width = 0.35  # ширина столбика

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, py_times, width, label='Python', color='blue')
    plt.bar(x + width/2, cy_times, width, label='Cython', color='green')

    plt.xlabel('Число')
    plt.ylabel('Время (сек.)')
    plt.title('Сравнение времени факторизации методом Ферма (Python vs Cython)')
    plt.xticks(ticks=x, labels=[str(n) for n in TEST_LST], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    #График сравнения потока и процесса
    labels = list(results.keys())
    times = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, times, color=['blue', 'green', 'orange', 'red'])
    plt.title('Время выполнения: Потоки vs Процессы (Python и Cython)')
    plt.ylabel('Секунды')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()