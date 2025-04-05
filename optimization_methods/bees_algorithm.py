import numpy as np
from .base import OptimizationMethod
import random


class BeesAlgorithm(OptimizationMethod):
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        super().__init__(f, initial_point, max_iterations, **kwargs)
        self.scoutbeecount = self.kwargs.get("scoutbeecount", 300)
        self.selectedbeecount = self.kwargs.get("selectedbeecount", 10)
        self.bestbeecount = self.kwargs.get("bestbeecount", 30)
        self.selsitescount = self.kwargs.get("selsitescount", 15)
        self.bestsitescount = self.kwargs.get("bestsitescount", 5)
        self.range_lower = self.kwargs.get("range_lower", -5.12)
        self.range_upper = self.kwargs.get("range_upper", 5.12)
        self.range_shrink = self.kwargs.get("range_shrink", 0.98)
        self.max_stagnation = self.kwargs.get("max_stagnation", 10)

    class FloatBee:
        def __init__(self, outer):
            self.minval = [outer.range_lower] * 2
            self.maxval = [outer.range_upper] * 2
            self.position = np.array([random.uniform(self.minval[n], self.maxval[n]) for n in range(2)])
            self.fitness = outer.f(self.position[0], self.position[1])

        def calcfitness(self, outer):
            self.fitness = outer.f(self.position[0], self.position[1])

        def goto(self, otherpos, range_list, outer):
            self.position = np.array([otherpos[n] + random.uniform(-range_list[n], range_list[n])
                                    for n in range(len(otherpos))])
            self.checkposition()
            self.calcfitness(outer)

        def gotorandom(self, outer):
            self.position = np.array([random.uniform(self.minval[n], self.maxval[n])
                                    for n in range(2)])
            self.checkposition()
            self.calcfitness(outer)

        def checkposition(self):
            self.position = np.clip(self.position, self.minval, self.maxval)

        def otherpatch(self, bee_list, range_list):
            if not bee_list:
                return True
            for curr_bee in bee_list:
                position = curr_bee.position
                if any(abs(self.position[n] - position[n]) > range_list[n] for n in range(2)):
                    return True
            return False

    def run(self):
        # Инициализация улья
        beecount = self.scoutbeecount + self.selectedbeecount * self.selsitescount + self.bestbeecount * self.bestsitescount
        swarm = [self.FloatBee(self) for _ in range(beecount)]
        range_list = [(self.range_upper - self.range_lower) / 2] * 2
        best_fitness = float('inf')
        best_position = None
        trajectory = []
        iterations_log = []
        stagnation_counter = 0

        for iteration in range(self.max_iterations):
            # Сортировка пчел по fitness (меньше — лучше)
            swarm.sort(key=lambda x: x.fitness)
            if swarm[0].fitness < best_fitness:
                best_fitness = swarm[0].fitness
                best_position = swarm[0].position.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Логирование
            iterations_log.append(f"Итерация {iteration}: x={best_position}, f(x)={best_fitness}")
            trajectory.append(best_position.copy())

            # Выбор лучших и перспективных участков
            bestsites = [swarm[0]]
            curr_index = 1
            while len(bestsites) < self.bestsitescount and curr_index < len(swarm):
                if swarm[curr_index].otherpatch(bestsites, range_list):
                    bestsites.append(swarm[curr_index])
                curr_index += 1

            selsites = []
            while len(selsites) < self.selsitescount and curr_index < len(swarm):
                if (swarm[curr_index].otherpatch(bestsites, range_list) and
                    swarm[curr_index].otherpatch(selsites, range_list)):
                    selsites.append(swarm[curr_index])
                curr_index += 1

            # Отправка пчел
            bee_index = 1
            for best_bee in bestsites:
                for _ in range(self.bestbeecount):
                    if bee_index >= len(swarm):
                        break
                    if swarm[bee_index] not in bestsites and swarm[bee_index] not in selsites:
                        swarm[bee_index].goto(best_bee.position, range_list, self)
                    bee_index += 1

            for sel_bee in selsites:
                for _ in range(self.selectedbeecount):
                    if bee_index >= len(swarm):
                        break
                    if swarm[bee_index] not in bestsites and swarm[bee_index] not in selsites:
                        swarm[bee_index].goto(sel_bee.position, range_list, self)
                    bee_index += 1

            for bee in swarm[bee_index:]:
                bee.gotorandom(self)

            # Сужение диапазона при стагнации
            if stagnation_counter >= self.max_stagnation:
                range_list = [r * self.range_shrink for r in range_list]
                stagnation_counter = 0

        return best_position, trajectory, "Bees Algorithm завершён", iterations_log
