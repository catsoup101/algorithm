# # -*- coding: utf-8 -*-
# import asyncio
# import random
# from typing import List
# import numpy as np
# import matplotlib.pyplot as plt

# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False


# class Equipment:
#     def __init__(
#         self,
#         slot: str,
#         attack: float = 0.0,
#         defense: float = 0.0,
#         health: float = 0.0,
#         attack_speed: float = 0.0,
#         crit_rate: float = 0.0,
#         lifesteal: float = 0.0,
#         dodge: float = 0.0,
#         stun_rate: float = 0.0,
#         counter_attack_rate: float = 0.0,
#         extra_attack_rate: float = 0.0,
#     ):
#         # 基础属性
#         self.slot = slot
#         self.attack = attack
#         self.defense = defense
#         self.health = health
#         self.attack_speed = attack_speed

#         # 特殊属性
#         self.crit_rate = crit_rate
#         self.lifesteal = lifesteal
#         self.dodge = dodge
#         self.stun_rate = stun_rate
#         self.counter_attack_rate = counter_attack_rate
#         self.extra_attack_rate = extra_attack_rate


# class Character:
#     def __init__(
#         self,
#         attack: float = 0.0,
#         defense: float = 0.0,
#         health: float = 0.0,
#         attack_speed: float = 0.0,
#         crit_rate: float = 0.0,
#         lifesteal: float = 0.0,
#         dodge: float = 0.0,
#         stun_rate: float = 0.0,
#         counter_attack_rate: float = 0.0,
#         extra_attack_rate: float = 0.0,
#     ):
#         # 基础属性
#         self.attack = attack
#         self.defense = defense
#         self.health = health
#         self.attack_speed = attack_speed

#         # 特殊属性
#         self.crit_rate = crit_rate
#         self.lifesteal = lifesteal
#         self.dodge = dodge
#         self.stun_rate = stun_rate
#         self.counter_attack_rate = counter_attack_rate
#         self.extra_attack_rate = extra_attack_rate

#         self.stunned = 0  # 眩晕轮数
#         self.equipment = {}  # 装备字典，键为部位

#     def is_alive(self) -> bool:
#         return self.health > 0

#     def is_stunned(self) -> bool:
#         return self.stunned > 0

#     def equip(self, equipment: Equipment):
#         self.equipment[equipment.slot] = equipment
#         self.recalculate_attributes()

#     def recalculate_attributes(self):
#         # 每次穿戴装备后重新计算属性，先重置为初始值
#         self.attack = 100
#         self.defense = 80
#         self.health = 400
#         self.attack_speed = 1.0
#         self.crit_rate = 0.0
#         self.lifesteal = 0.0
#         self.dodge = 0.0
#         self.stun_rate = 0.0
#         self.counter_attack_rate = 0.0
#         self.extra_attack_rate = 0.0

#         for eq in self.equipment.values():
#             self.attack += eq.attack
#             self.defense += eq.defense
#             self.health += eq.health
#             self.attack_speed += eq.attack_speed
#             self.crit_rate += eq.crit_rate
#             self.lifesteal += eq.lifesteal
#             self.dodge += eq.dodge
#             self.stun_rate += eq.stun_rate
#             self.counter_attack_rate += eq.counter_attack_rate
#             self.extra_attack_rate += eq.extra_attack_rate

#     async def take_action(
#         self, other: "Character", damage_calculator, stats: dict, role: str
#     ):
#         if self.is_stunned():
#             self.stunned -= 1
#             return
#         await damage_calculator.execute_damage_flow(self, other, stats, role)


# class DamageCalculator:
#     async def execute_damage_flow(
#         self, attacker: Character, defender: Character, stats: dict, role: str
#     ):
#         # 处理单次攻击的完整流程
#         for _i in range(6):  # 连击最多6次
#             stats[f"{role}_total_attacks"] += 1

#             # 计算基础伤害
#             damage = self.calculate_damage(attacker, defender)

#             # 计算暴击
#             if random.random() < attacker.crit_rate:
#                 damage *= 2
#                 stats[f"{role}_crits"] += 1

#             # 计算闪避
#             if random.random() < defender.dodge:
#                 stats[f"{role}_dodges"] += 1
#                 break  # 攻击流程结束

#             # 造成伤害
#             defender.health -= damage

#             # 计算吸血
#             healed_amount = damage * attacker.lifesteal
#             attacker.health += healed_amount
#             stats[f"{role}_total_lifesteal"] += healed_amount

#             # 计算眩晕
#             if random.random() < attacker.stun_rate and defender.is_alive():
#                 defender.stunned = 2
#                 stats[f"{role}_stuns"] += 1

#             # 计算反击
#             if random.random() < defender.counter_attack_rate and defender.is_alive():
#                 stats[f"{role}_counters"] += 1
#                 await self.execute_damage_flow(
#                     defender,
#                     attacker,
#                     stats,
#                     "defender" if role == "attacker" else "attacker",
#                 )
#                 break  # 攻击流程结束

#             # 计算连击
#             if random.random() < attacker.extra_attack_rate and defender.is_alive():
#                 stats[f"{role}_extra_attacks"] += 1
#                 continue  # 继续连击

#             break  # 攻击流程结束

#     def calculate_damage(self, attacker: Character, defender: Character) -> float:
#         # 计算基础伤害
#         return (
#             attacker.attack
#             * (1 / (1 + defender.defense / 100))
#             * (1 / max(0.2, 1 - 0.00001 * attacker.attack_speed))
#         )


# async def simulate_battle(
#     attacker: Character, defender: Character, damage_calculator: DamageCalculator
# ) -> dict:
#     stats = {
#         "rounds": 0,  # 记录战斗的回合数
#         "attacker_total_attacks": 0,  # 记录攻击方和防守方的总攻击次数
#         "defender_total_attacks": 0,
#         "attacker_crits": 0,  # 击方和防守方的暴击触发次数
#         "defender_crits": 0,
#         "attacker_dodges": 0,  # 记录攻击方和防守方的闪避触发次数
#         "defender_dodges": 0,
#         "attacker_stuns": 0,  # 记录攻击方和防守方的眩晕触发次数
#         "defender_stuns": 0,
#         "attacker_counters": 0,  # 记录攻击方和防守方的反击触发次数
#         "defender_counters": 0,
#         "attacker_extra_attacks": 0,  # 攻击方和防守方的连击触发次数
#         "defender_extra_attacks": 0,
#         "attacker_total_lifesteal": 0,  # 记录攻击方和防守方通过吸血恢复的总生命值
#         "defender_total_lifesteal": 0,
#         "attacker_starting_health": attacker.health,  # 分别记录攻击方和防守方在战斗开始时的初始生命值
#         "defender_starting_health": defender.health,
#     }

#     while attacker.is_alive() and defender.is_alive():
#         stats["rounds"] += 1

#         # 随机决定攻击顺序
#         if random.random() < 0.5:
#             task1 = asyncio.create_task(
#                 attacker.take_action(defender, damage_calculator, stats, "attacker")
#             )
#             task2 = asyncio.create_task(
#                 defender.take_action(attacker, damage_calculator, stats, "defender")
#             )
#         else:
#             task2 = asyncio.create_task(
#                 defender.take_action(attacker, damage_calculator, stats, "defender")
#             )
#             task1 = asyncio.create_task(
#                 attacker.take_action(defender, damage_calculator, stats, "attacker")
#             )

#         await asyncio.gather(task1, task2)

#     # 记录损失的生命值
#     stats["attacker_health_loss"] = stats["attacker_starting_health"] - attacker.health
#     stats["defender_health_loss"] = stats["defender_starting_health"] - defender.health

#     # 记录获胜方
#     stats["winner"] = "攻击方" if attacker.is_alive() else "防守方"

#     return stats


# async def simulate_multiple_battles(num_battles: int) -> List[dict]:
#     all_stats: List[dict] = []  # 存储每场战斗的统计结果
#     damage_calculator: DamageCalculator = DamageCalculator()  # 伤害流程计算

#     for _ in range(num_battles):
#         # 创建角色，初始属性
#         attacker = Character(100, 80, 400, 1)
#         defender = Character(100, 80, 400, 1)

#         # 攻击方装备
#         equipment_slot_1 = Equipment(slot="武器", attack=10, crit_rate=0.1)
#         equipment_slot_2 = Equipment(slot="副手", defense=15, counter_attack_rate=0.1)
#         equipment_slot_3 = Equipment(slot="头部", health=20, defense=5)
#         equipment_slot_4 = Equipment(slot="身体", health=50, defense=20)
#         equipment_slot_5 = Equipment(slot="脚部", dodge=0.2, attack_speed=0.1)
#         equipment_slot_6 = Equipment(slot="手部", attack=5, crit_rate=0.05)

#         # 穿戴装备
#         # attacker.equip(equipment_slot_1)
#         # attacker.equip(equipment_slot_2)
#         # attacker.equip(equipment_slot_3)
#         # attacker.equip(equipment_slot_4)
#         # attacker.equip(equipment_slot_5)
#         # attacker.equip(equipment_slot_6)

#         # 防守方装备
#         equipment_slot_7 = Equipment(slot="武器", attack=15, lifesteal=0.1)
#         equipment_slot_8 = Equipment(slot="副手", crit_rate=0.1, stun_rate=0.1)
#         equipment_slot_9 = Equipment(slot="头部", dodge=0.1, health=10)
#         equipment_slot_10 = Equipment(slot="身体", health=40, defense=10)
#         equipment_slot_11 = Equipment(slot="脚部", attack_speed=0.2, dodge=0.15)
#         equipment_slot_12 = Equipment(
#             slot="手部", lifesteal=0.05, extra_attack_rate=0.2
#         )

#         # 穿戴装备
#         # defender.equip(equipment_slot_7)
#         # defender.equip(equipment_slot_8)
#         # defender.equip(equipment_slot_9)
#         # defender.equip(equipment_slot_10)
#         # defender.equip(equipment_slot_11)
#         # defender.equip(equipment_slot_12)

#         # 模拟单场战斗
#         stats = await simulate_battle(attacker, defender, damage_calculator)
#         all_stats.append(stats)

#     return all_stats


# def analyze_results(all_stats: List[dict]):
#     # 分析10000次战斗的统计结果

#     # 胜率统计
#     attacker_wins = sum(1 for stat in all_stats if stat["winner"] == "攻击方")
#     defender_wins = sum(1 for stat in all_stats if stat["winner"] == "防守方")

#     # 平均数据
#     avg_rounds = np.mean([stat["rounds"] for stat in all_stats])
#     avg_attacker_total_attacks = np.mean(
#         [stat["attacker_total_attacks"] for stat in all_stats]
#     )
#     avg_defender_total_attacks = np.mean(
#         [stat["defender_total_attacks"] for stat in all_stats]
#     )
#     avg_attacker_crits = np.mean([stat["attacker_crits"] for stat in all_stats])
#     avg_defender_crits = np.mean([stat["defender_crits"] for stat in all_stats])
#     avg_attacker_dodges = np.mean([stat["attacker_dodges"] for stat in all_stats])
#     avg_defender_dodges = np.mean([stat["defender_dodges"] for stat in all_stats])
#     avg_attacker_stuns = np.mean([stat["attacker_stuns"] for stat in all_stats])
#     avg_defender_stuns = np.mean([stat["defender_stuns"] for stat in all_stats])
#     avg_attacker_counters = np.mean([stat["attacker_counters"] for stat in all_stats])
#     avg_defender_counters = np.mean([stat["defender_counters"] for stat in all_stats])
#     avg_attacker_extra_attacks = np.mean(
#         [stat["attacker_extra_attacks"] for stat in all_stats]
#     )
#     avg_defender_extra_attacks = np.mean(
#         [stat["defender_extra_attacks"] for stat in all_stats]
#     )
#     avg_attacker_total_lifesteal = np.mean(
#         [stat["attacker_total_lifesteal"] for stat in all_stats]
#     )
#     avg_defender_total_lifesteal = np.mean(
#         [stat["defender_total_lifesteal"] for stat in all_stats]
#     )
#     avg_attacker_health_loss = np.mean(
#         [stat["attacker_health_loss"] for stat in all_stats]
#     )
#     avg_defender_health_loss = np.mean(
#         [stat["defender_health_loss"] for stat in all_stats]
#     )

#     # 标准差数据
#     std_rounds = np.std([stat["rounds"] for stat in all_stats])
#     std_attacker_total_attacks = np.std(
#         [stat["attacker_total_attacks"] for stat in all_stats]
#     )
#     std_defender_total_attacks = np.std(
#         [stat["defender_total_attacks"] for stat in all_stats]
#     )
#     std_attacker_crits = np.std([stat["attacker_crits"] for stat in all_stats])
#     std_defender_crits = np.std([stat["defender_crits"] for stat in all_stats])
#     std_attacker_dodges = np.std([stat["attacker_dodges"] for stat in all_stats])
#     std_defender_dodges = np.std([stat["defender_dodges"] for stat in all_stats])
#     std_attacker_stuns = np.std([stat["attacker_stuns"] for stat in all_stats])
#     std_defender_stuns = np.std([stat["defender_stuns"] for stat in all_stats])
#     std_attacker_counters = np.std([stat["attacker_counters"] for stat in all_stats])
#     std_defender_counters = np.std([stat["defender_counters"] for stat in all_stats])
#     std_attacker_extra_attacks = np.std(
#         [stat["attacker_extra_attacks"] for stat in all_stats]
#     )
#     std_defender_extra_attacks = np.std(
#         [stat["defender_extra_attacks"] for stat in all_stats]
#     )
#     std_attacker_total_lifesteal = np.std(
#         [stat["attacker_total_lifesteal"] for stat in all_stats]
#     )
#     std_defender_total_lifesteal = np.std(
#         [stat["defender_total_lifesteal"] for stat in all_stats]
#     )
#     std_attacker_health_loss = np.std(
#         [stat["attacker_health_loss"] for stat in all_stats]
#     )
#     std_defender_health_loss = np.std(
#         [stat["defender_health_loss"] for stat in all_stats]
#     )

#     print(f"经过 {len(all_stats)} 次战斗模拟：")
#     print(
#         f"攻击方胜利次数: {attacker_wins}, 胜率: {attacker_wins / len(all_stats):.2%}"
#     )
#     print(
#         f"防守方胜利次数: {defender_wins}, 胜率: {defender_wins / len(all_stats):.2%}"
#     )
#     print(f"平均每场战斗回合数: {avg_rounds:.2f} (标准差: {std_rounds:.2f})")
#     print(
#         f"平均每场战斗攻击次数 - 攻击方: {avg_attacker_total_attacks:.2f} (标准差: {std_attacker_total_attacks:.2f}), 防守方: {avg_defender_total_attacks:.2f} (标准差: {std_defender_total_attacks:.2f})"
#     )
#     print(
#         f"暴击触发次数 - 攻击方: {avg_attacker_crits:.2f} (标准差: {std_attacker_crits:.2f}), 防守方: {avg_defender_crits:.2f} (标准差: {std_defender_crits:.2f})"
#     )
#     print(
#         f"闪避触发次数 - 攻击方: {avg_attacker_dodges:.2f} (标准差: {std_attacker_dodges:.2f}), 防守方: {avg_defender_dodges:.2f} (标准差: {std_defender_dodges:.2f})"
#     )
#     print(
#         f"眩晕触发次数 - 攻击方: {avg_attacker_stuns:.2f} (标准差: {std_attacker_stuns:.2f}), 防守方: {avg_defender_stuns:.2f} (标准差: {std_defender_stuns:.2f})"
#     )
#     print(
#         f"反击触发次数 - 攻击方: {avg_attacker_counters:.2f} (标准差: {std_attacker_counters:.2f}), 防守方: {avg_defender_counters:.2f} (标准差: {std_defender_counters:.2f})"
#     )
#     print(
#         f"连击触发次数 - 攻击方: {avg_attacker_extra_attacks:.2f} (标准差: {std_attacker_extra_attacks:.2f}), 防守方: {avg_defender_extra_attacks:.2f} (标准差: {std_defender_extra_attacks:.2f})"
#     )
#     print(
#         f"总吸血恢复量 - 攻击方: {avg_attacker_total_lifesteal:.2f} (标准差: {std_attacker_total_lifesteal:.2f}), 防守方: {avg_defender_total_lifesteal:.2f} (标准差: {std_defender_total_lifesteal:.2f})"
#     )
#     print(
#         f"损失的生命值 - 攻击方: {avg_attacker_health_loss:.2f} (标准差: {std_attacker_health_loss:.2f}), 防守方: {avg_defender_health_loss:.2f} (标准差: {std_defender_health_loss:.2f})"
#     )


# def main():
#     num_battles = 10000  # 模拟战斗次数
#     all_stats = asyncio.run(simulate_multiple_battles(num_battles))
#     analyze_results(all_stats)


# if __name__ == "__main__":
#     main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置常数
constant_new = 1_000_000


# 定义计算伤害函数
def compute_damage(attack, defense, attack_speed):
    attack_part = attack / (2 * (1 + defense / constant_new))
    speed_part = 1 / max(1 - 0.00002 * attack_speed, 0.2)
    return attack_part * speed_part


# 数据集
data = {
    "Group 1": {
        "攻击": [15, 10, 5, 10, 5, 15],
        "防御": [40, 17, 63, 40, 40, 17],
        "攻速": [2, 1, 1, 2, 1, 2],
    },
    "Group 2": {
        "攻击": [30, 20, 10, 20, 10, 30],
        "防御": [80, 33, 127, 80, 80, 33],
        "攻速": [4, 2, 2, 4, 2, 4],
    },
    "Group 3": {
        "攻击": [45, 30, 15, 30, 15, 45],
        "防御": [120, 50, 190, 120, 120, 50],
        "攻速": [6, 3, 3, 6, 3, 6],
    },
    "Group 4": {
        "攻击": [60, 40, 20, 40, 20, 60],
        "防御": [160, 66, 151, 160, 160, 66],
        "攻速": [8, 4, 4, 8, 4, 8],
    },
    "Group 5": {
        "攻击": [75, 50, 25, 50, 25, 75],
        "防御": [200, 83, 317, 200, 200, 83],
        "攻速": [10, 5, 5, 10, 5, 10],
    },
}


# 缩放函数
def scale_data(data, scale_factor):
    scaled_data = {}
    for key, values in data.items():
        scaled_data[key] = {
            column: [x * scale_factor for x in values[column]] for column in values
        }
    return scaled_data


# 读入并缩放数据
scale_factor = 500  # 可以修改这个值以探索不同缩放的影响
scaled_data = scale_data(data, scale_factor)

# 分组并计算伤害
groups = {}
for key, group_data in scaled_data.items():
    group_df = pd.DataFrame(group_data)
    group_df["伤害"] = group_df.apply(
        lambda row: compute_damage(row["攻击"], row["防御"], row["攻速"]), axis=1
    )
    groups[key] = group_df

# 绘制伤害曲线图
plt.figure(figsize=(12, 8))
for key, group_df in groups.items():
    plt.plot(group_df.index, group_df["伤害"], marker="o", label=key)
plt.xlabel("Index within Group")
plt.ylabel("Damage Output")
plt.title("Scaled Damage Output for Different Groups")
plt.legend()
plt.grid(True)
plt.show()
