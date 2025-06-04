#include "my_libraries.h"

struct Equipment
{
    int experience;
    int count;
};

int calculateTotalExperience(int targetLevel)
{
    long long n = targetLevel;
    return (n * (100 + n * 100)) / 2;
}

vector<vector<int>> setupDP(const vector<Equipment> &equipment, int totalExp)
{
    int n = equipment.size();
    vector<vector<int>> dp(n + 1, vector<int>(totalExp + 1, INT_MAX));

    // 初始化
    dp[0][0] = 0;

    // 填充dp表
    for (int i = 1; i <= n; ++i)
    {
        for (int exp = 0; exp <= totalExp; ++exp)
        {
            dp[i][exp] = dp[i - 1][exp]; // 不使用当前装备
            int maxEquipCount = min(equipment[i - 1].count, exp / equipment[i - 1].experience);
            for (int k = 1; k <= maxEquipCount; ++k)
            {
                int previousExp = exp - k * equipment[i - 1].experience;
                if (previousExp >= 0 && dp[i - 1][previousExp] != INT_MAX)
                {
                    dp[i][exp] = min(dp[i][exp], dp[i - 1][previousExp] + k);
                }
            }
        }
    }

    return dp;
}

int main()
{
    vector<Equipment> equipment = {
        {15, 65},  // Quality 1
        {300, 32}, // Quality 2
        {400, 25}, // Quality 3
        {600, 13}, // Quality 4
        {800, 5}   // Quality 5
    };

    int targetLevel = 200;
    int totalExpNeeded = calculateTotalExperience(targetLevel);
    cout << "Total experience needed to reach level " << targetLevel << ": " << totalExpNeeded << endl;

    auto dp = setupDP(equipment, totalExpNeeded);

    if (dp[equipment.size()][totalExpNeeded] == INT_MAX)
    {
        cout << "It is not possible to reach the target experience with the given equipment." << endl;
    }
    else
    {
        cout << "Minimum number of equipment pieces needed: " << dp[equipment.size()][totalExpNeeded] << endl;
    }

    system("pause");
}