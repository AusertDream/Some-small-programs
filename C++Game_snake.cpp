#include<cstdio>
#include<iostream>
#include<ctime>
#include<io.h>
#include<fcntl.h>
#include<stdlib.h>
#include<windows.h>
#include <conio.h>
#include<map>
#include<unordered_set>
#include<set>
#include<mutex>
#include<thread>
using namespace std;
typedef pair<int, int> PII;
#define frame_width 50
#define frame_height 25
const int INF = 0x3f3f3f3f;

typedef struct Food {
    int x, y;
    int score;
};
typedef struct Snake {
    int x[100]; //蛇身每一块的横坐标
    int y[100]; //蛇身每一块的纵坐标
    int len, state; //蛇长度，蛇头运动方向
    double speed; //蛇的速度 
    int score = 0;
    //简单的比较函数，用于set的正常使用，实际上没啥用
    bool operator < (const Snake& a) const {
        return score < a.score;
    }
    bool operator != (const Snake& other) const {
        if (len != other.len || state != other.state || speed != other.speed || score != other.score) {
            return true;
        }
        for (int i = 1; i <= len; i++) {
            if (x[i] != other.x[i] || y[i] != other.y[i]) {
                return true;
            }
        }
        return false;
    }
};

void gotoxy(int x, int y);  //最重要的一个函数，控制光标的位置
wchar_t getxy(int x, int y); //获取光标位置的字符
void print_map(); //打印地图
void get_newfood();//生成新食物
bool check_foodisok(Food);//检查新食物是否能够正常生成
void move_snake(Snake& snake);
void check_foodeating(Snake& snake, bool& isEaten);
bool check_snakealive(Snake& snake);
void SummonNewWall(); //生成新的墙壁
void AIsnake(); //AI蛇
void move_snake_ai(Snake& snake, bool& AIisEaten); //AI蛇移动
char AICheckWheretogo(Snake&); //AI蛇检查下一步该往哪走
void check_foodeating_ai(Snake& snake, bool& isEaten); //检查AI蛇是否吃到食物



Snake usersnake; //玩家操控的蛇
map<PII, Food> foodlist; //食物列表
set<PII> WallList;//墙壁列表
map<int,Snake> AIsnakeList; //AI蛇列表
bool check_eaten; //检查有没有吃到食物
int MaxScore = 0; //历史最高分
int AICnt = 0; //当前AI蛇数量
bool onGame = false;
int allAISnakeCnt = 0; //AI蛇总数

mutex Protectusersnake;//保护玩家的蛇
mutex ProtectallAISnakeCnt;
mutex Protectfoodlist;
mutex ProtectWallList, ProtectAIsnakeList, ProtectCursor;
mutex ProtectAICnt;
mutex ProtectonGame;

int main()
{
    system("color 0B");
    do
    {
        system("cls");
        cout << "欢迎游玩贪吃蛇小游戏！" << endl;
        cout << "输入1开始游戏，输入2查看历史最高分，输入3查看游戏规则，输入4退出游戏" << endl;
        int op;
        cin >> op;
        if (op == 1) {
            while (true) {
                system("cls");
                usersnake.score = 0, check_eaten = 0;
                print_map();
                ProtectonGame.lock();
                onGame = true;
                ProtectonGame.unlock();
                //贪吃蛇的每回合运行控制
                while (1)
                {
                    ProtectCursor.lock();
                    check_foodeating(usersnake, check_eaten);
                    move_snake(usersnake);
                    ProtectCursor.unlock();
                    Sleep(usersnake.speed * 1000);//控制速度
                    if (!check_snakealive(usersnake))
                        break;
                    srand(time(0));
                    Protectfoodlist.lock();
                    int n = foodlist.size();
                    Protectfoodlist.unlock();
                    if (rand() % 100 + 1 <= max(10, 100 - n * 10)) {
                        get_newfood();
                    }
                    if (usersnake.score >= 100) {
                        if (rand() % 100 + 1 <= 20) {
                            SummonNewWall();
                        }
                    }
                    if (usersnake.score >= 200) {
                        ProtectAICnt.lock();
                        int m = AICnt;
                        ProtectAICnt.unlock();
                        if (rand() % 100 + 1 <= max(1,10-4*m)) {
                            if (m < 3) {
                                thread thai(AIsnake);
                                thai.detach();
                            }
                        }
                    }
                }
                //清空食物列表
                Protectfoodlist.lock();
                foodlist.clear();
                Protectfoodlist.unlock();
                //清空墙壁列表
                ProtectWallList.lock();
                WallList.clear();
                ProtectWallList.unlock();
                //清空ai蛇列表
                ProtectAIsnakeList.lock();
                AIsnakeList.clear();
                ProtectAIsnakeList.unlock();

                ProtectCursor.lock();
                gotoxy(frame_height, 0);
                printf("Game Over!\n");
                ProtectCursor.unlock();
                ProtectonGame.lock();
                onGame = false;
                ProtectonGame.unlock();
                printf("1:重新开始\t2:退出\n");
                MaxScore = max(MaxScore, usersnake.score);
                char com2;
                Sleep(80);
                ProtectCursor.lock();
                gotoxy(frame_height + 3, 0);
                ProtectCursor.unlock();
                cin >> com2;
                if (com2 == '2')
                    break;
            }
        }
        else if (op == 2) {
            cout << "历史最高分为：" << MaxScore << endl;
            system("pause");
        }
        else if (op == 3) {
            system("cls");
            gotoxy(2,  3);
            cout << "欢迎来到贪吃蛇小游戏！";
            gotoxy(4,  3);
            cout << "操作说明：";
            gotoxy(6,  3);
            cout << "向上：W";
            gotoxy(8,  3);
            cout << "向下：S";
            gotoxy(10,  3);
            cout << "向左：A";
            gotoxy(12,  3);
            cout << "向右：D";
            
            gotoxy(4,  15);
            cout << "情景说明：";
            gotoxy(6,  15);
            cout << "￥:10-20分*一定基于你长度的倍率";
            gotoxy(8,  15);
            cout << "$:1-10分*一定基于你长度的倍率";
            gotoxy(10,  15);
            cout << "#:障碍物，碰到会死亡,100分之后额外随机生成";
            gotoxy(12,  15);
            cout << "%:AI蛇，死后有亡语，200分之后额外随机生成";
            gotoxy(16, 3);
            cout << "按数字键1,2,3,4进行变速";
            cout << endl;
            system("pause");
        }
        else if (op == 4) {
            break;
        }
        else {
            cout << "输入错误，请重新输入！" << endl;
            system("pause");
        }
    } while (1);
}

void gotoxy(int x, int y)
{
    COORD pos;//COORD是一种自带结构体，表示一个字符在控制台屏幕上的坐标
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE); //从标准输出设备里取出一个句柄
    pos.X = y, pos.Y = x;
    SetConsoleCursorPosition(hConsole, pos);//定位光标的函数
}

wchar_t getxy(int x, int y)
{
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD pos = { y,x };
    //将光标移动到指定位置
    SetConsoleCursorPosition(hConsole, pos);

    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(hConsole, &csbi);

    COORD cursorPosition = csbi.dwCursorPosition;

    CHAR_INFO ci;
    COORD bufferSize = { 1, 1 };  // 读取一个字符
    SMALL_RECT readRegion = { cursorPosition.X, cursorPosition.Y, cursorPosition.X, cursorPosition.Y };

    // 从光标位置读取字符和颜色信息
    ReadConsoleOutput(hConsole, &ci, bufferSize, { 0, 0 }, &readRegion);
    wchar_t res = ci.Char.UnicodeChar;
    return res;
}

void print_map()
{
    //打印墙壁
    for (int i = 0; i < frame_height; i++)
    {
        gotoxy(i, 0);
        cout << "#";
        gotoxy(i, frame_width - 1);//因为这个标记是长度，从零开始所以最后要减1
        cout << "#";
    }
    for (int i = 0; i < frame_width; i++)
    {
        gotoxy(0, i);
        cout << "#";
        gotoxy(frame_height - 1, i);
        cout << "#";
    }

    //蛇身初始化
    usersnake.len = 3;
    usersnake.state = 'w';
    usersnake.speed = 1;
    usersnake.x[1] = frame_height / 2;
    usersnake.y[1] = frame_width / 2;
    gotoxy(usersnake.x[1], usersnake.y[1]);
    cout << "@";//打印蛇头
    //打印蛇身
    for (int i = 2; i <= usersnake.len; i++)
    {
        usersnake.x[i] = usersnake.x[i - 1] + 1;
        usersnake.y[i] = usersnake.y[i - 1];
        gotoxy(usersnake.x[i], usersnake.y[i]);
        cout << "@";
    }
    //打印右边状态栏
    gotoxy(2, frame_width + 3);
    cout << "欢迎来到贪吃蛇小游戏！";
    gotoxy(4, frame_width + 3);
    cout << "操作说明：";
    gotoxy(6, frame_width + 3);
    cout << "向上：W";
    gotoxy(8, frame_width + 3);
    cout << "向下：S";
    gotoxy(10, frame_width + 3);
    cout << "向左：A";
    gotoxy(12, frame_width + 3);
    cout << "向右：D";
    gotoxy(16,frame_width + 3);
    cout << "按数字键1,2,3,4进行变速";
    gotoxy(20, frame_width + 3);
    printf("你的分数 : %d", usersnake.score);
    gotoxy(4, frame_width + 15);
    cout << "情景说明：";
    gotoxy(6, frame_width + 15);
    cout<<"￥:10-20分*一定基于你长度的倍率";
    gotoxy(8, frame_width + 15);
    cout << "$:1-10分*一定基于你长度的倍率";
    gotoxy(10, frame_width + 15);
    cout << "#:障碍物，碰到会死亡,100分之后额外随机生成";
    gotoxy(12, frame_width + 15);
    cout << "%:AI蛇，死后有亡语，200分之后额外随机生成";
}

bool check_foodisok(Food tempf)
{
    //检查是否和食物重合
    if (foodlist.find(make_pair(tempf.x, tempf.y)) != foodlist.end())
        return 1;
    //检查是否和玩家蛇重合
    for (int i = 1; i <= usersnake.len; i++)
        if (usersnake.x[i] == tempf.x && usersnake.y[i] == tempf.y)
            return 1;
    //检查是否和AI蛇重合
    for (auto& e : AIsnakeList) {
        for (int i = 1; i <= e.second.len; i++)
            if (e.second.x[i] == tempf.x && e.second.y[i] == tempf.y)
                return 1;
    }
    //检查是否和墙壁重合
    if (WallList.find(make_pair(tempf.x, tempf.y)) != WallList.end()) {
        return 1;
    }
    return 0;
}

void get_newfood()
{
    Food tempf;
    do {
        srand(time(0));
        tempf.x = rand() % (frame_height - 2) + 1;
        tempf.y = rand() % (frame_width - 2) + 1;

    } while (check_foodisok(tempf));
    ProtectCursor.lock();
    gotoxy(tempf.x, tempf.y);
    if (rand() % 100 + 1 <= 30) {
        cout << "￥";
        tempf.score = rand() % 10 + 10;
    }
    else {
        cout << "$";
        tempf.score = rand() % 10 + 1;
    }
    ProtectCursor.unlock();
    Protectfoodlist.lock();
    foodlist[make_pair(tempf.x, tempf.y)] = tempf;
    Protectfoodlist.unlock();
}

void move_snake(Snake& snake)
{
    char com = ' ';
    //kbhit()函数用于检测是否有键盘输入，有则返回一个非零值，否则返回零。
    //getch()函数用于不需要回车就可以获取键盘输入，但不显示在屏幕上。
    while (_kbhit())//键盘有输入
        com = _getch();//从控制台读取一个字符，但不显示在屏幕上
    //没有吃到去除蛇尾
    if (!check_eaten)
    {
        //光标移动到蛇尾
        gotoxy(snake.x[snake.len], snake.y[snake.len]);
        //蛇尾消失
        cout << " ";
    }
    //将除蛇头外的其他部分向前移动
    for (int i = snake.len; i > 1; i--) {
        snake.x[i] = snake.x[i - 1];
        snake.y[i] = snake.y[i - 1];
    }
    //移动蛇头
    switch (com)
    {
    case 'w':
    {
        if (snake.state == 's') //如果命令与当前方向相反不起作用
            snake.x[1]++;
        else
            snake.x[1]--, snake.state = 'w';
        break;
    }
    case 's':
    {
        if (snake.state == 'w')
            snake.x[1]--;
        else
            snake.x[1]++, snake.state = 's';
        break;
    }
    case 'a':
    {
        if (snake.state == 'd')
            snake.y[1]++;
        else
            snake.y[1]--, snake.state = 'a';
        break;
    }
    case 'd':
    {
        if (snake.state == 'a')
            snake.y[1]--;
        else
            snake.y[1]++, snake.state = 'd';
        break;
    }
    case '1': //按1切换不同的速度，默认缓慢速度，
        snake.speed = 1;
        if (snake.state == 's')
            snake.x[1]++;
        else if (snake.state == 'w')
            snake.x[1]--;
        else if (snake.state == 'd')
            snake.y[1]++;
        else if (snake.state == 'a')
            snake.y[1]--;
        break;
    case '2':
        snake.speed = 0.5;
        if (snake.state == 's')
            snake.x[1]++;
        else if (snake.state == 'w')
            snake.x[1]--;
        else if (snake.state == 'd')
            snake.y[1]++;
        else if (snake.state == 'a')
            snake.y[1]--;
        break;
    case '3':
        snake.speed = 0.1;
        if (snake.state == 's')
            snake.x[1]++;
        else if (snake.state == 'w')
            snake.x[1]--;
        else if (snake.state == 'd')
            snake.y[1]++;
        else if (snake.state == 'a')
            snake.y[1]--;
        break;
    case '4':
        snake.speed = 0.02;
        if (snake.state == 's')
            snake.x[1]++;
        else if (snake.state == 'w')
            snake.x[1]--;
        else if (snake.state == 'd')
            snake.y[1]++;
        else if (snake.state == 'a')
            snake.y[1]--;
        break;
    default: //按其余键保持状态前进
    {
        if (snake.state == 's')
            snake.x[1]++;
        else if (snake.state == 'w')
            snake.x[1]--;
        else if (snake.state == 'd')
            snake.y[1]++;
        else if (snake.state == 'a')
            snake.y[1]--;
        break;
    }
    }
    gotoxy(snake.x[1], snake.y[1]);
    printf("@");
    check_eaten = 0;
    gotoxy(frame_height, 0);
}

void check_foodeating(Snake& snake, bool& isEaten)
{
    //检查当前蛇头位置是否和食物重合
    PII cor = make_pair(snake.x[1], snake.y[1]);
    Protectfoodlist.lock();
    if (foodlist.find(cor) != foodlist.end())
    {
        Protectusersnake.lock();
        snake.score += foodlist[cor].score * (snake.len / 5 + 1);
        Protectusersnake.unlock(); 
        foodlist.erase(cor);
        Protectfoodlist.unlock();
        isEaten = 1;
        gotoxy(20, frame_width + 3);
        Protectusersnake.lock();
        printf("你的分数 : %d", snake.score);
        snake.len++;
        Protectusersnake.unlock();
        return;
    }
    Protectfoodlist.unlock();
}

void check_foodeating_ai(Snake& snake, bool& isEaten) {
    //检查当前蛇头位置是否和食物重合
    PII cor = make_pair(snake.x[1], snake.y[1]);
    Protectfoodlist.lock();
    if (foodlist.find(cor) != foodlist.end())
    {
        Protectusersnake.lock();
        snake.score += foodlist[cor].score * (snake.len / 5 + 1);
        Protectusersnake.unlock();
        foodlist.erase(cor);
        Protectfoodlist.unlock();
        isEaten = 1;
        gotoxy(20, frame_width + 3);
        Protectusersnake.lock();
        snake.len++;
        Protectusersnake.unlock();
        return;
    }
    Protectfoodlist.unlock();
}

bool check_snakealive(Snake& snake)
{
    //检查有没有撞到墙
    if (snake.x[1] == 0 || snake.x[1] == frame_height - 1 || snake.y[1] == 0 || snake.y[1] == frame_width - 1)//撞墙
        return 0;
    //检查有没有撞到新的墙壁
    for (auto e : WallList) {
        if (snake.x[1] == e.first && snake.y[1] == e.second)
            return 0;
    }
    //检查有没有吃到自己
    for (int i = 2; i <= snake.len; i++)
        if (snake.x[i] == snake.x[1] && snake.y[i] == snake.y[1]) {
            return 0;
        }
    //检查有没有撞到玩家蛇
    if (usersnake != snake) {
        for (int i = 1; i <= usersnake.len; i++) {
            if (snake.x[1] == usersnake.x[i] && snake.y[1] == usersnake.y[1]) {
                return 0;
            }
        }
    }
    //检测有没有撞到其他ai蛇
    for (auto& e : AIsnakeList) {
        if (e.second != snake) {
            for (int i = 1; i <= e.second.len; i++) {
                if (snake.x[1] == e.second.x[i] && snake.y[1] == e.second.y[i]) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

void SummonNewWall()
{
    srand(time(0));
    int x = rand() % (frame_height - 2) + 1;
    int y = rand() % (frame_width - 2) + 1;
    auto check = [&](int x, int y) {
        //检查是否和食物重合
        Protectfoodlist.lock();
        if (foodlist.find(make_pair(x, y)) != foodlist.end()) {
            Protectfoodlist.unlock();
            return false;
        }
        Protectfoodlist.unlock();
        //检查是否和玩家蛇重合
        for (int i = 1; i <= usersnake.len; i++)
            if (usersnake.x[i] == x && usersnake.y[i] == y)
                return false;
        //检查是否和墙壁重合
        ProtectWallList.lock();
        if (WallList.find(make_pair(x, y)) != WallList.end()) {
            ProtectWallList.unlock();
            return false;
        }
        ProtectWallList.unlock();
        //检查是否和AI蛇重合
        ProtectAIsnakeList.lock();
        for (auto& e : AIsnakeList) {
            for (int i = 1; i <= e.second.len; i++) {
                if (e.second.x[i] == x && e.second.y[i] == y) {
                    ProtectAIsnakeList.unlock();
                    return false;
                }
            }
        }
        ProtectAIsnakeList.unlock();
        return true;
        };
    //检查是否可以放置
    if (check(x, y)) {
        ProtectCursor.lock();
        gotoxy(x, y);
        cout << "#";
        ProtectCursor.unlock();
        ProtectWallList.lock();
        WallList.insert(make_pair(x, y));
        ProtectWallList.unlock();
    }
}

void AIsnake()
{
    int myid;
    ProtectallAISnakeCnt.lock();
    myid = allAISnakeCnt++;
    ProtectallAISnakeCnt.unlock();
    Snake aisnake;
    //蛇身初始化
    aisnake.len = 2;
    aisnake.state = 'w';
    srand(time(0));
    if (rand() % 100 + 1 <= 10) {
        aisnake.speed = 0.1;

    }
    else if (rand() % 100 + 1 <= 55) {
        aisnake.speed = 0.5;
    }
    else {
        aisnake.speed = 1;
    }
    //随机生成蛇头
    do {
        aisnake.x[1] = rand() % (frame_height - 2) + 1;
        aisnake.y[1] = rand() % (frame_width - 2) + 1;
        auto checkpos = [&](int x, int y) {
            //检查是否和食物墙壁重合
            if (foodlist.find(make_pair(x, y)) != foodlist.end() || WallList.find(make_pair(x, y)) != WallList.end())
                return false;
            //检查是否和玩家蛇重合
            for (int i = 1; i <= usersnake.len; i++)
                if (usersnake.x[i] == x && usersnake.y[i] == y)
                    return false;
            //检查是否和AI蛇重合
            for (auto& e : AIsnakeList) {
                for (int i = 1; i <= e.second.len; i++)
                    if (e.second.x[i] == x && e.second.y[i] == y)
                        return false;
            }
            //检查是否和边界重合
            if (x == 0 || x == frame_height - 1 || y == 0 || y == frame_width - 1)
                return false;
            return true;
            };
        //如果蛇头蛇身都没问题，就结束随机
        if (checkpos(aisnake.x[1], aisnake.y[1]) && checkpos(aisnake.x[1] + 1, aisnake.y[1])) {
            break;
        }
    } while (true);
    ProtectAICnt.lock();
    AICnt++;
    ProtectAICnt.unlock();

    ProtectAIsnakeList.lock();
    AIsnakeList[myid]=aisnake;
    ProtectAIsnakeList.unlock();

    ProtectCursor.lock();
    gotoxy(aisnake.x[1], aisnake.y[1]);
    cout << "%";//打印蛇头
    //打印蛇身
    for (int i = 2; i <= aisnake.len; i++)
    {
        aisnake.x[i] = aisnake.x[i - 1] + 1;
        aisnake.y[i] = aisnake.y[i - 1];
        gotoxy(aisnake.x[i], aisnake.y[i]);
        cout << "%";
    }
    ProtectCursor.unlock();
    //ai贪吃蛇的每回合运行控制
    bool AIisEaten = false;
    while (1)
    {
        ProtectonGame.lock();
        if (onGame == false) {
            ProtectonGame.unlock();
            break;
        }
        ProtectonGame.unlock();
        ProtectCursor.lock();
        check_foodeating_ai(aisnake, AIisEaten);
        move_snake_ai(aisnake, AIisEaten);
        ProtectCursor.unlock();

        ProtectAIsnakeList.lock();
        AIsnakeList[myid] = aisnake;
        ProtectAIsnakeList.unlock();

        Sleep(aisnake.speed * 1000);//控制速度（与长度呈反比）
        if (!check_snakealive(aisnake)) {
            ProtectCursor.lock();
            for (int i = 1; i <= aisnake.len; i++) {
                gotoxy(aisnake.x[i], aisnake.y[i]);
                cout << "#";
            }
            ProtectCursor.unlock();
            break;
        }
    }
    //死亡的时候将AI蛇的分数加到玩家上
    Protectusersnake.lock();
    usersnake.score += aisnake.score;
    Protectusersnake.unlock();

    ProtectAICnt.lock();
    AICnt--;
    ProtectAICnt.unlock();
    return;
}

void move_snake_ai(Snake& snake, bool& AIisEaten)
{
    char com = ' ';
    com = AICheckWheretogo(snake);
    //没有吃到去除蛇尾
    if (!AIisEaten)
    {
        //光标移动到蛇尾
        gotoxy(snake.x[snake.len], snake.y[snake.len]);
        //蛇尾消失
        cout << " ";
    }
    //将除蛇头外的其他部分向前移动
    for (int i = snake.len; i > 1; i--)
        snake.x[i] = snake.x[i - 1],
        snake.y[i] = snake.y[i - 1];
    //移动蛇头
    switch (com)
    {
    case 'w':
    {
        if (snake.state == 's') //如果命令与当前方向相反不起作用
            snake.x[1]++;
        else
            snake.x[1]--, snake.state = 'w';
        break;
    }
    case 's':
    {
        if (snake.state == 'w')
            snake.x[1]--;
        else
            snake.x[1]++, snake.state = 's';
        break;
    }
    case 'a':
    {
        if (snake.state == 'd')
            snake.y[1]++;
        else
            snake.y[1]--, snake.state = 'a';
        break;
    }
    case 'd':
    {
        if (snake.state == 'a')
            snake.y[1]--;
        else
            snake.y[1]++, snake.state = 'd';
        break;
    }
    default: //按其余键保持状态前进
    {
        if (snake.state == 's')
            snake.x[1]++;
        else if (snake.state == 'w')
            snake.x[1]--;
        else if (snake.state == 'd')
            snake.y[1]++;
        else if (snake.state == 'a')
            snake.y[1]--;
        break;
    }
    }
    gotoxy(snake.x[1], snake.y[1]);
    cout << "%";
    AIisEaten = 0;
    gotoxy(frame_height, 0);
}


char AICheckWheretogo(Snake& snake) {
    //寻找距离蛇头最近的食物
    Food myfoodpos = { -INF,-INF,0 };
    Protectfoodlist.lock();
    for (auto& e : foodlist) {
        if ((abs(e.second.x - snake.x[1]) + abs(e.second.y - snake.y[1])) <= (abs(myfoodpos.x - snake.x[1]) + abs(myfoodpos.y - snake.y[1]))) {
            myfoodpos = e.second;
        }
    }
    Protectfoodlist.unlock();
    //如果找不到
    if (myfoodpos.x == -INF && myfoodpos.y==-INF){
        return 'w';
	}
    else {
        //首先判定蛇头在食物的上方还是下方
        if (snake.x[1] < myfoodpos.x) {
			//如果在下方
            if (snake.state == 'w') {
				//如果当前方向是向上，那么就往右走
				return 'd';
			}
            else {
				//如果当前方向不是向上，那么就往下走
				return 's';
			}
		}
        else if (snake.x[1] > myfoodpos.x) {
			//如果在上方
            if (snake.state == 's') {
				//如果当前方向是向下，那么就往左走
				return 'a';
			}
            else {
				//如果当前方向不是向下，那么就往上走
				return 'w';
			}
		}
        else {
			//如果在同一行
            if (snake.y[1] < myfoodpos.y) {
				//如果在右边
                if (snake.state == 'a') {
					//如果当前方向是向左，那么就往下走
					return 's';
				}
                else {
					//如果当前方向不是向左，那么就往右走
					return 'd';
				}
			}
            else if (snake.y[1] > myfoodpos.y) {
				//如果在左边
                if (snake.state == 'd') {
					//如果当前方向是向右，那么就往上走
					return 'w';
				}
                else {
					//如果当前方向不是向右，那么就往左走
					return 'a';
				}
			}
            else {
                return 'w';
            }
		}
    }

}
