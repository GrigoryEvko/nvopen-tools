// Function: sub_B4EDA0
// Address: 0xb4eda0
//
__int64 __fastcall sub_B4EDA0(int *a1, __int64 a2, int a3)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  int v9; // edx

  if ( a3 != a2 )
    return 0;
  if ( (unsigned __int8)sub_B4ED30(a1, (unsigned int)a3, a3) == 1 && a3 > 1 )
  {
    v7 = 1;
    v8 = (unsigned int)(a3 - 1) + 2LL;
    while ( 1 )
    {
      v9 = a1[v7 - 1];
      if ( v9 != -1 && v9 != a3 - (_DWORD)v7 && v9 != 2 * a3 - (_DWORD)v7 )
        break;
      if ( ++v7 == v8 )
        return 1;
    }
  }
  return 0;
}
