// Function: sub_B4EF80
// Address: 0xb4ef80
//
__int64 __fastcall sub_B4EF80(__int64 a1, __int64 a2, int a3, int *a4)
{
  __int64 result; // rax
  __int64 v7; // r10
  __int64 v8; // rax
  int v9; // r8d
  int v10; // edx

  result = 0;
  if ( a3 == a2 && a3 )
  {
    v7 = (unsigned int)(a3 - 1);
    v8 = 0;
    v9 = -1;
    while ( 1 )
    {
      v10 = *(_DWORD *)(a1 + 4 * v8);
      if ( v10 != -1 )
      {
        if ( v9 == -1 )
        {
          if ( v10 < (int)v8 )
            return 0;
          v9 = v10 - v8;
          if ( v10 - (int)v8 >= a3 )
            return 0;
        }
        else if ( v9 + (_DWORD)v8 != v10 )
        {
          return 0;
        }
      }
      if ( v7 == v8 )
        break;
      ++v8;
    }
    if ( v9 == -1 )
      return 0;
    *a4 = v9;
    return 1;
  }
  return result;
}
