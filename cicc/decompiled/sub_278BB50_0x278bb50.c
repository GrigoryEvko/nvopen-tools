// Function: sub_278BB50
// Address: 0x278bb50
//
__int64 __fastcall sub_278BB50(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  int v5; // r11d
  __int64 v6; // rcx
  int v7; // r13d
  int v8; // ebx
  unsigned int v9; // edx
  __int64 v10; // r11
  unsigned int v11; // edx
  unsigned int v12; // edx

  result = *(_QWORD *)(a3 + 16);
  if ( result )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(result + 24);
      if ( (unsigned __int8)(*(_BYTE *)v4 - 30) <= 0xAu )
        break;
      result = *(_QWORD *)(result + 8);
      if ( !result )
        return result;
    }
    while ( 1 )
    {
      v5 = *(_DWORD *)(a1 + 176);
      v6 = *(_QWORD *)(v4 + 40);
      if ( v5 )
      {
        v7 = 1;
        v8 = v5 - 1;
        v9 = (v5 - 1)
           & (((0xBF58476D1CE4E5B9LL
              * (((unsigned __int64)(unsigned int)(37 * a2) << 32) | ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4))) >> 31)
            ^ (484763065 * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4))));
        while ( 1 )
        {
          v10 = *(_QWORD *)(a1 + 160) + 24LL * v9;
          if ( a2 == *(_DWORD *)v10 && v6 == *(_QWORD *)(v10 + 8) )
            break;
          if ( *(_DWORD *)v10 == -1 )
          {
            if ( *(_QWORD *)(v10 + 8) == -4096 )
              goto LABEL_10;
            v12 = v7 + v9;
            ++v7;
            v9 = v8 & v12;
          }
          else
          {
            v11 = v7 + v9;
            ++v7;
            v9 = v8 & v11;
          }
        }
        *(_DWORD *)v10 = -2;
        *(_QWORD *)(v10 + 8) = -8192;
        --*(_DWORD *)(a1 + 168);
        ++*(_DWORD *)(a1 + 172);
      }
LABEL_10:
      result = *(_QWORD *)(result + 8);
      if ( !result )
        break;
      while ( 1 )
      {
        v4 = *(_QWORD *)(result + 24);
        if ( (unsigned __int8)(*(_BYTE *)v4 - 30) <= 0xAu )
          break;
        result = *(_QWORD *)(result + 8);
        if ( !result )
          return result;
      }
    }
  }
  return result;
}
