// Function: sub_12E4800
// Address: 0x12e4800
//
__int64 __fastcall sub_12E4800(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v5; // rsi
  int v6; // ecx
  __int64 v7; // rdi
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 v10; // r9
  int v11; // r11d
  __int64 v12; // r10

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v5 = *(_QWORD *)(a2 + 24);
    v6 = result - 1;
    v7 = *(_QWORD *)(a1 + 8);
    v8 = (result - 1) & (((unsigned int)v5 >> 4) ^ ((unsigned int)v5 >> 9));
    v9 = v7 + ((unsigned __int64)v8 << 6);
    v10 = *(_QWORD *)(v9 + 24);
    if ( v5 == v10 )
    {
      *a3 = v9;
      return 1;
    }
    else
    {
      v11 = 1;
      v12 = 0;
      while ( v10 != -8 )
      {
        if ( !v12 && v10 == -16 )
          v12 = v9;
        v8 = v6 & (v11 + v8);
        v9 = v7 + ((unsigned __int64)v8 << 6);
        v10 = *(_QWORD *)(v9 + 24);
        if ( v5 == v10 )
        {
          *a3 = v9;
          return 1;
        }
        ++v11;
      }
      if ( !v12 )
        v12 = v9;
      *a3 = v12;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
  }
  return result;
}
