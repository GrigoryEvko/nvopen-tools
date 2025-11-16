// Function: sub_D8CCB0
// Address: 0xd8ccb0
//
__int64 __fastcall sub_D8CCB0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v4; // r9
  __int64 v5; // rdi
  int v6; // ecx
  unsigned int v7; // eax
  _QWORD *v8; // rsi
  __int64 v9; // r8
  int v10; // r11d
  _QWORD *v11; // r10

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = *a2;
    v6 = result - 1;
    v7 = (result - 1) & (((unsigned int)*a2 >> 4) ^ ((unsigned int)*a2 >> 9));
    v8 = (_QWORD *)(v4 + 56LL * v7);
    v9 = *v8;
    if ( v5 == *v8 )
    {
      *a3 = v8;
      return 1;
    }
    else
    {
      v10 = 1;
      v11 = 0;
      while ( v9 != -4096 )
      {
        if ( v9 == -8192 && !v11 )
          v11 = v8;
        v7 = v6 & (v10 + v7);
        v8 = (_QWORD *)(v4 + 56LL * v7);
        v9 = *v8;
        if ( *v8 == v5 )
        {
          *a3 = v8;
          return 1;
        }
        ++v10;
      }
      if ( !v11 )
        v11 = v8;
      *a3 = v11;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
  }
  return result;
}
