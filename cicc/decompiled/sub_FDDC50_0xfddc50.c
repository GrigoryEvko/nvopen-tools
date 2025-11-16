// Function: sub_FDDC50
// Address: 0xfddc50
//
__int64 __fastcall sub_FDDC50(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v5; // rsi
  __int64 v6; // r9
  int v7; // edx
  unsigned int v8; // eax
  __int64 v9; // rdi
  __int64 v10; // r8
  int v11; // r11d
  __int64 v12; // r10

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v5 = *(_QWORD *)(a2 + 16);
    v6 = *(_QWORD *)(a1 + 8);
    v7 = result - 1;
    v8 = (result - 1) & (((unsigned int)v5 >> 4) ^ ((unsigned int)v5 >> 9));
    v9 = v6 + 72LL * v8;
    v10 = *(_QWORD *)(v9 + 16);
    if ( v10 == v5 )
    {
      *a3 = v9;
      return 1;
    }
    else
    {
      v11 = 1;
      v12 = 0;
      while ( v10 != -4096 )
      {
        if ( !v12 && v10 == -8192 )
          v12 = v9;
        v8 = v7 & (v11 + v8);
        v9 = v6 + 72LL * v8;
        v10 = *(_QWORD *)(v9 + 16);
        if ( v10 == v5 )
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
