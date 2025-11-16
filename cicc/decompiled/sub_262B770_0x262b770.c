// Function: sub_262B770
// Address: 0x262b770
//
__int64 __fastcall sub_262B770(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v3; // ecx
  __int64 v4; // r8
  __int64 v5; // rdi
  int v6; // ecx
  unsigned int v7; // eax
  _QWORD *v8; // rsi
  __int64 v9; // r9
  int v11; // r11d
  _QWORD *v12; // r10

  v3 = *(_DWORD *)(a1 + 24);
  if ( v3 )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = *a2;
    v6 = v3 - 1;
    v7 = v6 & (((0xBF58476D1CE4E5B9LL * *a2) >> 31) ^ (484763065 * *(_DWORD *)a2));
    v8 = (_QWORD *)(v4 + 56LL * v7);
    v9 = *v8;
    if ( v5 == *v8 )
    {
      *a3 = v8;
      return 1;
    }
    else
    {
      v11 = 1;
      v12 = 0;
      while ( v9 != -1 )
      {
        if ( !v12 && v9 == -2 )
          v12 = v8;
        v7 = v6 & (v11 + v7);
        v8 = (_QWORD *)(v4 + 56LL * v7);
        v9 = *v8;
        if ( *v8 == v5 )
        {
          *a3 = v8;
          return 1;
        }
        ++v11;
      }
      if ( !v12 )
        v12 = v8;
      *a3 = v12;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
