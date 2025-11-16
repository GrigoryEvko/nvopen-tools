// Function: sub_37B5F80
// Address: 0x37b5f80
//
__int64 __fastcall sub_37B5F80(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v3; // r8d
  __int64 v4; // rsi
  int v5; // r8d
  __int64 v6; // rdi
  unsigned int v7; // ecx
  _QWORD *v8; // rax
  __int64 v9; // r9
  int v11; // r11d
  _QWORD *v12; // r10

  v3 = *(_DWORD *)(a1 + 24);
  if ( v3 )
  {
    v4 = *a2;
    v5 = v3 - 1;
    v6 = *(_QWORD *)(a1 + 8);
    v7 = v5 & ((484763065 * v4) ^ ((0xBF58476D1CE4E5B9LL * v4) >> 31));
    v8 = (_QWORD *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
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
        v7 = v5 & (v11 + v7);
        v8 = (_QWORD *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == v4 )
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
