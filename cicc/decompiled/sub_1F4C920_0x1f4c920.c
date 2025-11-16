// Function: sub_1F4C920
// Address: 0x1f4c920
//
__int64 __fastcall sub_1F4C920(__int64 a1, int a2, __int64 a3)
{
  int *v4; // rsi
  __int64 v5; // r10
  int v7; // edx
  __int64 v8; // r8
  __int64 v9; // rdi
  unsigned int v10; // ecx
  __int16 v11; // ax
  _WORD *v12; // rcx
  unsigned __int16 v13; // ax
  _WORD *v14; // r12
  _WORD *v15; // rcx
  int v16; // edx
  unsigned __int16 *v17; // r8
  unsigned int v18; // ecx
  unsigned int i; // edi
  bool v20; // cf
  int v22; // edi

  v4 = *(int **)a1;
  v5 = *(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8);
  if ( v5 == *(_QWORD *)a1 )
    return 0;
  do
  {
    v7 = *v4;
    if ( *v4 == a2 )
      return 1;
    if ( v7 >= 0 && a2 >= 0 )
    {
      v8 = *(_QWORD *)(a3 + 8);
      v9 = *(_QWORD *)(a3 + 56);
      v10 = *(_DWORD *)(v8 + 24LL * (unsigned int)v7 + 16);
      v11 = v7 * (v10 & 0xF);
      v12 = (_WORD *)(v9 + 2LL * (v10 >> 4));
      v13 = *v12 + v11;
      v14 = v12 + 1;
      v16 = a2 * (*(_DWORD *)(v8 + 24LL * (unsigned int)a2 + 16) & 0xF);
      v15 = (_WORD *)(v9 + 2LL * (*(_DWORD *)(v8 + 24LL * (unsigned int)a2 + 16) >> 4));
      LOWORD(v16) = *v15 + v16;
      v17 = v15 + 1;
      v18 = v13;
      for ( i = (unsigned __int16)v16; ; i = (unsigned __int16)v16 )
      {
        v20 = v18 < i;
        if ( v18 == i )
          break;
        while ( v20 )
        {
          v13 += *v14;
          if ( !*v14 )
            goto LABEL_13;
          v18 = v13;
          ++v14;
          v20 = v13 < i;
          if ( v13 == i )
            return 1;
        }
        v22 = *v17;
        if ( !(_WORD)v22 )
          goto LABEL_13;
        v16 += v22;
        ++v17;
      }
      return 1;
    }
LABEL_13:
    ++v4;
  }
  while ( (int *)v5 != v4 );
  return 0;
}
