// Function: sub_31590A0
// Address: 0x31590a0
//
_QWORD *__fastcall sub_31590A0(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  int v6; // r11d
  unsigned int v7; // edx
  _QWORD *v8; // r8
  _QWORD *v9; // rax
  __int64 v10; // rdi
  int v12; // ecx
  int v13; // ecx
  __int64 v14; // rdx
  int v15; // eax
  int v16; // esi
  __int64 v17; // r9
  unsigned int v18; // edx
  __int64 v19; // r8
  int v20; // r11d
  _QWORD *v21; // r10
  int v22; // eax
  int v23; // esi
  __int64 v24; // r9
  int v25; // r11d
  unsigned int v26; // edx
  __int64 v27; // r8

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = (_QWORD *)(v5 + 88LL * v7);
  v9 = 0;
  v10 = *v8;
  if ( *v8 == *a2 )
    return v8 + 1;
  while ( v10 != -4096 )
  {
    if ( !v9 && v10 == -8192 )
      v9 = v8;
    v7 = (v4 - 1) & (v6 + v7);
    v8 = (_QWORD *)(v5 + 88LL * v7);
    v10 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v6;
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( !v9 )
    v9 = v8;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * v13 >= 3 * v4 )
  {
LABEL_18:
    sub_3158A80(a1, 2 * v4);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = (v15 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v9 = (_QWORD *)(v17 + 88LL * v18);
      v19 = *v9;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v9 == *a2 )
        goto LABEL_14;
      v20 = 1;
      v21 = 0;
      while ( v19 != -4096 )
      {
        if ( !v21 && v19 == -8192 )
          v21 = v9;
        v18 = v16 & (v20 + v18);
        v9 = (_QWORD *)(v17 + 88LL * v18);
        v19 = *v9;
        if ( *a2 == *v9 )
          goto LABEL_14;
        ++v20;
      }
LABEL_22:
      if ( v21 )
        v9 = v21;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_3158A80(a1, v4);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 8);
      v21 = 0;
      v25 = 1;
      v26 = (v22 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v9 = (_QWORD *)(v24 + 88LL * v26);
      v27 = *v9;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v9 )
        goto LABEL_14;
      while ( v27 != -4096 )
      {
        if ( v27 == -8192 && !v21 )
          v21 = v9;
        v26 = v23 & (v25 + v26);
        v9 = (_QWORD *)(v24 + 88LL * v26);
        v27 = *v9;
        if ( *a2 == *v9 )
          goto LABEL_14;
        ++v25;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v9 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  v9[6] = 0x400000000LL;
  *v9 = v14;
  v9[5] = v9 + 7;
  *(_OWORD *)(v9 + 1) = 0;
  *(_OWORD *)(v9 + 3) = 0;
  *(_OWORD *)(v9 + 7) = 0;
  *(_OWORD *)(v9 + 9) = 0;
  return v9 + 1;
}
