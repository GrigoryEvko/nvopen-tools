// Function: sub_25D8EF0
// Address: 0x25d8ef0
//
_QWORD *__fastcall sub_25D8EF0(__int64 a1, _QWORD *a2)
{
  unsigned int v3; // esi
  __int64 v4; // r8
  int v5; // r11d
  unsigned int v6; // ecx
  _QWORD *v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // r10
  int v11; // eax
  int v12; // ecx
  int v13; // eax
  int v14; // esi
  __int64 v15; // r8
  unsigned int v16; // eax
  __int64 v17; // r9
  int v18; // r11d
  _QWORD *v19; // r10
  int v20; // eax
  int v21; // esi
  __int64 v22; // r8
  int v23; // r11d
  unsigned int v24; // eax
  __int64 v25; // r9

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v5 = 1;
  v6 = (v3 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v7 = 0;
  v8 = (_QWORD *)(v4 + 136LL * v6);
  v9 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  while ( v9 != -4096 )
  {
    if ( !v7 && v9 == -8192 )
      v7 = v8;
    v6 = (v3 - 1) & (v5 + v6);
    v8 = (_QWORD *)(v4 + 136LL * v6);
    v9 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v5;
  }
  if ( !v7 )
    v7 = v8;
  v11 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v3 )
  {
LABEL_18:
    sub_25D8AF0(a1, 2 * v3);
    v13 = *(_DWORD *)(a1 + 24);
    if ( v13 )
    {
      v14 = v13 - 1;
      v15 = *(_QWORD *)(a1 + 8);
      v16 = (v13 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v7 = (_QWORD *)(v15 + 136LL * v16);
      v17 = *v7;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v7 == *a2 )
        goto LABEL_14;
      v18 = 1;
      v19 = 0;
      while ( v17 != -4096 )
      {
        if ( !v19 && v17 == -8192 )
          v19 = v7;
        v16 = v14 & (v18 + v16);
        v7 = (_QWORD *)(v15 + 136LL * v16);
        v17 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_14;
        ++v18;
      }
LABEL_22:
      if ( v19 )
        v7 = v19;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v3 - *(_DWORD *)(a1 + 20) - v12 <= v3 >> 3 )
  {
    sub_25D8AF0(a1, v3);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 8);
      v19 = 0;
      v23 = 1;
      v24 = (v20 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v7 = (_QWORD *)(v22 + 136LL * v24);
      v25 = *v7;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v7 )
        goto LABEL_14;
      while ( v25 != -4096 )
      {
        if ( !v19 && v25 == -8192 )
          v19 = v7;
        v24 = v21 & (v23 + v24);
        v7 = (_QWORD *)(v22 + 136LL * v24);
        v25 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_14;
        ++v23;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v7 = *a2;
  memset(v7 + 1, 0, 0x80u);
  v7[1] = v7 + 3;
  v7[2] = 0x400000000LL;
  v7[14] = v7 + 12;
  v7[15] = v7 + 12;
  return v7 + 1;
}
