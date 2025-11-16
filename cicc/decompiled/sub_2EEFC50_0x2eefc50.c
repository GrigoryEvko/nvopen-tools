// Function: sub_2EEFC50
// Address: 0x2eefc50
//
_QWORD *__fastcall sub_2EEFC50(__int64 a1, _QWORD *a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdi
  int v6; // r11d
  _QWORD *v7; // rdx
  unsigned int v8; // eax
  _QWORD *v9; // r8
  __int64 v10; // r10
  int v12; // eax
  int v13; // ecx
  int v14; // eax
  int v15; // esi
  __int64 v16; // r9
  unsigned int v17; // eax
  __int64 v18; // r8
  int v19; // r11d
  _QWORD *v20; // r10
  int v21; // eax
  int v22; // esi
  __int64 v23; // r9
  int v24; // r11d
  unsigned int v25; // eax
  __int64 v26; // r8

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = 0;
  v8 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v9 = (_QWORD *)(v5 + 368LL * v8);
  v10 = *v9;
  if ( *a2 == *v9 )
    return v9 + 1;
  while ( v10 != -4096 )
  {
    if ( v10 == -8192 && !v7 )
      v7 = v9;
    v8 = (v4 - 1) & (v6 + v8);
    v9 = (_QWORD *)(v5 + 368LL * v8);
    v10 = *v9;
    if ( *a2 == *v9 )
      return v9 + 1;
    ++v6;
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = v9;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
  {
LABEL_18:
    sub_2EEEEA0(a1, 2 * v4);
    v14 = *(_DWORD *)(a1 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(a1 + 8);
      v13 = *(_DWORD *)(a1 + 16) + 1;
      v17 = (v14 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v7 = (_QWORD *)(v16 + 368LL * v17);
      v18 = *v7;
      if ( *v7 == *a2 )
        goto LABEL_14;
      v19 = 1;
      v20 = 0;
      while ( v18 != -4096 )
      {
        if ( !v20 && v18 == -8192 )
          v20 = v7;
        v17 = v15 & (v19 + v17);
        v7 = (_QWORD *)(v16 + 368LL * v17);
        v18 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_14;
        ++v19;
      }
LABEL_22:
      if ( v20 )
        v7 = v20;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_2EEEEA0(a1, v4);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      v20 = 0;
      v24 = 1;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      v25 = (v21 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v7 = (_QWORD *)(v23 + 368LL * v25);
      v26 = *v7;
      if ( *v7 == *a2 )
        goto LABEL_14;
      while ( v26 != -4096 )
      {
        if ( !v20 && v26 == -8192 )
          v20 = v7;
        v25 = v22 & (v24 + v25);
        v7 = (_QWORD *)(v23 + 368LL * v25);
        v26 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_14;
        ++v24;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v7 = *a2;
  memset(v7 + 1, 0, 0x168u);
  v7[24] = 8;
  v7[23] = v7 + 26;
  v7[35] = v7 + 38;
  *((_BYTE *)v7 + 204) = 1;
  v7[36] = 8;
  *((_BYTE *)v7 + 300) = 1;
  return v7 + 1;
}
