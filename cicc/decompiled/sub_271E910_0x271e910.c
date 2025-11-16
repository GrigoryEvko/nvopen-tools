// Function: sub_271E910
// Address: 0x271e910
//
_QWORD *__fastcall sub_271E910(__int64 a1, __int64 *a2)
{
  unsigned int v3; // esi
  __int64 v4; // r9
  _QWORD *v5; // rcx
  int v6; // r11d
  unsigned int v7; // edx
  _QWORD *v8; // rax
  __int64 v9; // r8
  int v11; // eax
  int v12; // edx
  __int64 v13; // rax
  int v14; // eax
  int v15; // eax
  __int64 v16; // r9
  unsigned int v17; // esi
  __int64 v18; // r8
  int v19; // r11d
  _QWORD *v20; // r10
  int v21; // eax
  int v22; // eax
  __int64 v23; // r9
  int v24; // r11d
  unsigned int v25; // esi
  __int64 v26; // r8

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v5 = 0;
  v6 = 1;
  v7 = (v3 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = (_QWORD *)(v4 + 24LL * v7);
  v9 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  while ( v9 != -4096 )
  {
    if ( !v5 && v9 == -8192 )
      v5 = v8;
    v7 = (v3 - 1) & (v6 + v7);
    v8 = (_QWORD *)(v4 + 24LL * v7);
    v9 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v6;
  }
  if ( !v5 )
    v5 = v8;
  v11 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v3 )
  {
LABEL_18:
    sub_271E3C0(a1, 2 * v3);
    v14 = *(_DWORD *)(a1 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(a1 + 8);
      v17 = v15 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v5 = (_QWORD *)(v16 + 24LL * v17);
      v18 = *v5;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v5 == *a2 )
        goto LABEL_14;
      v19 = 1;
      v20 = 0;
      while ( v18 != -4096 )
      {
        if ( !v20 && v18 == -8192 )
          v20 = v5;
        v17 = v15 & (v19 + v17);
        v5 = (_QWORD *)(v16 + 24LL * v17);
        v18 = *v5;
        if ( *a2 == *v5 )
          goto LABEL_14;
        ++v19;
      }
LABEL_22:
      if ( v20 )
        v5 = v20;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v3 - *(_DWORD *)(a1 + 20) - v12 <= v3 >> 3 )
  {
    sub_271E3C0(a1, v3);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      v20 = 0;
      v24 = 1;
      v25 = v22 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v5 = (_QWORD *)(v23 + 24LL * v25);
      v26 = *v5;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v5 )
        goto LABEL_14;
      while ( v26 != -4096 )
      {
        if ( !v20 && v26 == -8192 )
          v20 = v5;
        v25 = v22 & (v24 + v25);
        v5 = (_QWORD *)(v23 + 24LL * v25);
        v26 = *v5;
        if ( *a2 == *v5 )
          goto LABEL_14;
        ++v24;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v5 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v13 = *a2;
  *((_BYTE *)v5 + 8) = 0;
  v5[2] = 0;
  *v5 = v13;
  return v5 + 1;
}
