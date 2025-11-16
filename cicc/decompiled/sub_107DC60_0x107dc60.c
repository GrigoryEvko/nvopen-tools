// Function: sub_107DC60
// Address: 0x107dc60
//
_QWORD *__fastcall sub_107DC60(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  _QWORD *v6; // r10
  int v7; // r11d
  unsigned int v8; // ecx
  _QWORD *v9; // rax
  __int64 v10; // rdx
  int v12; // eax
  int v13; // edx
  __int64 v14; // rax
  int v15; // eax
  int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // ecx
  __int64 v19; // rax
  int v20; // r11d
  _QWORD *v21; // r9
  int v22; // eax
  int v23; // eax
  __int64 v24; // r8
  int v25; // r11d
  unsigned int v26; // ecx
  __int64 v27; // rdi

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  v8 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v9 = (_QWORD *)(v5 + 16LL * v8);
  v10 = *v9;
  if ( *a2 == *v9 )
    return v9 + 1;
  while ( v10 != -4096 )
  {
    if ( !v6 && v10 == -8192 )
      v6 = v9;
    v8 = (v4 - 1) & (v7 + v8);
    v9 = (_QWORD *)(v5 + 16LL * v8);
    v10 = *v9;
    if ( *a2 == *v9 )
      return v9 + 1;
    ++v7;
  }
  if ( !v6 )
    v6 = v9;
  v12 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
  {
LABEL_18:
    sub_107DA80(a1, 2 * v4);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v13 = *(_DWORD *)(a1 + 16) + 1;
      v18 = (v15 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v6 = (_QWORD *)(v17 + 16LL * v18);
      v19 = *v6;
      if ( *v6 == *a2 )
        goto LABEL_14;
      v20 = 1;
      v21 = 0;
      while ( v19 != -4096 )
      {
        if ( !v21 && v19 == -8192 )
          v21 = v6;
        v18 = v16 & (v20 + v18);
        v6 = (_QWORD *)(v17 + 16LL * v18);
        v19 = *v6;
        if ( *a2 == *v6 )
          goto LABEL_14;
        ++v20;
      }
LABEL_22:
      if ( v21 )
        v6 = v21;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_107DA80(a1, v4);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 8);
      v21 = 0;
      v25 = 1;
      v26 = v23 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v13 = *(_DWORD *)(a1 + 16) + 1;
      v6 = (_QWORD *)(v24 + 16LL * v26);
      v27 = *v6;
      if ( *a2 == *v6 )
        goto LABEL_14;
      while ( v27 != -4096 )
      {
        if ( !v21 && v27 == -8192 )
          v21 = v6;
        v26 = v23 & (v25 + v26);
        v6 = (_QWORD *)(v24 + 16LL * v26);
        v27 = *v6;
        if ( *a2 == *v6 )
          goto LABEL_14;
        ++v25;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v6 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  *((_DWORD *)v6 + 2) = 0;
  *v6 = v14;
  return v6 + 1;
}
