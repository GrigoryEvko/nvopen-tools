// Function: sub_35DFC90
// Address: 0x35dfc90
//
_QWORD *__fastcall sub_35DFC90(__int64 a1, _QWORD *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  int v6; // r11d
  unsigned int v7; // edx
  _QWORD *v8; // rcx
  _QWORD *v9; // rax
  __int64 v10; // r10
  int v12; // eax
  int v13; // edx
  int v14; // eax
  int v15; // edi
  __int64 v16; // r9
  unsigned int v17; // esi
  __int64 v18; // rax
  int v19; // r11d
  _QWORD *v20; // r10
  int v21; // eax
  int v22; // eax
  __int64 v23; // r8
  int v24; // r11d
  unsigned int v25; // esi
  __int64 v26; // r9

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = 0;
  v9 = (_QWORD *)(v5 + 56LL * v7);
  v10 = *v9;
  if ( *a2 == *v9 )
    return v9 + 1;
  while ( v10 != -4096 )
  {
    if ( !v8 && v10 == -8192 )
      v8 = v9;
    v7 = (v4 - 1) & (v6 + v7);
    v9 = (_QWORD *)(v5 + 56LL * v7);
    v10 = *v9;
    if ( *a2 == *v9 )
      return v9 + 1;
    ++v6;
  }
  if ( !v8 )
    v8 = v9;
  v12 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
  {
LABEL_18:
    sub_35DF930(a1, 2 * v4);
    v14 = *(_DWORD *)(a1 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(a1 + 8);
      v17 = (v14 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v8 = (_QWORD *)(v16 + 56LL * v17);
      v18 = *v8;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v8 == *a2 )
        goto LABEL_14;
      v19 = 1;
      v20 = 0;
      while ( v18 != -4096 )
      {
        if ( !v20 && v18 == -8192 )
          v20 = v8;
        v17 = v15 & (v19 + v17);
        v8 = (_QWORD *)(v16 + 56LL * v17);
        v18 = *v8;
        if ( *a2 == *v8 )
          goto LABEL_14;
        ++v19;
      }
LABEL_22:
      if ( v20 )
        v8 = v20;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_35DF930(a1, v4);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      v20 = 0;
      v24 = 1;
      v25 = v22 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v8 = (_QWORD *)(v23 + 56LL * v25);
      v26 = *v8;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v8 )
        goto LABEL_14;
      while ( v26 != -4096 )
      {
        if ( !v20 && v26 == -8192 )
          v20 = v8;
        v25 = v22 & (v24 + v25);
        v8 = (_QWORD *)(v23 + 56LL * v25);
        v26 = *v8;
        if ( *a2 == *v8 )
          goto LABEL_14;
        ++v24;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v8 = *a2;
  v8[1] = v8 + 3;
  v8[2] = 0x400000000LL;
  return v8 + 1;
}
