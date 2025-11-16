// Function: sub_9E1360
// Address: 0x9e1360
//
_DWORD *__fastcall sub_9E1360(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  _DWORD *v6; // rcx
  int v7; // r11d
  unsigned int v8; // edi
  _DWORD *v9; // rax
  int v10; // r8d
  int v12; // eax
  int v13; // edx
  int v14; // eax
  int v15; // eax
  int v16; // eax
  __int64 v17; // r9
  unsigned int v18; // esi
  int v19; // r8d
  int v20; // r11d
  _DWORD *v21; // r10
  int v22; // eax
  int v23; // eax
  __int64 v24; // r9
  int v25; // r11d
  unsigned int v26; // edi
  int v27; // r8d

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  v8 = (v4 - 1) & (37 * *a2);
  v9 = (_DWORD *)(v5 + 24LL * v8);
  v10 = *v9;
  if ( *a2 == *v9 )
    return v9 + 2;
  while ( v10 != -1 )
  {
    if ( !v6 && v10 == -2 )
      v6 = v9;
    v8 = (v4 - 1) & (v7 + v8);
    v9 = (_DWORD *)(v5 + 24LL * v8);
    v10 = *v9;
    if ( *a2 == *v9 )
      return v9 + 2;
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
    sub_9E0BD0(a1, 2 * v4);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = v16 & (37 * *a2);
      v6 = (_DWORD *)(v17 + 24LL * v18);
      v19 = *v6;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v6 == *a2 )
        goto LABEL_14;
      v20 = 1;
      v21 = 0;
      while ( v19 != -1 )
      {
        if ( !v21 && v19 == -2 )
          v21 = v6;
        v18 = v16 & (v20 + v18);
        v6 = (_DWORD *)(v17 + 24LL * v18);
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
    sub_9E0BD0(a1, v4);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 8);
      v21 = 0;
      v25 = 1;
      v26 = v23 & (37 * *a2);
      v6 = (_DWORD *)(v24 + 24LL * v26);
      v27 = *v6;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v6 )
        goto LABEL_14;
      while ( v27 != -1 )
      {
        if ( !v21 && v27 == -2 )
          v21 = v6;
        v26 = v23 & (v25 + v26);
        v6 = (_DWORD *)(v24 + 24LL * v26);
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
  if ( *v6 != -1 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  *((_QWORD *)v6 + 1) = 0;
  *((_QWORD *)v6 + 2) = 0;
  *v6 = v14;
  return v6 + 2;
}
