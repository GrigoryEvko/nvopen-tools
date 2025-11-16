// Function: sub_9E2360
// Address: 0x9e2360
//
_QWORD *__fastcall sub_9E2360(__int64 a1, _QWORD *a2)
{
  unsigned int v3; // esi
  __int64 v4; // r9
  _QWORD *v5; // rcx
  int v6; // r11d
  unsigned int v7; // eax
  _QWORD *v8; // rdx
  __int64 v9; // r8
  int v11; // eax
  int v12; // edx
  __int64 v13; // rax
  int v14; // edx
  int v15; // esi
  __int64 v16; // r9
  unsigned int v17; // eax
  __int64 v18; // r8
  int v19; // r11d
  _QWORD *v20; // r10
  int v21; // edx
  int v22; // esi
  __int64 v23; // r9
  int v24; // r11d
  unsigned int v25; // eax
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
  v7 = (v3 - 1) & (((0xBF58476D1CE4E5B9LL * *a2) >> 31) ^ (484763065 * *(_DWORD *)a2));
  v8 = (_QWORD *)(v4 + 24LL * v7);
  v9 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  while ( v9 != -1 )
  {
    if ( !v5 && v9 == -2 )
      v5 = v8;
    v7 = (v3 - 1) & (v6 + v7);
    v8 = (_QWORD *)(v4 + 24LL * v7);
    v9 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v6;
  }
  v11 = *(_DWORD *)(a1 + 16);
  if ( !v5 )
    v5 = v8;
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v3 )
  {
LABEL_18:
    sub_9E2150(a1, 2 * v3);
    v14 = *(_DWORD *)(a1 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(a1 + 8);
      v17 = (v14 - 1) & (((0xBF58476D1CE4E5B9LL * *a2) >> 31) ^ (484763065 * *(_DWORD *)a2));
      v5 = (_QWORD *)(v16 + 24LL * v17);
      v18 = *v5;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v5 == *a2 )
        goto LABEL_14;
      v19 = 1;
      v20 = 0;
      while ( v18 != -1 )
      {
        if ( !v20 && v18 == -2 )
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
    sub_9E2150(a1, v3);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      v20 = 0;
      v24 = 1;
      v25 = (v21 - 1) & (((0xBF58476D1CE4E5B9LL * *a2) >> 31) ^ (484763065 * *(_DWORD *)a2));
      v5 = (_QWORD *)(v23 + 24LL * v25);
      v26 = *v5;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v5 )
        goto LABEL_14;
      while ( v26 != -1 )
      {
        if ( !v20 && v26 == -2 )
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
  if ( *v5 != -1 )
    --*(_DWORD *)(a1 + 20);
  v13 = *a2;
  v5[1] = 0;
  v5[2] = 0;
  *v5 = v13;
  return v5 + 1;
}
