// Function: sub_307C080
// Address: 0x307c080
//
_DWORD *__fastcall sub_307C080(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  int v6; // r11d
  _DWORD *v7; // rdi
  unsigned int v8; // ecx
  _DWORD *v9; // rax
  int v10; // r8d
  int v12; // eax
  int v13; // edx
  int v14; // eax
  int v15; // eax
  int v16; // ecx
  __int64 v17; // r9
  unsigned int v18; // eax
  int v19; // r8d
  int v20; // r11d
  _DWORD *v21; // r10
  int v22; // eax
  int v23; // ecx
  __int64 v24; // r9
  int v25; // r11d
  unsigned int v26; // eax
  int v27; // r8d

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = 0;
  v8 = (v4 - 1) & (37 * *a2);
  v9 = (_DWORD *)(v5 + 8LL * v8);
  v10 = *v9;
  if ( *a2 == *v9 )
    return v9 + 1;
  while ( v10 != -1 )
  {
    if ( !v7 && v10 == -2 )
      v7 = v9;
    v8 = (v4 - 1) & (v6 + v8);
    v9 = (_DWORD *)(v5 + 8LL * v8);
    v10 = *v9;
    if ( *a2 == *v9 )
      return v9 + 1;
    ++v6;
  }
  if ( !v7 )
    v7 = v9;
  v12 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
  {
LABEL_18:
    sub_2E518D0(a1, 2 * v4);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = (v15 - 1) & (37 * *a2);
      v7 = (_DWORD *)(v17 + 8LL * (v16 & (unsigned int)(37 * *a2)));
      v19 = *v7;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v7 == *a2 )
        goto LABEL_14;
      v20 = 1;
      v21 = 0;
      while ( v19 != -1 )
      {
        if ( !v21 && v19 == -2 )
          v21 = v7;
        v18 = v16 & (v20 + v18);
        v7 = (_DWORD *)(v17 + 8LL * v18);
        v19 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_14;
        ++v20;
      }
LABEL_22:
      if ( v21 )
        v7 = v21;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_2E518D0(a1, v4);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 8);
      v21 = 0;
      v25 = 1;
      v26 = (v22 - 1) & (37 * *a2);
      v7 = (_DWORD *)(v24 + 8LL * (v23 & (unsigned int)(37 * *a2)));
      v27 = *v7;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v7 )
        goto LABEL_14;
      while ( v27 != -1 )
      {
        if ( !v21 && v27 == -2 )
          v21 = v7;
        v26 = v23 & (v25 + v26);
        v7 = (_DWORD *)(v24 + 8LL * v26);
        v27 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_14;
        ++v25;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != -1 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  v7[1] = 0;
  *v7 = v14;
  return v7 + 1;
}
