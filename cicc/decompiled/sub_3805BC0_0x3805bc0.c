// Function: sub_3805BC0
// Address: 0x3805bc0
//
_DWORD *__fastcall sub_3805BC0(__int64 a1, int *a2)
{
  char v4; // dl
  __int64 v5; // r9
  int v6; // esi
  unsigned int v7; // edi
  _DWORD *v8; // rax
  int v9; // r10d
  unsigned int v11; // esi
  unsigned int v12; // ecx
  _DWORD *v13; // r8
  int v14; // eax
  unsigned int v15; // edi
  int v16; // eax
  int v17; // r11d
  __int64 v18; // rsi
  int v19; // edx
  unsigned int v20; // ecx
  int v21; // edi
  __int64 v22; // rsi
  int v23; // ecx
  unsigned int v24; // eax
  int v25; // edi
  int v26; // r10d
  _DWORD *v27; // r9
  int v28; // edx
  int v29; // ecx
  int v30; // r10d

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( v4 )
  {
    v5 = a1 + 16;
    v6 = 7;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    if ( !v11 )
    {
      v12 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v13 = 0;
      v14 = (v12 >> 1) + 1;
LABEL_8:
      v15 = 3 * v11;
      goto LABEL_9;
    }
    v6 = v11 - 1;
  }
  v7 = v6 & (37 * *a2);
  v8 = (_DWORD *)(v5 + 8LL * v7);
  v9 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  v17 = 1;
  v13 = 0;
  while ( v9 != -1 )
  {
    if ( !v13 && v9 == -2 )
      v13 = v8;
    v7 = v6 & (v17 + v7);
    v8 = (_DWORD *)(v5 + 8LL * v7);
    v9 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v17;
  }
  v12 = *(_DWORD *)(a1 + 8);
  v15 = 24;
  v11 = 8;
  if ( !v13 )
    v13 = v8;
  ++*(_QWORD *)a1;
  v14 = (v12 >> 1) + 1;
  if ( !v4 )
  {
    v11 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
LABEL_9:
  if ( 4 * v14 >= v15 )
  {
    sub_375BDE0(a1, 2 * v11);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v18 = a1 + 16;
      v19 = 7;
    }
    else
    {
      v28 = *(_DWORD *)(a1 + 24);
      v18 = *(_QWORD *)(a1 + 16);
      if ( !v28 )
        goto LABEL_52;
      v19 = v28 - 1;
    }
    v20 = v19 & (37 * *a2);
    v13 = (_DWORD *)(v18 + 8LL * v20);
    v21 = *v13;
    if ( *a2 != *v13 )
    {
      v30 = 1;
      v27 = 0;
      while ( v21 != -1 )
      {
        if ( !v27 && v21 == -2 )
          v27 = v13;
        v20 = v19 & (v30 + v20);
        v13 = (_DWORD *)(v18 + 8LL * v20);
        v21 = *v13;
        if ( *a2 == *v13 )
          goto LABEL_23;
        ++v30;
      }
      goto LABEL_29;
    }
LABEL_23:
    v12 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  if ( v11 - *(_DWORD *)(a1 + 12) - v14 <= v11 >> 3 )
  {
    sub_375BDE0(a1, v11);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v22 = a1 + 16;
      v23 = 7;
      goto LABEL_26;
    }
    v29 = *(_DWORD *)(a1 + 24);
    v22 = *(_QWORD *)(a1 + 16);
    if ( v29 )
    {
      v23 = v29 - 1;
LABEL_26:
      v24 = v23 & (37 * *a2);
      v13 = (_DWORD *)(v22 + 8LL * v24);
      v25 = *v13;
      if ( *a2 != *v13 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -1 )
        {
          if ( v25 == -2 && !v27 )
            v27 = v13;
          v24 = v23 & (v26 + v24);
          v13 = (_DWORD *)(v22 + 8LL * v24);
          v25 = *v13;
          if ( *a2 == *v13 )
            goto LABEL_23;
          ++v26;
        }
LABEL_29:
        if ( v27 )
          v13 = v27;
        goto LABEL_23;
      }
      goto LABEL_23;
    }
LABEL_52:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 8) = (2 * (v12 >> 1) + 2) | v12 & 1;
  if ( *v13 != -1 )
    --*(_DWORD *)(a1 + 12);
  v16 = *a2;
  v13[1] = 0;
  *v13 = v16;
  return v13 + 1;
}
