// Function: sub_25A6840
// Address: 0x25a6840
//
_QWORD *__fastcall sub_25A6840(__int64 a1, __int64 *a2)
{
  int v4; // edi
  __int64 v5; // r9
  int v6; // esi
  unsigned int v7; // eax
  _QWORD *v8; // rdx
  __int64 v9; // r10
  unsigned int v11; // esi
  unsigned int v12; // ecx
  _QWORD *v13; // r8
  int v14; // eax
  unsigned int v15; // r9d
  __int64 v16; // rax
  int v17; // r11d
  __int64 v18; // rdi
  int v19; // ecx
  unsigned int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdi
  int v23; // ecx
  unsigned int v24; // edx
  __int64 v25; // rax
  int v26; // r10d
  _QWORD *v27; // r9
  int v28; // ecx
  int v29; // ecx
  int v30; // r10d

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( v4 )
  {
    v5 = a1 + 16;
    v6 = 15;
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
  v7 = v6 & (*a2 ^ ((unsigned __int64)*a2 >> 9));
  v8 = (_QWORD *)(v5 + 40LL * v7);
  v9 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  v17 = 1;
  v13 = 0;
  while ( v9 != -2 )
  {
    if ( !v13 && v9 == -16 )
      v13 = v8;
    v7 = v6 & (v17 + v7);
    v8 = (_QWORD *)(v5 + 40LL * v7);
    v9 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v17;
  }
  v12 = *(_DWORD *)(a1 + 8);
  v15 = 48;
  v11 = 16;
  if ( !v13 )
    v13 = v8;
  ++*(_QWORD *)a1;
  v14 = (v12 >> 1) + 1;
  if ( !(_BYTE)v4 )
  {
    v11 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
LABEL_9:
  if ( 4 * v14 >= v15 )
  {
    sub_25A5E40(a1, 2 * v11);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v18 = a1 + 16;
      v19 = 15;
    }
    else
    {
      v28 = *(_DWORD *)(a1 + 24);
      v18 = *(_QWORD *)(a1 + 16);
      if ( !v28 )
        goto LABEL_52;
      v19 = v28 - 1;
    }
    v20 = v19 & (*a2 ^ ((unsigned __int64)*a2 >> 9));
    v13 = (_QWORD *)(v18 + 40LL * v20);
    v21 = *v13;
    if ( *a2 != *v13 )
    {
      v30 = 1;
      v27 = 0;
      while ( v21 != -2 )
      {
        if ( !v27 && v21 == -16 )
          v27 = v13;
        v20 = v19 & (v30 + v20);
        v13 = (_QWORD *)(v18 + 40LL * v20);
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
    sub_25A5E40(a1, v11);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v22 = a1 + 16;
      v23 = 15;
      goto LABEL_26;
    }
    v29 = *(_DWORD *)(a1 + 24);
    v22 = *(_QWORD *)(a1 + 16);
    if ( v29 )
    {
      v23 = v29 - 1;
LABEL_26:
      v24 = v23 & (*a2 ^ ((unsigned __int64)*a2 >> 9));
      v13 = (_QWORD *)(v22 + 40LL * v24);
      v25 = *v13;
      if ( *a2 != *v13 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -2 )
        {
          if ( v25 == -16 && !v27 )
            v27 = v13;
          v24 = v23 & (v26 + v24);
          v13 = (_QWORD *)(v22 + 40LL * v24);
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
  if ( *v13 != -2 )
    --*(_DWORD *)(a1 + 12);
  v16 = *a2;
  *(_OWORD *)(v13 + 1) = 0;
  *v13 = v16;
  *(_OWORD *)(v13 + 3) = 0;
  return v13 + 1;
}
