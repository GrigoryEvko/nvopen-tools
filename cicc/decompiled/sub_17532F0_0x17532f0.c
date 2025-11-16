// Function: sub_17532F0
// Address: 0x17532f0
//
_QWORD *__fastcall sub_17532F0(__int64 a1, __int64 *a2)
{
  char v4; // cl
  __int64 v5; // r8
  int v6; // esi
  unsigned int v7; // edx
  _QWORD *result; // rax
  __int64 v9; // r9
  unsigned int v10; // esi
  unsigned int v11; // edx
  int v12; // edi
  unsigned int v13; // r8d
  __int64 v14; // rdx
  int v15; // r11d
  _QWORD *v16; // r10
  __int64 v17; // rdi
  int v18; // ecx
  unsigned int v19; // edx
  __int64 v20; // r8
  __int64 v21; // rdi
  int v22; // ecx
  unsigned int v23; // edx
  __int64 v24; // r8
  int v25; // r10d
  _QWORD *v26; // r9
  int v27; // ecx
  int v28; // ecx
  int v29; // r10d

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( v4 )
  {
    v5 = a1 + 16;
    v6 = 3;
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    if ( !v10 )
    {
      v11 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      result = 0;
      v12 = (v11 >> 1) + 1;
LABEL_8:
      v13 = 3 * v10;
      goto LABEL_9;
    }
    v6 = v10 - 1;
  }
  v7 = v6 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  result = (_QWORD *)(v5 + 16LL * v7);
  v9 = *result;
  if ( *a2 == *result )
    return result;
  v15 = 1;
  v16 = 0;
  while ( v9 != -8 )
  {
    if ( !v16 && v9 == -16 )
      v16 = result;
    v7 = v6 & (v15 + v7);
    result = (_QWORD *)(v5 + 16LL * v7);
    v9 = *result;
    if ( *a2 == *result )
      return result;
    ++v15;
  }
  v11 = *(_DWORD *)(a1 + 8);
  v13 = 12;
  v10 = 4;
  if ( v16 )
    result = v16;
  ++*(_QWORD *)a1;
  v12 = (v11 >> 1) + 1;
  if ( !v4 )
  {
    v10 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
LABEL_9:
  if ( 4 * v12 >= v13 )
  {
    sub_1752F10(a1, 2 * v10);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v17 = a1 + 16;
      v18 = 3;
    }
    else
    {
      v27 = *(_DWORD *)(a1 + 24);
      v17 = *(_QWORD *)(a1 + 16);
      if ( !v27 )
        goto LABEL_52;
      v18 = v27 - 1;
    }
    v19 = v18 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    result = (_QWORD *)(v17 + 16LL * v19);
    v20 = *result;
    if ( *a2 != *result )
    {
      v29 = 1;
      v26 = 0;
      while ( v20 != -8 )
      {
        if ( !v26 && v20 == -16 )
          v26 = result;
        v19 = v18 & (v29 + v19);
        result = (_QWORD *)(v17 + 16LL * v19);
        v20 = *result;
        if ( *a2 == *result )
          goto LABEL_23;
        ++v29;
      }
      goto LABEL_29;
    }
LABEL_23:
    v11 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  if ( v10 - *(_DWORD *)(a1 + 12) - v12 <= v10 >> 3 )
  {
    sub_1752F10(a1, v10);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v21 = a1 + 16;
      v22 = 3;
      goto LABEL_26;
    }
    v28 = *(_DWORD *)(a1 + 24);
    v21 = *(_QWORD *)(a1 + 16);
    if ( v28 )
    {
      v22 = v28 - 1;
LABEL_26:
      v23 = v22 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      result = (_QWORD *)(v21 + 16LL * v23);
      v24 = *result;
      if ( *a2 != *result )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -8 )
        {
          if ( v24 == -16 && !v26 )
            v26 = result;
          v23 = v22 & (v25 + v23);
          result = (_QWORD *)(v21 + 16LL * v23);
          v24 = *result;
          if ( *a2 == *result )
            goto LABEL_23;
          ++v25;
        }
LABEL_29:
        if ( v26 )
          result = v26;
        goto LABEL_23;
      }
      goto LABEL_23;
    }
LABEL_52:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 8) = (2 * (v11 >> 1) + 2) | v11 & 1;
  if ( *result != -8 )
    --*(_DWORD *)(a1 + 12);
  v14 = *a2;
  result[1] = 0;
  *result = v14;
  return result;
}
