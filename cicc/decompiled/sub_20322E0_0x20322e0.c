// Function: sub_20322E0
// Address: 0x20322e0
//
_DWORD *__fastcall sub_20322E0(__int64 a1, _DWORD *a2)
{
  char v4; // dl
  __int64 v5; // r8
  int v6; // esi
  unsigned int v7; // edi
  _DWORD *result; // rax
  int v9; // r9d
  unsigned int v10; // esi
  unsigned int v11; // ecx
  int v12; // edi
  unsigned int v13; // r8d
  int v14; // r11d
  _DWORD *v15; // r10
  __int64 v16; // rdi
  int v17; // ecx
  unsigned int v18; // esi
  int v19; // r8d
  __int64 v20; // rdi
  int v21; // esi
  unsigned int v22; // edx
  int v23; // r8d
  int v24; // r10d
  _DWORD *v25; // r9
  int v26; // ecx
  int v27; // esi
  int v28; // r10d

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( v4 )
  {
    v5 = a1 + 16;
    v6 = 7;
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
  v7 = v6 & (37 * *a2);
  result = (_DWORD *)(v5 + 8LL * v7);
  v9 = *result;
  if ( *a2 == *result )
    return result;
  v14 = 1;
  v15 = 0;
  while ( v9 != -1 )
  {
    if ( !v15 && v9 == -2 )
      v15 = result;
    v7 = v6 & (v14 + v7);
    result = (_DWORD *)(v5 + 8LL * v7);
    v9 = *result;
    if ( *a2 == *result )
      return result;
    ++v14;
  }
  v11 = *(_DWORD *)(a1 + 8);
  v13 = 24;
  v10 = 8;
  if ( v15 )
    result = v15;
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
    sub_20108A0(a1, 2 * v10);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v16 = a1 + 16;
      v17 = 7;
    }
    else
    {
      v26 = *(_DWORD *)(a1 + 24);
      v16 = *(_QWORD *)(a1 + 16);
      if ( !v26 )
        goto LABEL_52;
      v17 = v26 - 1;
    }
    v18 = v17 & (37 * *a2);
    result = (_DWORD *)(v16 + 8LL * v18);
    v19 = *result;
    if ( *a2 != *result )
    {
      v28 = 1;
      v25 = 0;
      while ( v19 != -1 )
      {
        if ( !v25 && v19 == -2 )
          v25 = result;
        v18 = v17 & (v28 + v18);
        result = (_DWORD *)(v16 + 8LL * v18);
        v19 = *result;
        if ( *a2 == *result )
          goto LABEL_23;
        ++v28;
      }
      goto LABEL_29;
    }
LABEL_23:
    v11 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  if ( v10 - *(_DWORD *)(a1 + 12) - v12 <= v10 >> 3 )
  {
    sub_20108A0(a1, v10);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v20 = a1 + 16;
      v21 = 7;
      goto LABEL_26;
    }
    v27 = *(_DWORD *)(a1 + 24);
    v20 = *(_QWORD *)(a1 + 16);
    if ( v27 )
    {
      v21 = v27 - 1;
LABEL_26:
      v22 = v21 & (37 * *a2);
      result = (_DWORD *)(v20 + 8LL * v22);
      v23 = *result;
      if ( *a2 != *result )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -1 )
        {
          if ( v23 == -2 && !v25 )
            v25 = result;
          v22 = v21 & (v24 + v22);
          result = (_DWORD *)(v20 + 8LL * v22);
          v23 = *result;
          if ( *a2 == *result )
            goto LABEL_23;
          ++v24;
        }
LABEL_29:
        if ( v25 )
          result = v25;
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
  if ( *result != -1 )
    --*(_DWORD *)(a1 + 12);
  *(_QWORD *)result = (unsigned int)*a2;
  return result;
}
