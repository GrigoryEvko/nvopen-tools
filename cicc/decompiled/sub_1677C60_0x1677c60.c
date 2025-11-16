// Function: sub_1677C60
// Address: 0x1677c60
//
__int64 __fastcall sub_1677C60(__int64 a1, _QWORD *a2)
{
  char v4; // dl
  __int64 v5; // rdi
  int v6; // esi
  unsigned int v7; // eax
  _QWORD *v8; // r9
  __int64 v9; // r8
  unsigned int v11; // esi
  unsigned int v12; // eax
  _QWORD *v13; // r10
  int v14; // ecx
  unsigned int v15; // edi
  __int64 v16; // rax
  int v17; // r11d
  __int64 v18; // rsi
  int v19; // edx
  unsigned int v20; // eax
  __int64 v21; // rdi
  __int64 v22; // rsi
  int v23; // edx
  unsigned int v24; // eax
  __int64 v25; // rdi
  int v26; // r9d
  _QWORD *v27; // r8
  int v28; // edx
  int v29; // edx
  int v30; // r9d

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
  v7 = v6 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = (_QWORD *)(v5 + 8LL * v7);
  v9 = *v8;
  if ( *a2 == *v8 )
    return 0;
  v17 = 1;
  v13 = 0;
  while ( v9 != -8 )
  {
    if ( v13 || v9 != -16 )
      v8 = v13;
    v7 = v6 & (v17 + v7);
    v9 = *(_QWORD *)(v5 + 8LL * v7);
    if ( *a2 == v9 )
      return 0;
    ++v17;
    v13 = v8;
    v8 = (_QWORD *)(v5 + 8LL * v7);
  }
  v12 = *(_DWORD *)(a1 + 8);
  if ( !v13 )
    v13 = v8;
  ++*(_QWORD *)a1;
  v14 = (v12 >> 1) + 1;
  if ( !v4 )
  {
    v11 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
  v15 = 48;
  v11 = 16;
LABEL_9:
  if ( 4 * v14 >= v15 )
  {
    sub_16778A0(a1, 2 * v11);
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
        goto LABEL_55;
      v19 = v28 - 1;
    }
    v20 = v19 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v13 = (_QWORD *)(v18 + 8LL * v20);
    v21 = *v13;
    if ( *v13 != *a2 )
    {
      v30 = 1;
      v27 = 0;
      while ( v21 != -8 )
      {
        if ( v21 == -16 && !v27 )
          v27 = v13;
        v20 = v19 & (v30 + v20);
        v13 = (_QWORD *)(v18 + 8LL * v20);
        v21 = *v13;
        if ( *a2 == *v13 )
          goto LABEL_25;
        ++v30;
      }
      goto LABEL_31;
    }
LABEL_25:
    v12 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  if ( v11 - *(_DWORD *)(a1 + 12) - v14 <= v11 >> 3 )
  {
    sub_16778A0(a1, v11);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v22 = a1 + 16;
      v23 = 15;
      goto LABEL_28;
    }
    v29 = *(_DWORD *)(a1 + 24);
    v22 = *(_QWORD *)(a1 + 16);
    if ( v29 )
    {
      v23 = v29 - 1;
LABEL_28:
      v24 = v23 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v13 = (_QWORD *)(v22 + 8LL * v24);
      v25 = *v13;
      if ( *v13 != *a2 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -8 )
        {
          if ( v25 == -16 && !v27 )
            v27 = v13;
          v24 = v23 & (v26 + v24);
          v13 = (_QWORD *)(v22 + 8LL * v24);
          v25 = *v13;
          if ( *a2 == *v13 )
            goto LABEL_25;
          ++v26;
        }
LABEL_31:
        if ( v27 )
          v13 = v27;
        goto LABEL_25;
      }
      goto LABEL_25;
    }
LABEL_55:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 8) = (2 * (v12 >> 1) + 2) | v12 & 1;
  if ( *v13 != -8 )
    --*(_DWORD *)(a1 + 12);
  *v13 = *a2;
  v16 = *(unsigned int *)(a1 + 152);
  if ( (unsigned int)v16 >= *(_DWORD *)(a1 + 156) )
  {
    sub_16CD150(a1 + 144, a1 + 160, 0, 8);
    v16 = *(unsigned int *)(a1 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * v16) = *a2;
  ++*(_DWORD *)(a1 + 152);
  return 1;
}
