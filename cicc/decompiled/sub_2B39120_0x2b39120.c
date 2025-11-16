// Function: sub_2B39120
// Address: 0x2b39120
//
void __fastcall sub_2B39120(__int64 a1)
{
  unsigned int v2; // r14d
  int v3; // r14d
  unsigned int v4; // ecx
  _QWORD *v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // r13
  _QWORD *i; // r13
  unsigned __int64 v9; // rdi
  _QWORD *v10; // r13
  unsigned __int64 v11; // rdi
  char v12; // al
  bool v13; // zf
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *j; // rdx
  unsigned int v17; // r14d
  unsigned int v18; // edx
  unsigned int v19; // ebx
  __int64 v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rdx
  _QWORD *k; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax

  v2 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v3 = v2 >> 1;
  if ( v3 )
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    {
      v4 = 4 * v3;
      goto LABEL_4;
    }
LABEL_17:
    v5 = (_QWORD *)(a1 + 16);
    v7 = 36;
LABEL_6:
    for ( i = &v5[v7]; v5 != i; v5 += 9 )
    {
      if ( *v5 != -4096 )
      {
        if ( *v5 != -8192 )
        {
          v9 = v5[1];
          if ( (_QWORD *)v9 != v5 + 3 )
            _libc_free(v9);
        }
        *v5 = -4096;
      }
    }
    *(_QWORD *)(a1 + 8) &= 1uLL;
    return;
  }
  if ( !*(_DWORD *)(a1 + 12) )
    return;
  v4 = 0;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    goto LABEL_17;
LABEL_4:
  v5 = *(_QWORD **)(a1 + 16);
  v6 = *(unsigned int *)(a1 + 24);
  v7 = 9 * v6;
  if ( (unsigned int)v6 <= v4 || (unsigned int)v6 <= 0x40 )
    goto LABEL_6;
  v10 = &v5[9 * v6];
  do
  {
    if ( *v5 != -4096 && *v5 != -8192 )
    {
      v11 = v5[1];
      if ( (_QWORD *)v11 != v5 + 3 )
        _libc_free(v11);
    }
    v5 += 9;
  }
  while ( v5 != v10 );
  v12 = *(_BYTE *)(a1 + 8);
  if ( !v3 )
  {
    if ( (v12 & 1) != 0 )
      goto LABEL_26;
LABEL_47:
    v25 = *(unsigned int *)(a1 + 24);
    if ( (_DWORD)v25 != v3 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 16), 72 * v25, 8);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_38;
    }
LABEL_26:
    v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    if ( v13 )
    {
      v14 = *(_QWORD **)(a1 + 16);
      v15 = 9LL * *(unsigned int *)(a1 + 24);
    }
    else
    {
      v14 = (_QWORD *)(a1 + 16);
      v15 = 36;
    }
    for ( j = &v14[v15]; j != v14; v14 += 9 )
    {
      if ( v14 )
        *v14 = -4096;
    }
    return;
  }
  v17 = v3 - 1;
  if ( !v17 )
  {
    v3 = 2;
    if ( (v12 & 1) != 0 )
      goto LABEL_26;
    goto LABEL_47;
  }
  _BitScanReverse(&v18, v17);
  v19 = 1 << (33 - (v18 ^ 0x1F));
  if ( v19 - 5 <= 0x3A )
  {
    if ( (v12 & 1) != 0 )
    {
      v20 = 4608;
      v19 = 64;
      goto LABEL_37;
    }
    v26 = *(unsigned int *)(a1 + 24);
    if ( (_DWORD)v26 == 64 )
      goto LABEL_26;
    v19 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 72 * v26, 8);
    v12 = *(_BYTE *)(a1 + 8);
    goto LABEL_57;
  }
  if ( (v12 & 1) != 0 )
  {
    if ( v19 <= 4 )
      goto LABEL_26;
    v20 = 72LL * v19;
LABEL_37:
    *(_BYTE *)(a1 + 8) = v12 & 0xFE;
    v21 = sub_C7D670(v20, 8);
    *(_DWORD *)(a1 + 24) = v19;
    *(_QWORD *)(a1 + 16) = v21;
    goto LABEL_38;
  }
  v27 = *(unsigned int *)(a1 + 24);
  if ( v19 == (_DWORD)v27 )
    goto LABEL_26;
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 72 * v27, 8);
  v12 = *(_BYTE *)(a1 + 8) | 1;
  *(_BYTE *)(a1 + 8) = v12;
  if ( v19 > 4 )
  {
LABEL_57:
    v20 = 72LL * v19;
    goto LABEL_37;
  }
LABEL_38:
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v13 )
  {
    v22 = *(_QWORD **)(a1 + 16);
    v23 = 9LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v22 = (_QWORD *)(a1 + 16);
    v23 = 36;
  }
  for ( k = &v22[v23]; k != v22; v22 += 9 )
  {
    if ( v22 )
      *v22 = -4096;
  }
}
