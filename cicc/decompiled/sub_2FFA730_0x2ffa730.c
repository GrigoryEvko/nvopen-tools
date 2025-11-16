// Function: sub_2FFA730
// Address: 0x2ffa730
//
void __fastcall sub_2FFA730(__int64 a1)
{
  unsigned int v2; // r14d
  int v3; // r14d
  unsigned int v4; // esi
  __int64 v5; // rcx
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 i; // r13
  unsigned __int64 v9; // rdi
  __int64 v10; // r13
  unsigned __int64 v11; // rdi
  char v12; // al
  bool v13; // zf
  _DWORD *v14; // rax
  __int64 v15; // rdx
  _DWORD *j; // rdx
  unsigned int v17; // r14d
  unsigned int v18; // edx
  unsigned int v19; // ebx
  __int64 v20; // rdi
  __int64 v21; // rax
  _DWORD *v22; // rax
  __int64 v23; // rdx
  _DWORD *k; // rdx
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
    v6 = a1 + 16;
    v7 = 224;
LABEL_6:
    for ( i = v6 + v7; v6 != i; v6 += 56 )
    {
      if ( *(_DWORD *)v6 != -1 )
      {
        if ( *(_DWORD *)v6 != -2 )
        {
          v9 = *(_QWORD *)(v6 + 8);
          if ( v9 != v6 + 24 )
            _libc_free(v9);
        }
        *(_DWORD *)v6 = -1;
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
  v5 = *(unsigned int *)(a1 + 24);
  v6 = *(_QWORD *)(a1 + 16);
  v7 = 56 * v5;
  if ( v4 >= (unsigned int)v5 || (unsigned int)v5 <= 0x40 )
    goto LABEL_6;
  v10 = v6 + 56 * v5;
  do
  {
    if ( *(_DWORD *)v6 <= 0xFFFFFFFD )
    {
      v11 = *(_QWORD *)(v6 + 8);
      if ( v11 != v6 + 24 )
        _libc_free(v11);
    }
    v6 += 56;
  }
  while ( v6 != v10 );
  v12 = *(_BYTE *)(a1 + 8);
  if ( !v3 )
  {
    if ( (v12 & 1) != 0 )
      goto LABEL_25;
LABEL_46:
    v25 = *(unsigned int *)(a1 + 24);
    if ( (_DWORD)v25 != v3 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 16), 56 * v25, 8);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_37;
    }
LABEL_25:
    v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    if ( v13 )
    {
      v14 = *(_DWORD **)(a1 + 16);
      v15 = 14LL * *(unsigned int *)(a1 + 24);
    }
    else
    {
      v14 = (_DWORD *)(a1 + 16);
      v15 = 56;
    }
    for ( j = &v14[v15]; j != v14; v14 += 14 )
    {
      if ( v14 )
        *v14 = -1;
    }
    return;
  }
  v17 = v3 - 1;
  if ( !v17 )
  {
    v3 = 2;
    if ( (v12 & 1) != 0 )
      goto LABEL_25;
    goto LABEL_46;
  }
  _BitScanReverse(&v18, v17);
  v19 = 1 << (33 - (v18 ^ 0x1F));
  if ( v19 - 5 <= 0x3A )
  {
    if ( (v12 & 1) != 0 )
    {
      v20 = 3584;
      v19 = 64;
      goto LABEL_36;
    }
    v26 = *(unsigned int *)(a1 + 24);
    if ( (_DWORD)v26 == 64 )
      goto LABEL_25;
    v19 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 56 * v26, 8);
    v12 = *(_BYTE *)(a1 + 8);
    goto LABEL_56;
  }
  if ( (v12 & 1) != 0 )
  {
    if ( v19 <= 4 )
      goto LABEL_25;
    v20 = 56LL * v19;
LABEL_36:
    *(_BYTE *)(a1 + 8) = v12 & 0xFE;
    v21 = sub_C7D670(v20, 8);
    *(_DWORD *)(a1 + 24) = v19;
    *(_QWORD *)(a1 + 16) = v21;
    goto LABEL_37;
  }
  v27 = *(unsigned int *)(a1 + 24);
  if ( v19 == (_DWORD)v27 )
    goto LABEL_25;
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 56 * v27, 8);
  v12 = *(_BYTE *)(a1 + 8) | 1;
  *(_BYTE *)(a1 + 8) = v12;
  if ( v19 > 4 )
  {
LABEL_56:
    v20 = 56LL * v19;
    goto LABEL_36;
  }
LABEL_37:
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v13 )
  {
    v22 = *(_DWORD **)(a1 + 16);
    v23 = 14LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v22 = (_DWORD *)(a1 + 16);
    v23 = 56;
  }
  for ( k = &v22[v23]; k != v22; v22 += 14 )
  {
    if ( v22 )
      *v22 = -1;
  }
}
