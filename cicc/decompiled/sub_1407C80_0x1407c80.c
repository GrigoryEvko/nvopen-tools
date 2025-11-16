// Function: sub_1407C80
// Address: 0x1407c80
//
void __fastcall sub_1407C80(__int64 a1)
{
  int v2; // r14d
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rdx
  int v9; // ebx
  unsigned int v10; // r14d
  unsigned int v11; // eax
  _QWORD *v12; // rdi
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *i; // rdx
  _QWORD *v18; // rax

  v2 = *(_DWORD *)(a1 + 184);
  ++*(_QWORD *)(a1 + 168);
  if ( !v2 && !*(_DWORD *)(a1 + 188) )
    goto LABEL_17;
  v3 = *(_QWORD *)(a1 + 176);
  v4 = v3 + 168LL * *(unsigned int *)(a1 + 192);
  v5 = 4 * v2;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v5 = 64;
  if ( *(_DWORD *)(a1 + 192) <= v5 )
  {
    while ( v3 != v4 )
    {
      if ( *(_QWORD *)v3 != -8 )
      {
        if ( *(_QWORD *)v3 != -16 )
        {
          v6 = *(_QWORD *)(v3 + 88);
          if ( v6 != v3 + 104 )
            _libc_free(v6);
          if ( (*(_BYTE *)(v3 + 16) & 1) == 0 )
            j___libc_free_0(*(_QWORD *)(v3 + 24));
        }
        *(_QWORD *)v3 = -8;
      }
      v3 += 168;
    }
LABEL_16:
    *(_QWORD *)(a1 + 184) = 0;
    goto LABEL_17;
  }
  do
  {
    while ( *(_QWORD *)v3 == -16 )
    {
LABEL_22:
      v3 += 168;
      if ( v3 == v4 )
        goto LABEL_26;
    }
    if ( *(_QWORD *)v3 != -8 )
    {
      v7 = *(_QWORD *)(v3 + 88);
      if ( v7 != v3 + 104 )
        _libc_free(v7);
      if ( (*(_BYTE *)(v3 + 16) & 1) == 0 )
        j___libc_free_0(*(_QWORD *)(v3 + 24));
      goto LABEL_22;
    }
    v3 += 168;
  }
  while ( v3 != v4 );
LABEL_26:
  v8 = *(unsigned int *)(a1 + 192);
  if ( !v2 )
  {
    if ( (_DWORD)v8 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 176));
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 184) = 0;
      *(_DWORD *)(a1 + 192) = 0;
      goto LABEL_17;
    }
    goto LABEL_16;
  }
  v9 = 64;
  v10 = v2 - 1;
  if ( v10 )
  {
    _BitScanReverse(&v11, v10);
    v9 = 1 << (33 - (v11 ^ 0x1F));
    if ( v9 < 64 )
      v9 = 64;
  }
  v12 = *(_QWORD **)(a1 + 176);
  if ( (_DWORD)v8 == v9 )
  {
    *(_QWORD *)(a1 + 184) = 0;
    v18 = &v12[21 * v8];
    do
    {
      if ( v12 )
        *v12 = -8;
      v12 += 21;
    }
    while ( v18 != v12 );
  }
  else
  {
    j___libc_free_0(v12);
    v13 = (((((((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
            | (4 * v9 / 3u + 1)
            | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 4)
          | (((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
          | (4 * v9 / 3u + 1)
          | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
          | (4 * v9 / 3u + 1)
          | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 4)
        | (((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
        | (4 * v9 / 3u + 1)
        | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1);
    v14 = ((v13 >> 16) | v13) + 1;
    *(_DWORD *)(a1 + 192) = v14;
    v15 = (_QWORD *)sub_22077B0(168 * v14);
    v16 = *(unsigned int *)(a1 + 192);
    *(_QWORD *)(a1 + 184) = 0;
    *(_QWORD *)(a1 + 176) = v15;
    for ( i = &v15[21 * v16]; i != v15; v15 += 21 )
    {
      if ( v15 )
        *v15 = -8;
    }
  }
LABEL_17:
  *(_QWORD *)(a1 + 160) = 0;
}
