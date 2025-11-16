// Function: sub_18E7390
// Address: 0x18e7390
//
void __fastcall sub_18E7390(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // rax
  unsigned __int64 v5; // r15
  unsigned __int64 *v6; // rbx
  unsigned int v7; // eax
  unsigned int v8; // eax
  unsigned int v9; // ecx
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *i; // rdx
  unsigned __int64 *v14; // r12
  unsigned __int64 *v15; // r14
  unsigned __int64 *v16; // rbx
  bool v17; // zf
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *j; // rdx
  unsigned int v21; // eax
  unsigned int v22; // ebx
  char v23; // al
  __int64 v24; // rdi
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rdx

  v2 = *(_QWORD *)(a1 + 296);
  v3 = v2 + 632LL * *(unsigned int *)(a1 + 304);
  while ( v2 != v3 )
  {
    while ( 1 )
    {
      v4 = *(unsigned int *)(v3 - 616);
      v5 = *(_QWORD *)(v3 - 624);
      v3 -= 632;
      v6 = (unsigned __int64 *)(v5 + 152 * v4);
      if ( (unsigned __int64 *)v5 != v6 )
      {
        do
        {
          v6 -= 19;
          if ( (unsigned __int64 *)*v6 != v6 + 2 )
            _libc_free(*v6);
        }
        while ( (unsigned __int64 *)v5 != v6 );
        v5 = *(_QWORD *)(v3 + 8);
      }
      if ( v5 == v3 + 24 )
        break;
      _libc_free(v5);
      if ( v2 == v3 )
        goto LABEL_10;
    }
  }
LABEL_10:
  v7 = *(_DWORD *)(a1 + 224);
  ++*(_QWORD *)(a1 + 216);
  *(_DWORD *)(a1 + 304) = 0;
  v8 = v7 >> 1;
  if ( v8 )
  {
    if ( (*(_BYTE *)(a1 + 224) & 1) == 0 )
    {
      v9 = 4 * v8;
      goto LABEL_13;
    }
LABEL_28:
    v11 = (_QWORD *)(a1 + 232);
    v12 = 8;
    goto LABEL_16;
  }
  if ( !*(_DWORD *)(a1 + 228) )
    goto LABEL_19;
  v9 = 0;
  if ( (*(_BYTE *)(a1 + 224) & 1) != 0 )
    goto LABEL_28;
LABEL_13:
  v10 = *(unsigned int *)(a1 + 240);
  if ( (unsigned int)v10 <= v9 || (unsigned int)v10 <= 0x40 )
  {
    v11 = *(_QWORD **)(a1 + 232);
    v12 = 2 * v10;
LABEL_16:
    for ( i = &v11[v12]; i != v11; v11 += 2 )
      *v11 = -8;
    *(_QWORD *)(a1 + 224) &= 1uLL;
    goto LABEL_19;
  }
  if ( !v8 || (v21 = v8 - 1) == 0 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 232));
    *(_BYTE *)(a1 + 224) |= 1u;
    goto LABEL_31;
  }
  _BitScanReverse(&v21, v21);
  v22 = 1 << (33 - (v21 ^ 0x1F));
  if ( v22 - 5 <= 0x3A )
  {
    v22 = 64;
    j___libc_free_0(*(_QWORD *)(a1 + 232));
    v23 = *(_BYTE *)(a1 + 224);
    v24 = 1024;
    goto LABEL_43;
  }
  if ( (_DWORD)v10 != v22 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 232));
    v23 = *(_BYTE *)(a1 + 224) | 1;
    *(_BYTE *)(a1 + 224) = v23;
    if ( v22 <= 4 )
      goto LABEL_31;
    v24 = 16LL * v22;
LABEL_43:
    *(_BYTE *)(a1 + 224) = v23 & 0xFE;
    v25 = sub_22077B0(v24);
    *(_DWORD *)(a1 + 240) = v22;
    *(_QWORD *)(a1 + 232) = v25;
LABEL_31:
    v17 = (*(_QWORD *)(a1 + 224) & 1LL) == 0;
    *(_QWORD *)(a1 + 224) &= 1uLL;
    if ( v17 )
    {
      v18 = *(_QWORD **)(a1 + 232);
      v19 = 2LL * *(unsigned int *)(a1 + 240);
    }
    else
    {
      v18 = (_QWORD *)(a1 + 232);
      v19 = 8;
    }
    for ( j = &v18[v19]; j != v18; v18 += 2 )
    {
      if ( v18 )
        *v18 = -8;
    }
    goto LABEL_19;
  }
  v17 = (*(_QWORD *)(a1 + 224) & 1LL) == 0;
  *(_QWORD *)(a1 + 224) &= 1uLL;
  if ( v17 )
  {
    v26 = *(_QWORD **)(a1 + 232);
    v27 = 2 * v10;
  }
  else
  {
    v26 = (_QWORD *)(a1 + 232);
    v27 = 8;
  }
  v28 = &v26[v27];
  do
  {
    if ( v26 )
      *v26 = -8;
    v26 += 2;
  }
  while ( v28 != v26 );
LABEL_19:
  v14 = *(unsigned __int64 **)(a1 + 192);
  v15 = *(unsigned __int64 **)(a1 + 200);
  if ( v14 != v15 )
  {
    v16 = *(unsigned __int64 **)(a1 + 192);
    do
    {
      if ( (unsigned __int64 *)*v16 != v16 + 2 )
        _libc_free(*v16);
      v16 += 20;
    }
    while ( v15 != v16 );
    *(_QWORD *)(a1 + 200) = v14;
  }
}
