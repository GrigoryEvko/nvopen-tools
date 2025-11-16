// Function: sub_349F740
// Address: 0x349f740
//
void __fastcall sub_349F740(__int64 a1)
{
  unsigned int v2; // ebx
  int v3; // ebx
  unsigned int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r13
  unsigned __int64 v9; // rdi
  char v10; // dl
  unsigned int v11; // ebx
  unsigned int v12; // eax
  unsigned int v13; // ebx
  __int64 v14; // rdi
  __int64 v15; // rax
  bool v16; // zf
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 i; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 j; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax

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
LABEL_25:
    v6 = a1 + 16;
    v7 = 576;
LABEL_7:
    v8 = v6 + v7;
    if ( v8 == v6 )
    {
LABEL_19:
      *(_QWORD *)(a1 + 8) &= 1uLL;
      return;
    }
    while ( 1 )
    {
      if ( !*(_QWORD *)v6 )
      {
        if ( *(_BYTE *)(v6 + 24) )
        {
          if ( !*(_QWORD *)(v6 + 8) && !*(_QWORD *)(v6 + 16) && !*(_QWORD *)(v6 + 32) )
            goto LABEL_11;
        }
        else if ( !*(_QWORD *)(v6 + 32) )
        {
          goto LABEL_12;
        }
      }
      v9 = *(_QWORD *)(v6 + 40);
      if ( v9 != v6 + 56 )
        _libc_free(v9);
LABEL_11:
      *(_QWORD *)v6 = 0;
      *(_BYTE *)(v6 + 24) = 0;
      *(_QWORD *)(v6 + 32) = 0;
LABEL_12:
      v6 += 72;
      if ( v6 == v8 )
        goto LABEL_19;
    }
  }
  if ( !*(_DWORD *)(a1 + 12) )
    return;
  v4 = 0;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    goto LABEL_25;
LABEL_4:
  v5 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)v5 <= v4 || (unsigned int)v5 <= 0x40 )
  {
    v6 = *(_QWORD *)(a1 + 16);
    v7 = 72 * v5;
    goto LABEL_7;
  }
  sub_349F6B0(a1);
  if ( !v3 )
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      goto LABEL_40;
LABEL_49:
    v23 = *(unsigned int *)(a1 + 24);
    if ( v3 == (_DWORD)v23 )
      goto LABEL_40;
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 72 * v23, 8);
    *(_BYTE *)(a1 + 8) |= 1u;
    goto LABEL_32;
  }
  v10 = *(_BYTE *)(a1 + 8);
  v11 = v3 - 1;
  if ( !v11 )
  {
    v3 = 2;
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      goto LABEL_40;
    goto LABEL_49;
  }
  _BitScanReverse(&v12, v11);
  v13 = 1 << (33 - (v12 ^ 0x1F));
  if ( v13 - 9 > 0x36 )
  {
    if ( (v10 & 1) != 0 )
    {
      if ( v13 > 8 )
      {
        v14 = 72LL * v13;
        goto LABEL_31;
      }
      goto LABEL_40;
    }
    v25 = *(unsigned int *)(a1 + 24);
    if ( v13 == (_DWORD)v25 )
      goto LABEL_40;
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 72 * v25, 8);
    v10 = *(_BYTE *)(a1 + 8) | 1;
    *(_BYTE *)(a1 + 8) = v10;
    if ( v13 > 8 )
      goto LABEL_59;
LABEL_32:
    v16 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    if ( v16 )
    {
      v17 = *(_QWORD *)(a1 + 16);
      v18 = 72LL * *(unsigned int *)(a1 + 24);
    }
    else
    {
      v17 = a1 + 16;
      v18 = 576;
    }
    for ( i = v17 + v18; i != v17; v17 += 72 )
    {
      if ( v17 )
      {
        *(_QWORD *)v17 = 0;
        *(_BYTE *)(v17 + 24) = 0;
        *(_QWORD *)(v17 + 32) = 0;
      }
    }
    return;
  }
  if ( (v10 & 1) != 0 )
  {
    v14 = 4608;
    v13 = 64;
LABEL_31:
    *(_BYTE *)(a1 + 8) = v10 & 0xFE;
    v15 = sub_C7D670(v14, 8);
    *(_DWORD *)(a1 + 24) = v13;
    *(_QWORD *)(a1 + 16) = v15;
    goto LABEL_32;
  }
  v24 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v24 != 64 )
  {
    v13 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 72 * v24, 8);
    v10 = *(_BYTE *)(a1 + 8);
LABEL_59:
    v14 = 72LL * v13;
    goto LABEL_31;
  }
LABEL_40:
  v16 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v16 )
  {
    v20 = *(_QWORD *)(a1 + 16);
    v21 = 72LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v20 = a1 + 16;
    v21 = 576;
  }
  for ( j = v20 + v21; j != v20; v20 += 72 )
  {
    if ( v20 )
    {
      *(_QWORD *)v20 = 0;
      *(_BYTE *)(v20 + 24) = 0;
      *(_QWORD *)(v20 + 32) = 0;
    }
  }
}
