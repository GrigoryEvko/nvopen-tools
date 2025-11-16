// Function: sub_1517390
// Address: 0x1517390
//
_QWORD *__fastcall sub_1517390(__int64 a1)
{
  int v2; // r14d
  _QWORD *v3; // rbx
  __int64 v4; // r13
  _QWORD *v5; // r13
  char v6; // al
  char v7; // dl
  unsigned int v8; // r14d
  unsigned int v9; // ebx
  __int64 v10; // rdi
  bool v11; // zf
  _QWORD *result; // rax
  __int64 v13; // rdx
  _QWORD *j; // rdx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  _QWORD *i; // rdx
  unsigned int v19; // r14d
  unsigned int v20; // eax

  v2 = *(_DWORD *)(a1 + 8) >> 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v3 = (_QWORD *)(a1 + 16);
    v4 = 2;
  }
  else
  {
    v15 = *(unsigned int *)(a1 + 24);
    if ( !(_DWORD)v15 )
    {
      if ( !v2 )
        goto LABEL_16;
      v19 = v2 - 1;
      if ( v19 )
      {
        _BitScanReverse(&v20, v19);
        v9 = 1 << (33 - (v20 ^ 0x1F));
        if ( v9 - 2 > 0x3D )
          goto LABEL_41;
      }
LABEL_38:
      if ( *(_DWORD *)(a1 + 24) == 64 )
        goto LABEL_16;
      j___libc_free_0(*(_QWORD *)(a1 + 16));
LABEL_26:
      v6 = *(_BYTE *)(a1 + 8);
      v10 = 1024;
      v9 = 64;
      goto LABEL_27;
    }
    v3 = *(_QWORD **)(a1 + 16);
    v4 = 2 * v15;
  }
  v5 = &v3[v4];
  do
  {
    if ( *v3 != -16 && *v3 != -8 && v3[1] )
      sub_16307F0();
    v3 += 2;
  }
  while ( v5 != v3 );
  v6 = *(_BYTE *)(a1 + 8);
  v7 = v6 & 1;
  if ( !v2 )
  {
    if ( v7 || !*(_DWORD *)(a1 + 24) )
      goto LABEL_16;
    j___libc_free_0(*(_QWORD *)(a1 + 16));
    *(_BYTE *)(a1 + 8) |= 1u;
    goto LABEL_28;
  }
  v8 = v2 - 1;
  if ( !v8 || (_BitScanReverse(&v8, v8), v9 = 1 << (33 - (v8 ^ 0x1F)), v9 - 2 <= 0x3D) )
  {
    if ( v7 )
      goto LABEL_26;
    goto LABEL_38;
  }
  if ( !v7 )
  {
LABEL_41:
    if ( *(_DWORD *)(a1 + 24) == v9 )
      goto LABEL_16;
    j___libc_free_0(*(_QWORD *)(a1 + 16));
    v6 = *(_BYTE *)(a1 + 8) | 1;
    *(_BYTE *)(a1 + 8) = v6;
    if ( v9 > 1 )
      goto LABEL_14;
LABEL_28:
    v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    if ( v11 )
    {
      result = *(_QWORD **)(a1 + 16);
      v17 = 2LL * *(unsigned int *)(a1 + 24);
    }
    else
    {
      result = (_QWORD *)(a1 + 16);
      v17 = 2;
    }
    for ( i = &result[v17]; i != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
    return result;
  }
  if ( v9 > 1 )
  {
LABEL_14:
    v10 = 16LL * v9;
LABEL_27:
    *(_BYTE *)(a1 + 8) = v6 & 0xFE;
    v16 = sub_22077B0(v10);
    *(_DWORD *)(a1 + 24) = v9;
    *(_QWORD *)(a1 + 16) = v16;
    goto LABEL_28;
  }
LABEL_16:
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    result = *(_QWORD **)(a1 + 16);
    v13 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = (_QWORD *)(a1 + 16);
    v13 = 2;
  }
  for ( j = &result[v13]; j != result; result += 2 )
  {
    if ( result )
      *result = -8;
  }
  return result;
}
