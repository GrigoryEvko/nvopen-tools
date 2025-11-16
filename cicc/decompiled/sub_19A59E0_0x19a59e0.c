// Function: sub_19A59E0
// Address: 0x19a59e0
//
_QWORD *__fastcall sub_19A59E0(__int64 a1)
{
  unsigned int v1; // eax
  _QWORD *result; // rax
  unsigned int v4; // ecx
  __int64 v5; // rdx
  __int64 v6; // rdx
  _QWORD *i; // rdx
  unsigned int v8; // eax
  unsigned int v9; // r12d
  char v10; // al
  __int64 v11; // rdi
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rdx
  _QWORD *j; // rdx
  __int64 v16; // rdx
  _QWORD *v17; // rdx

  v1 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  result = (_QWORD *)(v1 >> 1);
  if ( (_DWORD)result )
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    {
      v4 = 4 * (_DWORD)result;
      goto LABEL_4;
    }
LABEL_13:
    result = (_QWORD *)(a1 + 16);
    v6 = 4;
    goto LABEL_7;
  }
  if ( !*(_DWORD *)(a1 + 12) )
    return result;
  v4 = 0;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    goto LABEL_13;
LABEL_4:
  v5 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)v5 <= v4 || (unsigned int)v5 <= 0x40 )
  {
    result = *(_QWORD **)(a1 + 16);
    v6 = v5;
LABEL_7:
    for ( i = &result[v6]; result != i; ++result )
      *result = -8;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    return result;
  }
  if ( !(_DWORD)result || (v8 = (_DWORD)result - 1) == 0 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 16));
    *(_BYTE *)(a1 + 8) |= 1u;
LABEL_21:
    v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    if ( v13 )
    {
      result = *(_QWORD **)(a1 + 16);
      v14 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      result = (_QWORD *)(a1 + 16);
      v14 = 4;
    }
    for ( j = &result[v14]; j != result; ++result )
    {
      if ( result )
        *result = -8;
    }
    return result;
  }
  _BitScanReverse(&v8, v8);
  v9 = 1 << (33 - (v8 ^ 0x1F));
  if ( v9 - 5 <= 0x3A )
  {
    v9 = 64;
    j___libc_free_0(*(_QWORD *)(a1 + 16));
    v10 = *(_BYTE *)(a1 + 8);
    v11 = 512;
LABEL_20:
    *(_BYTE *)(a1 + 8) = v10 & 0xFE;
    v12 = sub_22077B0(v11);
    *(_DWORD *)(a1 + 24) = v9;
    *(_QWORD *)(a1 + 16) = v12;
    goto LABEL_21;
  }
  if ( (_DWORD)v5 != v9 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 16));
    v10 = *(_BYTE *)(a1 + 8) | 1;
    *(_BYTE *)(a1 + 8) = v10;
    if ( v9 <= 4 )
      goto LABEL_21;
    v11 = 8LL * v9;
    goto LABEL_20;
  }
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v13 )
  {
    result = *(_QWORD **)(a1 + 16);
    v16 = v5;
  }
  else
  {
    result = (_QWORD *)(a1 + 16);
    v16 = 4;
  }
  v17 = &result[v16];
  do
  {
    if ( result )
      *result = -8;
    ++result;
  }
  while ( v17 != result );
  return result;
}
