// Function: sub_37E7E80
// Address: 0x37e7e80
//
_QWORD *__fastcall sub_37E7E80(__int64 a1)
{
  unsigned int v1; // eax
  _QWORD *result; // rax
  unsigned int v4; // ecx
  __int64 v5; // rdx
  __int64 v6; // rdx
  _QWORD *i; // rdx
  __int64 v8; // rsi
  bool v9; // zf
  __int64 v10; // rdx
  _QWORD *j; // rdx
  unsigned int v12; // eax
  unsigned int v13; // r12d
  char v14; // al
  __int64 v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // rbx
  __int64 v18; // rax

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
    v6 = 8;
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
    v6 = 2 * v5;
LABEL_7:
    for ( i = &result[v6]; result != i; *(result - 1) = -4096 )
    {
      *result = -4096;
      result += 2;
    }
    *(_QWORD *)(a1 + 8) &= 1uLL;
    return result;
  }
  if ( !(_DWORD)result )
  {
    v8 = (unsigned int)v5;
LABEL_16:
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 16 * v8, 8);
    *(_BYTE *)(a1 + 8) |= 1u;
    goto LABEL_17;
  }
  v12 = (_DWORD)result - 1;
  if ( !v12 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    goto LABEL_16;
  }
  _BitScanReverse(&v12, v12);
  v13 = 1 << (33 - (v12 ^ 0x1F));
  if ( v13 - 5 <= 0x3A )
  {
    v13 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 24), 8);
    v14 = *(_BYTE *)(a1 + 8);
    v15 = 1024;
    goto LABEL_27;
  }
  if ( (_DWORD)v5 != v13 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 24), 8);
    v14 = *(_BYTE *)(a1 + 8) | 1;
    *(_BYTE *)(a1 + 8) = v14;
    if ( v13 <= 4 )
    {
LABEL_17:
      v9 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      if ( v9 )
      {
        result = *(_QWORD **)(a1 + 16);
        v10 = 2LL * *(unsigned int *)(a1 + 24);
      }
      else
      {
        result = (_QWORD *)(a1 + 16);
        v10 = 8;
      }
      for ( j = &result[v10]; j != result; result += 2 )
      {
        if ( result )
        {
          *result = -4096;
          result[1] = -4096;
        }
      }
      return result;
    }
    v15 = 16LL * v13;
LABEL_27:
    *(_BYTE *)(a1 + 8) = v14 & 0xFE;
    v16 = sub_C7D670(v15, 8);
    *(_DWORD *)(a1 + 24) = v13;
    *(_QWORD *)(a1 + 16) = v16;
    goto LABEL_17;
  }
  v9 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v9 )
  {
    v17 = *(_QWORD **)(a1 + 16);
    v18 = 2LL * (unsigned int)v5;
  }
  else
  {
    v17 = (_QWORD *)(a1 + 16);
    v18 = 8;
  }
  result = &v17[v18];
  do
  {
    if ( v17 )
    {
      *v17 = -4096;
      v17[1] = -4096;
    }
    v17 += 2;
  }
  while ( result != v17 );
  return result;
}
