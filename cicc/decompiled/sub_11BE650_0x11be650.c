// Function: sub_11BE650
// Address: 0x11be650
//
_QWORD *__fastcall sub_11BE650(__int64 a1, char a2)
{
  char v3; // al
  unsigned int v4; // ecx
  __int64 v5; // rbx
  __int64 v6; // r12
  _QWORD *v7; // r15
  __int64 v8; // rdi
  unsigned int v9; // r13d
  _QWORD *result; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rdx
  _QWORD *i; // rdx
  __int64 v15; // rbx
  __int64 v16; // rax
  bool v17; // zf
  __int64 v18; // rdx
  _QWORD *j; // rdx
  unsigned int v20; // eax
  unsigned int v21; // ebx
  char v22; // al
  __int64 v23; // rdi
  __int64 v24; // rax
  _QWORD *v25; // r14
  __int64 v26; // rax

  v3 = *(_BYTE *)(a1 + 40) & 1;
  v4 = *(_DWORD *)(a1 + 40) >> 1;
  if ( v4 )
  {
    if ( v3 )
    {
      v5 = a1 + 48;
      v6 = a1 + 80;
    }
    else
    {
      v5 = *(_QWORD *)(a1 + 48);
      v6 = v5 + 8LL * *(unsigned int *)(a1 + 56);
      if ( v5 == v6 )
        goto LABEL_7;
    }
    do
    {
      if ( *(_QWORD *)v5 != -4096 && *(_QWORD *)v5 != -8192 )
        break;
      v5 += 8;
    }
    while ( v5 != v6 );
  }
  else
  {
    if ( v3 )
    {
      v15 = a1 + 48;
      v16 = 32;
    }
    else
    {
      v15 = *(_QWORD *)(a1 + 48);
      v16 = 8LL * *(unsigned int *)(a1 + 56);
    }
    v5 = v16 + v15;
    v6 = v5;
  }
LABEL_7:
  if ( v6 != v5 )
  {
LABEL_8:
    v7 = *(_QWORD **)v5;
    v8 = *(_QWORD *)(*(_QWORD *)v5 - 32LL * (*(_DWORD *)(*(_QWORD *)v5 + 4LL) & 0x7FFFFFF));
    if ( *(_BYTE *)v8 != 17 )
      goto LABEL_14;
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
    {
      if ( !*(_QWORD *)(v8 + 24) )
        goto LABEL_14;
    }
    else if ( v9 == (unsigned int)sub_C444A0(v8 + 24) )
    {
      goto LABEL_14;
    }
    if ( a2 || sub_CF91F0((__int64)v7) )
    {
      *(_BYTE *)(a1 + 552) = 1;
      sub_B43D60(v7);
    }
LABEL_14:
    while ( 1 )
    {
      v5 += 8;
      if ( v5 == v6 )
        break;
      if ( *(_QWORD *)v5 != -8192 && *(_QWORD *)v5 != -4096 )
      {
        if ( v5 != v6 )
          goto LABEL_8;
        break;
      }
    }
    v4 = *(_DWORD *)(a1 + 40) >> 1;
  }
  ++*(_QWORD *)(a1 + 32);
  if ( v4 )
  {
    if ( (*(_BYTE *)(a1 + 40) & 1) == 0 )
    {
      v11 = 4 * v4;
      goto LABEL_22;
    }
LABEL_34:
    result = (_QWORD *)(a1 + 48);
    v13 = 4;
    goto LABEL_25;
  }
  result = (_QWORD *)*(unsigned int *)(a1 + 44);
  if ( !(_DWORD)result )
    return result;
  v11 = 0;
  if ( (*(_BYTE *)(a1 + 40) & 1) != 0 )
    goto LABEL_34;
LABEL_22:
  v12 = *(unsigned int *)(a1 + 56);
  if ( v11 >= (unsigned int)v12 || (unsigned int)v12 <= 0x40 )
  {
    result = *(_QWORD **)(a1 + 48);
    v13 = v12;
LABEL_25:
    for ( i = &result[v13]; i != result; ++result )
      *result = -4096;
    *(_QWORD *)(a1 + 40) &= 1uLL;
    return result;
  }
  if ( v4 <= 1 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 48), 8 * v12, 8);
    *(_BYTE *)(a1 + 40) |= 1u;
    goto LABEL_41;
  }
  _BitScanReverse(&v20, v4 - 1);
  v21 = 1 << (33 - (v20 ^ 0x1F));
  if ( v21 - 5 <= 0x3A )
  {
    v21 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 48), 8 * v12, 8);
    v22 = *(_BYTE *)(a1 + 40);
    v23 = 512;
    goto LABEL_50;
  }
  if ( (_DWORD)v12 != v21 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 48), 8 * v12, 8);
    v22 = *(_BYTE *)(a1 + 40) | 1;
    *(_BYTE *)(a1 + 40) = v22;
    if ( v21 <= 4 )
    {
LABEL_41:
      v17 = (*(_QWORD *)(a1 + 40) & 1LL) == 0;
      *(_QWORD *)(a1 + 40) &= 1uLL;
      if ( v17 )
      {
        result = *(_QWORD **)(a1 + 48);
        v18 = *(unsigned int *)(a1 + 56);
      }
      else
      {
        result = (_QWORD *)(a1 + 48);
        v18 = 4;
      }
      for ( j = &result[v18]; j != result; ++result )
      {
        if ( result )
          *result = -4096;
      }
      return result;
    }
    v23 = 8LL * v21;
LABEL_50:
    *(_BYTE *)(a1 + 40) = v22 & 0xFE;
    v24 = sub_C7D670(v23, 8);
    *(_DWORD *)(a1 + 56) = v21;
    *(_QWORD *)(a1 + 48) = v24;
    goto LABEL_41;
  }
  v17 = (*(_QWORD *)(a1 + 40) & 1LL) == 0;
  *(_QWORD *)(a1 + 40) &= 1uLL;
  if ( v17 )
  {
    v25 = *(_QWORD **)(a1 + 48);
    v26 = (unsigned int)v12;
  }
  else
  {
    v25 = (_QWORD *)(a1 + 48);
    v26 = 4;
  }
  result = &v25[v26];
  do
  {
    if ( v25 )
      *v25 = -4096;
    ++v25;
  }
  while ( result != v25 );
  return result;
}
