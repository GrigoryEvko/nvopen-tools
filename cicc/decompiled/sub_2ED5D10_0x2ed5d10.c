// Function: sub_2ED5D10
// Address: 0x2ed5d10
//
_DWORD *__fastcall sub_2ED5D10(__int64 a1)
{
  int v2; // r15d
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // r12
  __int64 v7; // rax
  unsigned __int64 *v8; // rax
  unsigned __int64 v9; // r14
  char v10; // al
  char v11; // dl
  bool v12; // zf
  _DWORD *result; // rax
  __int64 v14; // rdx
  _DWORD *i; // rdx
  unsigned int v16; // r15d
  unsigned int v17; // ebx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdx
  _DWORD *j; // rdx
  unsigned int v22; // r15d
  unsigned int v23; // eax

  v2 = *(_DWORD *)(a1 + 8) >> 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v4 = a1 + 16;
    v5 = 64;
LABEL_4:
    v6 = v4 + v5;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v4 <= 0xFFFFFFFD )
        {
          v7 = *(_QWORD *)(v4 + 8);
          if ( v7 )
          {
            if ( (v7 & 2) != 0 )
            {
              v8 = (unsigned __int64 *)(v7 & 0xFFFFFFFFFFFFFFFCLL);
              v9 = (unsigned __int64)v8;
              if ( v8 )
                break;
            }
          }
        }
        v4 += 16;
        if ( v6 == v4 )
          goto LABEL_13;
      }
      if ( (unsigned __int64 *)*v8 != v8 + 2 )
        _libc_free(*v8);
      v4 += 16;
      j_j___libc_free_0(v9);
    }
    while ( v6 != v4 );
LABEL_13:
    v10 = *(_BYTE *)(a1 + 8);
    v11 = v10 & 1;
    if ( !v2 )
    {
      if ( v11 )
        goto LABEL_15;
      LODWORD(v3) = *(_DWORD *)(a1 + 24);
LABEL_46:
      if ( (_DWORD)v3 != v2 )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * (unsigned int)v3, 8);
        *(_BYTE *)(a1 + 8) |= 1u;
        goto LABEL_26;
      }
LABEL_15:
      v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      if ( !v12 )
      {
LABEL_16:
        result = (_DWORD *)(a1 + 16);
        v14 = 16;
        goto LABEL_17;
      }
LABEL_41:
      result = *(_DWORD **)(a1 + 16);
      v14 = 4LL * *(unsigned int *)(a1 + 24);
LABEL_17:
      for ( i = &result[v14]; i != result; result += 4 )
      {
        if ( result )
          *result = -1;
      }
      return result;
    }
    v16 = v2 - 1;
    if ( v16 )
    {
      _BitScanReverse(&v16, v16);
      v17 = 1 << (33 - (v16 ^ 0x1F));
      if ( v17 - 5 <= 0x3A )
      {
        v18 = 1024;
        v17 = 64;
        if ( v11 )
        {
LABEL_25:
          *(_BYTE *)(a1 + 8) = v10 & 0xFE;
          v19 = sub_C7D670(v18, 8);
          *(_DWORD *)(a1 + 24) = v17;
          *(_QWORD *)(a1 + 16) = v19;
          goto LABEL_26;
        }
        LODWORD(v3) = *(_DWORD *)(a1 + 24);
        if ( (_DWORD)v3 == 64 )
        {
          v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
          *(_QWORD *)(a1 + 8) &= 1uLL;
          if ( !v12 )
            goto LABEL_16;
          goto LABEL_41;
        }
        goto LABEL_36;
      }
      if ( v11 )
      {
        if ( v17 <= 4 )
          goto LABEL_15;
        goto LABEL_44;
      }
      LODWORD(v3) = *(_DWORD *)(a1 + 24);
      goto LABEL_52;
    }
    if ( v11 )
      goto LABEL_15;
    LODWORD(v3) = *(_DWORD *)(a1 + 24);
LABEL_50:
    v2 = 2;
    goto LABEL_46;
  }
  v3 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    v5 = 16 * v3;
    goto LABEL_4;
  }
  if ( !v2 )
    goto LABEL_15;
  v22 = v2 - 1;
  if ( !v22 )
    goto LABEL_50;
  _BitScanReverse(&v23, v22);
  v17 = 1 << (33 - (v23 ^ 0x1F));
  if ( v17 - 5 <= 0x3A )
  {
LABEL_36:
    v17 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * (unsigned int)v3, 8);
    v10 = *(_BYTE *)(a1 + 8);
LABEL_44:
    v18 = 16LL * v17;
    goto LABEL_25;
  }
LABEL_52:
  if ( (_DWORD)v3 == v17 )
    goto LABEL_15;
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * (unsigned int)v3, 8);
  v10 = *(_BYTE *)(a1 + 8) | 1;
  *(_BYTE *)(a1 + 8) = v10;
  if ( v17 > 4 )
    goto LABEL_44;
LABEL_26:
  v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v12 )
  {
    result = *(_DWORD **)(a1 + 16);
    v20 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = (_DWORD *)(a1 + 16);
    v20 = 16;
  }
  for ( j = &result[v20]; j != result; result += 4 )
  {
    if ( result )
      *result = -1;
  }
  return result;
}
