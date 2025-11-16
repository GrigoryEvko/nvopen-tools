// Function: sub_A04FB0
// Address: 0xa04fb0
//
_QWORD *__fastcall sub_A04FB0(__int64 a1)
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
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  _QWORD *i; // rdx
  unsigned int v19; // r14d
  unsigned int v20; // edx
  unsigned int v21; // eax

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
          goto LABEL_38;
      }
LABEL_36:
      v9 = 64;
      sub_C7D6A0(*(_QWORD *)(a1 + 16), 16 * v15, 8);
      v6 = *(_BYTE *)(a1 + 8);
      goto LABEL_14;
    }
    v3 = *(_QWORD **)(a1 + 16);
    v4 = 2LL * (unsigned int)v15;
  }
  v5 = &v3[v4];
  do
  {
    if ( *v3 != -8192 && *v3 != -4096 && v3[1] )
      sub_BA65D0();
    v3 += 2;
  }
  while ( v5 != v3 );
  v6 = *(_BYTE *)(a1 + 8);
  v7 = v6 & 1;
  if ( !v2 )
  {
    if ( !v7 )
    {
      v21 = *(_DWORD *)(a1 + 24);
      if ( v21 )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * v21, 8);
        *(_BYTE *)(a1 + 8) |= 1u;
LABEL_28:
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        if ( !v11 )
        {
LABEL_29:
          result = (_QWORD *)(a1 + 16);
          v17 = 2;
          goto LABEL_30;
        }
LABEL_41:
        result = *(_QWORD **)(a1 + 16);
        v17 = 2LL * *(unsigned int *)(a1 + 24);
LABEL_30:
        for ( i = &result[v17]; i != result; result += 2 )
        {
          if ( result )
            *result = -4096;
        }
        return result;
      }
    }
    goto LABEL_16;
  }
  v8 = v2 - 1;
  if ( !v8 || (_BitScanReverse(&v8, v8), v9 = 1 << (33 - (v8 ^ 0x1F)), v9 - 2 <= 0x3D) )
  {
    if ( v7 )
    {
      v6 = *(_BYTE *)(a1 + 8);
      v10 = 1024;
      v9 = 64;
      goto LABEL_27;
    }
    v15 = *(unsigned int *)(a1 + 24);
    if ( (_DWORD)v15 == 64 )
      goto LABEL_16;
    goto LABEL_36;
  }
  if ( !v7 )
  {
    LODWORD(v15) = *(_DWORD *)(a1 + 24);
LABEL_38:
    if ( (_DWORD)v15 != v9 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * (unsigned int)v15, 8);
      v6 = *(_BYTE *)(a1 + 8) | 1;
      *(_BYTE *)(a1 + 8) = v6;
      if ( v9 <= 1 )
      {
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        if ( !v11 )
          goto LABEL_29;
        goto LABEL_41;
      }
      goto LABEL_14;
    }
    goto LABEL_16;
  }
  if ( v9 > 1 )
  {
LABEL_14:
    v10 = 16LL * v9;
LABEL_27:
    *(_BYTE *)(a1 + 8) = v6 & 0xFE;
    v16 = sub_C7D670(v10, 8);
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
      *result = -4096;
  }
  return result;
}
