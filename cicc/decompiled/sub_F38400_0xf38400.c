// Function: sub_F38400
// Address: 0xf38400
//
__int64 __fastcall sub_F38400(__int64 a1)
{
  unsigned int v1; // eax
  __int64 result; // rax
  unsigned int v4; // ecx
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 i; // rdx
  bool v8; // zf
  __int64 v9; // rdx
  __int64 j; // rdx
  unsigned int v11; // eax
  unsigned int v12; // r12d
  char v13; // al
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rax

  v1 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  result = v1 >> 1;
  if ( (_DWORD)result )
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    {
      v4 = 4 * result;
      goto LABEL_4;
    }
LABEL_13:
    result = a1 + 16;
    v6 = 160;
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
    result = *(_QWORD *)(a1 + 16);
    v6 = 40 * v5;
LABEL_7:
    for ( i = result + v6; result != i; *(_QWORD *)(result - 8) = 0 )
    {
      *(_QWORD *)result = 0;
      result += 40;
      *(_BYTE *)(result - 16) = 0;
    }
    *(_QWORD *)(a1 + 8) &= 1uLL;
    return result;
  }
  if ( !(_DWORD)result || (v11 = result - 1) == 0 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 40 * v5, 8);
    *(_BYTE *)(a1 + 8) |= 1u;
    goto LABEL_16;
  }
  _BitScanReverse(&v11, v11);
  v12 = 1 << (33 - (v11 ^ 0x1F));
  if ( v12 - 5 <= 0x3A )
  {
    v12 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 40 * v5, 8);
    v13 = *(_BYTE *)(a1 + 8);
    v14 = 2560;
    goto LABEL_26;
  }
  if ( (_DWORD)v5 != v12 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 40 * v5, 8);
    v13 = *(_BYTE *)(a1 + 8) | 1;
    *(_BYTE *)(a1 + 8) = v13;
    if ( v12 <= 4 )
    {
LABEL_16:
      v8 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      if ( v8 )
      {
        result = *(_QWORD *)(a1 + 16);
        v9 = 40LL * *(unsigned int *)(a1 + 24);
      }
      else
      {
        result = a1 + 16;
        v9 = 160;
      }
      for ( j = result + v9; j != result; result += 40 )
      {
        if ( result )
        {
          *(_QWORD *)result = 0;
          *(_BYTE *)(result + 24) = 0;
          *(_QWORD *)(result + 32) = 0;
        }
      }
      return result;
    }
    v14 = 40LL * v12;
LABEL_26:
    *(_BYTE *)(a1 + 8) = v13 & 0xFE;
    v15 = sub_C7D670(v14, 8);
    *(_DWORD *)(a1 + 24) = v12;
    *(_QWORD *)(a1 + 16) = v15;
    goto LABEL_16;
  }
  v8 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v8 )
  {
    v16 = *(_QWORD *)(a1 + 16);
    v17 = 40 * v5;
  }
  else
  {
    v16 = a1 + 16;
    v17 = 160;
  }
  result = v16 + v17;
  do
  {
    if ( v16 )
    {
      *(_QWORD *)v16 = 0;
      *(_BYTE *)(v16 + 24) = 0;
      *(_QWORD *)(v16 + 32) = 0;
    }
    v16 += 40;
  }
  while ( result != v16 );
  return result;
}
