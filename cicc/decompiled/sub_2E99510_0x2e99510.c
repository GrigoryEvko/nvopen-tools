// Function: sub_2E99510
// Address: 0x2e99510
//
signed __int64 __fastcall sub_2E99510(__int64 a1)
{
  unsigned int v1; // edx
  unsigned int v3; // edx
  signed __int64 result; // rax
  unsigned int v5; // ecx
  void *v6; // rdi
  size_t v7; // rdx
  bool v8; // zf
  __int64 v9; // rdx
  _DWORD *i; // rdx
  unsigned int v11; // edx
  unsigned int v12; // r12d
  char v13; // al
  __int64 v14; // rdi
  __int64 v15; // rax
  _DWORD *v16; // rbx
  __int64 v17; // rax

  v1 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v3 = v1 >> 1;
  if ( v3 )
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    {
      v5 = 4 * v3;
      goto LABEL_4;
    }
LABEL_12:
    v6 = (void *)(a1 + 16);
    v7 = 16;
    goto LABEL_7;
  }
  result = *(unsigned int *)(a1 + 12);
  if ( !(_DWORD)result )
    return result;
  v5 = 0;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    goto LABEL_12;
LABEL_4:
  result = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)result <= v5 || (unsigned int)result <= 0x40 )
  {
    v6 = *(void **)(a1 + 16);
    v7 = 4LL * (unsigned int)result;
    if ( !v7 )
    {
LABEL_8:
      *(_QWORD *)(a1 + 8) &= 1uLL;
      return result;
    }
LABEL_7:
    result = (signed __int64)memset(v6, 255, v7);
    goto LABEL_8;
  }
  if ( !v3 || (v11 = v3 - 1) == 0 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 4 * result, 4);
    *(_BYTE *)(a1 + 8) |= 1u;
    goto LABEL_15;
  }
  _BitScanReverse(&v11, v11);
  v12 = 1 << (33 - (v11 ^ 0x1F));
  if ( v12 - 5 <= 0x3A )
  {
    v12 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 4 * result, 4);
    v13 = *(_BYTE *)(a1 + 8);
    v14 = 256;
    goto LABEL_25;
  }
  if ( (_DWORD)result != v12 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 4 * result, 4);
    v13 = *(_BYTE *)(a1 + 8) | 1;
    *(_BYTE *)(a1 + 8) = v13;
    if ( v12 <= 4 )
    {
LABEL_15:
      v8 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      if ( v8 )
      {
        result = *(_QWORD *)(a1 + 16);
        v9 = 4LL * *(unsigned int *)(a1 + 24);
      }
      else
      {
        result = a1 + 16;
        v9 = 16;
      }
      for ( i = (_DWORD *)(result + v9); i != (_DWORD *)result; result += 4LL )
      {
        if ( result )
          *(_DWORD *)result = -1;
      }
      return result;
    }
    v14 = 4LL * v12;
LABEL_25:
    *(_BYTE *)(a1 + 8) = v13 & 0xFE;
    v15 = sub_C7D670(v14, 4);
    *(_DWORD *)(a1 + 24) = v12;
    *(_QWORD *)(a1 + 16) = v15;
    goto LABEL_15;
  }
  v8 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v8 )
  {
    v16 = *(_DWORD **)(a1 + 16);
    v17 = result;
  }
  else
  {
    v16 = (_DWORD *)(a1 + 16);
    v17 = 4;
  }
  result = (signed __int64)&v16[v17];
  do
  {
    if ( v16 )
      *v16 = -1;
    ++v16;
  }
  while ( (_DWORD *)result != v16 );
  return result;
}
