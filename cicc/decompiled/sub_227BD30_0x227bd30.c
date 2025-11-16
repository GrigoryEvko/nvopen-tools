// Function: sub_227BD30
// Address: 0x227bd30
//
_QWORD *__fastcall sub_227BD30(__int64 a1)
{
  unsigned int v2; // eax
  _QWORD *result; // rax
  unsigned int v4; // ecx
  __int64 v5; // rdx
  __int64 v6; // rdx
  _QWORD *i; // rdx
  unsigned int v8; // eax
  unsigned int v9; // ebx
  char v10; // al
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rsi

  v2 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  result = (_QWORD *)(v2 >> 1);
  if ( (_DWORD)result )
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    {
      v4 = 4 * (_DWORD)result;
      goto LABEL_4;
    }
LABEL_13:
    result = (_QWORD *)(a1 + 16);
    v6 = 32;
    goto LABEL_7;
  }
  if ( !*(_DWORD *)(a1 + 12) )
    return result;
  v4 = 0;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    goto LABEL_13;
LABEL_4:
  v5 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)v5 > v4 && (unsigned int)v5 > 0x40 )
  {
    if ( (_DWORD)result )
    {
      v8 = (_DWORD)result - 1;
      if ( v8 )
      {
        _BitScanReverse(&v8, v8);
        v9 = 1 << (33 - (v8 ^ 0x1F));
        if ( v9 - 17 > 0x2E )
        {
          if ( (_DWORD)v5 == v9 )
            return sub_227BCE0(a1);
          sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 24), 8);
          v10 = *(_BYTE *)(a1 + 8) | 1;
          *(_BYTE *)(a1 + 8) = v10;
          if ( v9 <= 0x10 )
            return sub_227BCE0(a1);
          v11 = 16LL * v9;
        }
        else
        {
          v9 = 64;
          sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 24), 8);
          v10 = *(_BYTE *)(a1 + 8);
          v11 = 1024;
        }
        *(_BYTE *)(a1 + 8) = v10 & 0xFE;
        v12 = sub_C7D670(v11, 8);
        *(_DWORD *)(a1 + 24) = v9;
        *(_QWORD *)(a1 + 16) = v12;
        return sub_227BCE0(a1);
      }
      v13 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      v13 = (unsigned int)v5;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 16 * v13, 8);
    *(_BYTE *)(a1 + 8) |= 1u;
    return sub_227BCE0(a1);
  }
  result = *(_QWORD **)(a1 + 16);
  v6 = 2 * v5;
LABEL_7:
  for ( i = &result[v6]; result != i; result += 2 )
    *result = -4096;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  return result;
}
