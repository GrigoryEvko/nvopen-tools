// Function: sub_21114A0
// Address: 0x21114a0
//
__int64 *__fastcall sub_21114A0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  int v3; // eax
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 *result; // rax
  _DWORD *v7; // rsi
  int v8; // r8d
  __int64 v9; // rdx
  __int64 *v10; // rsi
  __int64 *v11; // rdi
  __int64 v12; // rdx
  bool v13; // cl
  __int64 v14; // r8
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rcx

  v2 = *(_DWORD *)(a2 + 8) & 0xFFFFFFFE;
  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a1 + 8) & 0xFFFFFFFE | *(_DWORD *)(a2 + 8) & 1;
  *(_DWORD *)(a1 + 8) = v2 | *(_DWORD *)(a1 + 8) & 1;
  v3 = *(_DWORD *)(a1 + 12);
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
  *(_DWORD *)(a2 + 12) = v3;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
      goto LABEL_4;
    result = (__int64 *)(a1 + 16);
    v10 = (__int64 *)(a2 + 16);
    v11 = (__int64 *)(a1 + 80);
    while ( 1 )
    {
      v12 = *result;
      v13 = *result != -16 && *result != -8;
      v14 = *v10;
      if ( *v10 == -8 )
      {
        *result = -8;
        *v10 = v12;
        if ( v13 )
          goto LABEL_21;
      }
      else
      {
        if ( v14 != -16 )
        {
          *result = v14;
          if ( v13 )
          {
            v17 = result[1];
            result[1] = v10[1];
            *v10 = v12;
            v10[1] = v17;
          }
          else
          {
            *v10 = v12;
            result[1] = v10[1];
          }
          goto LABEL_16;
        }
        *result = -16;
        *v10 = v12;
        if ( v13 )
LABEL_21:
          v10[1] = result[1];
      }
LABEL_16:
      result += 2;
      v10 += 2;
      if ( v11 == result )
        return result;
    }
  }
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
  {
    v15 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    v16 = *(_DWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = v15;
    result = (__int64 *)*(unsigned int *)(a1 + 24);
    *(_DWORD *)(a1 + 24) = v16;
    *(_DWORD *)(a2 + 24) = (_DWORD)result;
    return result;
  }
  v4 = a2;
  a2 = a1;
  a1 = v4;
LABEL_4:
  *(_BYTE *)(a2 + 8) |= 1u;
  v5 = *(_QWORD *)(a2 + 16);
  result = (__int64 *)(a1 + 16);
  v7 = (_DWORD *)(a2 + 24);
  v8 = *v7;
  do
  {
    v9 = *result;
    *((_QWORD *)v7 - 1) = *result;
    if ( v9 != -16 && v9 != -8 )
      *(_QWORD *)v7 = result[1];
    result += 2;
    v7 += 4;
  }
  while ( (__int64 *)(a1 + 80) != result );
  *(_BYTE *)(a1 + 8) &= ~1u;
  *(_QWORD *)(a1 + 16) = v5;
  *(_DWORD *)(a1 + 24) = v8;
  return result;
}
