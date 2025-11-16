// Function: sub_15F75B0
// Address: 0x15f75b0
//
_QWORD *__fastcall sub_15F75B0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v4; // r8
  unsigned __int64 v5; // rcx
  __int64 v6; // rcx
  __int64 v7; // rsi
  unsigned __int64 v8; // rcx
  __int64 v9; // rsi

  if ( a3 )
    *(_WORD *)(a1 + 18) = *(_WORD *)(a1 + 18) & 0x8000 | *(_WORD *)(a1 + 18) & 0x7FFE | 1;
  result = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( *result )
  {
    v4 = result[1];
    v5 = result[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v5 = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = *(_QWORD *)(v4 + 16) & 3LL | v5;
  }
  *result = a2;
  if ( a2 )
  {
    v6 = *(_QWORD *)(a2 + 8);
    result[1] = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = (unsigned __int64)(result + 1) | *(_QWORD *)(v6 + 16) & 3LL;
    result[2] = (a2 + 8) | result[2] & 3LL;
    *(_QWORD *)(a2 + 8) = result;
  }
  if ( a3 )
  {
    result = (_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
    if ( *result )
    {
      v7 = result[1];
      v8 = result[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v8 = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = *(_QWORD *)(v7 + 16) & 3LL | v8;
    }
    *result = a3;
    v9 = *(_QWORD *)(a3 + 8);
    result[1] = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = (unsigned __int64)(result + 1) | *(_QWORD *)(v9 + 16) & 3LL;
    result[2] = result[2] & 3LL | (a3 + 8);
    *(_QWORD *)(a3 + 8) = result;
  }
  return result;
}
