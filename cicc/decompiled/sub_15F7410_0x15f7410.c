// Function: sub_15F7410
// Address: 0x15f7410
//
_QWORD *__fastcall sub_15F7410(__int64 a1, __int64 a2)
{
  __int16 v3; // dx
  _QWORD *result; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned __int64 v11; // rcx
  __int64 v12; // rcx

  sub_15F1EA0(a1, *(_QWORD *)a2, 8, a1 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF), *(_DWORD *)(a2 + 20) & 0xFFFFFFF, 0);
  v3 = *(_WORD *)(a2 + 18);
  HIBYTE(v3) &= ~0x80u;
  *(_WORD *)(a1 + 18) = v3 | *(_WORD *)(a1 + 18) & 0x8000;
  result = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v5 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( *result )
  {
    v6 = result[1];
    v7 = result[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v7 = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
  }
  *result = v5;
  if ( v5 )
  {
    v8 = *(_QWORD *)(v5 + 8);
    result[1] = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = (unsigned __int64)(result + 1) | *(_QWORD *)(v8 + 16) & 3LL;
    result[2] = (v5 + 8) | result[2] & 3LL;
    *(_QWORD *)(v5 + 8) = result;
  }
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    result = (_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
    v9 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( *result )
    {
      v10 = result[1];
      v11 = result[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v11 = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
    }
    *result = v9;
    if ( v9 )
    {
      v12 = *(_QWORD *)(v9 + 8);
      result[1] = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = (unsigned __int64)(result + 1) | *(_QWORD *)(v12 + 16) & 3LL;
      result[2] = (v9 + 8) | result[2] & 3LL;
      *(_QWORD *)(v9 + 8) = result;
    }
  }
  return result;
}
