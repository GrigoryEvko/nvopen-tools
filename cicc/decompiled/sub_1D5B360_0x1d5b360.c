// Function: sub_1D5B360
// Address: 0x1d5b360
//
_QWORD *__fastcall sub_1D5B360(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // rax
  _QWORD *result; // rax
  __int64 v5; // rsi
  unsigned __int64 v6; // rdx
  __int64 v7; // rdx

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_QWORD *)(a1 + 16);
  if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
    v3 = *(_QWORD *)(v1 - 8);
  else
    v3 = v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
  result = (_QWORD *)(v3 + 24LL * *(unsigned int *)(a1 + 24));
  if ( *result )
  {
    v5 = result[1];
    v6 = result[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v6 = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v5 + 16) & 3LL | v6;
  }
  *result = v2;
  if ( v2 )
  {
    v7 = *(_QWORD *)(v2 + 8);
    result[1] = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = (unsigned __int64)(result + 1) | *(_QWORD *)(v7 + 16) & 3LL;
    result[2] = (v2 + 8) | result[2] & 3LL;
    *(_QWORD *)(v2 + 8) = result;
  }
  return result;
}
