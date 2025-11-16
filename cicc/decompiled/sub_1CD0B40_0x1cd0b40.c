// Function: sub_1CD0B40
// Address: 0x1cd0b40
//
_QWORD *__fastcall sub_1CD0B40(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rdi
  _QWORD *result; // rax
  __int64 v5; // rsi
  unsigned __int64 v6; // rcx
  __int64 v7; // rcx

  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v3 = *(_QWORD *)(a1 - 8);
  else
    v3 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  result = (_QWORD *)(v3 + 24LL * a2);
  if ( *result )
  {
    v5 = result[1];
    v6 = result[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v6 = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v5 + 16) & 3LL | v6;
  }
  *result = a3;
  if ( a3 )
  {
    v7 = *(_QWORD *)(a3 + 8);
    result[1] = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = (unsigned __int64)(result + 1) | *(_QWORD *)(v7 + 16) & 3LL;
    result[2] = (a3 + 8) | result[2] & 3LL;
    *(_QWORD *)(a3 + 8) = result;
  }
  return result;
}
