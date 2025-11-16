// Function: sub_5F9380
// Address: 0x5f9380
//
_BYTE *__fastcall sub_5F9380(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // rax
  _BYTE *result; // rax

  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = *(_BYTE **)(*(_QWORD *)(i + 168) + 56LL);
  if ( result )
  {
    if ( (*result & 8) == 0 )
      return (_BYTE *)sub_5F90A0(a1, a2, a3, a4, a5);
  }
  return result;
}
