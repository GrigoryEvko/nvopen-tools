// Function: sub_6EB510
// Address: 0x6eb510
//
_BYTE *__fastcall sub_6EB510(__int64 a1)
{
  _BYTE *result; // rax
  __int64 v2; // rbx

  result = (_BYTE *)qword_4D03C50;
  if ( qword_4D03C50 )
  {
    if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
    {
      v2 = *(_QWORD *)(a1 + 56);
      sub_6EB4C0(v2);
      result = *(_BYTE **)(v2 + 24);
      if ( result )
      {
        if ( !*result )
          *(_BYTE *)(v2 + 49) |= 1u;
      }
    }
  }
  return result;
}
