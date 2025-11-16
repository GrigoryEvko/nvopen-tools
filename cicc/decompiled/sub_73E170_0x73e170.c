// Function: sub_73E170
// Address: 0x73e170
//
_BYTE *__fastcall sub_73E170(__int64 a1, __int64 a2)
{
  _BYTE *result; // rax

  if ( *(_BYTE *)(a1 + 24) == 1 && (*(_BYTE *)(a1 + 27) & 2) != 0 && *(_BYTE *)(a1 + 56) == 14 )
  {
    *(_QWORD *)a1 = a2;
    return (_BYTE *)a1;
  }
  else
  {
    result = sub_73DBF0(9u, a2, a1);
    result[27] |= 2u;
  }
  return result;
}
