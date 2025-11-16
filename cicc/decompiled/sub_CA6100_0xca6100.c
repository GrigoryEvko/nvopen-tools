// Function: sub_CA6100
// Address: 0xca6100
//
_BYTE *__fastcall sub_CA6100(__int64 a1, _BYTE *a2)
{
  _BYTE *result; // rax

  result = a2;
  if ( *(_BYTE **)(a1 + 48) != a2 && (*a2 == 32 || *a2 == 9) )
    return a2 + 1;
  return result;
}
