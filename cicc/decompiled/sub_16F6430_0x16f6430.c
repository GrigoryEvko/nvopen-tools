// Function: sub_16F6430
// Address: 0x16f6430
//
_BYTE *__fastcall sub_16F6430(__int64 a1, _BYTE *a2)
{
  _BYTE *result; // rax

  result = a2;
  if ( *(_BYTE **)(a1 + 48) != a2 && (*a2 == 32 || *a2 == 9) )
    return a2 + 1;
  return result;
}
