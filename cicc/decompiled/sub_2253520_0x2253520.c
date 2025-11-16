// Function: sub_2253520
// Address: 0x2253520
//
_BYTE *sub_2253520()
{
  _BYTE *result; // rax

  result = *(_BYTE **)sub_22529C0();
  if ( result )
  {
    if ( (result[80] & 1) != 0 )
      result = (_BYTE *)(*(_QWORD *)result - 112LL);
    return *(_BYTE **)result;
  }
  return result;
}
