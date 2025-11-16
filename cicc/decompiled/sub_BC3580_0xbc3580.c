// Function: sub_BC3580
// Address: 0xbc3580
//
_OWORD *sub_BC3580()
{
  _OWORD *result; // rax

  result = (_OWORD *)sub_22077B0(48);
  if ( result )
  {
    result[1] = 0;
    *((_DWORD *)result + 4) = 1;
    *result = 0;
    result[2] = 0;
  }
  return result;
}
