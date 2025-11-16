// Function: sub_16D5D90
// Address: 0x16d5d90
//
_QWORD *sub_16D5D90()
{
  _QWORD *result; // rax

  result = (_QWORD *)sub_22077B0(32);
  if ( result )
  {
    *result = 0;
    result[1] = 0;
    result[2] = 0x3000000000LL;
  }
  return result;
}
