// Function: sub_16F0F80
// Address: 0x16f0f80
//
_QWORD *sub_16F0F80()
{
  _QWORD *result; // rax

  result = (_QWORD *)sub_22077B0(32);
  if ( result )
  {
    *result = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
  }
  return result;
}
