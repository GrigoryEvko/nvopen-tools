// Function: sub_7274F0
// Address: 0x7274f0
//
_QWORD *sub_7274F0()
{
  _QWORD *result; // rax

  result = sub_7247C0(40);
  *((_BYTE *)result + 32) &= ~1u;
  *result = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;
  return result;
}
