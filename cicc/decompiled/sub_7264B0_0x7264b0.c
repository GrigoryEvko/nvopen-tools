// Function: sub_7264B0
// Address: 0x7264b0
//
_BYTE *sub_7264B0()
{
  _BYTE *result; // rax

  result = sub_7246D0(40);
  *(_QWORD *)result = 0;
  *((_QWORD *)result + 1) = 0;
  result[16] = 0;
  result[24] = 0;
  *((_QWORD *)result + 4) = 0;
  return result;
}
