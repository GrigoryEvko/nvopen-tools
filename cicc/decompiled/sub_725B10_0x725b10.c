// Function: sub_725B10
// Address: 0x725b10
//
_BYTE *sub_725B10()
{
  _BYTE *result; // rax

  result = sub_7246D0(40);
  *(_QWORD *)result = 0;
  *((_QWORD *)result + 1) = 0;
  result[16] = 0;
  *((_QWORD *)result + 4) = 0;
  return result;
}
