// Function: sub_7263C0
// Address: 0x7263c0
//
_BYTE *sub_7263C0()
{
  _BYTE *result; // rax

  result = sub_7246D0(24);
  *(_QWORD *)result = 0;
  result[8] = 0;
  *((_QWORD *)result + 2) = 0;
  return result;
}
