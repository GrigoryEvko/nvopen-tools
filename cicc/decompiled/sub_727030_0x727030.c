// Function: sub_727030
// Address: 0x727030
//
_BYTE *sub_727030()
{
  _BYTE *result; // rax

  result = sub_7246D0(32);
  *(_QWORD *)result = 0;
  *((_QWORD *)result + 1) = 0;
  result[16] = 0;
  *((_QWORD *)result + 3) = 0;
  return result;
}
