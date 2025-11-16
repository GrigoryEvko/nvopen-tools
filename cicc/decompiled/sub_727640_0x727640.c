// Function: sub_727640
// Address: 0x727640
//
_BYTE *sub_727640()
{
  _BYTE *result; // rax

  result = sub_7246D0(24);
  *(_QWORD *)result = 0;
  result[8] = 0;
  *((_QWORD *)result + 2) = 0;
  return result;
}
