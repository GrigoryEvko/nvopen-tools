// Function: sub_724980
// Address: 0x724980
//
_BYTE *sub_724980()
{
  _BYTE *result; // rax

  result = sub_7246D0(24);
  result[8] &= 0xFCu;
  *(_QWORD *)result = 0;
  *((_QWORD *)result + 2) = 0;
  return result;
}
