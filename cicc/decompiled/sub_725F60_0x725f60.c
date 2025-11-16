// Function: sub_725F60
// Address: 0x725f60
//
_QWORD *sub_725F60()
{
  _QWORD *result; // rax

  result = sub_7247C0(56);
  *((_BYTE *)result + 52) &= 0xFCu;
  *result = 0;
  result[1] = 0;
  result[2] = -1;
  result[3] = -1;
  result[4] = -1;
  result[5] = -1;
  *((_DWORD *)result + 12) = -1;
  return result;
}
