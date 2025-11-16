// Function: sub_726850
// Address: 0x726850
//
_BYTE *sub_726850()
{
  _BYTE *result; // rax

  result = sub_7246D0(128);
  result[120] &= 0xF8u;
  *(_QWORD *)result = 0;
  *((_QWORD *)result + 1) = 0;
  *((_QWORD *)result + 2) = 0;
  *((_QWORD *)result + 3) = 0;
  *((_QWORD *)result + 4) = 0;
  *((_QWORD *)result + 5) = 0;
  *((_QWORD *)result + 6) = 0;
  *((_QWORD *)result + 7) = 0;
  *((_QWORD *)result + 8) = 0;
  *((_QWORD *)result + 9) = 0;
  *((_QWORD *)result + 10) = 0;
  *((_QWORD *)result + 11) = 0;
  *((_QWORD *)result + 12) = 0;
  *((_QWORD *)result + 13) = 0;
  return result;
}
