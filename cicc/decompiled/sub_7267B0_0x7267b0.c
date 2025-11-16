// Function: sub_7267B0
// Address: 0x7267b0
//
_BYTE *sub_7267B0()
{
  _BYTE *result; // rax
  __int64 v1; // rdx

  result = sub_7246D0(72);
  *(_QWORD *)result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  result[64] |= 1u;
  *((_QWORD *)result + 1) = 0;
  *((_QWORD *)result + 2) = 0;
  *((_QWORD *)result + 3) = 0;
  *((_QWORD *)result + 4) = 0;
  *((_QWORD *)result + 5) = v1;
  *((_QWORD *)result + 6) = v1;
  *((_QWORD *)result + 7) = v1;
  return result;
}
