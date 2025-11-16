// Function: sub_7275F0
// Address: 0x7275f0
//
_BYTE *sub_7275F0()
{
  _BYTE *result; // rax
  __int64 v1; // rdx

  result = sub_7246D0(56);
  *(_QWORD *)result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  *((_WORD *)result + 16) &= 0xF800u;
  *((_QWORD *)result + 1) = 0;
  *((_QWORD *)result + 2) = 0;
  *((_QWORD *)result + 3) = 0;
  *(_QWORD *)(result + 36) = v1;
  *(_QWORD *)(result + 44) = v1;
  return result;
}
