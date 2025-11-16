// Function: sub_725B40
// Address: 0x725b40
//
_BYTE *sub_725B40()
{
  _BYTE *result; // rax
  __int64 v1; // rdx

  result = sub_7246D0(56);
  *(_QWORD *)result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  *((_QWORD *)result + 1) = 0;
  *((_QWORD *)result + 2) = 0;
  *((_QWORD *)result + 3) = 0;
  result[32] = 0;
  *(_QWORD *)(result + 36) = v1;
  *((_QWORD *)result + 6) = 0;
  return result;
}
