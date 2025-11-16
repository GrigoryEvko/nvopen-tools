// Function: sub_727210
// Address: 0x727210
//
_BYTE *sub_727210()
{
  _BYTE *result; // rax
  __int64 v1; // rdx

  result = sub_7246D0(32);
  *(_QWORD *)result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  result[8] &= 0xF8u;
  *(_QWORD *)(result + 12) = v1;
  *(_QWORD *)(result + 20) = v1;
  return result;
}
