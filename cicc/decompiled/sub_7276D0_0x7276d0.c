// Function: sub_7276D0
// Address: 0x7276d0
//
_QWORD *sub_7276D0()
{
  _QWORD *result; // rax
  __int64 v1; // rdx

  result = sub_7247C0(48);
  *result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  *((_DWORD *)result + 2) &= 0xFE000000;
  result[2] = 0;
  result[3] = v1;
  result[4] = v1;
  result[5] = 0;
  return result;
}
