// Function: sub_7271D0
// Address: 0x7271d0
//
_QWORD *sub_7271D0()
{
  _QWORD *result; // rax
  __int64 v1; // rdx

  result = sub_7247C0(48);
  *((_BYTE *)result + 8) = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  result[2] = 0;
  *((_BYTE *)result + 24) = 0;
  *result = v1;
  result[4] = 0;
  result[5] = 0;
  return result;
}
