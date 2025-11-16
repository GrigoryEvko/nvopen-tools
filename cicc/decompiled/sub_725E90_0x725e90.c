// Function: sub_725E90
// Address: 0x725e90
//
_QWORD *sub_725E90()
{
  _QWORD *result; // rax
  __int64 v1; // rdx

  result = sub_7247C0(32);
  *((_WORD *)result + 8) = 0;
  *result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  result[1] = 0;
  *(_QWORD *)((char *)result + 20) = v1;
  return result;
}
