// Function: sub_727170
// Address: 0x727170
//
_BYTE *sub_727170()
{
  _BYTE *result; // rax
  __int64 v1; // rdx

  result = sub_7246D0(24);
  result[8] = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  *((_QWORD *)result + 2) = 0;
  *(_QWORD *)result = v1;
  return result;
}
