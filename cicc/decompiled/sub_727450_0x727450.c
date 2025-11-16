// Function: sub_727450
// Address: 0x727450
//
_BYTE *sub_727450()
{
  _BYTE *result; // rax
  __int64 v1; // rdx

  result = sub_7246D0(24);
  *(_QWORD *)result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  result[16] = 0;
  *((_QWORD *)result + 1) = v1;
  return result;
}
