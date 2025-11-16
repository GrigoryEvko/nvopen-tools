// Function: sub_726370
// Address: 0x726370
//
_BYTE *sub_726370()
{
  _BYTE *result; // rax
  __int64 v1; // rdx

  result = sub_7246D0(48);
  *(_QWORD *)result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  *((_QWORD *)result + 1) = 0;
  *((_QWORD *)result + 2) = 0;
  result[24] = 0;
  *(_QWORD *)(result + 28) = v1;
  *((_QWORD *)result + 5) = 0;
  return result;
}
