// Function: sub_727110
// Address: 0x727110
//
_BYTE *sub_727110()
{
  _BYTE *result; // rax
  __int64 v1; // rdx

  result = sub_7246D0(64);
  result[58] &= 0xFCu;
  v1 = *(_QWORD *)&dword_4F077C8;
  *((_QWORD *)result + 1) = 0;
  result[16] = 0;
  *(_QWORD *)result = v1;
  *((_QWORD *)result + 3) = 0;
  *((_QWORD *)result + 4) = 0;
  *((_QWORD *)result + 5) = 0;
  *((_QWORD *)result + 6) = 0;
  *((_WORD *)result + 28) = 0;
  return result;
}
