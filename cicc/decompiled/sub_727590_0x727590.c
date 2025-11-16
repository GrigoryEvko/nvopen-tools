// Function: sub_727590
// Address: 0x727590
//
_BYTE *sub_727590()
{
  _BYTE *result; // rax
  __int64 v1; // rdx

  result = sub_7246D0(80);
  *(_QWORD *)result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  *((_WORD *)result + 12) &= 0xFE00u;
  *((_QWORD *)result + 1) = 0;
  result[25] &= 1u;
  *((_QWORD *)result + 2) = 0;
  *(_QWORD *)(result + 44) = v1;
  *(_QWORD *)(result + 52) = v1;
  *(_QWORD *)(result + 60) = v1;
  *(_QWORD *)(result + 68) = v1;
  *((_QWORD *)result + 4) = 0;
  *((_DWORD *)result + 10) = 0;
  return result;
}
