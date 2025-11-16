// Function: sub_726DD0
// Address: 0x726dd0
//
_BYTE *sub_726DD0()
{
  _BYTE *result; // rax
  __int64 v1; // rdx

  result = sub_7246D0(80);
  *(_QWORD *)result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  *((_WORD *)result + 20) &= 0xF000u;
  result[16] = 0;
  *((_QWORD *)result + 1) = v1;
  *((_QWORD *)result + 3) = 0;
  *((_QWORD *)result + 4) = 0;
  result[42] = 0;
  *((_QWORD *)result + 6) = 0;
  *((_QWORD *)result + 7) = 0;
  *((_QWORD *)result + 8) = 0;
  *((_QWORD *)result + 9) = 0;
  return result;
}
