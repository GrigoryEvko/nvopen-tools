// Function: sub_725A70
// Address: 0x725a70
//
_QWORD *__fastcall sub_725A70(unsigned __int8 a1)
{
  _BYTE *v1; // rax
  _QWORD *v2; // r12

  v1 = sub_7246D0(128);
  *(_DWORD *)(v1 + 49) &= 0xFE000000;
  v2 = v1;
  *(_QWORD *)v1 = 0;
  *((_QWORD *)v1 + 1) = 0;
  *((_QWORD *)v1 + 2) = 0;
  *((_QWORD *)v1 + 3) = 0;
  *((_QWORD *)v1 + 4) = 0;
  *((_QWORD *)v1 + 5) = 0;
  sub_7259F0((__int64)v1, a1);
  v2[10] = 0;
  v2[11] = 0;
  v2[12] = 0;
  v2[13] = 0;
  v2[14] = 0;
  v2[15] = 0;
  return v2;
}
