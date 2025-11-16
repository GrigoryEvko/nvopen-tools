// Function: sub_726B30
// Address: 0x726b30
//
_BYTE *__fastcall sub_726B30(char a1)
{
  _BYTE *v1; // r12
  __int64 v2; // rdx
  __int64 v3; // rax

  v1 = sub_7246D0(96);
  v1[41] &= 0xC0u;
  *((_QWORD *)v1 + 2) = 0;
  v2 = *(_QWORD *)&dword_4F077C8;
  *((_QWORD *)v1 + 3) = 0;
  *(_QWORD *)v1 = v2;
  v3 = *(_QWORD *)&dword_4F077C8;
  *((_QWORD *)v1 + 4) = 0;
  *((_QWORD *)v1 + 1) = v3;
  *((_QWORD *)v1 + 7) = 0;
  *((_QWORD *)v1 + 8) = 0;
  sub_7268E0((__int64)v1, a1);
  return v1;
}
