// Function: sub_1648A60
// Address: 0x1648a60
//
_QWORD *__fastcall sub_1648A60(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rax
  _QWORD *v4; // r12

  v2 = 3LL * a2;
  v3 = (_QWORD *)sub_22077B0(v2 * 8 + a1);
  v4 = &v3[v2];
  *((_DWORD *)v4 + 5) = *((_DWORD *)v4 + 5) & 0x30000000 | a2 & 0xFFFFFFF;
  sub_16485A0(v3, v4);
  return v4;
}
