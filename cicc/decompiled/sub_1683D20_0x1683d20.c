// Function: sub_1683D20
// Address: 0x1683d20
//
__int64 __fastcall sub_1683D20(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax

  v2 = (_QWORD *)malloc(0x18u);
  *v2 = a1;
  v3 = (__int64)v2;
  v2[1] = a2;
  sub_1688E30();
  v4 = qword_4F9F320;
  qword_4F9F320 = v3;
  *(_QWORD *)(v3 + 16) = v4;
  return sub_1688E70();
}
