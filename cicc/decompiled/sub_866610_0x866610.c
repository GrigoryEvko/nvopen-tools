// Function: sub_866610
// Address: 0x866610
//
__int64 __fastcall sub_866610(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 v3; // rbx
  __int64 result; // rax

  v2 = sub_8663A0();
  v3 = sub_866270(0);
  *(_QWORD *)(v3 + 8) = *(_QWORD *)(a1 + 8);
  result = sub_892BC0(a1);
  *(_QWORD *)(v3 + 64) = result;
  v2[3] = v3;
  *(_QWORD *)(a2 + 16) = v2;
  return result;
}
