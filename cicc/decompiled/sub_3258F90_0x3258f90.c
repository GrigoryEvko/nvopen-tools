// Function: sub_3258F90
// Address: 0x3258f90
//
unsigned __int64 __fastcall sub_3258F90(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r14
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rax

  v2 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 216LL);
  v3 = sub_E81A90(1, v2, 0, 0);
  v4 = sub_E808D0(a2, 0x72u, *(_QWORD **)(*(_QWORD *)(a1 + 8) + 216LL), 0);
  return sub_E81A00(0, v4, v3, v2, 0);
}
