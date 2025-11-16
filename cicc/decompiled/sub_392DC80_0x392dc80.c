// Function: sub_392DC80
// Address: 0x392dc80
//
__int64 __fastcall sub_392DC80(__int64 a1, unsigned int a2)
{
  _QWORD *v2; // r13
  __int64 v3; // rcx

  v2 = *(_QWORD **)(a1 + 8);
  v3 = (*(__int64 (__fastcall **)(_QWORD *))(*v2 + 64LL))(v2) + v2[3] - v2[1];
  return sub_16E8900(*(_QWORD *)(a1 + 8), a2 * (((unsigned __int64)a2 + v3 - 1) / a2) - v3);
}
