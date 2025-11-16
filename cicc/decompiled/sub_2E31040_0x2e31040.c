// Function: sub_2E31040
// Address: 0x2e31040
//
__int64 __fastcall sub_2E31040(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13

  v2 = *a1;
  *(_QWORD *)(a2 + 24) = *a1;
  v3 = *(_QWORD *)(v2 + 32);
  sub_2E86750(a2, *(_QWORD *)(v3 + 32));
  return sub_2E78D20(v3, a2);
}
