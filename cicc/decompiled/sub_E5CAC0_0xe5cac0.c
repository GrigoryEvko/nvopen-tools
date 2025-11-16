// Function: sub_E5CAC0
// Address: 0xe5cac0
//
__int64 __fastcall sub_E5CAC0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx

  v2 = *(_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL);
  v3 = sub_E5C2C0((__int64)a1, v2);
  return v3 + sub_E5BD20(a1, v2);
}
