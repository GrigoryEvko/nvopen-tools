// Function: sub_B43D60
// Address: 0xb43d60
//
__int64 __fastcall sub_B43D60(_QWORD *a1)
{
  __int64 v1; // r13
  unsigned __int64 *v2; // rcx
  unsigned __int64 v3; // rdx

  sub_B43CE0((__int64)a1);
  v1 = a1[4];
  sub_AA4910(a1[5] + 48LL, (__int64)a1);
  v2 = (unsigned __int64 *)a1[4];
  v3 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  *v2 = v3 | *v2 & 7;
  *(_QWORD *)(v3 + 8) = v2;
  a1[3] &= 7uLL;
  a1[4] = 0;
  sub_BD72D0(a1);
  return v1;
}
