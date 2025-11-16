// Function: sub_B43D10
// Address: 0xb43d10
//
__int64 __fastcall sub_B43D10(_QWORD *a1)
{
  __int64 *v1; // rcx
  unsigned __int64 v2; // rdx
  __int64 result; // rax

  sub_B43CE0((__int64)a1);
  sub_AA4910(a1[5] + 48LL, (__int64)a1);
  v1 = (__int64 *)a1[4];
  v2 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  result = v2 | *v1 & 7;
  *v1 = result;
  *(_QWORD *)(v2 + 8) = v1;
  a1[4] = 0;
  a1[3] &= 7uLL;
  return result;
}
