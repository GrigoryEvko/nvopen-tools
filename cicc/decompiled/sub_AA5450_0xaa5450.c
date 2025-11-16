// Function: sub_AA5450
// Address: 0xaa5450
//
__int64 __fastcall sub_AA5450(_QWORD *a1)
{
  __int64 v1; // r13
  unsigned __int64 *v2; // rcx
  unsigned __int64 v3; // rdx

  v1 = a1[4];
  sub_B2B7E0(a1[9] + 72LL, a1);
  v2 = (unsigned __int64 *)a1[4];
  v3 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  *v2 = v3 | *v2 & 7;
  *(_QWORD *)(v3 + 8) = v2;
  a1[3] &= 7uLL;
  a1[4] = 0;
  sub_AA5290((__int64)a1);
  j_j___libc_free_0(a1, 80);
  return v1;
}
