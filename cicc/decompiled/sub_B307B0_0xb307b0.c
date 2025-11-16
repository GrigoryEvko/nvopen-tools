// Function: sub_B307B0
// Address: 0xb307b0
//
__int64 __fastcall sub_B307B0(_QWORD *a1)
{
  unsigned __int64 *v1; // rcx
  unsigned __int64 v2; // rdx

  sub_BA86F0(a1[5] + 56LL, a1);
  v1 = (unsigned __int64 *)a1[8];
  v2 = a1[7] & 0xFFFFFFFFFFFFFFF8LL;
  *v1 = v2 | *v1 & 7;
  *(_QWORD *)(v2 + 8) = v1;
  a1[7] &= 7uLL;
  a1[8] = 0;
  sub_B2F9E0((__int64)a1, (__int64)a1, v2, (__int64)v1);
  return sub_BD2DD0(a1);
}
