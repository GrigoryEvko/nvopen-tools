// Function: sub_B30340
// Address: 0xb30340
//
__int64 __fastcall sub_B30340(_QWORD *a1)
{
  unsigned __int64 *v1; // rcx
  unsigned __int64 v2; // rdx

  sub_BA8670(a1[5] + 40LL, a1);
  v1 = (unsigned __int64 *)a1[7];
  v2 = a1[6] & 0xFFFFFFFFFFFFFFF8LL;
  *v1 = v2 | *v1 & 7;
  *(_QWORD *)(v2 + 8) = v1;
  a1[6] &= 7uLL;
  a1[7] = 0;
  sub_AD0030((__int64)a1);
  sub_BD7260(a1);
  return sub_BD2DD0(a1);
}
