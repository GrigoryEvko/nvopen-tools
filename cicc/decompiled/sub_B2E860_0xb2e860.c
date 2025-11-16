// Function: sub_B2E860
// Address: 0xb2e860
//
__int64 __fastcall sub_B2E860(_QWORD *a1)
{
  unsigned __int64 *v1; // rcx
  unsigned __int64 v2; // rdx

  sub_BA8570(a1[5] + 24LL, a1);
  v1 = (unsigned __int64 *)a1[8];
  v2 = a1[7] & 0xFFFFFFFFFFFFFFF8LL;
  *v1 = v2 | *v1 & 7;
  *(_QWORD *)(v2 + 8) = v1;
  a1[7] &= 7uLL;
  a1[8] = 0;
  sub_B2E780(a1);
  return sub_BD2DD0(a1);
}
