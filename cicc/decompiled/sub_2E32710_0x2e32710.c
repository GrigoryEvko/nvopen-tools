// Function: sub_2E32710
// Address: 0x2e32710
//
__int64 __fastcall sub_2E32710(_QWORD *a1)
{
  __int64 v1; // r13
  unsigned __int64 *v2; // rcx
  unsigned __int64 v3; // rdx

  v1 = a1[4] + 320LL;
  sub_2E31020(v1, (__int64)a1);
  v2 = (unsigned __int64 *)a1[1];
  v3 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  *v2 = v3 | *v2 & 7;
  *(_QWORD *)(v3 + 8) = v2;
  *a1 &= 7uLL;
  a1[1] = 0;
  return sub_2E79D60(v1, a1);
}
