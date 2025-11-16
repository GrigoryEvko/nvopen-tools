// Function: sub_2E88DB0
// Address: 0x2e88db0
//
_QWORD *__fastcall sub_2E88DB0(_QWORD *a1)
{
  unsigned __int64 *v1; // rcx
  unsigned __int64 v2; // rdx

  sub_2E31080(a1[3] + 40LL, (__int64)a1);
  v1 = (unsigned __int64 *)a1[1];
  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  *v1 = v2 | *v1 & 7;
  *(_QWORD *)(v2 + 8) = v1;
  a1[1] = 0;
  *a1 &= 7uLL;
  return a1;
}
