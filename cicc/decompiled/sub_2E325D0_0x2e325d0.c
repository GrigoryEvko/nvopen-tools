// Function: sub_2E325D0
// Address: 0x2e325d0
//
__int64 __fastcall sub_2E325D0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r14
  unsigned __int64 *v3; // rcx
  unsigned __int64 v4; // rdx

  sub_2E2F9E0((__int64)a2);
  v2 = a2[1];
  sub_2E31080(a1 + 40, (__int64)a2);
  v3 = (unsigned __int64 *)a2[1];
  v4 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  *v3 = v4 | *v3 & 7;
  *(_QWORD *)(v4 + 8) = v3;
  *a2 &= 7uLL;
  a2[1] = 0;
  sub_2E310F0(a1 + 40);
  return v2;
}
