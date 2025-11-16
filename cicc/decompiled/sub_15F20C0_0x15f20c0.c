// Function: sub_15F20C0
// Address: 0x15f20c0
//
__int64 __fastcall sub_15F20C0(_QWORD *a1)
{
  __int64 v1; // r13
  unsigned __int64 *v2; // rcx
  unsigned __int64 v3; // rdx
  __int64 v4; // r8

  v1 = a1[4];
  sub_157EA20(a1[5] + 40LL, (__int64)a1);
  v2 = (unsigned __int64 *)a1[4];
  v3 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  *v2 = v3 | *v2 & 7;
  *(_QWORD *)(v3 + 8) = v2;
  a1[3] &= 7uLL;
  a1[4] = 0;
  sub_164BEC0(a1, a1, v3, v2, v4);
  return v1;
}
