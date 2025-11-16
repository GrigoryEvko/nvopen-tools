// Function: sub_15E5AB0
// Address: 0x15e5ab0
//
__int64 __fastcall sub_15E5AB0(_QWORD *a1)
{
  unsigned __int64 *v1; // rcx
  unsigned __int64 v2; // rdx

  sub_1631D10(a1[5] + 56LL, a1);
  v1 = (unsigned __int64 *)a1[7];
  v2 = a1[6] & 0xFFFFFFFFFFFFFFF8LL;
  *v1 = v2 | *v1 & 7;
  *(_QWORD *)(v2 + 8) = v1;
  a1[6] &= 7uLL;
  a1[7] = 0;
  sub_159D9E0((__int64)a1);
  sub_164BE60(a1);
  return sub_1648B90(a1);
}
