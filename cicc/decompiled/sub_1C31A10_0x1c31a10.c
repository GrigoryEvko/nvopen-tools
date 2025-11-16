// Function: sub_1C31A10
// Address: 0x1c31a10
//
__int64 __fastcall sub_1C31A10(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49F77D0;
  j___libc_free_0(a1[39]);
  j___libc_free_0(a1[35]);
  sub_16E7BC0(a1 + 28);
  v2 = (_QWORD *)a1[24];
  if ( v2 != a1 + 26 )
    j_j___libc_free_0(v2, a1[26] + 1LL);
  sub_1636790(a1);
  return j_j___libc_free_0(a1, 336);
}
