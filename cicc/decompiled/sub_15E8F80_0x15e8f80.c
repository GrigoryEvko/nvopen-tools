// Function: sub_15E8F80
// Address: 0x15e8f80
//
__int64 __fastcall sub_15E8F80(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49ED1D8;
  v2 = (_QWORD *)a1[21];
  if ( v2 != a1 + 23 )
    j_j___libc_free_0(v2, a1[23] + 1LL);
  sub_1636790(a1);
  return j_j___libc_free_0(a1, 208);
}
