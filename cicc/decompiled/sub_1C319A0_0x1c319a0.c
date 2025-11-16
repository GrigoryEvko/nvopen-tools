// Function: sub_1C319A0
// Address: 0x1c319a0
//
void *__fastcall sub_1C319A0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49F77D0;
  j___libc_free_0(a1[39]);
  j___libc_free_0(a1[35]);
  sub_16E7BC0(a1 + 28);
  v2 = (_QWORD *)a1[24];
  if ( v2 != a1 + 26 )
    j_j___libc_free_0(v2, a1[26] + 1LL);
  return sub_1636790(a1);
}
