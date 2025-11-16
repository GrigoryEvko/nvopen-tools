// Function: sub_1802280
// Address: 0x1802280
//
void *__fastcall sub_1802280(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49F0848;
  v2 = (_QWORD *)a1[28];
  if ( v2 != a1 + 30 )
    j_j___libc_free_0(v2, a1[30] + 1LL);
  j___libc_free_0(a1[22]);
  return sub_1636790(a1);
}
