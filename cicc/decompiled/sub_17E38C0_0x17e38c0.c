// Function: sub_17E38C0
// Address: 0x17e38c0
//
void *__fastcall sub_17E38C0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49F0580;
  v2 = (_QWORD *)a1[20];
  if ( v2 != a1 + 22 )
    j_j___libc_free_0(v2, a1[22] + 1LL);
  return sub_1636790(a1);
}
