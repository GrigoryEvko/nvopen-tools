// Function: sub_1802350
// Address: 0x1802350
//
void *__fastcall sub_1802350(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49F08F0;
  j___libc_free_0(a1[97]);
  j___libc_free_0(a1[93]);
  v2 = (_QWORD *)a1[21];
  if ( v2 != a1 + 23 )
    j_j___libc_free_0(v2, a1[23] + 1LL);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
