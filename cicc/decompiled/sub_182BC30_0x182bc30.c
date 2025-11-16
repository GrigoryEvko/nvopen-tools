// Function: sub_182BC30
// Address: 0x182bc30
//
void *__fastcall sub_182BC30(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49F0AE8;
  v2 = (_QWORD *)a1[21];
  if ( v2 != a1 + 23 )
    j_j___libc_free_0(v2, a1[23] + 1LL);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
