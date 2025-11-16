// Function: sub_1DF1DF0
// Address: 0x1df1df0
//
void *__fastcall sub_1DF1DF0(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 71;
  v3 = a1[69];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
