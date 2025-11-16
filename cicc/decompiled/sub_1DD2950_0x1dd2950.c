// Function: sub_1DD2950
// Address: 0x1dd2950
//
__int64 __fastcall sub_1DD2950(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 31;
  v3 = a1[29];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 376);
}
