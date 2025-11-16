// Function: sub_1F2A610
// Address: 0x1f2a610
//
void *__fastcall sub_1F2A610(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  _libc_free(a1[37]);
  v2 = a1[31];
  if ( (_QWORD *)v2 != a1 + 33 )
    _libc_free(v2);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
