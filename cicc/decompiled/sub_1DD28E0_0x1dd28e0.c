// Function: sub_1DD28E0
// Address: 0x1dd28e0
//
void *__fastcall sub_1DD28E0(_QWORD *a1)
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
  return sub_16366C0(a1);
}
