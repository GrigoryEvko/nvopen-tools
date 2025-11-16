// Function: sub_1F11C70
// Address: 0x1f11c70
//
void *__fastcall sub_1F11C70(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *a1 = &unk_49FE828;
  sub_1F11BC0((__int64)a1);
  _libc_free(a1[64]);
  v2 = a1[58];
  if ( (_QWORD *)v2 != a1 + 60 )
    _libc_free(v2);
  v3 = a1[47];
  if ( (_QWORD *)v3 != a1 + 49 )
    _libc_free(v3);
  v4 = a1[41];
  if ( (_QWORD *)v4 != a1 + 43 )
    _libc_free(v4);
  v5 = a1[35];
  if ( (_QWORD *)v5 != a1 + 37 )
    _libc_free(v5);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
