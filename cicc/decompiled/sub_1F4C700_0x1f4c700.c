// Function: sub_1F4C700
// Address: 0x1f4c700
//
void *__fastcall sub_1F4C700(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *a1 = off_49FF950;
  j___libc_free_0(a1[74]);
  j___libc_free_0(a1[70]);
  v2 = a1[58];
  if ( v2 != a1[57] )
    _libc_free(v2);
  v3 = a1[45];
  if ( v3 != a1[44] )
    _libc_free(v3);
  j___libc_free_0(a1[40]);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
