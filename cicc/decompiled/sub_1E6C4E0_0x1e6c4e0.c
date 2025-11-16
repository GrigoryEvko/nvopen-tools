// Function: sub_1E6C4E0
// Address: 0x1e6c4e0
//
void *__fastcall sub_1E6C4E0(_QWORD *a1)
{
  *(a1 - 8) = &unk_49FC7A0;
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return sub_1E6C000(a1 - 8);
}
