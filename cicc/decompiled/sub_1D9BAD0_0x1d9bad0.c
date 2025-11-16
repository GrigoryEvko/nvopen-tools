// Function: sub_1D9BAD0
// Address: 0x1d9bad0
//
void *__fastcall sub_1D9BAD0(_QWORD *a1)
{
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
