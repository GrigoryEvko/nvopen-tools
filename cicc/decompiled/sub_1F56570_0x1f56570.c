// Function: sub_1F56570
// Address: 0x1f56570
//
void *__fastcall sub_1F56570(_QWORD *a1)
{
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
