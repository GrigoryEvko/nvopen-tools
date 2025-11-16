// Function: sub_2204E70
// Address: 0x2204e70
//
void *__fastcall sub_2204E70(_QWORD *a1)
{
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
