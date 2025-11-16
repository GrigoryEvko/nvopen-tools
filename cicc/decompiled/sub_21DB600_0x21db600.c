// Function: sub_21DB600
// Address: 0x21db600
//
void *__fastcall sub_21DB600(_QWORD *a1)
{
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
