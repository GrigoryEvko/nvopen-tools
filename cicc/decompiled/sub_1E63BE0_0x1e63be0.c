// Function: sub_1E63BE0
// Address: 0x1e63be0
//
void *__fastcall sub_1E63BE0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  v2 = a1 + 29;
  *(v2 - 29) = &unk_49FC390;
  sub_1E63B90(v2);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
