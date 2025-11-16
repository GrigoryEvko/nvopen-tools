// Function: sub_1E6C640
// Address: 0x1e6c640
//
__int64 __fastcall sub_1E6C640(_QWORD *a1)
{
  *a1 = &unk_49FC7A0;
  _libc_free(a1[34]);
  _libc_free(a1[31]);
  _libc_free(a1[28]);
  a1[8] = &unk_49EE078;
  sub_16366C0(a1 + 8);
  sub_1E6C000(a1);
  return j_j___libc_free_0(a1, 296);
}
