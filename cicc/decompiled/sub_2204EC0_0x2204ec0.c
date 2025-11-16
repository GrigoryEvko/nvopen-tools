// Function: sub_2204EC0
// Address: 0x2204ec0
//
__int64 __fastcall sub_2204EC0(_QWORD *a1)
{
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 232);
}
