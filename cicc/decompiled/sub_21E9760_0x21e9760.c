// Function: sub_21E9760
// Address: 0x21e9760
//
__int64 __fastcall sub_21E9760(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = &unk_4A03F50;
  v2 = (_QWORD *)a1[30];
  if ( v2 != a1 + 32 )
    j_j___libc_free_0(v2, a1[32] + 1LL);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 272);
}
