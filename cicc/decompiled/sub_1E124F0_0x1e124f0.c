// Function: sub_1E124F0
// Address: 0x1e124f0
//
__int64 __fastcall sub_1E124F0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49FB858;
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
