// Function: sub_1EADEE0
// Address: 0x1eadee0
//
__int64 __fastcall sub_1EADEE0(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = a1 + 39;
  v3 = a1[37];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[31];
  if ( (_QWORD *)v4 != a1 + 33 )
    _libc_free(v4);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 360);
}
