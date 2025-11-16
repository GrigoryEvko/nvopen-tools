// Function: sub_1F2ACE0
// Address: 0x1f2ace0
//
__int64 __fastcall sub_1F2ACE0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  _QWORD *v3; // rdi

  *a1 = &unk_49FEA80;
  v2 = a1[39];
  if ( v2 != a1[38] )
    _libc_free(v2);
  j___libc_free_0(a1[33]);
  v3 = (_QWORD *)a1[22];
  if ( v3 != a1 + 24 )
    j_j___libc_free_0(v3, a1[24] + 1LL);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 472);
}
