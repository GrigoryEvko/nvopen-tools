// Function: sub_1E42060
// Address: 0x1e42060
//
__int64 __fastcall sub_1E42060(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  __int64 i; // rbx
  __int64 v7; // rdi

  *a1 = off_49FC128;
  v2 = a1[48];
  if ( (_QWORD *)v2 != a1 + 50 )
    _libc_free(v2);
  v3 = a1[45];
  if ( v3 )
    j_j___libc_free_0_0(v3);
  _libc_free(a1[42]);
  v4 = a1[39];
  if ( (_QWORD *)v4 != a1 + 41 )
    _libc_free(v4);
  v5 = a1[34];
  if ( v5 )
  {
    for ( i = v5 + 24LL * *(_QWORD *)(v5 - 8); v5 != i; i -= 24 )
    {
      v7 = *(_QWORD *)(i - 8);
      if ( v7 )
        j_j___libc_free_0_0(v7);
    }
    j_j_j___libc_free_0_0(v5 - 8);
  }
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 576);
}
