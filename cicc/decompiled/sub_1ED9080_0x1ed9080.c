// Function: sub_1ED9080
// Address: 0x1ed9080
//
__int64 __fastcall sub_1ED9080(_QWORD *a1)
{
  _QWORD *v1; // r14
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // r13
  __int64 i; // r12
  __int64 v12; // rdi

  v1 = a1 - 29;
  *(a1 - 29) = off_49FDFC0;
  *a1 = &unk_49FE090;
  v3 = a1[64];
  if ( (_QWORD *)v3 != a1 + 66 )
    _libc_free(v3);
  v4 = a1[54];
  if ( (_QWORD *)v4 != a1 + 56 )
    _libc_free(v4);
  v5 = a1[43];
  if ( v5 != a1[42] )
    _libc_free(v5);
  v6 = a1[31];
  if ( (_QWORD *)v6 != a1 + 33 )
    _libc_free(v6);
  v7 = a1[21];
  if ( (_QWORD *)v7 != a1 + 23 )
    _libc_free(v7);
  v8 = a1[19];
  if ( v8 )
    j_j___libc_free_0_0(v8);
  _libc_free(a1[16]);
  v9 = a1[13];
  if ( (_QWORD *)v9 != a1 + 15 )
    _libc_free(v9);
  v10 = a1[8];
  if ( v10 )
  {
    for ( i = v10 + 24LL * *(_QWORD *)(v10 - 8); v10 != i; i -= 24 )
    {
      v12 = *(_QWORD *)(i - 8);
      if ( v12 )
        j_j___libc_free_0_0(v12);
    }
    j_j_j___libc_free_0_0(v10 - 8);
  }
  _libc_free(*(a1 - 3));
  _libc_free(*(a1 - 6));
  _libc_free(*(a1 - 9));
  *(a1 - 29) = &unk_49EE078;
  sub_16366C0(v1);
  return j_j___libc_free_0(v1, 792);
}
