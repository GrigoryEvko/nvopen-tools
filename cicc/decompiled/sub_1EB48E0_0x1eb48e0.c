// Function: sub_1EB48E0
// Address: 0x1eb48e0
//
void *__fastcall sub_1EB48E0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // r13
  __int64 i; // r12
  __int64 v9; // rdi

  *(a1 - 29) = off_49FD818;
  *a1 = &unk_49FD910;
  a1[55] = &unk_49FD968;
  _libc_free(a1[62]);
  v2 = a1[58];
  if ( v2 )
    j_j___libc_free_0(v2, a1[60] - v2);
  v3 = a1[57];
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 16LL))(v3);
  v4 = a1[20];
  *a1 = &unk_4A00E78;
  if ( v4 != a1[19] )
    _libc_free(v4);
  v5 = a1[17];
  if ( v5 )
    j_j___libc_free_0_0(v5);
  _libc_free(a1[14]);
  v6 = a1[11];
  if ( (_QWORD *)v6 != a1 + 13 )
    _libc_free(v6);
  v7 = a1[6];
  if ( v7 )
  {
    for ( i = v7 + 24LL * *(_QWORD *)(v7 - 8); v7 != i; i -= 24 )
    {
      v9 = *(_QWORD *)(i - 8);
      if ( v9 )
        j_j___libc_free_0_0(v9);
    }
    j_j_j___libc_free_0_0(v7 - 8);
  }
  _libc_free(*(a1 - 3));
  _libc_free(*(a1 - 6));
  _libc_free(*(a1 - 9));
  *(a1 - 29) = &unk_49EE078;
  return sub_16366C0(a1 - 29);
}
