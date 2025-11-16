// Function: sub_1EB4780
// Address: 0x1eb4780
//
__int64 __fastcall sub_1EB4780(_QWORD *a1)
{
  _QWORD *v1; // r14
  __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // r13
  __int64 i; // r12
  __int64 v10; // rdi

  v1 = a1 - 29;
  *(a1 - 29) = off_49FD818;
  *a1 = &unk_49FD910;
  a1[55] = &unk_49FD968;
  _libc_free(a1[62]);
  v3 = a1[58];
  if ( v3 )
    j_j___libc_free_0(v3, a1[60] - v3);
  v4 = a1[57];
  if ( v4 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 16LL))(v4);
  v5 = a1[20];
  *a1 = &unk_4A00E78;
  if ( v5 != a1[19] )
    _libc_free(v5);
  v6 = a1[17];
  if ( v6 )
    j_j___libc_free_0_0(v6);
  _libc_free(a1[14]);
  v7 = a1[11];
  if ( (_QWORD *)v7 != a1 + 13 )
    _libc_free(v7);
  v8 = a1[6];
  if ( v8 )
  {
    for ( i = v8 + 24LL * *(_QWORD *)(v8 - 8); v8 != i; i -= 24 )
    {
      v10 = *(_QWORD *)(i - 8);
      if ( v10 )
        j_j___libc_free_0_0(v10);
    }
    j_j_j___libc_free_0_0(v8 - 8);
  }
  _libc_free(*(a1 - 3));
  _libc_free(*(a1 - 6));
  _libc_free(*(a1 - 9));
  *(a1 - 29) = &unk_49EE078;
  sub_16366C0(v1);
  return j_j___libc_free_0(v1, 752);
}
