// Function: sub_1EB4610
// Address: 0x1eb4610
//
__int64 __fastcall sub_1EB4610(_QWORD *a1)
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

  v1 = a1 - 84;
  *(a1 - 84) = off_49FD818;
  *(a1 - 55) = &unk_49FD910;
  *a1 = &unk_49FD968;
  _libc_free(a1[7]);
  v3 = a1[3];
  if ( v3 )
    j_j___libc_free_0(v3, a1[5] - v3);
  v4 = a1[2];
  if ( v4 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 16LL))(v4);
  v5 = *(a1 - 35);
  *(a1 - 55) = &unk_4A00E78;
  if ( v5 != *(a1 - 36) )
    _libc_free(v5);
  v6 = *(a1 - 38);
  if ( v6 )
    j_j___libc_free_0_0(v6);
  _libc_free(*(a1 - 41));
  v7 = *(a1 - 44);
  if ( (_QWORD *)v7 != a1 - 42 )
    _libc_free(v7);
  v8 = *(a1 - 49);
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
  _libc_free(*(a1 - 58));
  _libc_free(*(a1 - 61));
  _libc_free(*(a1 - 64));
  *(a1 - 84) = &unk_49EE078;
  sub_16366C0(v1);
  return j_j___libc_free_0(v1, 752);
}
