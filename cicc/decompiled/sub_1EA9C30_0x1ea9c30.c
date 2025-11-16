// Function: sub_1EA9C30
// Address: 0x1ea9c30
//
void *__fastcall sub_1EA9C30(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // r13
  __int64 i; // rbx
  __int64 v6; // rdi

  *a1 = off_49FD440;
  v2 = a1[41];
  if ( v2 )
    j_j___libc_free_0_0(v2);
  _libc_free(a1[38]);
  v3 = a1[35];
  if ( (_QWORD *)v3 != a1 + 37 )
    _libc_free(v3);
  v4 = a1[30];
  if ( v4 )
  {
    for ( i = v4 + 24LL * *(_QWORD *)(v4 - 8); v4 != i; i -= 24 )
    {
      v6 = *(_QWORD *)(i - 8);
      if ( v6 )
        j_j___libc_free_0_0(v6);
    }
    j_j_j___libc_free_0_0(v4 - 8);
  }
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
