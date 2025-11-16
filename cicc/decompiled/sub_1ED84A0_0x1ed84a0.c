// Function: sub_1ED84A0
// Address: 0x1ed84a0
//
void *__fastcall sub_1ED84A0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // r13
  __int64 i; // rbx
  __int64 v11; // rdi

  *a1 = off_49FDFC0;
  a1[29] = &unk_49FE090;
  v2 = a1[93];
  if ( (_QWORD *)v2 != a1 + 95 )
    _libc_free(v2);
  v3 = a1[83];
  if ( (_QWORD *)v3 != a1 + 85 )
    _libc_free(v3);
  v4 = a1[72];
  if ( v4 != a1[71] )
    _libc_free(v4);
  v5 = a1[60];
  if ( (_QWORD *)v5 != a1 + 62 )
    _libc_free(v5);
  v6 = a1[50];
  if ( (_QWORD *)v6 != a1 + 52 )
    _libc_free(v6);
  v7 = a1[48];
  if ( v7 )
    j_j___libc_free_0_0(v7);
  _libc_free(a1[45]);
  v8 = a1[42];
  if ( (_QWORD *)v8 != a1 + 44 )
    _libc_free(v8);
  v9 = a1[37];
  if ( v9 )
  {
    for ( i = v9 + 24LL * *(_QWORD *)(v9 - 8); v9 != i; i -= 24 )
    {
      v11 = *(_QWORD *)(i - 8);
      if ( v11 )
        j_j___libc_free_0_0(v11);
    }
    j_j_j___libc_free_0_0(v9 - 8);
  }
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
