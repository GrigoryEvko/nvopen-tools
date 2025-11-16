// Function: sub_3528850
// Address: 0x3528850
//
__int64 __fastcall sub_3528850(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // r13
  __int64 i; // rbx
  unsigned __int64 v10; // rdi

  *a1 = off_4A38CF0;
  v2 = a1[110];
  if ( (_QWORD *)v2 != a1 + 112 )
    _libc_free(v2);
  v3 = a1[81];
  if ( v3 )
    j_j___libc_free_0_0(v3);
  v4 = a1[72];
  if ( (_QWORD *)v4 != a1 + 74 )
    _libc_free(v4);
  v5 = a1[63];
  if ( (_QWORD *)v5 != a1 + 65 )
    _libc_free(v5);
  v6 = a1[55];
  if ( (_QWORD *)v6 != a1 + 58 )
    _libc_free(v6);
  v7 = a1[48];
  if ( (_QWORD *)v7 != a1 + 51 )
    _libc_free(v7);
  v8 = a1[44];
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
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
