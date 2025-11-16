// Function: sub_37E9CF0
// Address: 0x37e9cf0
//
__int64 __fastcall sub_37E9CF0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // r13
  __int64 i; // rbx
  unsigned __int64 v12; // rdi

  *a1 = &unk_4A3D600;
  v2 = a1[77];
  if ( v2 )
    _libc_free(v2);
  v3 = a1[72];
  if ( (_QWORD *)v3 != a1 + 75 )
    _libc_free(v3);
  v4 = a1[68];
  if ( v4 )
    j_j___libc_free_0(v4);
  v5 = a1[65];
  if ( v5 )
    j_j___libc_free_0_0(v5);
  v6 = a1[56];
  if ( (_QWORD *)v6 != a1 + 58 )
    _libc_free(v6);
  v7 = a1[47];
  if ( (_QWORD *)v7 != a1 + 49 )
    _libc_free(v7);
  v8 = a1[39];
  if ( (_QWORD *)v8 != a1 + 42 )
    _libc_free(v8);
  v9 = a1[32];
  if ( (_QWORD *)v9 != a1 + 35 )
    _libc_free(v9);
  v10 = a1[28];
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
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
