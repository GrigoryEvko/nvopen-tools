// Function: sub_1F012F0
// Address: 0x1f012f0
//
void __fastcall sub_1F012F0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  *a1 = &unk_49FE548;
  v2 = a1[57];
  if ( (_QWORD *)v2 != a1 + 59 )
    _libc_free(v2);
  v3 = a1[47];
  if ( (_QWORD *)v3 != a1 + 49 )
    _libc_free(v3);
  v4 = a1[23];
  if ( (_QWORD *)v4 != a1 + 25 )
    _libc_free(v4);
  v5 = a1[13];
  if ( (_QWORD *)v5 != a1 + 15 )
    _libc_free(v5);
  v6 = a1[7];
  v7 = a1[6];
  if ( v6 != v7 )
  {
    do
    {
      v8 = *(_QWORD *)(v7 + 112);
      if ( v8 != v7 + 128 )
        _libc_free(v8);
      v9 = *(_QWORD *)(v7 + 32);
      if ( v9 != v7 + 48 )
        _libc_free(v9);
      v7 += 272;
    }
    while ( v6 != v7 );
    v7 = a1[6];
  }
  if ( v7 )
    j_j___libc_free_0(v7, a1[8] - v7);
}
