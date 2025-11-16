// Function: sub_2FAEF10
// Address: 0x2faef10
//
void __fastcall sub_2FAEF10(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // r12
  __int64 v8; // rbx
  unsigned __int64 v9; // rdi

  v2 = a1[34];
  if ( v2 )
    _libc_free(v2);
  v3 = a1[28];
  if ( (_QWORD *)v3 != a1 + 30 )
    _libc_free(v3);
  v4 = a1[17];
  if ( (_QWORD *)v4 != a1 + 19 )
    _libc_free(v4);
  v5 = a1[11];
  if ( (_QWORD *)v5 != a1 + 13 )
    _libc_free(v5);
  v6 = a1[5];
  if ( (_QWORD *)v6 != a1 + 7 )
    _libc_free(v6);
  v7 = a1[3];
  if ( v7 )
  {
    v8 = v7 + 112LL * *(_QWORD *)(v7 - 8);
    while ( v7 != v8 )
    {
      v8 -= 112;
      v9 = *(_QWORD *)(v8 + 24);
      if ( v9 != v8 + 40 )
        _libc_free(v9);
    }
    j_j_j___libc_free_0_0(v7 - 8);
  }
}
