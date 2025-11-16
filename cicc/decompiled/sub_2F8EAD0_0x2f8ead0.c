// Function: sub_2F8EAD0
// Address: 0x2f8ead0
//
void __fastcall sub_2F8EAD0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  *a1 = &unk_4A2BC80;
  v2 = a1[56];
  if ( (_QWORD *)v2 != a1 + 58 )
    _libc_free(v2);
  v3 = a1[46];
  if ( (_QWORD *)v3 != a1 + 48 )
    _libc_free(v3);
  v4 = a1[24];
  if ( (_QWORD *)v4 != a1 + 26 )
    _libc_free(v4);
  v5 = a1[14];
  if ( (_QWORD *)v5 != a1 + 16 )
    _libc_free(v5);
  v6 = a1[7];
  v7 = a1[6];
  if ( v6 != v7 )
  {
    do
    {
      v8 = *(_QWORD *)(v7 + 120);
      if ( v8 != v7 + 136 )
        _libc_free(v8);
      v9 = *(_QWORD *)(v7 + 40);
      if ( v9 != v7 + 56 )
        _libc_free(v9);
      v7 += 256LL;
    }
    while ( v6 != v7 );
    v7 = a1[6];
  }
  if ( v7 )
    j_j___libc_free_0(v7);
}
