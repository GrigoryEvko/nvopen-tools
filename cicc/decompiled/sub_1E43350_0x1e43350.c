// Function: sub_1E43350
// Address: 0x1e43350
//
__int64 __fastcall sub_1E43350(_QWORD *a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r13
  unsigned __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi

  v2 = (_QWORD *)a1[303];
  v3 = (_QWORD *)a1[302];
  *a1 = off_49FC1F0;
  if ( v2 != v3 )
  {
    do
    {
      if ( *v3 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v3 + 16LL))(*v3);
      ++v3;
    }
    while ( v2 != v3 );
    v3 = (_QWORD *)a1[302];
  }
  if ( v3 )
    j_j___libc_free_0(v3, a1[304] - (_QWORD)v3);
  v4 = a1[295];
  if ( v4 != a1[294] )
    _libc_free(v4);
  j___libc_free_0(a1[290]);
  v5 = a1[286];
  if ( v5 )
    j_j___libc_free_0(v5, a1[288] - v5);
  j___libc_free_0(a1[283]);
  v6 = a1[279];
  if ( v6 )
    j_j___libc_free_0(v6, a1[281] - v6);
  _libc_free(a1[276]);
  v7 = a1[273];
  if ( v7 )
    j_j___libc_free_0(v7, a1[275] - v7);
  v8 = a1[270];
  if ( v8 )
    j_j___libc_free_0(v8, a1[272] - v8);
  v9 = a1[261];
  *a1 = &unk_49FE610;
  _libc_free(v9);
  v10 = a1[255];
  if ( (_QWORD *)v10 != a1 + 257 )
    _libc_free(v10);
  v11 = a1[250];
  if ( v11 )
    j_j___libc_free_0(v11, a1[252] - v11);
  _libc_free(a1[244]);
  v12 = a1[210];
  if ( (_QWORD *)v12 != a1 + 212 )
    _libc_free(v12);
  _libc_free(a1[207]);
  v13 = a1[181];
  if ( (_QWORD *)v13 != a1 + 183 )
    _libc_free(v13);
  _libc_free(a1[178]);
  v14 = a1[152];
  if ( (_QWORD *)v14 != a1 + 154 )
    _libc_free(v14);
  _libc_free(a1[149]);
  v15 = a1[123];
  if ( (_QWORD *)v15 != a1 + 125 )
    _libc_free(v15);
  j___libc_free_0(a1[120]);
  v16 = a1[103];
  if ( (_QWORD *)v16 != a1 + 105 )
    _libc_free(v16);
  return sub_1F012F0(a1);
}
