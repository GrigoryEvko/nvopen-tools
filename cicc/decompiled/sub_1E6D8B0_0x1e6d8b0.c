// Function: sub_1E6D8B0
// Address: 0x1e6d8b0
//
__int64 __fastcall sub_1E6D8B0(_QWORD *a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi

  v2 = (_QWORD *)a1[278];
  v3 = (_QWORD *)a1[277];
  *a1 = &unk_49FC890;
  if ( v2 != v3 )
  {
    do
    {
      if ( *v3 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v3 + 16LL))(*v3);
      ++v3;
    }
    while ( v2 != v3 );
    v3 = (_QWORD *)a1[277];
  }
  if ( v3 )
    j_j___libc_free_0(v3, a1[279] - (_QWORD)v3);
  _libc_free(a1[274]);
  v4 = a1[271];
  if ( v4 )
    j_j___libc_free_0(v4, a1[273] - v4);
  v5 = a1[268];
  if ( v5 )
    j_j___libc_free_0(v5, a1[270] - v5);
  v6 = a1[265];
  if ( v6 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 16LL))(v6);
  v7 = a1[261];
  *a1 = &unk_49FE610;
  _libc_free(v7);
  v8 = a1[255];
  if ( (_QWORD *)v8 != a1 + 257 )
    _libc_free(v8);
  v9 = a1[250];
  if ( v9 )
    j_j___libc_free_0(v9, a1[252] - v9);
  _libc_free(a1[244]);
  v10 = a1[210];
  if ( (_QWORD *)v10 != a1 + 212 )
    _libc_free(v10);
  _libc_free(a1[207]);
  v11 = a1[181];
  if ( (_QWORD *)v11 != a1 + 183 )
    _libc_free(v11);
  _libc_free(a1[178]);
  v12 = a1[152];
  if ( (_QWORD *)v12 != a1 + 154 )
    _libc_free(v12);
  _libc_free(a1[149]);
  v13 = a1[123];
  if ( (_QWORD *)v13 != a1 + 125 )
    _libc_free(v13);
  j___libc_free_0(a1[120]);
  v14 = a1[103];
  if ( (_QWORD *)v14 != a1 + 105 )
    _libc_free(v14);
  return sub_1F012F0(a1);
}
