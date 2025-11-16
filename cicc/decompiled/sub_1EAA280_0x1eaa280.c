// Function: sub_1EAA280
// Address: 0x1eaa280
//
__int64 __fastcall sub_1EAA280(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi

  *a1 = off_49FD3B8;
  v2 = a1[276];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = a1[277];
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  v4 = (_QWORD *)a1[283];
  v5 = (_QWORD *)a1[282];
  if ( v4 != v5 )
  {
    do
    {
      if ( *v5 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v5 + 16LL))(*v5);
      ++v5;
    }
    while ( v4 != v5 );
    v5 = (_QWORD *)a1[282];
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[284] - (_QWORD)v5);
  v6 = a1[279];
  if ( v6 )
    j_j___libc_free_0(v6, a1[281] - v6);
  v7 = a1[273];
  if ( v7 )
    j_j___libc_free_0(v7, a1[275] - v7);
  v8 = a1[269];
  a1[263] = &unk_4A00AB0;
  if ( v8 )
    j_j___libc_free_0(v8, a1[271] - v8);
  v9 = a1[266];
  if ( v9 )
    j_j___libc_free_0(v9, a1[268] - v9);
  v10 = a1[261];
  *a1 = &unk_49FE610;
  _libc_free(v10);
  v11 = a1[255];
  if ( (_QWORD *)v11 != a1 + 257 )
    _libc_free(v11);
  v12 = a1[250];
  if ( v12 )
    j_j___libc_free_0(v12, a1[252] - v12);
  _libc_free(a1[244]);
  v13 = a1[210];
  if ( (_QWORD *)v13 != a1 + 212 )
    _libc_free(v13);
  _libc_free(a1[207]);
  v14 = a1[181];
  if ( (_QWORD *)v14 != a1 + 183 )
    _libc_free(v14);
  _libc_free(a1[178]);
  v15 = a1[152];
  if ( (_QWORD *)v15 != a1 + 154 )
    _libc_free(v15);
  _libc_free(a1[149]);
  v16 = a1[123];
  if ( (_QWORD *)v16 != a1 + 125 )
    _libc_free(v16);
  j___libc_free_0(a1[120]);
  v17 = a1[103];
  if ( (_QWORD *)v17 != a1 + 105 )
    _libc_free(v17);
  return sub_1F012F0(a1);
}
