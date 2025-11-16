// Function: sub_1E6DAC0
// Address: 0x1e6dac0
//
__int64 __fastcall sub_1E6DAC0(_QWORD *a1)
{
  _QWORD *v2; // r14
  __int64 v3; // rdi
  unsigned __int64 *v4; // rbx
  unsigned __int64 *v5; // r13
  unsigned __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  __int64 v29; // rdi
  unsigned __int64 v30; // rdi

  v2 = (_QWORD *)a1[285];
  *a1 = &unk_49FC920;
  if ( v2 )
  {
    v3 = v2[25];
    if ( v3 )
      j_j___libc_free_0(v3, v2[27] - v3);
    v4 = (unsigned __int64 *)v2[23];
    v5 = (unsigned __int64 *)v2[22];
    if ( v4 != v5 )
    {
      do
      {
        if ( (unsigned __int64 *)*v5 != v5 + 2 )
          _libc_free(*v5);
        v5 += 6;
      }
      while ( v4 != v5 );
      v5 = (unsigned __int64 *)v2[22];
    }
    if ( v5 )
      j_j___libc_free_0(v5, v2[24] - (_QWORD)v5);
    v6 = v2[4];
    if ( (_QWORD *)v6 != v2 + 6 )
      _libc_free(v6);
    v7 = v2[1];
    if ( v7 )
      j_j___libc_free_0(v7, v2[3] - v7);
    j_j___libc_free_0(v2, 224);
  }
  v8 = a1[505];
  if ( v8 )
    j_j___libc_free_0(v8, a1[507] - v8);
  _libc_free(a1[503]);
  v9 = a1[497];
  if ( (_QWORD *)v9 != a1 + 499 )
    _libc_free(v9);
  _libc_free(a1[494]);
  v10 = a1[484];
  if ( (_QWORD *)v10 != a1 + 486 )
    _libc_free(v10);
  v11 = a1[481];
  if ( v11 )
    j_j___libc_free_0(v11, a1[483] - v11);
  v12 = a1[460];
  if ( (_QWORD *)v12 != a1 + 462 )
    _libc_free(v12);
  v13 = a1[450];
  if ( (_QWORD *)v13 != a1 + 452 )
    _libc_free(v13);
  v14 = a1[447];
  if ( v14 )
    j_j___libc_free_0(v14, a1[449] - v14);
  v15 = a1[444];
  if ( v15 )
    j_j___libc_free_0(v15, a1[446] - v15);
  _libc_free(a1[442]);
  v16 = a1[436];
  if ( (_QWORD *)v16 != a1 + 438 )
    _libc_free(v16);
  _libc_free(a1[433]);
  v17 = a1[423];
  if ( (_QWORD *)v17 != a1 + 425 )
    _libc_free(v17);
  v18 = a1[420];
  if ( v18 )
    j_j___libc_free_0(v18, a1[422] - v18);
  v19 = a1[399];
  if ( (_QWORD *)v19 != a1 + 401 )
    _libc_free(v19);
  v20 = a1[389];
  if ( (_QWORD *)v20 != a1 + 391 )
    _libc_free(v20);
  v21 = a1[386];
  if ( v21 )
    j_j___libc_free_0(v21, a1[388] - v21);
  v22 = a1[383];
  if ( v22 )
    j_j___libc_free_0(v22, a1[385] - v22);
  v23 = a1[380];
  if ( v23 )
    j_j___libc_free_0(v23, a1[382] - v23);
  _libc_free(a1[378]);
  v24 = a1[372];
  if ( (_QWORD *)v24 != a1 + 374 )
    _libc_free(v24);
  _libc_free(a1[369]);
  v25 = a1[359];
  if ( (_QWORD *)v25 != a1 + 361 )
    _libc_free(v25);
  v26 = a1[356];
  if ( v26 )
    j_j___libc_free_0(v26, a1[358] - v26);
  v27 = a1[335];
  if ( (_QWORD *)v27 != a1 + 337 )
    _libc_free(v27);
  v28 = a1[325];
  if ( (_QWORD *)v28 != a1 + 327 )
    _libc_free(v28);
  v29 = a1[322];
  if ( v29 )
    j_j___libc_free_0(v29, a1[324] - v29);
  _libc_free(a1[319]);
  _libc_free(a1[316]);
  v30 = a1[290];
  if ( (_QWORD *)v30 != a1 + 292 )
    _libc_free(v30);
  _libc_free(a1[286]);
  return sub_1E6D8B0(a1);
}
