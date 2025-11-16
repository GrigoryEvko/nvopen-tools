// Function: sub_2EC48E0
// Address: 0x2ec48e0
//
__int64 __fastcall sub_2EC48E0(_QWORD *a1)
{
  _QWORD *v2; // r14
  unsigned __int64 v3; // rdi
  unsigned __int64 *v4; // rbx
  unsigned __int64 *v5; // r13
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi

  v2 = (_QWORD *)a1[444];
  *a1 = &unk_4A29E58;
  if ( v2 )
  {
    v3 = v2[25];
    if ( v3 )
      j_j___libc_free_0(v3);
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
      j_j___libc_free_0((unsigned __int64)v5);
    v6 = v2[4];
    if ( (_QWORD *)v6 != v2 + 6 )
      _libc_free(v6);
    v7 = v2[1];
    if ( v7 )
      j_j___libc_free_0(v7);
    j_j___libc_free_0((unsigned __int64)v2);
  }
  v8 = a1[830];
  if ( v8 )
    j_j___libc_free_0(v8);
  v9 = a1[828];
  if ( v9 )
    _libc_free(v9);
  v10 = a1[822];
  if ( (_QWORD *)v10 != a1 + 824 )
    _libc_free(v10);
  v11 = a1[819];
  if ( v11 )
    _libc_free(v11);
  v12 = a1[793];
  if ( (_QWORD *)v12 != a1 + 795 )
    _libc_free(v12);
  v13 = a1[790];
  if ( v13 )
    j_j___libc_free_0(v13);
  v14 = a1[753];
  if ( (_QWORD *)v14 != a1 + 755 )
    _libc_free(v14);
  v15 = a1[727];
  if ( (_QWORD *)v15 != a1 + 729 )
    _libc_free(v15);
  v16 = a1[724];
  if ( v16 )
    j_j___libc_free_0(v16);
  v17 = a1[721];
  if ( v17 )
    j_j___libc_free_0(v17);
  v18 = a1[719];
  if ( v18 )
    _libc_free(v18);
  v19 = a1[713];
  if ( (_QWORD *)v19 != a1 + 715 )
    _libc_free(v19);
  v20 = a1[710];
  if ( v20 )
    _libc_free(v20);
  v21 = a1[684];
  if ( (_QWORD *)v21 != a1 + 686 )
    _libc_free(v21);
  v22 = a1[681];
  if ( v22 )
    j_j___libc_free_0(v22);
  v23 = a1[644];
  if ( (_QWORD *)v23 != a1 + 646 )
    _libc_free(v23);
  v24 = a1[618];
  if ( (_QWORD *)v24 != a1 + 620 )
    _libc_free(v24);
  v25 = a1[615];
  if ( v25 )
    j_j___libc_free_0(v25);
  v26 = a1[612];
  if ( v26 )
    j_j___libc_free_0(v26);
  v27 = a1[609];
  if ( v27 )
    j_j___libc_free_0(v27);
  v28 = a1[607];
  if ( v28 )
    _libc_free(v28);
  v29 = a1[601];
  if ( (_QWORD *)v29 != a1 + 603 )
    _libc_free(v29);
  v30 = a1[598];
  if ( v30 )
    _libc_free(v30);
  v31 = a1[572];
  if ( (_QWORD *)v31 != a1 + 574 )
    _libc_free(v31);
  v32 = a1[569];
  if ( v32 )
    j_j___libc_free_0(v32);
  v33 = a1[532];
  if ( (_QWORD *)v33 != a1 + 534 )
    _libc_free(v33);
  v34 = a1[506];
  if ( (_QWORD *)v34 != a1 + 508 )
    _libc_free(v34);
  v35 = a1[503];
  if ( v35 )
    j_j___libc_free_0(v35);
  _libc_free(a1[500]);
  _libc_free(a1[497]);
  v36 = a1[455];
  if ( (_QWORD *)v36 != a1 + 457 )
    _libc_free(v36);
  v37 = a1[445];
  if ( (_QWORD *)v37 != a1 + 447 )
    _libc_free(v37);
  return sub_2EC4810(a1);
}
