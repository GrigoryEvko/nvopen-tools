// Function: sub_1E11810
// Address: 0x1e11810
//
void __fastcall sub_1E11810(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // r12
  _QWORD *v10; // r13
  unsigned __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // r13
  _QWORD *v17; // r12
  __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  __int64 v22; // r13
  __int64 v23; // r12
  __int64 v24; // rdi
  __int64 v25; // rdi
  _QWORD *v26; // r13
  _QWORD *v27; // r12
  unsigned __int64 *v28; // rcx
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rdi
  unsigned __int64 *v31; // r12
  unsigned __int64 *v32; // r13
  unsigned __int64 v33; // rdi
  unsigned __int64 *v34; // r12
  unsigned __int64 v35; // r13
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  __int64 v38; // rdi

  sub_1E0FED0(a1);
  v3 = *(_QWORD *)(a1 + 608);
  if ( v3 != a1 + 624 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 576);
  if ( v4 )
  {
    a2 = *(_QWORD *)(a1 + 592) - v4;
    j_j___libc_free_0(v4, a2);
  }
  v5 = *(_QWORD *)(a1 + 552);
  if ( v5 )
  {
    a2 = *(_QWORD *)(a1 + 568) - v5;
    j_j___libc_free_0(v5, a2);
  }
  v6 = *(_QWORD *)(a1 + 528);
  if ( v6 )
  {
    a2 = *(_QWORD *)(a1 + 544) - v6;
    j_j___libc_free_0(v6, a2);
  }
  v7 = *(_QWORD *)(a1 + 496);
  if ( v7 )
  {
    a2 = *(_QWORD *)(a1 + 512) - v7;
    j_j___libc_free_0(v7, a2);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 472));
  v8 = *(unsigned int *)(a1 + 456);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD **)(a1 + 440);
    v10 = &v9[5 * v8];
    do
    {
      if ( *v9 != -8 && *v9 != -16 )
      {
        v11 = v9[1];
        if ( (_QWORD *)v11 != v9 + 3 )
          _libc_free(v11);
      }
      v9 += 5;
    }
    while ( v10 != v9 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 440));
  v16 = *(_QWORD **)(a1 + 416);
  v17 = *(_QWORD **)(a1 + 408);
  if ( v16 != v17 )
  {
    do
    {
      v18 = v17[12];
      if ( v18 )
      {
        a2 = v17[14] - v18;
        j_j___libc_free_0(v18, a2);
      }
      v19 = v17[7];
      if ( (_QWORD *)v19 != v17 + 9 )
        _libc_free(v19);
      v20 = v17[4];
      if ( (_QWORD *)v20 != v17 + 6 )
        _libc_free(v20);
      v21 = v17[1];
      if ( (_QWORD *)v21 != v17 + 3 )
        _libc_free(v21);
      v17 += 15;
    }
    while ( v16 != v17 );
    v17 = *(_QWORD **)(a1 + 408);
  }
  if ( v17 )
  {
    a2 = *(_QWORD *)(a1 + 424) - (_QWORD)v17;
    j_j___libc_free_0(v17, a2);
  }
  v22 = *(_QWORD *)(a1 + 392);
  v23 = *(_QWORD *)(a1 + 384);
  if ( v22 != v23 )
  {
    do
    {
      v24 = *(_QWORD *)(v23 + 24);
      if ( v24 )
      {
        a2 = *(_QWORD *)(v23 + 40) - v24;
        j_j___libc_free_0(v24, a2);
      }
      v23 += 48;
    }
    while ( v22 != v23 );
    v23 = *(_QWORD *)(a1 + 384);
  }
  if ( v23 )
  {
    a2 = *(_QWORD *)(a1 + 400) - v23;
    j_j___libc_free_0(v23, a2);
  }
  v25 = *(_QWORD *)(a1 + 376);
  if ( v25 )
    sub_1E10D30(v25, a2, v12, v13, v14, v15);
  _libc_free(*(_QWORD *)(a1 + 352));
  v26 = *(_QWORD **)(a1 + 328);
  while ( (_QWORD *)(a1 + 320) != v26 )
  {
    v27 = v26;
    v26 = (_QWORD *)v26[1];
    sub_1DD5B80(a1 + 320, (__int64)v27);
    v28 = (unsigned __int64 *)v27[1];
    v29 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
    *v28 = v29 | *v28 & 7;
    *(_QWORD *)(v29 + 8) = v28;
    *v27 &= 7uLL;
    v27[1] = 0;
    sub_1E0A230(a1 + 320, v27);
  }
  v30 = *(_QWORD *)(a1 + 232);
  if ( v30 != a1 + 248 )
    _libc_free(v30);
  v31 = *(unsigned __int64 **)(a1 + 136);
  v32 = &v31[*(unsigned int *)(a1 + 144)];
  while ( v32 != v31 )
  {
    v33 = *v31++;
    _libc_free(v33);
  }
  v34 = *(unsigned __int64 **)(a1 + 184);
  v35 = (unsigned __int64)&v34[2 * *(unsigned int *)(a1 + 192)];
  if ( v34 != (unsigned __int64 *)v35 )
  {
    do
    {
      v36 = *v34;
      v34 += 2;
      _libc_free(v36);
    }
    while ( (unsigned __int64 *)v35 != v34 );
    v35 = *(_QWORD *)(a1 + 184);
  }
  if ( v35 != a1 + 200 )
    _libc_free(v35);
  v37 = *(_QWORD *)(a1 + 136);
  if ( v37 != a1 + 152 )
    _libc_free(v37);
  v38 = *(_QWORD *)(a1 + 96);
  if ( v38 )
    j_j___libc_free_0(v38, *(_QWORD *)(a1 + 112) - v38);
}
