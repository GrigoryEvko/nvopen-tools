// Function: sub_2E81F20
// Address: 0x2e81f20
//
void __fastcall sub_2E81F20(__int64 a1, __int64 a2, void (*a3)(), __int64 a4)
{
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rsi
  _QWORD *v8; // r12
  _QWORD *v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 v15; // rax
  _QWORD *v16; // r12
  _QWORD *v17; // r13
  unsigned __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  _QWORD *v24; // r13
  _QWORD *v25; // r12
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  _QWORD *v32; // r13
  _QWORD *v33; // r12
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  _QWORD *v37; // r13
  _QWORD *v38; // r12
  unsigned __int64 *v39; // rcx
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rdi
  __int64 *v42; // r14
  __int64 *v43; // r12
  __int64 i; // rax
  __int64 v45; // rdi
  unsigned int v46; // ecx
  __int64 v47; // rsi
  __int64 *v48; // r12
  unsigned __int64 v49; // r13
  __int64 v50; // rsi
  __int64 v51; // rdi
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // rdi

  sub_2E80380(a1, a2, a3, a4);
  sub_C7D6A0(*(_QWORD *)(a1 + 1088), 24LL * *(unsigned int *)(a1 + 1104), 8);
  v5 = *(_QWORD *)(a1 + 904);
  if ( v5 != a1 + 920 )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 752);
  if ( v6 != a1 + 768 )
    _libc_free(v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 728), 24LL * *(unsigned int *)(a1 + 744), 8);
  v7 = *(unsigned int *)(a1 + 712);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD **)(a1 + 696);
    v9 = &v8[4 * v7];
    do
    {
      if ( *v8 != -8192 && *v8 != -4096 )
      {
        v10 = v8[1];
        if ( (_QWORD *)v10 != v8 + 3 )
          _libc_free(v10);
      }
      v8 += 4;
    }
    while ( v9 != v8 );
    v7 = *(unsigned int *)(a1 + 712);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 696), 32 * v7, 8);
  v11 = *(_QWORD *)(a1 + 640);
  if ( v11 )
    j_j___libc_free_0(v11);
  v12 = *(_QWORD *)(a1 + 616);
  if ( v12 )
    j_j___libc_free_0(v12);
  v13 = *(_QWORD *)(a1 + 592);
  if ( v13 )
    j_j___libc_free_0(v13);
  v14 = *(_QWORD *)(a1 + 552);
  if ( v14 )
    j_j___libc_free_0(v14);
  sub_C7D6A0(*(_QWORD *)(a1 + 528), 16LL * *(unsigned int *)(a1 + 544), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 496), 16LL * *(unsigned int *)(a1 + 512), 8);
  v15 = *(unsigned int *)(a1 + 480);
  if ( (_DWORD)v15 )
  {
    v16 = *(_QWORD **)(a1 + 464);
    v17 = &v16[5 * v15];
    do
    {
      if ( *v16 != -4096 && *v16 != -8192 )
      {
        v18 = v16[1];
        if ( (_QWORD *)v18 != v16 + 3 )
          _libc_free(v18);
      }
      v16 += 5;
    }
    while ( v17 != v16 );
    v15 = *(unsigned int *)(a1 + 480);
  }
  v19 = 40 * v15;
  sub_C7D6A0(*(_QWORD *)(a1 + 464), 40 * v15, 8);
  v24 = *(_QWORD **)(a1 + 440);
  v25 = *(_QWORD **)(a1 + 432);
  if ( v24 != v25 )
  {
    do
    {
      v26 = v25[12];
      if ( v26 )
      {
        v19 = v25[14] - v26;
        j_j___libc_free_0(v26);
      }
      v27 = v25[7];
      if ( (_QWORD *)v27 != v25 + 9 )
        _libc_free(v27);
      v28 = v25[4];
      if ( (_QWORD *)v28 != v25 + 6 )
        _libc_free(v28);
      v29 = v25[1];
      if ( (_QWORD *)v29 != v25 + 3 )
        _libc_free(v29);
      v25 += 15;
    }
    while ( v24 != v25 );
    v25 = *(_QWORD **)(a1 + 432);
  }
  if ( v25 )
  {
    v19 = *(_QWORD *)(a1 + 448) - (_QWORD)v25;
    j_j___libc_free_0((unsigned __int64)v25);
  }
  v30 = *(_QWORD *)(a1 + 408);
  if ( v30 )
  {
    v19 = *(_QWORD *)(a1 + 424) - v30;
    j_j___libc_free_0(v30);
  }
  v31 = *(_QWORD *)(a1 + 384);
  if ( v31 )
  {
    v19 = *(_QWORD *)(a1 + 400) - v31;
    j_j___libc_free_0(v31);
  }
  v32 = *(_QWORD **)(a1 + 368);
  v33 = *(_QWORD **)(a1 + 360);
  if ( v32 != v33 )
  {
    do
    {
      v34 = v33[9];
      if ( (_QWORD *)v34 != v33 + 11 )
      {
        v19 = v33[11] + 1LL;
        j_j___libc_free_0(v34);
      }
      v35 = v33[6];
      if ( v35 )
      {
        v19 = v33[8] - v35;
        j_j___libc_free_0(v35);
      }
      v33 += 13;
    }
    while ( v32 != v33 );
    v33 = *(_QWORD **)(a1 + 360);
  }
  if ( v33 )
  {
    v19 = *(_QWORD *)(a1 + 376) - (_QWORD)v33;
    j_j___libc_free_0((unsigned __int64)v33);
  }
  v36 = *(_QWORD *)(a1 + 352);
  if ( v36 )
    sub_2E81360(v36, v19, v20, v21, v22, v23);
  v37 = *(_QWORD **)(a1 + 328);
  while ( (_QWORD *)(a1 + 320) != v37 )
  {
    v38 = v37;
    v37 = (_QWORD *)v37[1];
    sub_2E31020(a1 + 320, (__int64)v38);
    v39 = (unsigned __int64 *)v38[1];
    v40 = *v38 & 0xFFFFFFFFFFFFFFF8LL;
    *v39 = v40 | *v39 & 7;
    *(_QWORD *)(v40 + 8) = v39;
    *v38 &= 7uLL;
    v38[1] = 0;
    sub_2E79D60(a1 + 320, v38);
  }
  v41 = *(_QWORD *)(a1 + 232);
  if ( v41 != a1 + 248 )
    _libc_free(v41);
  v42 = *(__int64 **)(a1 + 144);
  v43 = &v42[*(unsigned int *)(a1 + 152)];
  if ( v42 != v43 )
  {
    for ( i = *(_QWORD *)(a1 + 144); ; i = *(_QWORD *)(a1 + 144) )
    {
      v45 = *v42;
      v46 = (unsigned int)(((__int64)v42 - i) >> 3) >> 7;
      v47 = 4096LL << v46;
      if ( v46 >= 0x1E )
        v47 = 0x40000000000LL;
      ++v42;
      sub_C7D6A0(v45, v47, 16);
      if ( v43 == v42 )
        break;
    }
  }
  v48 = *(__int64 **)(a1 + 192);
  v49 = (unsigned __int64)&v48[2 * *(unsigned int *)(a1 + 200)];
  if ( v48 != (__int64 *)v49 )
  {
    do
    {
      v50 = v48[1];
      v51 = *v48;
      v48 += 2;
      sub_C7D6A0(v51, v50, 16);
    }
    while ( (__int64 *)v49 != v48 );
    v49 = *(_QWORD *)(a1 + 192);
  }
  if ( v49 != a1 + 208 )
    _libc_free(v49);
  v52 = *(_QWORD *)(a1 + 144);
  if ( v52 != a1 + 160 )
    _libc_free(v52);
  v53 = *(_QWORD *)(a1 + 96);
  if ( v53 )
    j_j___libc_free_0(v53);
}
