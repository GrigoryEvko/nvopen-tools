// Function: sub_CB34B0
// Address: 0xcb34b0
//
void __fastcall sub_CB34B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  _QWORD *v4; // r13
  __int64 v5; // rsi
  _QWORD *v6; // r14
  __int64 v7; // rax
  unsigned int v8; // ecx
  __int64 v9; // rdx
  unsigned __int64 v10; // r15
  unsigned __int64 i; // rbx
  __int64 v12; // rdi
  _QWORD *v13; // r14
  _QWORD *v14; // r15
  unsigned __int64 v15; // r13
  unsigned __int64 j; // rbx
  __int64 v17; // rdi
  __int64 *v18; // rbx
  __int64 *v19; // r13
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 *v22; // r13
  __int64 v23; // rdi
  __int64 *v24; // r14
  __int64 *v25; // rbx
  __int64 k; // rax
  __int64 v27; // rdi
  unsigned int v28; // ecx
  __int64 *v29; // rbx
  __int64 *v30; // r13
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 *v33; // rbx
  __int64 v34; // rdx
  __int64 *v35; // r13
  __int64 v36; // rdi
  __int64 *v37; // r13
  __int64 v38; // rdi
  __int64 *v39; // rbx
  __int64 v40; // rdx
  __int64 *v41; // r13
  __int64 v42; // rdi
  __int64 *v43; // r13
  __int64 v44; // rdi
  __int64 *v45; // r14
  __int64 *v46; // rbx
  __int64 m; // rax
  __int64 v48; // rdi
  unsigned int v49; // ecx
  __int64 *v50; // rbx
  __int64 *v51; // r13
  __int64 v52; // rdi
  __int64 v53; // rdi
  __int64 v54; // r13
  _QWORD *v55; // rbx
  _QWORD *v56; // r13
  __int64 *v57; // rbx
  __int64 *v58; // r13
  __int64 *v59; // rdi
  __int64 *v60; // rax
  __int64 v61; // rcx
  __int64 *v62; // rbx
  __int64 *v63; // r14
  __int64 v64; // rdi
  unsigned int v65; // ecx
  __int64 v66; // rsi
  __int64 *v67; // rbx
  __int64 v68; // rdi
  __int64 *v69; // rax
  __int64 v70; // rcx
  __int64 *v71; // rbx
  __int64 *v72; // r14
  __int64 v73; // rdi
  unsigned int v74; // ecx
  __int64 v75; // rsi
  __int64 *v76; // rbx
  __int64 v77; // rax
  __int64 v78; // rdi
  __int64 *v79; // rax
  __int64 v80; // rcx
  __int64 *v81; // rbx
  __int64 *v82; // r14
  __int64 v83; // rdi
  unsigned int v84; // ecx
  __int64 v85; // rsi
  __int64 *v86; // rbx
  __int64 v87; // rax
  __int64 v88; // rdi

  *(_QWORD *)a1 = &unk_49DCE78;
  v3 = *(_QWORD *)(a1 + 600);
  if ( v3 != a1 + 616 )
    _libc_free(v3, a2);
  v4 = *(_QWORD **)(a1 + 512);
  v5 = *(unsigned int *)(a1 + 520);
  v6 = &v4[v5];
  if ( v4 != v6 )
  {
    v7 = *(_QWORD *)(a1 + 512);
    while ( 1 )
    {
      v8 = (unsigned int)(((__int64)v4 - v7) >> 3) >> 7;
      v9 = 4096LL << v8;
      if ( v8 >= 0x1E )
        v9 = 0x40000000000LL;
      v10 = *v4 + v9;
      if ( *v4 == *(_QWORD *)(v7 + 8 * v5 - 8) )
        v10 = *(_QWORD *)(a1 + 496);
      for ( i = ((*v4 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 32; i <= v10; i += 32LL )
      {
        v12 = *(_QWORD *)(i - 24);
        if ( v12 )
        {
          v5 = *(_QWORD *)(i - 8) - v12;
          j_j___libc_free_0(v12, v5);
        }
      }
      if ( v6 == ++v4 )
        break;
      v7 = *(_QWORD *)(a1 + 512);
      v5 = *(unsigned int *)(a1 + 520);
    }
  }
  v13 = *(_QWORD **)(a1 + 560);
  v14 = &v13[2 * *(unsigned int *)(a1 + 568)];
  if ( v13 != v14 )
  {
    do
    {
      v15 = *v13 + v13[1];
      for ( j = ((*v13 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 32; v15 >= j; j += 32LL )
      {
        v17 = *(_QWORD *)(j - 24);
        if ( v17 )
        {
          v5 = *(_QWORD *)(j - 8) - v17;
          j_j___libc_free_0(v17, v5);
        }
      }
      v13 += 2;
    }
    while ( v14 != v13 );
    v18 = *(__int64 **)(a1 + 560);
    v19 = &v18[2 * *(unsigned int *)(a1 + 568)];
    while ( v19 != v18 )
    {
      v5 = v18[1];
      v20 = *v18;
      v18 += 2;
      sub_C7D6A0(v20, v5, 16);
    }
  }
  v21 = *(unsigned int *)(a1 + 520);
  *(_DWORD *)(a1 + 568) = 0;
  if ( !(_DWORD)v21 )
    goto LABEL_24;
  v60 = *(__int64 **)(a1 + 512);
  *(_QWORD *)(a1 + 576) = 0;
  v61 = *v60;
  v62 = &v60[v21];
  v63 = v60 + 1;
  *(_QWORD *)(a1 + 496) = *v60;
  for ( *(_QWORD *)(a1 + 504) = v61 + 4096; v62 != v63; v60 = *(__int64 **)(a1 + 512) )
  {
    v64 = *v63;
    v65 = (unsigned int)(v63 - v60) >> 7;
    v66 = 4096LL << v65;
    if ( v65 >= 0x1E )
      v66 = 0x40000000000LL;
    ++v63;
    sub_C7D6A0(v64, v66, 16);
  }
  v5 = 4096;
  *(_DWORD *)(a1 + 520) = 1;
  sub_C7D6A0(*v60, 4096, 16);
  v67 = *(__int64 **)(a1 + 560);
  v22 = &v67[2 * *(unsigned int *)(a1 + 568)];
  if ( v67 != v22 )
  {
    do
    {
      v5 = v67[1];
      v68 = *v67;
      v67 += 2;
      sub_C7D6A0(v68, v5, 16);
    }
    while ( v22 != v67 );
LABEL_24:
    v22 = *(__int64 **)(a1 + 560);
  }
  if ( v22 != (__int64 *)(a1 + 576) )
    _libc_free(v22, v5);
  v23 = *(_QWORD *)(a1 + 512);
  if ( v23 != a1 + 528 )
    _libc_free(v23, v5);
  sub_CB3120(a1 + 400);
  v24 = *(__int64 **)(a1 + 416);
  v25 = &v24[*(unsigned int *)(a1 + 424)];
  if ( v24 != v25 )
  {
    for ( k = *(_QWORD *)(a1 + 416); ; k = *(_QWORD *)(a1 + 416) )
    {
      v27 = *v24;
      v28 = (unsigned int)(((__int64)v24 - k) >> 3) >> 7;
      v5 = 4096LL << v28;
      if ( v28 >= 0x1E )
        v5 = 0x40000000000LL;
      ++v24;
      sub_C7D6A0(v27, v5, 16);
      if ( v25 == v24 )
        break;
    }
  }
  v29 = *(__int64 **)(a1 + 464);
  v30 = &v29[2 * *(unsigned int *)(a1 + 472)];
  if ( v29 != v30 )
  {
    do
    {
      v5 = v29[1];
      v31 = *v29;
      v29 += 2;
      sub_C7D6A0(v31, v5, 16);
    }
    while ( v30 != v29 );
    v30 = *(__int64 **)(a1 + 464);
  }
  if ( v30 != (__int64 *)(a1 + 480) )
    _libc_free(v30, v5);
  v32 = *(_QWORD *)(a1 + 416);
  if ( v32 != a1 + 432 )
    _libc_free(v32, v5);
  v33 = *(__int64 **)(a1 + 368);
  v34 = *(unsigned int *)(a1 + 328);
  v35 = &v33[2 * *(unsigned int *)(a1 + 376)];
  if ( v33 != v35 )
  {
    do
    {
      v5 = v33[1];
      v36 = *v33;
      v33 += 2;
      sub_C7D6A0(v36, v5, 16);
    }
    while ( v35 != v33 );
    v34 = *(unsigned int *)(a1 + 328);
  }
  *(_DWORD *)(a1 + 376) = 0;
  if ( !(_DWORD)v34 )
    goto LABEL_46;
  v79 = *(__int64 **)(a1 + 320);
  *(_QWORD *)(a1 + 384) = 0;
  v80 = *v79;
  v81 = &v79[v34];
  v82 = v79 + 1;
  *(_QWORD *)(a1 + 304) = *v79;
  for ( *(_QWORD *)(a1 + 312) = v80 + 4096; v81 != v82; v79 = *(__int64 **)(a1 + 320) )
  {
    v83 = *v82;
    v84 = (unsigned int)(v82 - v79) >> 7;
    v85 = 4096LL << v84;
    if ( v84 >= 0x1E )
      v85 = 0x40000000000LL;
    ++v82;
    sub_C7D6A0(v83, v85, 16);
  }
  v5 = 4096;
  *(_DWORD *)(a1 + 328) = 1;
  sub_C7D6A0(*v79, 4096, 16);
  v86 = *(__int64 **)(a1 + 368);
  v87 = 2LL * *(unsigned int *)(a1 + 376);
  v37 = &v86[v87];
  if ( v86 != &v86[v87] )
  {
    do
    {
      v5 = v86[1];
      v88 = *v86;
      v86 += 2;
      sub_C7D6A0(v88, v5, 16);
    }
    while ( v37 != v86 );
LABEL_46:
    v37 = *(__int64 **)(a1 + 368);
  }
  if ( v37 != (__int64 *)(a1 + 384) )
    _libc_free(v37, v5);
  v38 = *(_QWORD *)(a1 + 320);
  if ( v38 != a1 + 336 )
    _libc_free(v38, v5);
  v39 = *(__int64 **)(a1 + 272);
  v40 = *(unsigned int *)(a1 + 232);
  v41 = &v39[2 * *(unsigned int *)(a1 + 280)];
  if ( v39 != v41 )
  {
    do
    {
      v5 = v39[1];
      v42 = *v39;
      v39 += 2;
      sub_C7D6A0(v42, v5, 16);
    }
    while ( v41 != v39 );
    v40 = *(unsigned int *)(a1 + 232);
  }
  *(_DWORD *)(a1 + 280) = 0;
  if ( !(_DWORD)v40 )
    goto LABEL_55;
  v69 = *(__int64 **)(a1 + 224);
  *(_QWORD *)(a1 + 288) = 0;
  v70 = *v69;
  v71 = &v69[v40];
  v72 = v69 + 1;
  *(_QWORD *)(a1 + 208) = *v69;
  for ( *(_QWORD *)(a1 + 216) = v70 + 4096; v71 != v72; v69 = *(__int64 **)(a1 + 224) )
  {
    v73 = *v72;
    v74 = (unsigned int)(v72 - v69) >> 7;
    v75 = 4096LL << v74;
    if ( v74 >= 0x1E )
      v75 = 0x40000000000LL;
    ++v72;
    sub_C7D6A0(v73, v75, 16);
  }
  v5 = 4096;
  *(_DWORD *)(a1 + 232) = 1;
  sub_C7D6A0(*v69, 4096, 16);
  v76 = *(__int64 **)(a1 + 272);
  v77 = 2LL * *(unsigned int *)(a1 + 280);
  v43 = &v76[v77];
  if ( v76 != &v76[v77] )
  {
    do
    {
      v5 = v76[1];
      v78 = *v76;
      v76 += 2;
      sub_C7D6A0(v78, v5, 16);
    }
    while ( v43 != v76 );
LABEL_55:
    v43 = *(__int64 **)(a1 + 272);
  }
  if ( v43 != (__int64 *)(a1 + 288) )
    _libc_free(v43, v5);
  v44 = *(_QWORD *)(a1 + 224);
  if ( v44 != a1 + 240 )
    _libc_free(v44, v5);
  v45 = *(__int64 **)(a1 + 128);
  v46 = &v45[*(unsigned int *)(a1 + 136)];
  if ( v45 != v46 )
  {
    for ( m = *(_QWORD *)(a1 + 128); ; m = *(_QWORD *)(a1 + 128) )
    {
      v48 = *v45;
      v49 = (unsigned int)(((__int64)v45 - m) >> 3) >> 7;
      v5 = 4096LL << v49;
      if ( v49 >= 0x1E )
        v5 = 0x40000000000LL;
      ++v45;
      sub_C7D6A0(v48, v5, 16);
      if ( v46 == v45 )
        break;
    }
  }
  v50 = *(__int64 **)(a1 + 176);
  v51 = &v50[2 * *(unsigned int *)(a1 + 184)];
  if ( v50 != v51 )
  {
    do
    {
      v5 = v50[1];
      v52 = *v50;
      v50 += 2;
      sub_C7D6A0(v52, v5, 16);
    }
    while ( v51 != v50 );
    v51 = *(__int64 **)(a1 + 176);
  }
  if ( v51 != (__int64 *)(a1 + 192) )
    _libc_free(v51, v5);
  v53 = *(_QWORD *)(a1 + 128);
  if ( v53 != a1 + 144 )
    _libc_free(v53, v5);
  v54 = *(_QWORD *)(a1 + 80);
  if ( v54 )
  {
    sub_CA8840(*(__int64 **)(a1 + 80), v5);
    j_j___libc_free_0(v54, 16);
  }
  v55 = *(_QWORD **)(a1 + 48);
  v56 = *(_QWORD **)(a1 + 40);
  if ( v55 != v56 )
  {
    do
    {
      if ( (_QWORD *)*v56 != v56 + 2 )
        j_j___libc_free_0(*v56, v56[2] + 1LL);
      v56 += 4;
    }
    while ( v55 != v56 );
    v56 = *(_QWORD **)(a1 + 40);
  }
  if ( v56 )
    j_j___libc_free_0(v56, *(_QWORD *)(a1 + 56) - (_QWORD)v56);
  v57 = *(__int64 **)(a1 + 24);
  v58 = *(__int64 **)(a1 + 16);
  if ( v57 != v58 )
  {
    do
    {
      v59 = v58;
      v58 += 3;
      sub_C8EE20(v59);
    }
    while ( v57 != v58 );
    v58 = *(__int64 **)(a1 + 16);
  }
  if ( v58 )
    j_j___libc_free_0(v58, *(_QWORD *)(a1 + 32) - (_QWORD)v58);
  nullsub_175();
}
