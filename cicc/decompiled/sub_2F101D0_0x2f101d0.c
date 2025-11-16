// Function: sub_2F101D0
// Address: 0x2f101d0
//
void __fastcall sub_2F101D0(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // r13
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 *v7; // r13
  unsigned __int64 *v8; // r12
  __int64 v9; // r15
  unsigned __int64 v10; // r14
  unsigned __int64 *v11; // r13
  unsigned __int64 *v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v14; // r15
  unsigned __int64 v15; // r14
  unsigned __int64 *v16; // r13
  unsigned __int64 *v17; // r12
  unsigned __int64 v18; // rdi
  void (*v19)(void); // rax
  __int64 v20; // r13
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // rdi
  _QWORD *v23; // r13
  _QWORD *v24; // r12
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 *v30; // r13
  unsigned __int64 *v31; // r12
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  _QWORD *v35; // r13
  _QWORD *v36; // r12
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rdi
  unsigned __int64 *v45; // r13
  unsigned __int64 *v46; // r12
  unsigned __int64 v47; // rdi
  _QWORD *v48; // r14
  _QWORD *v49; // r15
  unsigned __int64 *v50; // r13
  unsigned __int64 *v51; // r12
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // rdi
  unsigned __int64 *v54; // r13
  unsigned __int64 *v55; // r12

  v2 = a1 + 624;
  v3 = *(_QWORD *)(a1 + 608);
  if ( v3 != v2 )
    j_j___libc_free_0(v3);
  v4 = *(_QWORD *)(a1 + 592);
  v5 = *(_QWORD *)(a1 + 584);
  if ( v4 != v5 )
  {
    do
    {
      v6 = *(_QWORD *)(v5 + 8);
      if ( v6 != v5 + 24 )
        j_j___libc_free_0(v6);
      v5 += 64LL;
    }
    while ( v4 != v5 );
    v5 = *(_QWORD *)(a1 + 584);
  }
  if ( v5 )
    j_j___libc_free_0(v5);
  v7 = *(unsigned __int64 **)(a1 + 568);
  v8 = *(unsigned __int64 **)(a1 + 560);
  if ( v7 != v8 )
  {
    do
    {
      if ( (unsigned __int64 *)*v8 != v8 + 2 )
        j_j___libc_free_0(*v8);
      v8 += 6;
    }
    while ( v7 != v8 );
    v8 = *(unsigned __int64 **)(a1 + 560);
  }
  if ( v8 )
    j_j___libc_free_0((unsigned __int64)v8);
  v9 = *(_QWORD *)(a1 + 544);
  v10 = *(_QWORD *)(a1 + 536);
  if ( v9 != v10 )
  {
    do
    {
      v11 = *(unsigned __int64 **)(v10 + 32);
      v12 = *(unsigned __int64 **)(v10 + 24);
      if ( v11 != v12 )
      {
        do
        {
          if ( (unsigned __int64 *)*v12 != v12 + 2 )
            j_j___libc_free_0(*v12);
          v12 += 6;
        }
        while ( v11 != v12 );
        v12 = *(unsigned __int64 **)(v10 + 24);
      }
      if ( v12 )
        j_j___libc_free_0((unsigned __int64)v12);
      v10 += 48LL;
    }
    while ( v9 != v10 );
    v10 = *(_QWORD *)(a1 + 536);
  }
  if ( v10 )
    j_j___libc_free_0(v10);
  v13 = *(_QWORD *)(a1 + 504);
  if ( v13 )
    j_j___libc_free_0(v13);
  v14 = *(_QWORD *)(a1 + 488);
  v15 = *(_QWORD *)(a1 + 480);
  if ( v14 != v15 )
  {
    do
    {
      v16 = *(unsigned __int64 **)(v15 + 16);
      v17 = *(unsigned __int64 **)(v15 + 8);
      if ( v16 != v17 )
      {
        do
        {
          if ( (unsigned __int64 *)*v17 != v17 + 2 )
            j_j___libc_free_0(*v17);
          v17 += 7;
        }
        while ( v16 != v17 );
        v17 = *(unsigned __int64 **)(v15 + 8);
      }
      if ( v17 )
        j_j___libc_free_0((unsigned __int64)v17);
      v15 += 32LL;
    }
    while ( v14 != v15 );
    v15 = *(_QWORD *)(a1 + 480);
  }
  if ( v15 )
    j_j___libc_free_0(v15);
  v18 = *(_QWORD *)(a1 + 472);
  if ( v18 )
  {
    v19 = *(void (**)(void))(*(_QWORD *)v18 + 8LL);
    if ( (char *)v19 == (char *)sub_2F07240 )
      j_j___libc_free_0(v18);
    else
      v19();
  }
  v20 = *(_QWORD *)(a1 + 456);
  v21 = *(_QWORD *)(a1 + 448);
  if ( v20 != v21 )
  {
    do
    {
      v22 = *(_QWORD *)(v21 + 24);
      if ( v22 != v21 + 40 )
        j_j___libc_free_0(v22);
      v21 += 80LL;
    }
    while ( v20 != v21 );
    v21 = *(_QWORD *)(a1 + 448);
  }
  if ( v21 )
    j_j___libc_free_0(v21);
  v23 = *(_QWORD **)(a1 + 432);
  v24 = *(_QWORD **)(a1 + 424);
  if ( v23 != v24 )
  {
    do
    {
      v25 = v24[34];
      if ( (_QWORD *)v25 != v24 + 36 )
        j_j___libc_free_0(v25);
      v26 = v24[28];
      if ( (_QWORD *)v26 != v24 + 30 )
        j_j___libc_free_0(v26);
      v27 = v24[22];
      if ( (_QWORD *)v27 != v24 + 24 )
        j_j___libc_free_0(v27);
      v28 = v24[13];
      if ( (_QWORD *)v28 != v24 + 15 )
        j_j___libc_free_0(v28);
      v29 = v24[3];
      if ( (_QWORD *)v29 != v24 + 5 )
        j_j___libc_free_0(v29);
      v24 += 40;
    }
    while ( v23 != v24 );
    v24 = *(_QWORD **)(a1 + 424);
  }
  if ( v24 )
    j_j___libc_free_0((unsigned __int64)v24);
  v30 = *(unsigned __int64 **)(a1 + 408);
  v31 = *(unsigned __int64 **)(a1 + 400);
  if ( v30 != v31 )
  {
    do
    {
      v32 = v31[18];
      if ( (unsigned __int64 *)v32 != v31 + 20 )
        j_j___libc_free_0(v32);
      v33 = v31[12];
      if ( (unsigned __int64 *)v33 != v31 + 14 )
        j_j___libc_free_0(v33);
      v34 = v31[6];
      if ( (unsigned __int64 *)v34 != v31 + 8 )
        j_j___libc_free_0(v34);
      if ( (unsigned __int64 *)*v31 != v31 + 2 )
        j_j___libc_free_0(*v31);
      v31 += 24;
    }
    while ( v30 != v31 );
    v31 = *(unsigned __int64 **)(a1 + 400);
  }
  if ( v31 )
    j_j___libc_free_0((unsigned __int64)v31);
  v35 = *(_QWORD **)(a1 + 384);
  v36 = *(_QWORD **)(a1 + 376);
  if ( v35 != v36 )
  {
    do
    {
      v37 = v36[27];
      if ( (_QWORD *)v37 != v36 + 29 )
        j_j___libc_free_0(v37);
      v38 = v36[21];
      if ( (_QWORD *)v38 != v36 + 23 )
        j_j___libc_free_0(v38);
      v39 = v36[15];
      if ( (_QWORD *)v39 != v36 + 17 )
        j_j___libc_free_0(v39);
      v40 = v36[8];
      if ( (_QWORD *)v40 != v36 + 10 )
        j_j___libc_free_0(v40);
      v36 += 33;
    }
    while ( v35 != v36 );
    v36 = *(_QWORD **)(a1 + 376);
  }
  if ( v36 )
    j_j___libc_free_0((unsigned __int64)v36);
  v41 = *(_QWORD *)(a1 + 328);
  if ( v41 != a1 + 344 )
    j_j___libc_free_0(v41);
  v42 = *(_QWORD *)(a1 + 280);
  if ( v42 != a1 + 296 )
    j_j___libc_free_0(v42);
  v43 = *(_QWORD *)(a1 + 208);
  if ( v43 != a1 + 224 )
    j_j___libc_free_0(v43);
  v44 = *(_QWORD *)(a1 + 160);
  if ( v44 != a1 + 176 )
    j_j___libc_free_0(v44);
  if ( *(_BYTE *)(a1 + 120) )
  {
    v54 = *(unsigned __int64 **)(a1 + 104);
    v55 = *(unsigned __int64 **)(a1 + 96);
    *(_BYTE *)(a1 + 120) = 0;
    if ( v54 != v55 )
    {
      do
      {
        if ( (unsigned __int64 *)*v55 != v55 + 2 )
          j_j___libc_free_0(*v55);
        v55 += 6;
      }
      while ( v54 != v55 );
      v55 = *(unsigned __int64 **)(a1 + 96);
    }
    if ( v55 )
      j_j___libc_free_0((unsigned __int64)v55);
  }
  v45 = *(unsigned __int64 **)(a1 + 80);
  v46 = *(unsigned __int64 **)(a1 + 72);
  if ( v45 != v46 )
  {
    do
    {
      v47 = v46[6];
      if ( (unsigned __int64 *)v47 != v46 + 8 )
        j_j___libc_free_0(v47);
      if ( (unsigned __int64 *)*v46 != v46 + 2 )
        j_j___libc_free_0(*v46);
      v46 += 12;
    }
    while ( v45 != v46 );
    v46 = *(unsigned __int64 **)(a1 + 72);
  }
  if ( v46 )
    j_j___libc_free_0((unsigned __int64)v46);
  v48 = *(_QWORD **)(a1 + 56);
  v49 = *(_QWORD **)(a1 + 48);
  if ( v48 != v49 )
  {
    do
    {
      v50 = (unsigned __int64 *)v49[16];
      v51 = (unsigned __int64 *)v49[15];
      if ( v50 != v51 )
      {
        do
        {
          if ( (unsigned __int64 *)*v51 != v51 + 2 )
            j_j___libc_free_0(*v51);
          v51 += 6;
        }
        while ( v50 != v51 );
        v51 = (unsigned __int64 *)v49[15];
      }
      if ( v51 )
        j_j___libc_free_0((unsigned __int64)v51);
      v52 = v49[9];
      if ( (_QWORD *)v52 != v49 + 11 )
        j_j___libc_free_0(v52);
      v53 = v49[3];
      if ( (_QWORD *)v53 != v49 + 5 )
        j_j___libc_free_0(v53);
      v49 += 18;
    }
    while ( v48 != v49 );
    v49 = *(_QWORD **)(a1 + 48);
  }
  if ( v49 )
    j_j___libc_free_0((unsigned __int64)v49);
}
