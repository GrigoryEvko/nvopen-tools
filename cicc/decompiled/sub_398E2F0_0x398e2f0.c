// Function: sub_398E2F0
// Address: 0x398e2f0
//
void __fastcall sub_398E2F0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // r8
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // r13
  unsigned __int64 *v13; // rbx
  unsigned __int64 *v14; // r13
  __int64 v15; // rbx
  unsigned __int64 v16; // r13
  _QWORD *v17; // r14
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 *v20; // rbx
  unsigned __int64 *v21; // r13
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 v25; // rbx
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // r14
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned __int64 *v31; // r13
  unsigned __int64 *v32; // rbx
  unsigned __int64 v33; // rdi
  unsigned __int64 *v34; // r13
  unsigned __int64 v35; // r14
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi

  *(_QWORD *)a1 = &unk_4A3FBA0;
  v2 = *(_QWORD *)(a1 + 6680);
  if ( v2 != *(_QWORD *)(a1 + 6672) )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 6608);
  if ( v3 != *(_QWORD *)(a1 + 6600) )
    _libc_free(v3);
  j___libc_free_0(*(_QWORD *)(a1 + 6560));
  sub_3988E70(a1 + 6352);
  sub_3988E70(a1 + 6152);
  sub_3988E70(a1 + 5952);
  sub_3988E70(a1 + 5752);
  sub_3988E70(a1 + 5552);
  j___libc_free_0(*(_QWORD *)(a1 + 5520));
  j___libc_free_0(*(_QWORD *)(a1 + 5488));
  j___libc_free_0(*(_QWORD *)(a1 + 5456));
  j___libc_free_0(*(_QWORD *)(a1 + 5424));
  v4 = *(_QWORD *)(a1 + 5328);
  if ( v4 != a1 + 5344 )
    j_j___libc_free_0(v4);
  v5 = *(_QWORD *)(a1 + 5296);
  if ( v5 != a1 + 5312 )
    j_j___libc_free_0(v5);
  v6 = *(_QWORD *)(a1 + 5264);
  if ( *(_DWORD *)(a1 + 5276) )
  {
    v7 = *(unsigned int *)(a1 + 5272);
    if ( (_DWORD)v7 )
    {
      v8 = 8 * v7;
      v9 = 0;
      do
      {
        v10 = *(_QWORD *)(v6 + v9);
        if ( v10 != -8 && v10 )
        {
          _libc_free(v10);
          v6 = *(_QWORD *)(a1 + 5264);
        }
        v9 += 8;
      }
      while ( v8 != v9 );
    }
  }
  _libc_free(v6);
  v11 = *(unsigned __int64 **)(a1 + 5032);
  v12 = &v11[9 * *(unsigned int *)(a1 + 5040)];
  if ( v11 != v12 )
  {
    do
    {
      v12 -= 9;
      if ( (unsigned __int64 *)*v12 != v12 + 2 )
        j_j___libc_free_0(*v12);
    }
    while ( v11 != v12 );
    v12 = *(unsigned __int64 **)(a1 + 5032);
  }
  if ( v12 != (unsigned __int64 *)(a1 + 5048) )
    _libc_free((unsigned __int64)v12);
  v13 = *(unsigned __int64 **)(a1 + 4920);
  v14 = &v13[4 * *(unsigned int *)(a1 + 4928)];
  if ( v13 != v14 )
  {
    do
    {
      v14 -= 4;
      if ( (unsigned __int64 *)*v14 != v14 + 2 )
        j_j___libc_free_0(*v14);
    }
    while ( v13 != v14 );
    v14 = *(unsigned __int64 **)(a1 + 4920);
  }
  if ( v14 != (unsigned __int64 *)(a1 + 4936) )
    _libc_free((unsigned __int64)v14);
  sub_3988C60(a1 + 4520);
  v15 = *(_QWORD *)(a1 + 4464);
  v16 = v15 + 16LL * *(unsigned int *)(a1 + 4472);
  if ( v15 != v16 )
  {
    do
    {
      v17 = *(_QWORD **)(v16 - 16);
      v16 -= 16LL;
      if ( v17 )
      {
        *v17 = &unk_4A3FCC0;
        sub_39A20E0(v17);
        j_j___libc_free_0((unsigned __int64)v17);
      }
    }
    while ( v15 != v16 );
    v16 = *(_QWORD *)(a1 + 4464);
  }
  if ( v16 != a1 + 4480 )
    _libc_free(v16);
  j___libc_free_0(*(_QWORD *)(a1 + 4440));
  sub_3988C60(a1 + 4040);
  v18 = *(_QWORD *)(a1 + 3864);
  if ( v18 != a1 + 3880 )
    _libc_free(v18);
  v19 = *(_QWORD *)(a1 + 3712);
  if ( v19 != *(_QWORD *)(a1 + 3704) )
    _libc_free(v19);
  v20 = *(unsigned __int64 **)(a1 + 2648);
  v21 = &v20[4 * *(unsigned int *)(a1 + 2656)];
  if ( v20 != v21 )
  {
    do
    {
      v21 -= 4;
      if ( (unsigned __int64 *)*v21 != v21 + 2 )
        j_j___libc_free_0(*v21);
    }
    while ( v20 != v21 );
    v21 = *(unsigned __int64 **)(a1 + 2648);
  }
  if ( v21 != (unsigned __int64 *)(a1 + 2664) )
    _libc_free((unsigned __int64)v21);
  v22 = *(_QWORD *)(a1 + 2376);
  if ( v22 != a1 + 2392 )
    _libc_free(v22);
  v23 = *(_QWORD *)(a1 + 1336);
  if ( v23 != a1 + 1352 )
    _libc_free(v23);
  v24 = *(_QWORD *)(a1 + 1192);
  if ( v24 != a1 + 1208 )
    _libc_free(v24);
  v25 = *(_QWORD *)(a1 + 664);
  v26 = v25 + 8LL * *(unsigned int *)(a1 + 672);
  if ( v25 != v26 )
  {
    do
    {
      v27 = *(_QWORD *)(v26 - 8);
      v26 -= 8LL;
      if ( v27 )
      {
        v28 = *(_QWORD *)(v27 + 40);
        if ( v28 != v27 + 56 )
          _libc_free(v28);
        j_j___libc_free_0(v27);
      }
    }
    while ( v25 != v26 );
    v26 = *(_QWORD *)(a1 + 664);
  }
  if ( v26 != a1 + 680 )
    _libc_free(v26);
  j___libc_free_0(*(_QWORD *)(a1 + 640));
  v29 = *(_QWORD *)(a1 + 608);
  if ( v29 )
    j_j___libc_free_0(v29);
  j___libc_free_0(*(_QWORD *)(a1 + 584));
  v30 = *(_QWORD *)(a1 + 552);
  if ( v30 )
    j_j___libc_free_0(v30);
  j___libc_free_0(*(_QWORD *)(a1 + 528));
  v31 = *(unsigned __int64 **)(a1 + 432);
  v32 = &v31[*(unsigned int *)(a1 + 440)];
  while ( v32 != v31 )
  {
    v33 = *v31++;
    _libc_free(v33);
  }
  v34 = *(unsigned __int64 **)(a1 + 480);
  v35 = (unsigned __int64)&v34[2 * *(unsigned int *)(a1 + 488)];
  if ( v34 != (unsigned __int64 *)v35 )
  {
    do
    {
      v36 = *v34;
      v34 += 2;
      _libc_free(v36);
    }
    while ( (unsigned __int64 *)v35 != v34 );
    v35 = *(_QWORD *)(a1 + 480);
  }
  if ( v35 != a1 + 496 )
    _libc_free(v35);
  v37 = *(_QWORD *)(a1 + 432);
  if ( v37 != a1 + 448 )
    _libc_free(v37);
  sub_398E0E0(a1);
}
