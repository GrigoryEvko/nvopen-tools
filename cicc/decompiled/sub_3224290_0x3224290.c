// Function: sub_3224290
// Address: 0x3224290
//
void __fastcall sub_3224290(__int64 a1)
{
  bool v2; // zf
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r8
  __int64 v8; // r13
  __int64 v9; // r13
  __int64 v10; // rbx
  _QWORD *v11; // rdi
  unsigned __int64 *v12; // rbx
  unsigned __int64 *v13; // r13
  unsigned __int64 *v14; // rbx
  unsigned __int64 *v15; // r13
  __int64 v16; // rbx
  unsigned __int64 v17; // r13
  _QWORD *v18; // r14
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r13
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 v25; // rsi
  unsigned __int64 *v26; // rbx
  unsigned __int64 *v27; // r13
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v31; // rbx
  unsigned __int64 v32; // r13
  __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  __int64 *v36; // r14
  __int64 *v37; // rbx
  __int64 i; // rax
  __int64 v39; // rdi
  unsigned int v40; // ecx
  __int64 v41; // rsi
  __int64 *v42; // rbx
  unsigned __int64 v43; // r13
  __int64 v44; // rsi
  __int64 v45; // rdi
  unsigned __int64 v46; // rdi

  v2 = *(_BYTE *)(a1 + 6292) == 0;
  *(_QWORD *)a1 = &unk_4A35BF8;
  if ( v2 )
    _libc_free(*(_QWORD *)(a1 + 6272));
  sub_C7D6A0(*(_QWORD *)(a1 + 6240), 8LL * *(unsigned int *)(a1 + 6256), 8);
  sub_321E250(a1 + 6016);
  sub_321E250(a1 + 5808);
  sub_321E250(a1 + 5600);
  sub_321E250(a1 + 5392);
  v3 = *(_QWORD *)(a1 + 5344);
  if ( v3 != a1 + 5360 )
    _libc_free(v3);
  sub_321E250(a1 + 5136);
  v4 = *(_QWORD *)(a1 + 5096);
  if ( v4 != a1 + 5112 )
    _libc_free(v4);
  sub_321E250(a1 + 4888);
  sub_C7D6A0(*(_QWORD *)(a1 + 4848), 16LL * *(unsigned int *)(a1 + 4864), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 4816), 16LL * *(unsigned int *)(a1 + 4832), 8);
  v5 = *(_QWORD *)(a1 + 4704);
  if ( v5 != a1 + 4720 )
    j_j___libc_free_0(v5);
  v6 = *(_QWORD *)(a1 + 4672);
  if ( v6 != a1 + 4688 )
    j_j___libc_free_0(v6);
  v7 = *(_QWORD *)(a1 + 4648);
  if ( *(_DWORD *)(a1 + 4660) )
  {
    v8 = *(unsigned int *)(a1 + 4656);
    if ( (_DWORD)v8 )
    {
      v9 = 8 * v8;
      v10 = 0;
      do
      {
        v11 = *(_QWORD **)(v7 + v10);
        if ( v11 && v11 != (_QWORD *)-8LL )
        {
          sub_C7D6A0((__int64)v11, *v11 + 17LL, 8);
          v7 = *(_QWORD *)(a1 + 4648);
        }
        v10 += 8;
      }
      while ( v9 != v10 );
    }
  }
  _libc_free(v7);
  v12 = *(unsigned __int64 **)(a1 + 4392);
  v13 = &v12[10 * *(unsigned int *)(a1 + 4400)];
  if ( v12 != v13 )
  {
    do
    {
      v13 -= 10;
      if ( (unsigned __int64 *)*v13 != v13 + 2 )
        j_j___libc_free_0(*v13);
    }
    while ( v12 != v13 );
    v13 = *(unsigned __int64 **)(a1 + 4392);
  }
  if ( v13 != (unsigned __int64 *)(a1 + 4408) )
    _libc_free((unsigned __int64)v13);
  v14 = *(unsigned __int64 **)(a1 + 4280);
  v15 = &v14[4 * *(unsigned int *)(a1 + 4288)];
  if ( v14 != v15 )
  {
    do
    {
      v15 -= 4;
      if ( (unsigned __int64 *)*v15 != v15 + 2 )
        j_j___libc_free_0(*v15);
    }
    while ( v14 != v15 );
    v15 = *(unsigned __int64 **)(a1 + 4280);
  }
  if ( v15 != (unsigned __int64 *)(a1 + 4296) )
    _libc_free((unsigned __int64)v15);
  sub_3223F20(a1 + 3776);
  if ( !*(_BYTE *)(a1 + 3724) )
    _libc_free(*(_QWORD *)(a1 + 3704));
  v16 = *(_QWORD *)(a1 + 3640);
  v17 = v16 + 16LL * *(unsigned int *)(a1 + 3648);
  if ( v16 != v17 )
  {
    do
    {
      v18 = *(_QWORD **)(v17 - 16);
      v17 -= 16LL;
      if ( v18 )
      {
        *v18 = &unk_4A35D40;
        sub_32478E0(v18);
        j_j___libc_free_0((unsigned __int64)v18);
      }
    }
    while ( v16 != v17 );
    v17 = *(_QWORD *)(a1 + 3640);
  }
  if ( v17 != a1 + 3656 )
    _libc_free(v17);
  sub_C7D6A0(*(_QWORD *)(a1 + 3616), 16LL * *(unsigned int *)(a1 + 3632), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 3584), 16LL * *(unsigned int *)(a1 + 3600), 8);
  sub_3223F20(a1 + 3080);
  if ( (*(_BYTE *)(a1 + 3008) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 3016), 8LL * *(unsigned int *)(a1 + 3024), 8);
  v19 = *(unsigned int *)(a1 + 2992);
  if ( (_DWORD)v19 )
  {
    v20 = *(_QWORD *)(a1 + 2976);
    v21 = v20 + 88 * v19;
    do
    {
      while ( 1 )
      {
        if ( *(_QWORD *)v20 != -8192 && *(_QWORD *)v20 != -4096 )
        {
          v22 = *(_QWORD *)(v20 + 56);
          if ( v22 != v20 + 72 )
            _libc_free(v22);
          if ( !*(_BYTE *)(v20 + 36) )
            break;
        }
        v20 += 88;
        if ( v21 == v20 )
          goto LABEL_52;
      }
      v23 = *(_QWORD *)(v20 + 16);
      v20 += 88;
      _libc_free(v23);
    }
    while ( v21 != v20 );
LABEL_52:
    v19 = *(unsigned int *)(a1 + 2992);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 2976), 88 * v19, 8);
  v24 = *(_QWORD *)(a1 + 2824);
  if ( v24 != a1 + 2840 )
    _libc_free(v24);
  v25 = 8LL * *(unsigned int *)(a1 + 2816);
  sub_C7D6A0(*(_QWORD *)(a1 + 2800), v25, 8);
  v26 = *(unsigned __int64 **)(a1 + 2760);
  v27 = *(unsigned __int64 **)(a1 + 2752);
  if ( v26 != v27 )
  {
    do
    {
      if ( (unsigned __int64 *)*v27 != v27 + 2 )
      {
        v25 = v27[2] + 1;
        j_j___libc_free_0(*v27);
      }
      v27 += 4;
    }
    while ( v26 != v27 );
    v27 = *(unsigned __int64 **)(a1 + 2752);
  }
  if ( v27 )
  {
    v25 = *(_QWORD *)(a1 + 2768) - (_QWORD)v27;
    j_j___libc_free_0((unsigned __int64)v27);
  }
  v28 = *(_QWORD *)(a1 + 2472);
  if ( v28 != a1 + 2496 )
    _libc_free(v28);
  v29 = *(_QWORD *)(a1 + 1432);
  if ( v29 != a1 + 1448 )
    _libc_free(v29);
  v30 = *(_QWORD *)(a1 + 1288);
  if ( v30 != a1 + 1304 )
    _libc_free(v30);
  v31 = *(_QWORD *)(a1 + 760);
  v32 = v31 + 8LL * *(unsigned int *)(a1 + 768);
  if ( v31 != v32 )
  {
    do
    {
      v33 = *(_QWORD *)(v32 - 8);
      v32 -= 8LL;
      if ( v33 )
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v33 + 8LL))(v33, v25);
    }
    while ( v31 != v32 );
    v32 = *(_QWORD *)(a1 + 760);
  }
  if ( v32 != a1 + 776 )
    _libc_free(v32);
  sub_C7D6A0(*(_QWORD *)(a1 + 736), 16LL * *(unsigned int *)(a1 + 752), 8);
  v34 = *(_QWORD *)(a1 + 704);
  if ( v34 )
    j_j___libc_free_0(v34);
  sub_C7D6A0(*(_QWORD *)(a1 + 680), 16LL * *(unsigned int *)(a1 + 696), 8);
  v35 = *(_QWORD *)(a1 + 656);
  if ( v35 != a1 + 672 )
    _libc_free(v35);
  sub_C7D6A0(*(_QWORD *)(a1 + 632), 16LL * *(unsigned int *)(a1 + 648), 8);
  v36 = *(__int64 **)(a1 + 544);
  v37 = &v36[*(unsigned int *)(a1 + 552)];
  if ( v36 != v37 )
  {
    for ( i = *(_QWORD *)(a1 + 544); ; i = *(_QWORD *)(a1 + 544) )
    {
      v39 = *v36;
      v40 = (unsigned int)(((__int64)v36 - i) >> 3) >> 7;
      v41 = 4096LL << v40;
      if ( v40 >= 0x1E )
        v41 = 0x40000000000LL;
      ++v36;
      sub_C7D6A0(v39, v41, 16);
      if ( v37 == v36 )
        break;
    }
  }
  v42 = *(__int64 **)(a1 + 592);
  v43 = (unsigned __int64)&v42[2 * *(unsigned int *)(a1 + 600)];
  if ( v42 != (__int64 *)v43 )
  {
    do
    {
      v44 = v42[1];
      v45 = *v42;
      v42 += 2;
      sub_C7D6A0(v45, v44, 16);
    }
    while ( (__int64 *)v43 != v42 );
    v43 = *(_QWORD *)(a1 + 592);
  }
  if ( v43 != a1 + 608 )
    _libc_free(v43);
  v46 = *(_QWORD *)(a1 + 544);
  if ( v46 != a1 + 560 )
    _libc_free(v46);
  sub_3212B10(a1);
}
