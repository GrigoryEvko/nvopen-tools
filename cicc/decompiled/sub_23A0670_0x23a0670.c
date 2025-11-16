// Function: sub_23A0670
// Address: 0x23a0670
//
void __fastcall sub_23A0670(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 *v3; // r14
  __int64 *v4; // rbx
  __int64 i; // rax
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 v8; // rsi
  __int64 *v9; // rbx
  unsigned __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // r14
  _QWORD *v16; // r15
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  _QWORD *v21; // r14
  _QWORD *v22; // r15
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // rdi
  __int64 *v29; // r14
  __int64 *v30; // rbx
  __int64 j; // rax
  __int64 v32; // rdi
  unsigned int v33; // ecx
  __int64 v34; // rsi
  __int64 *v35; // rbx
  unsigned __int64 v36; // r13
  __int64 v37; // rsi
  __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // r8
  __int64 v41; // r13
  __int64 v42; // r13
  __int64 v43; // rbx
  _QWORD *v44; // rdi

  sub_C7D6A0(*(_QWORD *)(a1 + 560), 16LL * *(unsigned int *)(a1 + 576), 8);
  v2 = *(_QWORD *)(a1 + 528);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(__int64 **)(a1 + 432);
  v4 = &v3[*(unsigned int *)(a1 + 440)];
  if ( v3 != v4 )
  {
    for ( i = *(_QWORD *)(a1 + 432); ; i = *(_QWORD *)(a1 + 432) )
    {
      v6 = *v3;
      v7 = (unsigned int)(((__int64)v3 - i) >> 3) >> 7;
      v8 = 4096LL << v7;
      if ( v7 >= 0x1E )
        v8 = 0x40000000000LL;
      ++v3;
      sub_C7D6A0(v6, v8, 16);
      if ( v4 == v3 )
        break;
    }
  }
  v9 = *(__int64 **)(a1 + 480);
  v10 = (unsigned __int64)&v9[2 * *(unsigned int *)(a1 + 488)];
  if ( v9 != (__int64 *)v10 )
  {
    do
    {
      v11 = v9[1];
      v12 = *v9;
      v9 += 2;
      sub_C7D6A0(v12, v11, 16);
    }
    while ( (__int64 *)v10 != v9 );
    v10 = *(_QWORD *)(a1 + 480);
  }
  if ( v10 != a1 + 496 )
    _libc_free(v10);
  v13 = *(_QWORD *)(a1 + 432);
  if ( v13 != a1 + 448 )
    _libc_free(v13);
  v14 = *(unsigned int *)(a1 + 408);
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD **)(a1 + 392);
    v16 = &v15[7 * v14];
    do
    {
      while ( 1 )
      {
        if ( *v15 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v17 = v15[3];
          if ( v17 )
            break;
        }
        v15 += 7;
        if ( v16 == v15 )
          goto LABEL_25;
      }
      do
      {
        v18 = v17;
        sub_23A0370(*(_QWORD **)(v17 + 24));
        v19 = *(_QWORD *)(v17 + 32);
        v17 = *(_QWORD *)(v17 + 16);
        if ( v19 != v18 + 48 )
          j_j___libc_free_0(v19);
        j_j___libc_free_0(v18);
      }
      while ( v17 );
      v15 += 7;
    }
    while ( v16 != v15 );
LABEL_25:
    v14 = *(unsigned int *)(a1 + 408);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 392), 56 * v14, 8);
  v20 = *(unsigned int *)(a1 + 376);
  if ( (_DWORD)v20 )
  {
    v21 = *(_QWORD **)(a1 + 360);
    v22 = &v21[7 * v20];
    do
    {
      while ( 1 )
      {
        if ( *v21 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v23 = v21[3];
          if ( v23 )
            break;
        }
        v21 += 7;
        if ( v22 == v21 )
          goto LABEL_35;
      }
      do
      {
        v24 = v23;
        sub_23A0370(*(_QWORD **)(v23 + 24));
        v25 = *(_QWORD *)(v23 + 32);
        v23 = *(_QWORD *)(v23 + 16);
        if ( v25 != v24 + 48 )
          j_j___libc_free_0(v25);
        j_j___libc_free_0(v24);
      }
      while ( v23 );
      v21 += 7;
    }
    while ( v22 != v21 );
LABEL_35:
    v20 = *(unsigned int *)(a1 + 376);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 360), 56 * v20, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 312), 16LL * *(unsigned int *)(a1 + 328), 8);
  v26 = *(_QWORD *)(a1 + 272);
  while ( v26 )
  {
    v27 = v26;
    sub_23A0050(*(_QWORD **)(v26 + 24));
    v28 = *(_QWORD *)(v26 + 48);
    v26 = *(_QWORD *)(v26 + 16);
    if ( v28 )
      j_j___libc_free_0(v28);
    j_j___libc_free_0(v27);
  }
  sub_239FF70(*(_QWORD *)(a1 + 224));
  sub_C7D6A0(*(_QWORD *)(a1 + 184), 16LL * *(unsigned int *)(a1 + 200), 8);
  v29 = *(__int64 **)(a1 + 88);
  v30 = &v29[*(unsigned int *)(a1 + 96)];
  if ( v29 != v30 )
  {
    for ( j = *(_QWORD *)(a1 + 88); ; j = *(_QWORD *)(a1 + 88) )
    {
      v32 = *v29;
      v33 = (unsigned int)(((__int64)v29 - j) >> 3) >> 7;
      v34 = 4096LL << v33;
      if ( v33 >= 0x1E )
        v34 = 0x40000000000LL;
      ++v29;
      sub_C7D6A0(v32, v34, 16);
      if ( v30 == v29 )
        break;
    }
  }
  v35 = *(__int64 **)(a1 + 136);
  v36 = (unsigned __int64)&v35[2 * *(unsigned int *)(a1 + 144)];
  if ( v35 != (__int64 *)v36 )
  {
    do
    {
      v37 = v35[1];
      v38 = *v35;
      v35 += 2;
      sub_C7D6A0(v38, v37, 16);
    }
    while ( (__int64 *)v36 != v35 );
    v36 = *(_QWORD *)(a1 + 136);
  }
  if ( v36 != a1 + 152 )
    _libc_free(v36);
  v39 = *(_QWORD *)(a1 + 88);
  if ( v39 != a1 + 104 )
    _libc_free(v39);
  v40 = *(_QWORD *)(a1 + 48);
  if ( *(_DWORD *)(a1 + 60) )
  {
    v41 = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)v41 )
    {
      v42 = 8 * v41;
      v43 = 0;
      do
      {
        v44 = *(_QWORD **)(v40 + v43);
        if ( v44 != (_QWORD *)-8LL && v44 )
        {
          sub_C7D6A0((__int64)v44, *v44 + 33LL, 8);
          v40 = *(_QWORD *)(a1 + 48);
        }
        v43 += 8;
      }
      while ( v43 != v42 );
    }
  }
  _libc_free(v40);
  sub_239EC70(*(_QWORD **)(a1 + 16));
  j_j___libc_free_0(a1);
}
