// Function: sub_9CD560
// Address: 0x9cd560
//
__int64 __fastcall sub_9CD560(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rdi
  __int64 *v4; // r14
  __int64 *v5; // r12
  __int64 i; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // r12
  __int64 *v10; // r13
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // r14
  _QWORD *v15; // r15
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // r14
  _QWORD *v21; // r15
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 v24; // rdi
  __int64 v25; // r12
  __int64 v26; // r13
  __int64 v27; // rdi
  __int64 v28; // rsi
  __int64 *v29; // r14
  __int64 *v30; // r12
  __int64 j; // rax
  __int64 v32; // rdi
  unsigned int v33; // ecx
  __int64 *v34; // r12
  __int64 *v35; // r13
  __int64 v36; // rdi
  __int64 v37; // rdi
  __int64 v38; // r8
  __int64 v39; // r13
  __int64 v40; // r13
  __int64 v41; // r12
  _QWORD *v42; // rdi

  v2 = 16LL * *(unsigned int *)(a1 + 576);
  sub_C7D6A0(*(_QWORD *)(a1 + 560), v2, 8);
  v3 = *(_QWORD *)(a1 + 528);
  if ( v3 )
  {
    v2 = *(_QWORD *)(a1 + 544) - v3;
    j_j___libc_free_0(v3, v2);
  }
  v4 = *(__int64 **)(a1 + 432);
  v5 = &v4[*(unsigned int *)(a1 + 440)];
  if ( v4 != v5 )
  {
    for ( i = *(_QWORD *)(a1 + 432); ; i = *(_QWORD *)(a1 + 432) )
    {
      v7 = *v4;
      v8 = (unsigned int)(((__int64)v4 - i) >> 3) >> 7;
      v2 = 4096LL << v8;
      if ( v8 >= 0x1E )
        v2 = 0x40000000000LL;
      ++v4;
      sub_C7D6A0(v7, v2, 16);
      if ( v5 == v4 )
        break;
    }
  }
  v9 = *(__int64 **)(a1 + 480);
  v10 = &v9[2 * *(unsigned int *)(a1 + 488)];
  if ( v9 != v10 )
  {
    do
    {
      v2 = v9[1];
      v11 = *v9;
      v9 += 2;
      sub_C7D6A0(v11, v2, 16);
    }
    while ( v10 != v9 );
    v10 = *(__int64 **)(a1 + 480);
  }
  if ( v10 != (__int64 *)(a1 + 496) )
    _libc_free(v10, v2);
  v12 = *(_QWORD *)(a1 + 432);
  if ( v12 != a1 + 448 )
    _libc_free(v12, v2);
  v13 = *(unsigned int *)(a1 + 408);
  if ( (_DWORD)v13 )
  {
    v14 = *(_QWORD **)(a1 + 392);
    v15 = &v14[7 * v13];
    do
    {
      while ( 1 )
      {
        if ( *v14 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v16 = v14[3];
          if ( v16 )
            break;
        }
        v14 += 7;
        if ( v15 == v14 )
          goto LABEL_25;
      }
      do
      {
        v17 = v16;
        sub_9C4EF0(*(_QWORD **)(v16 + 24));
        v18 = *(_QWORD *)(v16 + 32);
        v16 = *(_QWORD *)(v16 + 16);
        if ( v18 != v17 + 48 )
          j_j___libc_free_0(v18, *(_QWORD *)(v17 + 48) + 1LL);
        j_j___libc_free_0(v17, 64);
      }
      while ( v16 );
      v14 += 7;
    }
    while ( v15 != v14 );
LABEL_25:
    v13 = *(unsigned int *)(a1 + 408);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 392), 56 * v13, 8);
  v19 = *(unsigned int *)(a1 + 376);
  if ( (_DWORD)v19 )
  {
    v20 = *(_QWORD **)(a1 + 360);
    v21 = &v20[7 * v19];
    do
    {
      while ( 1 )
      {
        if ( *v20 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v22 = v20[3];
          if ( v22 )
            break;
        }
        v20 += 7;
        if ( v21 == v20 )
          goto LABEL_35;
      }
      do
      {
        v23 = v22;
        sub_9C4EF0(*(_QWORD **)(v22 + 24));
        v24 = *(_QWORD *)(v22 + 32);
        v22 = *(_QWORD *)(v22 + 16);
        if ( v24 != v23 + 48 )
          j_j___libc_free_0(v24, *(_QWORD *)(v23 + 48) + 1LL);
        j_j___libc_free_0(v23, 64);
      }
      while ( v22 );
      v20 += 7;
    }
    while ( v21 != v20 );
LABEL_35:
    v19 = *(unsigned int *)(a1 + 376);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 360), 56 * v19, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 312), 16LL * *(unsigned int *)(a1 + 328), 8);
  v25 = *(_QWORD *)(a1 + 272);
  while ( v25 )
  {
    v26 = v25;
    sub_9C4B30(*(_QWORD **)(v25 + 24));
    v27 = *(_QWORD *)(v25 + 48);
    v25 = *(_QWORD *)(v25 + 16);
    if ( v27 )
      j_j___libc_free_0(v27, *(_QWORD *)(v26 + 64) - v27);
    j_j___libc_free_0(v26, 72);
  }
  sub_9C48E0(*(_QWORD *)(a1 + 224));
  v28 = 16LL * *(unsigned int *)(a1 + 200);
  sub_C7D6A0(*(_QWORD *)(a1 + 184), v28, 8);
  v29 = *(__int64 **)(a1 + 88);
  v30 = &v29[*(unsigned int *)(a1 + 96)];
  if ( v29 != v30 )
  {
    for ( j = *(_QWORD *)(a1 + 88); ; j = *(_QWORD *)(a1 + 88) )
    {
      v32 = *v29;
      v33 = (unsigned int)(((__int64)v29 - j) >> 3) >> 7;
      v28 = 4096LL << v33;
      if ( v33 >= 0x1E )
        v28 = 0x40000000000LL;
      ++v29;
      sub_C7D6A0(v32, v28, 16);
      if ( v30 == v29 )
        break;
    }
  }
  v34 = *(__int64 **)(a1 + 136);
  v35 = &v34[2 * *(unsigned int *)(a1 + 144)];
  if ( v34 != v35 )
  {
    do
    {
      v28 = v34[1];
      v36 = *v34;
      v34 += 2;
      sub_C7D6A0(v36, v28, 16);
    }
    while ( v35 != v34 );
    v35 = *(__int64 **)(a1 + 136);
  }
  if ( v35 != (__int64 *)(a1 + 152) )
    _libc_free(v35, v28);
  v37 = *(_QWORD *)(a1 + 88);
  if ( v37 != a1 + 104 )
    _libc_free(v37, v28);
  v38 = *(_QWORD *)(a1 + 48);
  if ( *(_DWORD *)(a1 + 60) )
  {
    v39 = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)v39 )
    {
      v40 = 8 * v39;
      v41 = 0;
      do
      {
        v42 = *(_QWORD **)(v38 + v41);
        if ( v42 != (_QWORD *)-8LL && v42 )
        {
          v28 = *v42 + 33LL;
          sub_C7D6A0(v42, v28, 8);
          v38 = *(_QWORD *)(a1 + 48);
        }
        v41 += 8;
      }
      while ( v41 != v40 );
    }
  }
  _libc_free(v38, v28);
  return sub_9C38B0(*(_QWORD **)(a1 + 16));
}
