// Function: sub_93EDA0
// Address: 0x93eda0
//
__int64 __fastcall sub_93EDA0(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  __int64 v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // r13
  __int64 v8; // rsi
  __int64 *v9; // r13
  __int64 **v10; // r14
  __int64 *v11; // r15
  __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 *v14; // rbx
  unsigned __int64 v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // r13
  __int64 v22; // rsi
  __int64 v23; // r14
  __int64 v24; // r13
  __int64 v25; // r14
  __int64 v26; // r13
  __int64 v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // rbx
  __int64 v30; // r13
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 v33; // rbx
  __int64 v34; // r13
  __int64 v35; // rbx
  __int64 result; // rax
  __int64 v37; // r13
  __int64 v38; // [rsp+8h] [rbp-58h]
  __int64 *v39; // [rsp+10h] [rbp-50h]
  unsigned __int64 i; // [rsp+18h] [rbp-48h]
  __int64 *v41; // [rsp+20h] [rbp-40h]
  __int64 *v42; // [rsp+28h] [rbp-38h]

  v2 = *(unsigned int *)(a1 + 600);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 584);
    v4 = &v3[2 * v2];
    do
    {
      if ( *v3 != -8192 && *v3 != -4096 && v3[1] )
        sub_B91220(v3 + 1);
      v3 += 2;
    }
    while ( v4 != v3 );
    LODWORD(v2) = *(_DWORD *)(a1 + 600);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 584), 16LL * (unsigned int)v2, 8);
  v5 = *(unsigned int *)(a1 + 568);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD **)(a1 + 552);
    v7 = &v6[2 * v5];
    do
    {
      if ( *v6 != -8192 && *v6 != -4096 && v6[1] )
        sub_B91220(v6 + 1);
      v6 += 2;
    }
    while ( v7 != v6 );
    LODWORD(v5) = *(_DWORD *)(a1 + 568);
  }
  v8 = 16LL * (unsigned int)v5;
  sub_C7D6A0(*(_QWORD *)(a1 + 552), v8, 8);
  v9 = *(__int64 **)(a1 + 480);
  v41 = *(__int64 **)(a1 + 512);
  v10 = (__int64 **)(*(_QWORD *)(a1 + 504) + 8LL);
  v39 = *(__int64 **)(a1 + 496);
  v42 = *(__int64 **)(a1 + 520);
  v38 = *(_QWORD *)(a1 + 504);
  for ( i = *(_QWORD *)(a1 + 536); i > (unsigned __int64)v10; ++v10 )
  {
    v11 = *v10;
    v12 = (__int64)(*v10 + 64);
    do
    {
      v8 = *v11;
      if ( *v11 )
        sub_B91220(v11);
      ++v11;
    }
    while ( (__int64 *)v12 != v11 );
  }
  if ( i == v38 )
  {
    while ( v41 != v9 )
    {
      v8 = *v9;
      if ( *v9 )
        sub_B91220(v9);
      ++v9;
    }
  }
  else
  {
    for ( ; v39 != v9; ++v9 )
    {
      v8 = *v9;
      if ( *v9 )
        sub_B91220(v9);
    }
    for ( ; v41 != v42; ++v42 )
    {
      v8 = *v42;
      if ( *v42 )
        sub_B91220(v42);
    }
  }
  v13 = *(_QWORD *)(a1 + 464);
  if ( v13 )
  {
    v14 = *(__int64 **)(a1 + 504);
    v15 = *(_QWORD *)(a1 + 536) + 8LL;
    if ( v15 > (unsigned __int64)v14 )
    {
      do
      {
        v16 = *v14++;
        j_j___libc_free_0(v16, 512);
      }
      while ( v15 > (unsigned __int64)v14 );
      v13 = *(_QWORD *)(a1 + 464);
    }
    v8 = 8LL * *(_QWORD *)(a1 + 472);
    j_j___libc_free_0(v13, v8);
  }
  v17 = *(unsigned int *)(a1 + 440);
  if ( (_DWORD)v17 )
  {
    v18 = *(_QWORD *)(a1 + 424);
    v19 = v18 + 56 * v17;
    do
    {
      if ( *(_QWORD *)v18 != -8192 && *(_QWORD *)v18 != -4096 )
      {
        v20 = *(_QWORD *)(v18 + 8);
        v21 = v20 + 8LL * *(unsigned int *)(v18 + 16);
        if ( v20 != v21 )
        {
          do
          {
            v8 = *(_QWORD *)(v21 - 8);
            v21 -= 8;
            if ( v8 )
              sub_B91220(v21);
          }
          while ( v20 != v21 );
          v21 = *(_QWORD *)(v18 + 8);
        }
        if ( v21 != v18 + 24 )
          _libc_free(v21, v8);
      }
      v18 += 56;
    }
    while ( v19 != v18 );
    v17 = *(unsigned int *)(a1 + 440);
  }
  v22 = 56 * v17;
  sub_C7D6A0(*(_QWORD *)(a1 + 424), 56 * v17, 8);
  v23 = *(_QWORD *)(a1 + 360);
  v24 = v23 + 8LL * *(unsigned int *)(a1 + 368);
  if ( v23 != v24 )
  {
    do
    {
      v22 = *(_QWORD *)(v24 - 8);
      v24 -= 8;
      if ( v22 )
        sub_B91220(v24);
    }
    while ( v23 != v24 );
    v24 = *(_QWORD *)(a1 + 360);
  }
  if ( v24 != a1 + 376 )
    _libc_free(v24, v22);
  v25 = *(_QWORD *)(a1 + 344);
  v26 = v25 + 56LL * *(unsigned int *)(a1 + 352);
  if ( v25 != v26 )
  {
    do
    {
      v26 -= 56;
      v27 = *(_QWORD *)(v26 + 40);
      if ( v27 != v26 + 56 )
        _libc_free(v27, v22);
      v22 = 8LL * *(unsigned int *)(v26 + 32);
      sub_C7D6A0(*(_QWORD *)(v26 + 16), v22, 8);
    }
    while ( v25 != v26 );
    v26 = *(_QWORD *)(a1 + 344);
  }
  if ( a1 + 360 != v26 )
    _libc_free(v26, v22);
  v28 = 16LL * *(unsigned int *)(a1 + 336);
  sub_C7D6A0(*(_QWORD *)(a1 + 320), v28, 8);
  v29 = *(_QWORD *)(a1 + 264);
  v30 = v29 + 8LL * *(unsigned int *)(a1 + 272);
  if ( v29 != v30 )
  {
    do
    {
      v28 = *(_QWORD *)(v30 - 8);
      v30 -= 8;
      if ( v28 )
        sub_B91220(v30);
    }
    while ( v29 != v30 );
    v30 = *(_QWORD *)(a1 + 264);
  }
  if ( v30 != a1 + 280 )
    _libc_free(v30, v28);
  v31 = *(_QWORD *)(a1 + 216);
  if ( v31 != a1 + 232 )
    _libc_free(v31, v28);
  v32 = *(_QWORD *)(a1 + 168);
  if ( v32 != a1 + 184 )
    _libc_free(v32, v28);
  v33 = *(_QWORD *)(a1 + 120);
  v34 = v33 + 8LL * *(unsigned int *)(a1 + 128);
  if ( v33 != v34 )
  {
    do
    {
      v28 = *(_QWORD *)(v34 - 8);
      v34 -= 8;
      if ( v28 )
        sub_B91220(v34);
    }
    while ( v33 != v34 );
    v34 = *(_QWORD *)(a1 + 120);
  }
  if ( v34 != a1 + 136 )
    _libc_free(v34, v28);
  v35 = *(_QWORD *)(a1 + 72);
  result = *(unsigned int *)(a1 + 80);
  v37 = v35 + 8 * result;
  if ( v35 != v37 )
  {
    do
    {
      v28 = *(_QWORD *)(v37 - 8);
      v37 -= 8;
      if ( v28 )
        result = sub_B91220(v37);
    }
    while ( v35 != v37 );
    v37 = *(_QWORD *)(a1 + 72);
  }
  if ( v37 != a1 + 88 )
    return _libc_free(v37, v28);
  return result;
}
