// Function: sub_23DCD20
// Address: 0x23dcd20
//
void __fastcall sub_23DCD20(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // r14
  __int64 v13; // r15
  unsigned __int64 v14; // r12
  __int64 v15; // rsi
  __int64 v16; // r14
  unsigned __int64 v17; // r12
  __int64 v18; // rsi
  __int64 v19; // r14
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // rdi
  __int64 v22; // r13
  unsigned __int64 v23; // r12
  __int64 v24; // rsi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  __int64 v27; // r13
  unsigned __int64 v28; // r12
  __int64 v29; // rsi
  __int64 v30; // r13
  unsigned __int64 v31; // r12
  __int64 v32; // rsi

  v2 = a1 + 5976;
  v3 = *(_QWORD *)(a1 + 5960);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 5936);
  if ( v4 != a1 + 5952 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 5656);
  if ( v5 != a1 + 5672 )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 5384);
  if ( v6 != a1 + 5400 )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 792);
  if ( v7 != a1 + 808 )
    _libc_free(v7);
  v8 = *(_QWORD *)(a1 + 648);
  if ( v8 != a1 + 664 )
    _libc_free(v8);
  v9 = *(_QWORD *)(a1 + 504);
  if ( v9 != a1 + 520 )
    _libc_free(v9);
  v10 = *(unsigned int *)(a1 + 448);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD *)(a1 + 432);
    v12 = v11 + 56 * v10;
    do
    {
      if ( *(_QWORD *)v11 != -8192 && *(_QWORD *)v11 != -4096 )
      {
        v13 = *(_QWORD *)(v11 + 8);
        v14 = v13 + 8LL * *(unsigned int *)(v11 + 16);
        if ( v13 != v14 )
        {
          do
          {
            v15 = *(_QWORD *)(v14 - 8);
            v14 -= 8LL;
            if ( v15 )
              sub_B91220(v14, v15);
          }
          while ( v13 != v14 );
          v14 = *(_QWORD *)(v11 + 8);
        }
        if ( v14 != v11 + 24 )
          _libc_free(v14);
      }
      v11 += 56;
    }
    while ( v12 != v11 );
    v10 = *(unsigned int *)(a1 + 448);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 432), 56 * v10, 8);
  v16 = *(_QWORD *)(a1 + 368);
  v17 = v16 + 8LL * *(unsigned int *)(a1 + 376);
  if ( v16 != v17 )
  {
    do
    {
      v18 = *(_QWORD *)(v17 - 8);
      v17 -= 8LL;
      if ( v18 )
        sub_B91220(v17, v18);
    }
    while ( v16 != v17 );
    v17 = *(_QWORD *)(a1 + 368);
  }
  if ( v17 != a1 + 384 )
    _libc_free(v17);
  v19 = *(_QWORD *)(a1 + 352);
  v20 = v19 + 56LL * *(unsigned int *)(a1 + 360);
  if ( v19 != v20 )
  {
    do
    {
      v20 -= 56LL;
      v21 = *(_QWORD *)(v20 + 40);
      if ( v21 != v20 + 56 )
        _libc_free(v21);
      sub_C7D6A0(*(_QWORD *)(v20 + 16), 8LL * *(unsigned int *)(v20 + 32), 8);
    }
    while ( v19 != v20 );
    v20 = *(_QWORD *)(a1 + 352);
  }
  if ( a1 + 368 != v20 )
    _libc_free(v20);
  sub_C7D6A0(*(_QWORD *)(a1 + 328), 16LL * *(unsigned int *)(a1 + 344), 8);
  v22 = *(_QWORD *)(a1 + 272);
  v23 = v22 + 8LL * *(unsigned int *)(a1 + 280);
  if ( v22 != v23 )
  {
    do
    {
      v24 = *(_QWORD *)(v23 - 8);
      v23 -= 8LL;
      if ( v24 )
        sub_B91220(v23, v24);
    }
    while ( v22 != v23 );
    v23 = *(_QWORD *)(a1 + 272);
  }
  if ( v23 != a1 + 288 )
    _libc_free(v23);
  v25 = *(_QWORD *)(a1 + 224);
  if ( v25 != a1 + 240 )
    _libc_free(v25);
  v26 = *(_QWORD *)(a1 + 176);
  if ( v26 != a1 + 192 )
    _libc_free(v26);
  v27 = *(_QWORD *)(a1 + 128);
  v28 = v27 + 8LL * *(unsigned int *)(a1 + 136);
  if ( v27 != v28 )
  {
    do
    {
      v29 = *(_QWORD *)(v28 - 8);
      v28 -= 8LL;
      if ( v29 )
        sub_B91220(v28, v29);
    }
    while ( v27 != v28 );
    v28 = *(_QWORD *)(a1 + 128);
  }
  if ( v28 != a1 + 144 )
    _libc_free(v28);
  v30 = *(_QWORD *)(a1 + 80);
  v31 = v30 + 8LL * *(unsigned int *)(a1 + 88);
  if ( v30 != v31 )
  {
    do
    {
      v32 = *(_QWORD *)(v31 - 8);
      v31 -= 8LL;
      if ( v32 )
        sub_B91220(v31, v32);
    }
    while ( v30 != v31 );
    v31 = *(_QWORD *)(a1 + 80);
  }
  if ( v31 != a1 + 96 )
    _libc_free(v31);
}
