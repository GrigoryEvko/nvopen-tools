// Function: sub_2989910
// Address: 0x2989910
//
void __fastcall sub_2989910(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // r13
  __int64 v5; // rsi
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // r13
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 v14; // r14
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // r15
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // rdi
  _QWORD *v24; // r13
  _QWORD *v25; // r12
  __int64 v26; // rax
  unsigned __int64 v27; // rdi

  v2 = *(unsigned int *)(a1 + 904);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 888);
    v4 = &v3[2 * v2];
    do
    {
      if ( *v3 != -8192 && *v3 != -4096 )
      {
        v5 = v3[1];
        if ( v5 )
          sub_B91220((__int64)(v3 + 1), v5);
      }
      v3 += 2;
    }
    while ( v4 != v3 );
    LODWORD(v2) = *(_DWORD *)(a1 + 904);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 888), 16LL * (unsigned int)v2, 8);
  v6 = *(_QWORD *)(a1 + 800);
  if ( v6 != a1 + 816 )
    _libc_free(v6);
  v7 = *(unsigned int *)(a1 + 792);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD *)(a1 + 776);
    v9 = v8 + 40 * v7;
    do
    {
      if ( *(_QWORD *)v8 != -8192 && *(_QWORD *)v8 != -4096 )
        sub_C7D6A0(*(_QWORD *)(v8 + 16), 32LL * *(unsigned int *)(v8 + 32), 8);
      v8 += 40;
    }
    while ( v9 != v8 );
    v7 = *(unsigned int *)(a1 + 792);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 776), 40 * v7, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 744), 16LL * *(unsigned int *)(a1 + 760), 8);
  v10 = *(_QWORD *)(a1 + 656);
  if ( v10 != a1 + 672 )
    _libc_free(v10);
  v11 = *(unsigned int *)(a1 + 648);
  if ( (_DWORD)v11 )
  {
    v12 = *(_QWORD *)(a1 + 632);
    v13 = v12 + 40 * v11;
    do
    {
      if ( *(_QWORD *)v12 != -8192 && *(_QWORD *)v12 != -4096 )
        sub_C7D6A0(*(_QWORD *)(v12 + 16), 32LL * *(unsigned int *)(v12 + 32), 8);
      v12 += 40;
    }
    while ( v13 != v12 );
    v11 = *(unsigned int *)(a1 + 648);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 632), 40 * v11, 8);
  v14 = *(_QWORD *)(a1 + 608);
  v15 = v14 + 88LL * *(unsigned int *)(a1 + 616);
  if ( v14 != v15 )
  {
    do
    {
      v15 -= 88LL;
      v16 = *(_QWORD *)(v15 + 8);
      if ( v16 != v15 + 24 )
        _libc_free(v16);
    }
    while ( v14 != v15 );
    v15 = *(_QWORD *)(a1 + 608);
  }
  if ( a1 + 624 != v15 )
    _libc_free(v15);
  sub_C7D6A0(*(_QWORD *)(a1 + 584), 16LL * *(unsigned int *)(a1 + 600), 8);
  v17 = *(unsigned int *)(a1 + 568);
  if ( (_DWORD)v17 )
  {
    v18 = *(_QWORD *)(a1 + 552);
    v19 = v18 + 56 * v17;
    do
    {
      v20 = v18 + 56;
      if ( *(_QWORD *)v18 != -8192 && *(_QWORD *)v18 != -4096 )
      {
        v21 = *(_QWORD *)(v18 + 40);
        v22 = v21 + 56LL * *(unsigned int *)(v18 + 48);
        if ( v21 != v22 )
        {
          do
          {
            v22 -= 56LL;
            v23 = *(_QWORD *)(v22 + 8);
            if ( v23 != v22 + 24 )
              _libc_free(v23);
          }
          while ( v21 != v22 );
          v22 = *(_QWORD *)(v18 + 40);
        }
        v20 = v18 + 56;
        if ( v22 != v18 + 56 )
          _libc_free(v22);
        sub_C7D6A0(*(_QWORD *)(v18 + 16), 16LL * *(unsigned int *)(v18 + 32), 8);
      }
      v18 = v20;
    }
    while ( v19 != v20 );
    v17 = *(unsigned int *)(a1 + 568);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 552), 56 * v17, 8);
  v24 = *(_QWORD **)(a1 + 336);
  v25 = &v24[3 * *(unsigned int *)(a1 + 344)];
  if ( v24 != v25 )
  {
    do
    {
      v26 = *(v25 - 1);
      v25 -= 3;
      if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
        sub_BD60C0(v25);
    }
    while ( v24 != v25 );
    v25 = *(_QWORD **)(a1 + 336);
  }
  if ( v25 != (_QWORD *)(a1 + 352) )
    _libc_free((unsigned __int64)v25);
  if ( !*(_BYTE *)(a1 + 268) )
    _libc_free(*(_QWORD *)(a1 + 248));
  if ( !*(_BYTE *)(a1 + 172) )
    _libc_free(*(_QWORD *)(a1 + 152));
  v27 = *(_QWORD *)(a1 + 64);
  if ( v27 != a1 + 80 )
    _libc_free(v27);
}
