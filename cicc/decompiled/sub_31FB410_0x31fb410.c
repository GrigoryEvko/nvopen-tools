// Function: sub_31FB410
// Address: 0x31fb410
//
void __fastcall sub_31FB410(__int64 a1)
{
  unsigned __int64 v1; // rdi
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // r13
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rbx
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 v15; // r13
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // rdi
  __int64 v24; // rbx
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // r14
  unsigned __int64 v27; // r15
  unsigned __int64 v28; // rdi
  void *v29; // rdi
  bool v30; // cc
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  _QWORD *v35; // [rsp+8h] [rbp-38h]
  _QWORD *v36; // [rsp+8h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 416);
  if ( v1 )
    j_j___libc_free_0(v1);
  v2 = *(_QWORD *)(a1 + 392);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 368);
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = *(_QWORD *)(a1 + 344);
  if ( v4 != a1 + 360 )
    _libc_free(v4);
  v35 = *(_QWORD **)(a1 + 304);
  while ( v35 )
  {
    v5 = (unsigned __int64)v35;
    v6 = v35[19];
    v35 = (_QWORD *)*v35;
    if ( v6 != v5 + 168 )
      _libc_free(v6);
    v7 = *(_QWORD *)(v5 + 120);
    if ( v7 != v5 + 136 )
      _libc_free(v7);
    v8 = *(_QWORD *)(v5 + 16);
    v9 = v8 + 88LL * *(unsigned int *)(v5 + 24);
    if ( v8 != v9 )
    {
      do
      {
        v9 -= 88LL;
        if ( *(_BYTE *)(v9 + 80) )
        {
          v30 = *(_DWORD *)(v9 + 72) <= 0x40u;
          *(_BYTE *)(v9 + 80) = 0;
          if ( !v30 )
          {
            v31 = *(_QWORD *)(v9 + 64);
            if ( v31 )
              j_j___libc_free_0_0(v31);
          }
        }
        v10 = *(_QWORD *)(v9 + 40);
        v11 = v10 + 40LL * *(unsigned int *)(v9 + 48);
        if ( v10 != v11 )
        {
          do
          {
            v11 -= 40LL;
            v12 = *(_QWORD *)(v11 + 8);
            if ( v12 != v11 + 24 )
              _libc_free(v12);
          }
          while ( v10 != v11 );
          v10 = *(_QWORD *)(v9 + 40);
        }
        if ( v10 != v9 + 56 )
          _libc_free(v10);
        sub_C7D6A0(*(_QWORD *)(v9 + 16), 12LL * *(unsigned int *)(v9 + 32), 4);
      }
      while ( v8 != v9 );
      v9 = *(_QWORD *)(v5 + 16);
    }
    if ( v9 != v5 + 32 )
      _libc_free(v9);
    j_j___libc_free_0(v5);
  }
  memset(*(void **)(a1 + 288), 0, 8LL * *(_QWORD *)(a1 + 296));
  v13 = *(_QWORD *)(a1 + 288);
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  if ( v13 != a1 + 336 )
    j_j___libc_free_0(v13);
  v14 = *(_QWORD *)(a1 + 256);
  if ( v14 != a1 + 272 )
    _libc_free(v14);
  v15 = *(_QWORD *)(a1 + 152);
  v16 = v15 + 88LL * *(unsigned int *)(a1 + 160);
  if ( v15 != v16 )
  {
    do
    {
      v16 -= 88LL;
      if ( *(_BYTE *)(v16 + 80) )
      {
        v30 = *(_DWORD *)(v16 + 72) <= 0x40u;
        *(_BYTE *)(v16 + 80) = 0;
        if ( !v30 )
        {
          v33 = *(_QWORD *)(v16 + 64);
          if ( v33 )
            j_j___libc_free_0_0(v33);
        }
      }
      v17 = *(_QWORD *)(v16 + 40);
      v18 = v17 + 40LL * *(unsigned int *)(v16 + 48);
      if ( v17 != v18 )
      {
        do
        {
          v18 -= 40LL;
          v19 = *(_QWORD *)(v18 + 8);
          if ( v19 != v18 + 24 )
            _libc_free(v19);
        }
        while ( v17 != v18 );
        v17 = *(_QWORD *)(v16 + 40);
      }
      if ( v17 != v16 + 56 )
        _libc_free(v17);
      sub_C7D6A0(*(_QWORD *)(v16 + 16), 12LL * *(unsigned int *)(v16 + 32), 4);
    }
    while ( v15 != v16 );
    v16 = *(_QWORD *)(a1 + 152);
  }
  if ( v16 != a1 + 168 )
    _libc_free(v16);
  sub_31F5020(*(_QWORD *)(a1 + 120));
  v20 = *(_QWORD *)(a1 + 80);
  if ( v20 != a1 + 96 )
    _libc_free(v20);
  v21 = *(_QWORD *)(a1 + 56);
  if ( v21 != a1 + 72 )
    _libc_free(v21);
  v36 = *(_QWORD **)(a1 + 16);
  while ( v36 )
  {
    v22 = (unsigned __int64)v36;
    v23 = v36[15];
    v36 = (_QWORD *)*v36;
    if ( v23 != v22 + 136 )
      _libc_free(v23);
    v24 = *(_QWORD *)(v22 + 16);
    v25 = v24 + 88LL * *(unsigned int *)(v22 + 24);
    if ( v24 != v25 )
    {
      do
      {
        v25 -= 88LL;
        if ( *(_BYTE *)(v25 + 80) )
        {
          v30 = *(_DWORD *)(v25 + 72) <= 0x40u;
          *(_BYTE *)(v25 + 80) = 0;
          if ( !v30 )
          {
            v32 = *(_QWORD *)(v25 + 64);
            if ( v32 )
              j_j___libc_free_0_0(v32);
          }
        }
        v26 = *(_QWORD *)(v25 + 40);
        v27 = v26 + 40LL * *(unsigned int *)(v25 + 48);
        if ( v26 != v27 )
        {
          do
          {
            v27 -= 40LL;
            v28 = *(_QWORD *)(v27 + 8);
            if ( v28 != v27 + 24 )
              _libc_free(v28);
          }
          while ( v26 != v27 );
          v26 = *(_QWORD *)(v25 + 40);
        }
        if ( v26 != v25 + 56 )
          _libc_free(v26);
        sub_C7D6A0(*(_QWORD *)(v25 + 16), 12LL * *(unsigned int *)(v25 + 32), 4);
      }
      while ( v24 != v25 );
      v25 = *(_QWORD *)(v22 + 16);
    }
    if ( v25 != v22 + 32 )
      _libc_free(v25);
    j_j___libc_free_0(v22);
  }
  memset(*(void **)a1, 0, 8LL * *(_QWORD *)(a1 + 8));
  v29 = *(void **)a1;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  if ( v29 != (void *)(a1 + 48) )
    j_j___libc_free_0((unsigned __int64)v29);
}
