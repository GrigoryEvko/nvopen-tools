// Function: sub_129E320
// Address: 0x129e320
//
__int64 __fastcall sub_129E320(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // r12
  _QWORD *v15; // r13
  _QWORD *v16; // r12
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // r13
  __int64 v20; // r12
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 result; // rax
  __int64 v26; // rdi

  v3 = *(unsigned int *)(a1 + 456);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 440);
    v5 = v4 + 32 * v3;
    do
    {
      if ( *(_QWORD *)v4 != -16 && *(_QWORD *)v4 != -8 )
      {
        v6 = *(_QWORD *)(v4 + 8);
        v7 = v6 + 8LL * *(unsigned int *)(v4 + 16);
        if ( v6 != v7 )
        {
          do
          {
            a2 = *(_QWORD *)(v7 - 8);
            v7 -= 8;
            if ( a2 )
              sub_161E7C0(v7);
          }
          while ( v6 != v7 );
          v7 = *(_QWORD *)(v4 + 8);
        }
        if ( v7 != v4 + 24 )
          _libc_free(v7, a2);
      }
      v4 += 32;
    }
    while ( v5 != v4 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 440));
  v8 = *(unsigned int *)(a1 + 424);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD *)(a1 + 408);
    v10 = v9 + 32 * v8;
    do
    {
      if ( *(_QWORD *)v9 != -16 && *(_QWORD *)v9 != -8 )
      {
        v11 = *(_QWORD *)(v9 + 8);
        v12 = v11 + 8LL * *(unsigned int *)(v9 + 16);
        if ( v11 != v12 )
        {
          do
          {
            a2 = *(_QWORD *)(v12 - 8);
            v12 -= 8;
            if ( a2 )
              sub_161E7C0(v12);
          }
          while ( v11 != v12 );
          v12 = *(_QWORD *)(v9 + 8);
        }
        if ( v12 != v9 + 24 )
          _libc_free(v12, a2);
      }
      v9 += 32;
    }
    while ( v10 != v9 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 408));
  v13 = *(_QWORD *)(a1 + 344);
  v14 = v13 + 8LL * *(unsigned int *)(a1 + 352);
  if ( v13 != v14 )
  {
    do
    {
      a2 = *(_QWORD *)(v14 - 8);
      v14 -= 8;
      if ( a2 )
        sub_161E7C0(v14);
    }
    while ( v13 != v14 );
    v14 = *(_QWORD *)(a1 + 344);
  }
  if ( v14 != a1 + 360 )
    _libc_free(v14, a2);
  v15 = *(_QWORD **)(a1 + 328);
  v16 = *(_QWORD **)(a1 + 320);
  if ( v15 != v16 )
  {
    do
    {
      v17 = v16[5];
      if ( v17 )
      {
        a2 = v16[7] - v17;
        j_j___libc_free_0(v17, a2);
      }
      v18 = v16[2];
      v16 += 8;
      j___libc_free_0(v18);
    }
    while ( v15 != v16 );
    v16 = *(_QWORD **)(a1 + 320);
  }
  if ( v16 )
  {
    a2 = *(_QWORD *)(a1 + 336) - (_QWORD)v16;
    j_j___libc_free_0(v16, a2);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 296));
  v19 = *(_QWORD *)(a1 + 240);
  v20 = v19 + 8LL * *(unsigned int *)(a1 + 248);
  if ( v19 != v20 )
  {
    do
    {
      a2 = *(_QWORD *)(v20 - 8);
      v20 -= 8;
      if ( a2 )
        sub_161E7C0(v20);
    }
    while ( v19 != v20 );
    v20 = *(_QWORD *)(a1 + 240);
  }
  if ( v20 != a1 + 256 )
    _libc_free(v20, a2);
  v21 = *(_QWORD *)(a1 + 192);
  if ( v21 != a1 + 208 )
    _libc_free(v21, a2);
  v22 = *(_QWORD *)(a1 + 144);
  if ( v22 != a1 + 160 )
    _libc_free(v22, a2);
  v23 = *(_QWORD *)(a1 + 96);
  v24 = v23 + 8LL * *(unsigned int *)(a1 + 104);
  if ( v23 != v24 )
  {
    do
    {
      a2 = *(_QWORD *)(v24 - 8);
      v24 -= 8;
      if ( a2 )
        sub_161E7C0(v24);
    }
    while ( v23 != v24 );
    v24 = *(_QWORD *)(a1 + 96);
  }
  result = a1 + 112;
  if ( v24 != a1 + 112 )
    result = _libc_free(v24, a2);
  v26 = *(_QWORD *)(a1 + 48);
  if ( v26 != a1 + 64 )
    return _libc_free(v26, a2);
  return result;
}
