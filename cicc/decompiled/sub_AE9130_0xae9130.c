// Function: sub_AE9130
// Address: 0xae9130
//
__int64 __fastcall sub_AE9130(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // rsi
  __int64 v9; // r14
  __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // r13
  __int64 v20; // r12
  __int64 v21; // r13
  __int64 result; // rax
  __int64 v23; // r12

  v3 = *(unsigned int *)(a1 + 424);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 408);
    v5 = v4 + 56 * v3;
    do
    {
      if ( *(_QWORD *)v4 != -8192 && *(_QWORD *)v4 != -4096 )
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
              sub_B91220(v7);
          }
          while ( v6 != v7 );
          v7 = *(_QWORD *)(v4 + 8);
        }
        if ( v7 != v4 + 24 )
          _libc_free(v7, a2);
      }
      v4 += 56;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 424);
  }
  v8 = 56 * v3;
  sub_C7D6A0(*(_QWORD *)(a1 + 408), 56 * v3, 8);
  v9 = *(_QWORD *)(a1 + 344);
  v10 = v9 + 8LL * *(unsigned int *)(a1 + 352);
  if ( v9 != v10 )
  {
    do
    {
      v8 = *(_QWORD *)(v10 - 8);
      v10 -= 8;
      if ( v8 )
        sub_B91220(v10);
    }
    while ( v9 != v10 );
    v10 = *(_QWORD *)(a1 + 344);
  }
  if ( v10 != a1 + 360 )
    _libc_free(v10, v8);
  v11 = *(_QWORD *)(a1 + 328);
  v12 = v11 + 56LL * *(unsigned int *)(a1 + 336);
  if ( v11 != v12 )
  {
    do
    {
      v12 -= 56;
      v13 = *(_QWORD *)(v12 + 40);
      if ( v13 != v12 + 56 )
        _libc_free(v13, v8);
      v8 = 8LL * *(unsigned int *)(v12 + 32);
      sub_C7D6A0(*(_QWORD *)(v12 + 16), v8, 8);
    }
    while ( v11 != v12 );
    v12 = *(_QWORD *)(a1 + 328);
  }
  if ( a1 + 344 != v12 )
    _libc_free(v12, v8);
  v14 = 16LL * *(unsigned int *)(a1 + 320);
  sub_C7D6A0(*(_QWORD *)(a1 + 304), v14, 8);
  v15 = *(_QWORD *)(a1 + 248);
  v16 = v15 + 8LL * *(unsigned int *)(a1 + 256);
  if ( v15 != v16 )
  {
    do
    {
      v14 = *(_QWORD *)(v16 - 8);
      v16 -= 8;
      if ( v14 )
        sub_B91220(v16);
    }
    while ( v15 != v16 );
    v16 = *(_QWORD *)(a1 + 248);
  }
  if ( v16 != a1 + 264 )
    _libc_free(v16, v14);
  v17 = *(_QWORD *)(a1 + 200);
  if ( v17 != a1 + 216 )
    _libc_free(v17, v14);
  v18 = *(_QWORD *)(a1 + 152);
  if ( v18 != a1 + 168 )
    _libc_free(v18, v14);
  v19 = *(_QWORD *)(a1 + 104);
  v20 = v19 + 8LL * *(unsigned int *)(a1 + 112);
  if ( v19 != v20 )
  {
    do
    {
      v14 = *(_QWORD *)(v20 - 8);
      v20 -= 8;
      if ( v14 )
        sub_B91220(v20);
    }
    while ( v19 != v20 );
    v20 = *(_QWORD *)(a1 + 104);
  }
  if ( v20 != a1 + 120 )
    _libc_free(v20, v14);
  v21 = *(_QWORD *)(a1 + 56);
  result = *(unsigned int *)(a1 + 64);
  v23 = v21 + 8 * result;
  if ( v21 != v23 )
  {
    do
    {
      v14 = *(_QWORD *)(v23 - 8);
      v23 -= 8;
      if ( v14 )
        result = sub_B91220(v23);
    }
    while ( v21 != v23 );
    v23 = *(_QWORD *)(a1 + 56);
  }
  if ( v23 != a1 + 72 )
    return _libc_free(v23, v14);
  return result;
}
