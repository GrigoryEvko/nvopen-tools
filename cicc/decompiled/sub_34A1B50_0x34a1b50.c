// Function: sub_34A1B50
// Address: 0x34a1b50
//
void __fastcall sub_34A1B50(__int64 a1)
{
  __int64 *v1; // r14
  __int64 v3; // rax
  __int64 *v4; // r12
  __int64 *i; // rax
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 v8; // rsi
  __int64 *v9; // r12
  unsigned __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rsi
  _QWORD *v15; // r12
  _QWORD *v16; // r13
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  _QWORD *v20; // r13
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi

  v1 = *(__int64 **)(a1 + 400);
  *(_QWORD *)a1 = off_49D8CC8;
  v3 = *(unsigned int *)(a1 + 408);
  *(_QWORD *)(a1 + 376) = 0;
  v4 = &v1[v3];
  if ( v1 != v4 )
  {
    for ( i = v1; ; i = *(__int64 **)(a1 + 400) )
    {
      v6 = *v1;
      v7 = (unsigned int)(v1 - i) >> 7;
      v8 = 4096LL << v7;
      if ( v7 >= 0x1E )
        v8 = 0x40000000000LL;
      ++v1;
      sub_C7D6A0(v6, v8, 16);
      if ( v4 == v1 )
        break;
    }
  }
  v9 = *(__int64 **)(a1 + 448);
  v10 = (unsigned __int64)&v9[2 * *(unsigned int *)(a1 + 456)];
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
    v10 = *(_QWORD *)(a1 + 448);
  }
  if ( v10 != a1 + 464 )
    _libc_free(v10);
  v13 = *(_QWORD *)(a1 + 400);
  if ( v13 != a1 + 416 )
    _libc_free(v13);
  v14 = *(unsigned int *)(a1 + 368);
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD **)(a1 + 352);
    v16 = &v15[2 * v14];
    do
    {
      if ( *v15 != -8192 && *v15 != -4096 )
      {
        v17 = v15[1];
        if ( v17 )
        {
          if ( !*(_BYTE *)(v17 + 28) )
            _libc_free(*(_QWORD *)(v17 + 8));
          j_j___libc_free_0(v17);
        }
      }
      v15 += 2;
    }
    while ( v16 != v15 );
    v14 = *(unsigned int *)(a1 + 368);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 352), 16 * v14, 8);
  v18 = *(_QWORD *)(a1 + 288);
  if ( v18 != a1 + 304 )
    _libc_free(v18);
  sub_2DFA900(a1 + 232);
  v19 = *(_QWORD *)(a1 + 232);
  if ( v19 != a1 + 280 )
    j_j___libc_free_0(v19);
  v20 = *(_QWORD **)(a1 + 192);
  while ( v20 )
  {
    v21 = (unsigned __int64)v20;
    v20 = (_QWORD *)*v20;
    v22 = *(_QWORD *)(v21 + 104);
    if ( v22 != v21 + 120 )
      _libc_free(v22);
    v23 = *(_QWORD *)(v21 + 56);
    if ( v23 != v21 + 72 )
      _libc_free(v23);
    j_j___libc_free_0(v21);
  }
  memset(*(void **)(a1 + 176), 0, 8LL * *(_QWORD *)(a1 + 184));
  v24 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  if ( v24 != a1 + 224 )
    j_j___libc_free_0(v24);
  sub_2DFA900(a1 + 120);
  v25 = *(_QWORD *)(a1 + 120);
  if ( v25 != a1 + 168 )
    j_j___libc_free_0(v25);
  v26 = *(_QWORD *)(a1 + 40);
  if ( v26 != a1 + 56 )
    _libc_free(v26);
}
