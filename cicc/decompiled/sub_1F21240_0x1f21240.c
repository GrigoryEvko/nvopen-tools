// Function: sub_1F21240
// Address: 0x1f21240
//
void *__fastcall sub_1F21240(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // rbx
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 *v10; // rbx
  unsigned __int64 *v11; // r12
  __int64 v12; // rcx
  unsigned __int64 v13; // r14
  unsigned __int64 *v14; // r12
  unsigned __int64 v15; // r15
  __int64 v16; // rbx
  __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  _QWORD *v21; // rbx
  _QWORD *v22; // r12
  __int64 v24; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = off_49FE8F0;
  _libc_free(*(_QWORD *)(a1 + 1536));
  _libc_free(*(_QWORD *)(a1 + 1512));
  v2 = *(_QWORD *)(a1 + 1432);
  if ( v2 != a1 + 1448 )
    _libc_free(v2);
  v3 = *(unsigned __int64 **)(a1 + 1336);
  v4 = &v3[*(unsigned int *)(a1 + 1344)];
  while ( v4 != v3 )
  {
    v5 = *v3++;
    _libc_free(v5);
  }
  v6 = *(unsigned __int64 **)(a1 + 1384);
  v7 = (unsigned __int64)&v6[2 * *(unsigned int *)(a1 + 1392)];
  if ( v6 != (unsigned __int64 *)v7 )
  {
    do
    {
      v8 = *v6;
      v6 += 2;
      _libc_free(v8);
    }
    while ( (unsigned __int64 *)v7 != v6 );
    v7 = *(_QWORD *)(a1 + 1384);
  }
  if ( v7 != a1 + 1400 )
    _libc_free(v7);
  v9 = *(_QWORD *)(a1 + 1336);
  if ( v9 != a1 + 1352 )
    _libc_free(v9);
  v10 = *(unsigned __int64 **)(a1 + 536);
  v11 = &v10[6 * *(unsigned int *)(a1 + 544)];
  if ( v10 != v11 )
  {
    do
    {
      v11 -= 6;
      if ( (unsigned __int64 *)*v11 != v11 + 2 )
        _libc_free(*v11);
    }
    while ( v10 != v11 );
    v11 = *(unsigned __int64 **)(a1 + 536);
  }
  if ( v11 != (unsigned __int64 *)(a1 + 552) )
    _libc_free((unsigned __int64)v11);
  v12 = *(_QWORD *)(a1 + 392);
  v13 = v12 + 8LL * *(unsigned int *)(a1 + 400);
  v24 = v12;
  if ( v12 != v13 )
  {
    do
    {
      v14 = *(unsigned __int64 **)(v13 - 8);
      v13 -= 8LL;
      if ( v14 )
      {
        sub_1DB4CE0((__int64)v14);
        v15 = v14[12];
        if ( v15 )
        {
          v16 = *(_QWORD *)(v15 + 16);
          while ( v16 )
          {
            sub_1F21070(*(_QWORD *)(v16 + 24));
            v17 = v16;
            v16 = *(_QWORD *)(v16 + 16);
            j_j___libc_free_0(v17, 56);
          }
          j_j___libc_free_0(v15, 48);
        }
        v18 = v14[8];
        if ( (unsigned __int64 *)v18 != v14 + 10 )
          _libc_free(v18);
        if ( (unsigned __int64 *)*v14 != v14 + 2 )
          _libc_free(*v14);
        j_j___libc_free_0(v14, 120);
      }
    }
    while ( v24 != v13 );
    v13 = *(_QWORD *)(a1 + 392);
  }
  if ( v13 != a1 + 408 )
    _libc_free(v13);
  v19 = *(_QWORD *)(a1 + 312);
  if ( v19 != a1 + 328 )
    _libc_free(v19);
  j___libc_free_0(*(_QWORD *)(a1 + 288));
  v20 = *(unsigned int *)(a1 + 272);
  if ( (_DWORD)v20 )
  {
    v21 = *(_QWORD **)(a1 + 256);
    v22 = &v21[13 * v20];
    do
    {
      if ( *v21 != -16 && *v21 != -8 )
      {
        _libc_free(v21[10]);
        _libc_free(v21[7]);
        _libc_free(v21[4]);
        _libc_free(v21[1]);
      }
      v21 += 13;
    }
    while ( v22 != v21 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 256));
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
