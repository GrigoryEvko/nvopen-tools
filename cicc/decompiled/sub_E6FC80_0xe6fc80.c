// Function: sub_E6FC80
// Address: 0xe6fc80
//
__int64 __fastcall sub_E6FC80(_QWORD *a1, __int64 a2, unsigned int **a3)
{
  __int64 v5; // r15
  unsigned int v6; // r9d
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rcx
  _BOOL8 v10; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // r9
  __int64 v16; // r14
  __int64 v17; // r14
  __int64 v18; // rbx
  _QWORD *v19; // rdi
  _QWORD *v20; // rbx
  _QWORD *v21; // r14
  _QWORD *v22; // rbx
  _QWORD *v23; // r13
  unsigned int v24; // [rsp+14h] [rbp-3Ch]
  __int64 v25; // [rsp+18h] [rbp-38h]

  v5 = sub_22077B0(608);
  v6 = **a3;
  *(_DWORD *)(v5 + 32) = v6;
  memset((void *)(v5 + 40), 0, 0x238u);
  v24 = v6;
  *(_QWORD *)(v5 + 56) = 0x300000000LL;
  *(_QWORD *)(v5 + 168) = 0x300000000LL;
  *(_QWORD *)(v5 + 432) = 0x1000000000LL;
  *(_QWORD *)(v5 + 472) = v5 + 488;
  *(_QWORD *)(v5 + 48) = v5 + 64;
  *(_QWORD *)(v5 + 160) = v5 + 176;
  *(_QWORD *)(v5 + 440) = v5 + 456;
  *(_BYTE *)(v5 + 553) = 1;
  *(_QWORD *)(v5 + 592) = v5 + 608;
  v7 = sub_E55F30(a1, a2, (unsigned int *)(v5 + 32));
  v25 = v7;
  if ( v8 )
  {
    v9 = a1 + 1;
    v10 = 1;
    if ( !v7 && (_QWORD *)v8 != v9 )
      v10 = v24 < *(_DWORD *)(v8 + 32);
    sub_220F040(v10, v5, v8, v9);
    ++a1[5];
    return v5;
  }
  else
  {
    v12 = 0;
    sub_C7D6A0(0, 0, 8);
    v13 = *(_QWORD *)(v5 + 472);
    if ( v5 + 488 != v13 )
    {
      v12 = *(_QWORD *)(v5 + 488) + 1LL;
      j_j___libc_free_0(v13, v12);
    }
    v14 = *(_QWORD *)(v5 + 440);
    if ( v5 + 456 != v14 )
    {
      v12 = *(_QWORD *)(v5 + 456) + 1LL;
      j_j___libc_free_0(v14, v12);
    }
    v15 = *(_QWORD *)(v5 + 416);
    if ( *(_DWORD *)(v5 + 428) )
    {
      v16 = *(unsigned int *)(v5 + 424);
      if ( (_DWORD)v16 )
      {
        v17 = 8 * v16;
        v18 = 0;
        do
        {
          v19 = *(_QWORD **)(v15 + v18);
          if ( v19 != (_QWORD *)-8LL && v19 )
          {
            v12 = *v19 + 17LL;
            sub_C7D6A0((__int64)v19, v12, 8);
            v15 = *(_QWORD *)(v5 + 416);
          }
          v18 += 8;
        }
        while ( v17 != v18 );
      }
    }
    _libc_free(v15, v12);
    v20 = *(_QWORD **)(v5 + 160);
    v21 = &v20[10 * *(unsigned int *)(v5 + 168)];
    if ( v20 != v21 )
    {
      do
      {
        v21 -= 10;
        if ( (_QWORD *)*v21 != v21 + 2 )
        {
          v12 = v21[2] + 1LL;
          j_j___libc_free_0(*v21, v12);
        }
      }
      while ( v20 != v21 );
      v21 = *(_QWORD **)(v5 + 160);
    }
    if ( (_QWORD *)(v5 + 176) != v21 )
      _libc_free(v21, v12);
    v22 = *(_QWORD **)(v5 + 48);
    v23 = &v22[4 * *(unsigned int *)(v5 + 56)];
    if ( v22 != v23 )
    {
      do
      {
        v23 -= 4;
        if ( (_QWORD *)*v23 != v23 + 2 )
        {
          v12 = v23[2] + 1LL;
          j_j___libc_free_0(*v23, v12);
        }
      }
      while ( v22 != v23 );
      v23 = *(_QWORD **)(v5 + 48);
    }
    if ( (_QWORD *)(v5 + 64) != v23 )
      _libc_free(v23, v12);
    j_j___libc_free_0(v5, 608);
    return v25;
  }
}
