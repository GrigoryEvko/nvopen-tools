// Function: sub_233AEF0
// Address: 0x233aef0
//
void __fastcall sub_233AEF0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // r13
  __int64 v5; // rax
  unsigned __int64 v6; // r12
  __int64 v7; // r15
  __int64 v8; // rdx
  unsigned __int64 v9; // r14
  unsigned __int64 *v10; // rbx
  __int64 v11; // r14
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  unsigned __int64 v14; // r15
  unsigned __int64 *v15; // r12
  __int64 v16; // r14
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // r12
  unsigned __int64 *v20; // r15
  unsigned __int64 *v21; // r13
  unsigned __int64 *v22; // r12
  __int64 v24; // [rsp+18h] [rbp-38h]

  v1 = a1 + 5576;
  v2 = a1 + 5624;
  v3 = *(_QWORD *)(a1 + 5608);
  if ( v3 != v2 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 5584), 16LL * *(unsigned int *)(a1 + 5600), 8);
  v24 = *(_QWORD *)(a1 + 5560);
  v4 = v24 + 5400LL * *(unsigned int *)(a1 + 5568);
  if ( v24 != v4 )
  {
    do
    {
      v5 = *(unsigned int *)(v4 - 5384);
      v6 = *(_QWORD *)(v4 - 5392);
      v4 -= 5400LL;
      v5 *= 672;
      v7 = v6 + v5;
      if ( v6 != v6 + v5 )
      {
        do
        {
          v8 = *(unsigned int *)(v7 - 648);
          v9 = *(_QWORD *)(v7 - 656);
          v7 -= 672;
          v8 *= 160;
          v10 = (unsigned __int64 *)(v9 + v8);
          if ( v9 != v9 + v8 )
          {
            do
            {
              v10 -= 20;
              if ( (unsigned __int64 *)*v10 != v10 + 2 )
                _libc_free(*v10);
            }
            while ( (unsigned __int64 *)v9 != v10 );
            v9 = *(_QWORD *)(v7 + 16);
          }
          if ( v9 != v7 + 32 )
            _libc_free(v9);
        }
        while ( v6 != v7 );
        v6 = *(_QWORD *)(v4 + 8);
      }
      if ( v6 != v4 + 24 )
        _libc_free(v6);
    }
    while ( v24 != v4 );
    v4 = *(_QWORD *)(a1 + 5560);
  }
  if ( v1 != v4 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 5536), 16LL * *(unsigned int *)(a1 + 5552), 8);
  v11 = *(_QWORD *)(a1 + 136);
  v12 = v11 + 672LL * *(unsigned int *)(a1 + 144);
  if ( v11 != v12 )
  {
    do
    {
      v13 = *(unsigned int *)(v12 - 648);
      v14 = *(_QWORD *)(v12 - 656);
      v12 -= 672LL;
      v15 = (unsigned __int64 *)(v14 + 160 * v13);
      if ( (unsigned __int64 *)v14 != v15 )
      {
        do
        {
          v15 -= 20;
          if ( (unsigned __int64 *)*v15 != v15 + 2 )
            _libc_free(*v15);
        }
        while ( (unsigned __int64 *)v14 != v15 );
        v14 = *(_QWORD *)(v12 + 16);
      }
      if ( v14 != v12 + 32 )
        _libc_free(v14);
    }
    while ( v11 != v12 );
    v12 = *(_QWORD *)(a1 + 136);
  }
  if ( v12 != a1 + 152 )
    _libc_free(v12);
  v16 = *(_QWORD *)(a1 + 120);
  v17 = v16 + 32LL * *(unsigned int *)(a1 + 128);
  if ( v16 != v17 )
  {
    do
    {
      v18 = *(_QWORD *)(v17 - 24);
      v19 = *(unsigned __int64 **)(v17 - 16);
      v17 -= 32LL;
      v20 = (unsigned __int64 *)v18;
      if ( v19 != (unsigned __int64 *)v18 )
      {
        do
        {
          if ( (unsigned __int64 *)*v20 != v20 + 2 )
            _libc_free(*v20);
          v20 += 21;
        }
        while ( v19 != v20 );
        v18 = *(_QWORD *)(v17 + 8);
      }
      if ( v18 )
        j_j___libc_free_0(v18);
    }
    while ( v16 != v17 );
    v17 = *(_QWORD *)(a1 + 120);
  }
  if ( a1 + 136 != v17 )
    _libc_free(v17);
  sub_C7D6A0(*(_QWORD *)(a1 + 96), 16LL * *(unsigned int *)(a1 + 112), 8);
  v21 = *(unsigned __int64 **)(a1 + 72);
  v22 = *(unsigned __int64 **)(a1 + 64);
  if ( v21 != v22 )
  {
    do
    {
      if ( (unsigned __int64 *)*v22 != v22 + 2 )
        _libc_free(*v22);
      v22 += 21;
    }
    while ( v21 != v22 );
    v22 = *(unsigned __int64 **)(a1 + 64);
  }
  if ( v22 )
    j_j___libc_free_0((unsigned __int64)v22);
}
