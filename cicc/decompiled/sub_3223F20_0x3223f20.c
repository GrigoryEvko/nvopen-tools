// Function: sub_3223F20
// Address: 0x3223f20
//
void __fastcall sub_3223F20(__int64 a1)
{
  __int64 v2; // rsi
  _QWORD *v3; // r12
  _QWORD *v4; // r13
  __int64 v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // r13
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // r13
  _QWORD *v12; // r14
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // r13
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdi
  __int64 v20; // r14
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // r13
  __int64 *v23; // r14
  __int64 *v24; // r12
  __int64 i; // rax
  __int64 v26; // rdi
  unsigned int v27; // ecx
  __int64 v28; // rsi
  __int64 *v29; // r12
  unsigned __int64 v30; // r13
  __int64 v31; // rsi
  __int64 v32; // rdi
  unsigned __int64 v33; // rdi

  sub_C7D6A0(*(_QWORD *)(a1 + 472), 16LL * *(unsigned int *)(a1 + 488), 8);
  v2 = *(unsigned int *)(a1 + 456);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 440);
    v4 = &v3[2 * v2];
    do
    {
      if ( *v3 != -8192 && *v3 != -4096 )
      {
        v5 = v3[1];
        if ( v5 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
      }
      v3 += 2;
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 456);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 440), 16 * v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 408), 16LL * *(unsigned int *)(a1 + 424), 8);
  v6 = *(unsigned int *)(a1 + 392);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD **)(a1 + 376);
    v8 = &v7[7 * v6];
    do
    {
      if ( *v7 != -8192 && *v7 != -4096 )
      {
        v9 = v7[1];
        if ( (_QWORD *)v9 != v7 + 3 )
          _libc_free(v9);
      }
      v7 += 7;
    }
    while ( v8 != v7 );
    v6 = *(unsigned int *)(a1 + 392);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 376), 56 * v6, 8);
  v10 = *(unsigned int *)(a1 + 360);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 344);
    v12 = &v11[17 * v10];
    do
    {
      if ( *v11 != -4096 && *v11 != -8192 )
      {
        v13 = v11[7];
        if ( (_QWORD *)v13 != v11 + 9 )
          _libc_free(v13);
        v14 = v11[3];
        while ( v14 )
        {
          sub_321A090(*(_QWORD *)(v14 + 24));
          v15 = v14;
          v14 = *(_QWORD *)(v14 + 16);
          j_j___libc_free_0(v15);
        }
      }
      v11 += 17;
    }
    while ( v12 != v11 );
    v10 = *(unsigned int *)(a1 + 360);
  }
  v16 = 136 * v10;
  sub_C7D6A0(*(_QWORD *)(a1 + 344), 136 * v10, 8);
  v17 = *(_QWORD *)(a1 + 240);
  v18 = v17 + ((unsigned __int64)*(unsigned int *)(a1 + 248) << 6);
  if ( v17 != v18 )
  {
    do
    {
      v18 -= 64LL;
      v19 = *(_QWORD *)(v18 + 16);
      if ( v19 != v18 + 32 )
        _libc_free(v19);
    }
    while ( v17 != v18 );
    v18 = *(_QWORD *)(a1 + 240);
  }
  if ( v18 != a1 + 256 )
    _libc_free(v18);
  _libc_free(*(_QWORD *)(a1 + 176));
  v20 = *(_QWORD *)(a1 + 152);
  v21 = v20 + 8LL * *(unsigned int *)(a1 + 160);
  if ( v20 != v21 )
  {
    do
    {
      v22 = *(_QWORD *)(v21 - 8);
      v21 -= 8LL;
      if ( v22 )
      {
        sub_3223CF0(v22);
        v16 = 784;
        j_j___libc_free_0(v22);
      }
    }
    while ( v20 != v21 );
    v21 = *(_QWORD *)(a1 + 152);
  }
  if ( v21 != a1 + 168 )
    _libc_free(v21);
  sub_3214DE0((_QWORD *)(a1 + 104), v16);
  v23 = *(__int64 **)(a1 + 24);
  v24 = &v23[*(unsigned int *)(a1 + 32)];
  if ( v23 != v24 )
  {
    for ( i = *(_QWORD *)(a1 + 24); ; i = *(_QWORD *)(a1 + 24) )
    {
      v26 = *v23;
      v27 = (unsigned int)(((__int64)v23 - i) >> 3) >> 7;
      v28 = 4096LL << v27;
      if ( v27 >= 0x1E )
        v28 = 0x40000000000LL;
      ++v23;
      sub_C7D6A0(v26, v28, 16);
      if ( v24 == v23 )
        break;
    }
  }
  v29 = *(__int64 **)(a1 + 72);
  v30 = (unsigned __int64)&v29[2 * *(unsigned int *)(a1 + 80)];
  if ( v29 != (__int64 *)v30 )
  {
    do
    {
      v31 = v29[1];
      v32 = *v29;
      v29 += 2;
      sub_C7D6A0(v32, v31, 16);
    }
    while ( (__int64 *)v30 != v29 );
    v30 = *(_QWORD *)(a1 + 72);
  }
  if ( v30 != a1 + 88 )
    _libc_free(v30);
  v33 = *(_QWORD *)(a1 + 24);
  if ( v33 != a1 + 40 )
    _libc_free(v33);
}
