// Function: sub_24A58D0
// Address: 0x24a58d0
//
__int64 __fastcall sub_24A58D0(__int64 a1)
{
  __int64 v2; // rsi
  _QWORD *v3; // r12
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  __int64 v6; // rsi
  unsigned __int64 *v7; // r13
  unsigned __int64 *v8; // r12
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 *v11; // r13
  unsigned __int64 *v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r13
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // r13
  unsigned __int64 v22; // rdi

  if ( *(_BYTE *)(a1 + 392) )
  {
    v15 = *(unsigned int *)(a1 + 384);
    *(_BYTE *)(a1 + 392) = 0;
    if ( (_DWORD)v15 )
    {
      v16 = *(_QWORD *)(a1 + 368);
      v17 = v16 + 88 * v15;
      do
      {
        if ( *(_QWORD *)v16 != -4096 && *(_QWORD *)v16 != -8192 )
        {
          v18 = *(_QWORD *)(v16 + 40);
          if ( v18 != v16 + 56 )
            _libc_free(v18);
          sub_C7D6A0(*(_QWORD *)(v16 + 16), 8LL * *(unsigned int *)(v16 + 32), 8);
        }
        v16 += 88;
      }
      while ( v17 != v16 );
      v15 = *(unsigned int *)(a1 + 384);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 368), 88 * v15, 8);
    v19 = *(unsigned int *)(a1 + 352);
    if ( (_DWORD)v19 )
    {
      v20 = *(_QWORD *)(a1 + 336);
      v21 = v20 + 88 * v19;
      do
      {
        if ( *(_QWORD *)v20 != -8192 && *(_QWORD *)v20 != -4096 )
        {
          v22 = *(_QWORD *)(v20 + 40);
          if ( v22 != v20 + 56 )
            _libc_free(v22);
          sub_C7D6A0(*(_QWORD *)(v20 + 16), 8LL * *(unsigned int *)(v20 + 32), 8);
        }
        v20 += 88;
      }
      while ( v21 != v20 );
      v19 = *(unsigned int *)(a1 + 352);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 336), 88 * v19, 8);
  }
  v2 = *(unsigned int *)(a1 + 264);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 248);
    v4 = &v3[2 * v2];
    do
    {
      if ( *v3 != -4096 && *v3 != -8192 )
      {
        v5 = v3[1];
        if ( v5 )
          j_j___libc_free_0(v5);
      }
      v3 += 2;
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 264);
  }
  v6 = 16 * v2;
  sub_C7D6A0(*(_QWORD *)(a1 + 248), v6, 8);
  v7 = *(unsigned __int64 **)(a1 + 224);
  v8 = *(unsigned __int64 **)(a1 + 216);
  if ( v7 != v8 )
  {
    do
    {
      if ( *v8 )
      {
        v6 = 32;
        j_j___libc_free_0(*v8);
      }
      ++v8;
    }
    while ( v7 != v8 );
    v8 = *(unsigned __int64 **)(a1 + 216);
  }
  if ( v8 )
  {
    v6 = *(_QWORD *)(a1 + 232) - (_QWORD)v8;
    j_j___libc_free_0((unsigned __int64)v8);
  }
  v9 = *(_QWORD *)(a1 + 160);
  if ( v9 != a1 + 176 )
  {
    v6 = *(_QWORD *)(a1 + 176) + 1LL;
    j_j___libc_free_0(v9);
  }
  v10 = *(_QWORD *)(a1 + 128);
  if ( v10 != a1 + 144 )
  {
    v6 = *(_QWORD *)(a1 + 144) + 1LL;
    j_j___libc_free_0(v10);
  }
  v11 = *(unsigned __int64 **)(a1 + 48);
  v12 = *(unsigned __int64 **)(a1 + 40);
  if ( v11 != v12 )
  {
    do
    {
      v13 = *v12;
      if ( *v12 )
      {
        v6 = v12[2] - v13;
        j_j___libc_free_0(v13);
      }
      v12 += 3;
    }
    while ( v11 != v12 );
    v12 = *(unsigned __int64 **)(a1 + 40);
  }
  if ( v12 )
  {
    v6 = *(_QWORD *)(a1 + 56) - (_QWORD)v12;
    j_j___libc_free_0((unsigned __int64)v12);
  }
  return sub_24DABD0(a1 + 24, v6);
}
