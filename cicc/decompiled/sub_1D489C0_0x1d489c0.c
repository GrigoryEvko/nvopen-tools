// Function: sub_1D489C0
// Address: 0x1d489c0
//
__int64 __fastcall sub_1D489C0(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // r12
  unsigned __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // rdi
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 v15; // rsi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 *v18; // r12
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  _QWORD *v21; // r15
  _QWORD *v22; // r14
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 v25; // rsi

  v2 = *(unsigned int *)(a1 + 752);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 736);
    v4 = &v3[5 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = v3[1];
        if ( (_QWORD *)v5 != v3 + 3 )
          _libc_free(v5);
      }
      v3 += 5;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 736));
  j___libc_free_0(*(_QWORD *)(a1 + 688));
  v6 = *(_QWORD *)(a1 + 640);
  v7 = *(_QWORD *)(a1 + 632);
  if ( v6 != v7 )
  {
    do
    {
      v8 = *(_QWORD *)(v7 + 64);
      if ( v8 != v7 + 80 )
        _libc_free(v8);
      if ( *(_DWORD *)(v7 + 24) > 0x40u )
      {
        v9 = *(_QWORD *)(v7 + 16);
        if ( v9 )
          j_j___libc_free_0_0(v9);
      }
      if ( *(_DWORD *)(v7 + 8) > 0x40u && *(_QWORD *)v7 )
        j_j___libc_free_0_0(*(_QWORD *)v7);
      v7 += 184;
    }
    while ( v6 != v7 );
    v7 = *(_QWORD *)(a1 + 632);
  }
  if ( v7 )
    j_j___libc_free_0(v7, *(_QWORD *)(a1 + 648) - v7);
  v10 = *(_QWORD *)(a1 + 616);
  v11 = *(_QWORD *)(a1 + 608);
  if ( v10 != v11 )
  {
    do
    {
      if ( *(_DWORD *)(v11 + 24) > 0x40u )
      {
        v12 = *(_QWORD *)(v11 + 16);
        if ( v12 )
          j_j___libc_free_0_0(v12);
      }
      if ( *(_DWORD *)(v11 + 8) > 0x40u && *(_QWORD *)v11 )
        j_j___libc_free_0_0(*(_QWORD *)v11);
      v11 += 80;
    }
    while ( v10 != v11 );
    v11 = *(_QWORD *)(a1 + 608);
  }
  if ( v11 )
    j_j___libc_free_0(v11, *(_QWORD *)(a1 + 624) - v11);
  v13 = *(_QWORD *)(a1 + 592);
  v14 = *(_QWORD *)(a1 + 584);
  if ( v13 != v14 )
  {
    do
    {
      v15 = *(_QWORD *)(v14 + 56);
      if ( v15 )
        sub_161E7C0(v14 + 56, v15);
      v14 += 80;
    }
    while ( v13 != v14 );
    v14 = *(_QWORD *)(a1 + 584);
  }
  if ( v14 )
    j_j___libc_free_0(v14, *(_QWORD *)(a1 + 600) - v14);
  v16 = *(_QWORD *)(a1 + 392);
  if ( v16 != a1 + 408 )
    _libc_free(v16);
  v17 = *(_QWORD *)(a1 + 296);
  if ( v17 != a1 + 312 )
    _libc_free(v17);
  v18 = *(unsigned __int64 **)(a1 + 280);
  if ( ((unsigned __int8)v18 & 1) == 0 && v18 )
  {
    _libc_free(*v18);
    j_j___libc_free_0(v18, 24);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 256));
  v19 = *(_QWORD *)(a1 + 104);
  if ( v19 != a1 + 120 )
    _libc_free(v19);
  v20 = *(unsigned int *)(a1 + 96);
  if ( (_DWORD)v20 )
  {
    v21 = *(_QWORD **)(a1 + 80);
    v22 = &v21[4 * v20];
    do
    {
      if ( *v21 != -16 && *v21 != -8 )
      {
        v23 = v21[2];
        v24 = v21[1];
        if ( v23 != v24 )
        {
          do
          {
            v25 = *(_QWORD *)(v24 + 8);
            if ( v25 )
              sub_161E7C0(v24 + 8, v25);
            v24 += 24;
          }
          while ( v23 != v24 );
          v24 = v21[1];
        }
        if ( v24 )
          j_j___libc_free_0(v24, v21[3] - v24);
      }
      v21 += 4;
    }
    while ( v22 != v21 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 80));
  j___libc_free_0(*(_QWORD *)(a1 + 48));
  return j___libc_free_0(*(_QWORD *)(a1 + 16));
}
