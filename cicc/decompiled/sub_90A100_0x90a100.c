// Function: sub_90A100
// Address: 0x90a100
//
__int64 __fastcall sub_90A100(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  _QWORD *v4; // r14
  _QWORD *i; // rbx
  _QWORD *v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // r8
  __int64 v11; // r12
  __int64 v12; // rbx
  _QWORD *v13; // rdi
  __int64 v14; // r12
  __int64 v15; // r8
  __int64 v16; // r12
  __int64 v17; // rbx
  _QWORD *v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // rax

  v3 = *(_QWORD *)(a1 + 368);
  if ( v3 )
  {
    sub_93EDA0(*(_QWORD *)(a1 + 368));
    a2 = 784;
    j_j___libc_free_0(v3, 784);
  }
  v4 = *(_QWORD **)(a1 + 440);
  for ( i = *(_QWORD **)(a1 + 432); v4 != i; ++i )
  {
    v6 = (_QWORD *)*i;
    if ( *i )
    {
      v7 = v6[3];
      if ( v7 != v6[4] )
        v6[4] = v7;
      if ( v7 )
        j_j___libc_free_0(v7, v6[5] - v7);
      v8 = v6[2];
      if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
        sub_BD60C0(v6);
      a2 = 48;
      j_j___libc_free_0(v6, 48);
    }
  }
  sub_9093B0(*(_QWORD *)(a1 + 664));
  sub_909580(*(_QWORD *)(a1 + 616));
  sub_909750(*(_QWORD *)(a1 + 568));
  sub_909920(*(_QWORD *)(a1 + 520));
  if ( *(_DWORD *)(a1 + 492) )
  {
    v9 = *(unsigned int *)(a1 + 488);
    v10 = *(_QWORD *)(a1 + 480);
    if ( (_DWORD)v9 )
    {
      v11 = 8 * v9;
      v12 = 0;
      do
      {
        v13 = *(_QWORD **)(v10 + v12);
        if ( v13 != (_QWORD *)-8LL && v13 )
        {
          a2 = *v13 + 17LL;
          sub_C7D6A0(v13, a2, 8);
          v10 = *(_QWORD *)(a1 + 480);
        }
        v12 += 8;
      }
      while ( v11 != v12 );
    }
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 480);
  }
  _libc_free(v10, a2);
  if ( *(_DWORD *)(a1 + 468) )
  {
    v14 = *(unsigned int *)(a1 + 464);
    v15 = *(_QWORD *)(a1 + 456);
    if ( (_DWORD)v14 )
    {
      v16 = 8 * v14;
      v17 = 0;
      do
      {
        v18 = *(_QWORD **)(v15 + v17);
        if ( v18 != (_QWORD *)-8LL && v18 )
        {
          a2 = *v18 + 17LL;
          sub_C7D6A0(v18, a2, 8);
          v15 = *(_QWORD *)(a1 + 456);
        }
        v17 += 8;
      }
      while ( v16 != v17 );
    }
  }
  else
  {
    v15 = *(_QWORD *)(a1 + 456);
  }
  _libc_free(v15, a2);
  v19 = *(_QWORD *)(a1 + 432);
  if ( v19 )
    j_j___libc_free_0(v19, *(_QWORD *)(a1 + 448) - v19);
  v20 = *(_QWORD *)(a1 + 416);
  v21 = *(_QWORD *)(a1 + 408);
  if ( v20 != v21 )
  {
    do
    {
      v22 = *(_QWORD *)(v21 + 16);
      if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
        sub_BD60C0(v21);
      v21 += 24;
    }
    while ( v20 != v21 );
    v21 = *(_QWORD *)(a1 + 408);
  }
  if ( v21 )
    j_j___libc_free_0(v21, *(_QWORD *)(a1 + 424) - v21);
  sub_C7D6A0(*(_QWORD *)(a1 + 384), 16LL * *(unsigned int *)(a1 + 400), 8);
  return sub_917E00(a1 + 8);
}
