// Function: sub_313FB90
// Address: 0x313fb90
//
void __fastcall sub_313FB90(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 *v4; // r14
  __int64 *v5; // r12
  __int64 i; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 v9; // rsi
  __int64 *v10; // r12
  unsigned __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  _QWORD *v15; // r12
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  __int64 v18; // r13
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // rdi
  void (__fastcall *v21)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 v24; // r14
  __int64 v25; // r14
  __int64 v26; // r13
  _QWORD *v27; // r12
  unsigned __int64 v28; // rdi
  __int64 v29; // r15
  __int64 v30; // rax
  __int64 v31; // r13
  unsigned __int64 v32; // r8
  __int64 v33; // r13
  __int64 v34; // r12
  _QWORD *v35; // rdi
  unsigned __int64 v36; // rdi
  unsigned __int64 *v37; // r13
  unsigned __int64 *v38; // r12
  unsigned __int64 v39; // r13
  unsigned __int64 v40; // r12
  void (__fastcall *v41)(unsigned __int64, unsigned __int64, __int64); // rax

  v2 = a1 + 3096;
  v3 = *(_QWORD *)(a1 + 3080);
  if ( v3 != v2 )
    j_j___libc_free_0(v3);
  v4 = *(__int64 **)(a1 + 2520);
  v5 = &v4[*(unsigned int *)(a1 + 2528)];
  if ( v4 != v5 )
  {
    for ( i = *(_QWORD *)(a1 + 2520); ; i = *(_QWORD *)(a1 + 2520) )
    {
      v7 = *v4;
      v8 = (unsigned int)(((__int64)v4 - i) >> 3) >> 7;
      v9 = 4096LL << v8;
      if ( v8 >= 0x1E )
        v9 = 0x40000000000LL;
      ++v4;
      sub_C7D6A0(v7, v9, 16);
      if ( v5 == v4 )
        break;
    }
  }
  v10 = *(__int64 **)(a1 + 2568);
  v11 = (unsigned __int64)&v10[2 * *(unsigned int *)(a1 + 2576)];
  if ( v10 != (__int64 *)v11 )
  {
    do
    {
      v12 = v10[1];
      v13 = *v10;
      v10 += 2;
      sub_C7D6A0(v13, v12, 16);
    }
    while ( (__int64 *)v11 != v10 );
    v11 = *(_QWORD *)(a1 + 2568);
  }
  if ( v11 != a1 + 2584 )
    _libc_free(v11);
  v14 = *(_QWORD *)(a1 + 2520);
  if ( v14 != a1 + 2536 )
    _libc_free(v14);
  _libc_free(*(_QWORD *)(a1 + 2480));
  v15 = *(_QWORD **)(a1 + 2472);
  while ( v15 )
  {
    v16 = (unsigned __int64)v15;
    v15 = (_QWORD *)*v15;
    j_j___libc_free_0(v16);
  }
  v17 = *(_QWORD *)(a1 + 2328);
  if ( v17 != a1 + 2344 )
    _libc_free(v17);
  v18 = *(_QWORD *)(a1 + 904);
  v19 = v18 + 88LL * *(unsigned int *)(a1 + 912);
  if ( v18 != v19 )
  {
    do
    {
      v19 -= 88LL;
      v20 = *(_QWORD *)(v19 + 56);
      if ( v20 != v19 + 72 )
        _libc_free(v20);
      v21 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v19 + 16);
      if ( v21 )
        v21(v19, v19, 3);
    }
    while ( v18 != v19 );
    v19 = *(_QWORD *)(a1 + 904);
  }
  if ( v19 != a1 + 920 )
    _libc_free(v19);
  v22 = *(_QWORD *)(a1 + 848);
  if ( v22 != a1 + 864 )
    j_j___libc_free_0(v22);
  v23 = *(_QWORD *)(a1 + 824);
  if ( *(_DWORD *)(a1 + 836) )
  {
    v24 = *(unsigned int *)(a1 + 832);
    if ( (_DWORD)v24 )
    {
      v25 = 8 * v24;
      v26 = 0;
      do
      {
        v27 = *(_QWORD **)(v23 + v26);
        if ( v27 != (_QWORD *)-8LL && v27 )
        {
          v28 = v27[8];
          v29 = *v27 + 97LL;
          if ( (_QWORD *)v28 != v27 + 10 )
            j_j___libc_free_0(v28);
          v30 = v27[3];
          if ( v30 != -4096 && v30 != 0 && v30 != -8192 )
            sub_BD60C0(v27 + 1);
          sub_C7D6A0((__int64)v27, v29, 8);
          v23 = *(_QWORD *)(a1 + 824);
        }
        v26 += 8;
      }
      while ( v25 != v26 );
    }
  }
  _libc_free(v23);
  sub_3121850(*(_QWORD **)(a1 + 792));
  sub_3121450(*(_QWORD **)(a1 + 744));
  sub_C7D6A0(*(_QWORD *)(a1 + 688), 24LL * *(unsigned int *)(a1 + 704), 8);
  if ( *(_DWORD *)(a1 + 668) )
  {
    v31 = *(unsigned int *)(a1 + 664);
    v32 = *(_QWORD *)(a1 + 656);
    if ( (_DWORD)v31 )
    {
      v33 = 8 * v31;
      v34 = 0;
      do
      {
        v35 = *(_QWORD **)(v32 + v34);
        if ( v35 != (_QWORD *)-8LL && v35 )
        {
          sub_C7D6A0((__int64)v35, *v35 + 17LL, 8);
          v32 = *(_QWORD *)(a1 + 656);
        }
        v34 += 8;
      }
      while ( v33 != v34 );
    }
  }
  else
  {
    v32 = *(_QWORD *)(a1 + 656);
  }
  _libc_free(v32);
  nullsub_61();
  *(_QWORD *)(a1 + 640) = &unk_49DA100;
  nullsub_63();
  v36 = *(_QWORD *)(a1 + 512);
  if ( v36 != a1 + 528 )
    _libc_free(v36);
  v37 = *(unsigned __int64 **)(a1 + 424);
  v38 = &v37[7 * *(unsigned int *)(a1 + 432)];
  if ( v37 != v38 )
  {
    do
    {
      v38 -= 7;
      if ( (unsigned __int64 *)*v38 != v38 + 2 )
        j_j___libc_free_0(*v38);
    }
    while ( v37 != v38 );
    v38 = *(unsigned __int64 **)(a1 + 424);
  }
  if ( v38 != (unsigned __int64 *)(a1 + 440) )
    _libc_free((unsigned __int64)v38);
  v39 = *(_QWORD *)a1;
  v40 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v40 )
  {
    do
    {
      v41 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v40 - 24);
      v40 -= 40LL;
      if ( v41 )
        v41(v40, v40, 3);
    }
    while ( v39 != v40 );
    v40 = *(_QWORD *)a1;
  }
  if ( v40 != a1 + 16 )
    _libc_free(v40);
}
