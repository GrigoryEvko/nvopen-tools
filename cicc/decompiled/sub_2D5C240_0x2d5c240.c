// Function: sub_2D5C240
// Address: 0x2d5c240
//
void __fastcall sub_2D5C240(__int64 a1)
{
  unsigned __int64 v2; // r15
  __int64 v3; // rbx
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rdi
  __int64 v7; // rbx
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  __int64 v13; // rax
  _QWORD *v14; // r13
  __int64 v15; // r8
  _QWORD *v16; // r14
  __int64 v17; // rbx
  unsigned __int64 v18; // r15
  _QWORD *v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  __int64 *v25; // r12
  _QWORD *v26; // rbx
  _QWORD *v27; // r13
  __int64 v28; // rax
  _QWORD *v29; // rbx
  _QWORD *v30; // r13
  __int64 v31; // rax
  _QWORD *v32; // rbx
  _QWORD *v33; // r13
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  unsigned int v37; // eax
  _QWORD *v38; // rbx
  _QWORD *v39; // r13
  __int64 v40; // rsi
  void *v41; // [rsp+10h] [rbp-90h] BYREF
  __int64 v42; // [rsp+18h] [rbp-88h] BYREF
  __int64 v43; // [rsp+20h] [rbp-80h]
  __int64 v44; // [rsp+28h] [rbp-78h]
  __int64 v45; // [rsp+30h] [rbp-70h]
  void *v46; // [rsp+40h] [rbp-60h] BYREF
  __int64 v47; // [rsp+48h] [rbp-58h] BYREF
  __int64 v48; // [rsp+50h] [rbp-50h]
  __int64 v49; // [rsp+58h] [rbp-48h]
  __int64 v50; // [rsp+60h] [rbp-40h]

  if ( !*(_BYTE *)(a1 + 868) )
    _libc_free(*(_QWORD *)(a1 + 848));
  v2 = *(_QWORD *)(a1 + 824);
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 24);
    v4 = v3 + 8LL * *(unsigned int *)(v2 + 32);
    if ( v3 != v4 )
    {
      do
      {
        v5 = *(_QWORD *)(v4 - 8);
        v4 -= 8LL;
        if ( v5 )
        {
          v6 = *(_QWORD *)(v5 + 24);
          if ( v6 != v5 + 40 )
            _libc_free(v6);
          j_j___libc_free_0(v5);
        }
      }
      while ( v3 != v4 );
      v4 = *(_QWORD *)(v2 + 24);
    }
    if ( v4 != v2 + 40 )
      _libc_free(v4);
    if ( *(_QWORD *)v2 != v2 + 16 )
      _libc_free(*(_QWORD *)v2);
    j_j___libc_free_0(v2);
  }
  v7 = *(_QWORD *)(a1 + 792);
  v8 = v7 + 152LL * *(unsigned int *)(a1 + 800);
  if ( v7 != v8 )
  {
    do
    {
      v8 -= 152LL;
      v9 = *(_QWORD *)(v8 + 8);
      if ( v9 != v8 + 24 )
        _libc_free(v9);
    }
    while ( v7 != v8 );
    v8 = *(_QWORD *)(a1 + 792);
  }
  if ( v8 != a1 + 808 )
    _libc_free(v8);
  sub_C7D6A0(*(_QWORD *)(a1 + 768), 16LL * *(unsigned int *)(a1 + 784), 8);
  v10 = *(unsigned int *)(a1 + 752);
  if ( (_DWORD)v10 )
  {
    v29 = *(_QWORD **)(a1 + 736);
    v43 = -4096;
    v41 = 0;
    v42 = 0;
    v30 = &v29[4 * v10];
    v46 = 0;
    v47 = 0;
    v48 = -8192;
    do
    {
      v31 = v29[2];
      if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
        sub_BD60C0(v29);
      v29 += 4;
    }
    while ( v30 != v29 );
    sub_D68D70(&v46);
    sub_D68D70(&v41);
    LODWORD(v10) = *(_DWORD *)(a1 + 752);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 736), 32LL * (unsigned int)v10, 8);
  sub_2D587C0(*(_QWORD **)(a1 + 696));
  v11 = *(_QWORD **)(a1 + 616);
  v12 = &v11[3 * *(unsigned int *)(a1 + 624)];
  if ( v11 != v12 )
  {
    do
    {
      v13 = *(v12 - 1);
      v12 -= 3;
      if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
        sub_BD60C0(v12);
    }
    while ( v11 != v12 );
    v12 = *(_QWORD **)(a1 + 616);
  }
  if ( v12 != (_QWORD *)(a1 + 632) )
    _libc_free((unsigned __int64)v12);
  v14 = *(_QWORD **)(a1 + 600);
  v15 = 133LL * *(unsigned int *)(a1 + 608);
  v16 = &v14[v15];
  if ( v14 != &v14[v15] )
  {
    do
    {
      v17 = *((unsigned int *)v16 - 258);
      v18 = *(v16 - 130);
      v16 -= 133;
      v19 = (_QWORD *)(v18 + 32 * v17);
      if ( (_QWORD *)v18 != v19 )
      {
        do
        {
          v20 = *(v19 - 2);
          v19 -= 4;
          if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
            sub_BD60C0(v19);
        }
        while ( (_QWORD *)v18 != v19 );
        v18 = v16[3];
      }
      if ( (_QWORD *)v18 != v16 + 5 )
        _libc_free(v18);
      v21 = v16[2];
      if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
        sub_BD60C0(v16);
    }
    while ( v14 != v16 );
    v16 = *(_QWORD **)(a1 + 600);
  }
  if ( (_QWORD *)(a1 + 616) != v16 )
    _libc_free((unsigned __int64)v16);
  v22 = *(unsigned int *)(a1 + 592);
  if ( (_DWORD)v22 )
  {
    v26 = *(_QWORD **)(a1 + 576);
    v43 = -4096;
    v41 = 0;
    v42 = 0;
    v27 = &v26[4 * v22];
    v46 = 0;
    v47 = 0;
    v48 = -8192;
    do
    {
      v28 = v26[2];
      if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
        sub_BD60C0(v26);
      v26 += 4;
    }
    while ( v27 != v26 );
    sub_D68D70(&v46);
    sub_D68D70(&v41);
    LODWORD(v22) = *(_DWORD *)(a1 + 592);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 576), 32LL * (unsigned int)v22, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 544), 16LL * *(unsigned int *)(a1 + 560), 8);
  if ( !*(_BYTE *)(a1 + 404) )
    _libc_free(*(_QWORD *)(a1 + 384));
  sub_C7D6A0(*(_QWORD *)(a1 + 352), 16LL * *(unsigned int *)(a1 + 368), 8);
  if ( !*(_BYTE *)(a1 + 212) )
    _libc_free(*(_QWORD *)(a1 + 192));
  if ( *(_BYTE *)(a1 + 168) )
  {
    v37 = *(_DWORD *)(a1 + 160);
    *(_BYTE *)(a1 + 168) = 0;
    if ( v37 )
    {
      v38 = *(_QWORD **)(a1 + 144);
      v39 = &v38[2 * v37];
      do
      {
        if ( *v38 != -4096 && *v38 != -8192 )
        {
          v40 = v38[1];
          if ( v40 )
            sub_B91220((__int64)(v38 + 1), v40);
        }
        v38 += 2;
      }
      while ( v39 != v38 );
      v37 = *(_DWORD *)(a1 + 160);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 144), 16LL * v37, 8);
  }
  v23 = *(unsigned int *)(a1 + 128);
  if ( (_DWORD)v23 )
  {
    v32 = *(_QWORD **)(a1 + 112);
    v42 = 2;
    v43 = 0;
    v33 = &v32[8 * v23];
    v44 = -4096;
    v34 = -4096;
    v41 = &unk_4A26638;
    v45 = 0;
    v47 = 2;
    v48 = 0;
    v49 = -8192;
    v46 = &unk_4A26638;
    v50 = 0;
    while ( 1 )
    {
      v35 = v32[3];
      if ( v35 != v34 )
      {
        v34 = v49;
        if ( v35 != v49 )
        {
          v36 = v32[7];
          if ( v36 != -4096 && v36 != 0 && v36 != -8192 )
          {
            sub_BD60C0(v32 + 5);
            v35 = v32[3];
          }
          v34 = v35;
        }
      }
      *v32 = &unk_49DB368;
      if ( v34 != -4096 && v34 != 0 && v34 != -8192 )
        sub_BD60C0(v32 + 1);
      v32 += 8;
      if ( v33 == v32 )
        break;
      v34 = v44;
    }
    v46 = &unk_49DB368;
    sub_D68D70(&v47);
    v41 = &unk_49DB368;
    sub_D68D70(&v42);
    LODWORD(v23) = *(_DWORD *)(a1 + 128);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 112), (unsigned __int64)(unsigned int)v23 << 6, 8);
  v24 = *(_QWORD *)(a1 + 72);
  if ( v24 )
    sub_2D59CD0(v24);
  v25 = *(__int64 **)(a1 + 64);
  if ( v25 )
  {
    sub_FDC110(v25);
    j_j___libc_free_0((unsigned __int64)v25);
  }
}
