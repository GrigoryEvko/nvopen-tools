// Function: sub_37BDF30
// Address: 0x37bdf30
//
void __fastcall sub_37BDF30(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rax
  _BYTE *v4; // r12
  _BYTE *v5; // r13
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  _QWORD *v10; // r13
  _QWORD *v11; // r14
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 v15; // rax
  _QWORD *v16; // r12
  _QWORD *v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  __int64 v21; // rsi
  _QWORD *v22; // r12
  _QWORD *v23; // r13
  unsigned __int64 v24; // r14
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  _QWORD *v27; // r13
  unsigned __int64 v28; // r12
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  _QWORD v34[6]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v35[10]; // [rsp+30h] [rbp-50h] BYREF

  *(_QWORD *)a1 = &unk_4A3D510;
  v2 = *(_QWORD *)(a1 + 2296);
  if ( v2 != a1 + 2312 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 2272), 48LL * *(unsigned int *)(a1 + 2288), 8);
  v3 = *(unsigned int *)(a1 + 2256);
  if ( (_DWORD)v3 )
  {
    v4 = *(_BYTE **)(a1 + 2240);
    v34[0] = 21;
    v34[2] = 0;
    v35[0] = 22;
    v5 = &v4[48 * v3];
    v35[2] = 0;
    do
    {
      while ( (unsigned __int8)(*v4 - 21) <= 1u
           || (unsigned __int8)sub_2EAB6C0((__int64)v4, (char *)v34)
           || (unsigned __int8)(*v4 - 21) <= 1u )
      {
        v4 += 48;
        if ( v5 == v4 )
          goto LABEL_10;
      }
      v6 = (__int64)v4;
      v4 += 48;
      sub_2EAB6C0(v6, (char *)v35);
    }
    while ( v5 != v4 );
LABEL_10:
    v3 = *(unsigned int *)(a1 + 2256);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 2240), 48 * v3, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 2208), 16LL * *(unsigned int *)(a1 + 2224), 8);
  v7 = *(_QWORD *)(a1 + 2184);
  if ( v7 != a1 + 2200 )
    _libc_free(v7);
  v8 = *(_QWORD *)(a1 + 2168);
  if ( a1 + 2184 != v8 )
    _libc_free(v8);
  sub_C7D6A0(*(_QWORD *)(a1 + 2144), 32LL * *(unsigned int *)(a1 + 2160), 8);
  v9 = *(unsigned int *)(a1 + 2128);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD **)(a1 + 2112);
    v11 = &v10[17 * v9];
    do
    {
      if ( *v10 != -4096 && *v10 != -8192 )
      {
        v12 = v10[13];
        while ( v12 )
        {
          sub_37B75D0(*(_QWORD *)(v12 + 24));
          v13 = v12;
          v12 = *(_QWORD *)(v12 + 16);
          j_j___libc_free_0(v13);
        }
        v14 = v10[1];
        if ( (_QWORD *)v14 != v10 + 3 )
          _libc_free(v14);
      }
      v10 += 17;
    }
    while ( v11 != v10 );
    v9 = *(unsigned int *)(a1 + 2128);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 2112), 136 * v9, 8);
  v15 = *(unsigned int *)(a1 + 2096);
  if ( (_DWORD)v15 )
  {
    v16 = *(_QWORD **)(a1 + 2080);
    v17 = &v16[7 * v15];
    while ( 1 )
    {
      while ( *v16 == -4096 )
      {
        if ( v16[1] != -1 || v16[2] != -1 )
          goto LABEL_28;
        v16 += 7;
        if ( v17 == v16 )
        {
LABEL_35:
          v15 = *(unsigned int *)(a1 + 2096);
          goto LABEL_36;
        }
      }
      if ( *v16 != -8192 || v16[1] != -2 || v16[2] != -2 )
      {
LABEL_28:
        v18 = v16[3];
        if ( (_QWORD *)v18 != v16 + 5 )
          _libc_free(v18);
      }
      v16 += 7;
      if ( v17 == v16 )
        goto LABEL_35;
    }
  }
LABEL_36:
  sub_C7D6A0(*(_QWORD *)(a1 + 2080), 56 * v15, 8);
  v19 = *(_QWORD *)(a1 + 776);
  if ( v19 != a1 + 792 )
    _libc_free(v19);
  sub_37B7D10(*(_QWORD *)(a1 + 744));
  sub_C7D6A0(*(_QWORD *)(a1 + 704), 8LL * *(unsigned int *)(a1 + 720), 4);
  sub_C7D6A0(*(_QWORD *)(a1 + 672), 16LL * *(unsigned int *)(a1 + 688), 8);
  v20 = *(_QWORD *)(a1 + 600);
  if ( v20 != a1 + 616 )
    _libc_free(v20);
  if ( !*(_BYTE *)(a1 + 468) )
    _libc_free(*(_QWORD *)(a1 + 448));
  v21 = *(unsigned int *)(a1 + 384);
  if ( (_DWORD)v21 )
  {
    v22 = *(_QWORD **)(a1 + 368);
    v23 = &v22[2 * v21];
    do
    {
      if ( *v22 != -8192 && *v22 != -4096 )
      {
        v24 = v22[1];
        if ( v24 )
        {
          if ( !*(_BYTE *)(v24 + 28) )
            _libc_free(*(_QWORD *)(v24 + 8));
          j_j___libc_free_0(v24);
        }
      }
      v22 += 2;
    }
    while ( v23 != v22 );
    v21 = *(unsigned int *)(a1 + 384);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 368), 16 * v21, 8);
  v25 = *(_QWORD *)(a1 + 304);
  if ( v25 != a1 + 320 )
    _libc_free(v25);
  sub_2DFA900(a1 + 248);
  v26 = *(_QWORD *)(a1 + 248);
  if ( v26 != a1 + 296 )
    j_j___libc_free_0(v26);
  v27 = *(_QWORD **)(a1 + 208);
  while ( v27 )
  {
    v28 = (unsigned __int64)v27;
    v27 = (_QWORD *)*v27;
    v29 = *(_QWORD *)(v28 + 104);
    if ( v29 != v28 + 120 )
      _libc_free(v29);
    v30 = *(_QWORD *)(v28 + 56);
    if ( v30 != v28 + 72 )
      _libc_free(v30);
    j_j___libc_free_0(v28);
  }
  memset(*(void **)(a1 + 192), 0, 8LL * *(_QWORD *)(a1 + 200));
  v31 = *(_QWORD *)(a1 + 192);
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  if ( v31 != a1 + 240 )
    j_j___libc_free_0(v31);
  sub_2DFA900(a1 + 136);
  v32 = *(_QWORD *)(a1 + 136);
  if ( v32 != a1 + 184 )
    j_j___libc_free_0(v32);
  v33 = *(_QWORD *)(a1 + 56);
  if ( v33 != a1 + 72 )
    _libc_free(v33);
}
