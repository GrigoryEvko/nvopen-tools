// Function: sub_3186A70
// Address: 0x3186a70
//
__int64 __fastcall sub_3186A70(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // r13
  unsigned __int64 v4; // r12
  void (__fastcall *v5)(unsigned __int64, unsigned __int64, __int64); // rax
  __int64 v6; // r13
  unsigned __int64 v7; // r12
  void (__fastcall *v8)(unsigned __int64, unsigned __int64, __int64); // rax
  __int64 v9; // r13
  unsigned __int64 v10; // r12
  void (__fastcall *v11)(unsigned __int64, unsigned __int64, __int64); // rax
  __int64 v12; // r13
  unsigned __int64 v13; // r12
  void (__fastcall *v14)(unsigned __int64, unsigned __int64, __int64); // rax
  __int64 v15; // rsi
  _QWORD *v16; // r12
  _QWORD *v17; // r13
  unsigned __int64 v18; // rdi
  __int64 v19; // rsi
  _QWORD *v20; // r12
  _QWORD *v21; // r13
  unsigned __int64 v22; // rdi
  __int64 v23; // rsi
  _QWORD *v24; // r12
  _QWORD *v25; // r13
  __int64 v26; // rdi

  nullsub_61();
  *(_QWORD *)(a1 + 512) = &unk_49DA100;
  nullsub_63();
  v2 = *(_QWORD *)(a1 + 384);
  if ( v2 != a1 + 400 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 360);
  v4 = v3 + 40LL * *(unsigned int *)(a1 + 368);
  if ( v3 != v4 )
  {
    do
    {
      v5 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v4 - 16);
      v4 -= 40LL;
      if ( v5 )
        v5(v4 + 8, v4 + 8, 3);
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 360);
  }
  if ( v4 != a1 + 376 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 336), 16LL * *(unsigned int *)(a1 + 352), 8);
  v6 = *(_QWORD *)(a1 + 312);
  v7 = v6 + 40LL * *(unsigned int *)(a1 + 320);
  if ( v6 != v7 )
  {
    do
    {
      v8 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v7 - 16);
      v7 -= 40LL;
      if ( v8 )
        v8(v7 + 8, v7 + 8, 3);
    }
    while ( v6 != v7 );
    v7 = *(_QWORD *)(a1 + 312);
  }
  if ( a1 + 328 != v7 )
    _libc_free(v7);
  sub_C7D6A0(*(_QWORD *)(a1 + 288), 16LL * *(unsigned int *)(a1 + 304), 8);
  v9 = *(_QWORD *)(a1 + 264);
  v10 = v9 + 40LL * *(unsigned int *)(a1 + 272);
  if ( v9 != v10 )
  {
    do
    {
      v11 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v10 - 16);
      v10 -= 40LL;
      if ( v11 )
        v11(v10 + 8, v10 + 8, 3);
    }
    while ( v9 != v10 );
    v10 = *(_QWORD *)(a1 + 264);
  }
  if ( a1 + 280 != v10 )
    _libc_free(v10);
  sub_C7D6A0(*(_QWORD *)(a1 + 240), 16LL * *(unsigned int *)(a1 + 256), 8);
  v12 = *(_QWORD *)(a1 + 216);
  v13 = v12 + 40LL * *(unsigned int *)(a1 + 224);
  if ( v12 != v13 )
  {
    do
    {
      v14 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v13 - 16);
      v13 -= 40LL;
      if ( v14 )
        v14(v13 + 8, v13 + 8, 3);
    }
    while ( v12 != v13 );
    v13 = *(_QWORD *)(a1 + 216);
  }
  if ( a1 + 232 != v13 )
    _libc_free(v13);
  sub_C7D6A0(*(_QWORD *)(a1 + 192), 16LL * *(unsigned int *)(a1 + 208), 8);
  v15 = *(unsigned int *)(a1 + 176);
  if ( (_DWORD)v15 )
  {
    v16 = *(_QWORD **)(a1 + 160);
    v17 = &v16[2 * v15];
    do
    {
      if ( *v16 != -8192 && *v16 != -4096 )
      {
        v18 = v16[1];
        if ( v18 )
          j_j___libc_free_0(v18);
      }
      v16 += 2;
    }
    while ( v17 != v16 );
    v15 = *(unsigned int *)(a1 + 176);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 160), 16 * v15, 8);
  v19 = *(unsigned int *)(a1 + 144);
  if ( (_DWORD)v19 )
  {
    v20 = *(_QWORD **)(a1 + 128);
    v21 = &v20[2 * v19];
    do
    {
      if ( *v20 != -8192 && *v20 != -4096 )
      {
        v22 = v20[1];
        if ( v22 )
          j_j___libc_free_0(v22);
      }
      v20 += 2;
    }
    while ( v21 != v20 );
    v19 = *(unsigned int *)(a1 + 144);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 128), 16 * v19, 8);
  v23 = *(unsigned int *)(a1 + 112);
  if ( (_DWORD)v23 )
  {
    v24 = *(_QWORD **)(a1 + 96);
    v25 = &v24[2 * v23];
    do
    {
      if ( *v24 != -4096 && *v24 != -8192 )
      {
        v26 = v24[1];
        if ( v26 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
      }
      v24 += 2;
    }
    while ( v25 != v24 );
    v23 = *(unsigned int *)(a1 + 112);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 96), 16 * v23, 8);
  return sub_318DE50(a1 + 8);
}
