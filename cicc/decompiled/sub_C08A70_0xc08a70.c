// Function: sub_C08A70
// Address: 0xc08a70
//
__int64 (__fastcall *__fastcall sub_C08A70(__int64 a1, __int64 a2))(_QWORD *, _QWORD *, __int64)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 *v6; // r14
  __int64 *v7; // r13
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rdi
  void (__fastcall *v14)(__int64, __int64, __int64); // rax
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rsi
  _QWORD *v19; // r12
  __int64 v20; // rsi
  _QWORD *v21; // r13
  __int64 v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // r15
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rsi
  __int64 v29; // r14
  __int64 v30; // r12
  __int64 v31; // r13
  __int64 v32; // rdi
  __int64 v33; // rdi

  v3 = a1 + 2288;
  v4 = *(_QWORD *)(a1 + 2272);
  if ( v4 != v3 )
    _libc_free(v4, a2);
  v5 = 16LL * *(unsigned int *)(a1 + 2256);
  sub_C7D6A0(*(_QWORD *)(a1 + 2240), v5, 8);
  v6 = *(__int64 **)(a1 + 2200);
  v7 = *(__int64 **)(a1 + 2192);
  if ( v6 != v7 )
  {
    do
    {
      v8 = *v7;
      if ( *v7 )
      {
        v9 = *(_QWORD *)(v8 + 176);
        if ( v9 != v8 + 192 )
          _libc_free(v9, v5);
        v10 = *(_QWORD *)(v8 + 88);
        if ( v10 != v8 + 104 )
          _libc_free(v10, v5);
        sub_C7D6A0(*(_QWORD *)(v8 + 64), 8LL * *(unsigned int *)(v8 + 80), 8);
        v11 = *(_QWORD *)(v8 + 40);
        sub_BF0670(*(__int64 **)(v8 + 32), v11);
        v12 = *(_QWORD *)(v8 + 32);
        if ( v12 )
        {
          v11 = *(_QWORD *)(v8 + 48) - v12;
          j_j___libc_free_0(v12, v11);
        }
        v13 = *(_QWORD *)(v8 + 8);
        if ( v13 != v8 + 24 )
          _libc_free(v13, v11);
        v5 = 224;
        j_j___libc_free_0(v8, 224);
      }
      ++v7;
    }
    while ( v6 != v7 );
    v7 = *(__int64 **)(a1 + 2192);
  }
  if ( v7 )
    j_j___libc_free_0(v7, *(_QWORD *)(a1 + 2208) - (_QWORD)v7);
  sub_C7D6A0(*(_QWORD *)(a1 + 2168), 16LL * *(unsigned int *)(a1 + 2184), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 2136), 16LL * *(unsigned int *)(a1 + 2152), 8);
  v14 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 2096);
  if ( v14 )
    v14(a1 + 2080, a1 + 2080, 3);
  sub_C7D6A0(*(_QWORD *)(a1 + 2048), 16LL * *(unsigned int *)(a1 + 2064), 8);
  v15 = 16LL * *(unsigned int *)(a1 + 2032);
  sub_C7D6A0(*(_QWORD *)(a1 + 2016), v15, 8);
  v16 = *(_QWORD *)(a1 + 1856);
  if ( v16 != a1 + 1872 )
    _libc_free(v16, v15);
  if ( !*(_BYTE *)(a1 + 1596) )
    _libc_free(*(_QWORD *)(a1 + 1576), v15);
  if ( !*(_BYTE *)(a1 + 1308) )
    _libc_free(*(_QWORD *)(a1 + 1288), v15);
  v17 = *(_QWORD *)(a1 + 1232);
  if ( v17 != a1 + 1248 )
    _libc_free(v17, v15);
  if ( !*(_BYTE *)(a1 + 972) )
    _libc_free(*(_QWORD *)(a1 + 952), v15);
  v18 = *(unsigned int *)(a1 + 936);
  if ( (_DWORD)v18 )
  {
    v19 = *(_QWORD **)(a1 + 920);
    v20 = 2 * v18;
    v21 = &v19[v20];
    do
    {
      if ( *v19 != -8192 && *v19 != -4096 )
      {
        v22 = v19[1];
        if ( v22 )
        {
          if ( (v22 & 4) != 0 )
          {
            v23 = (_QWORD *)(v22 & 0xFFFFFFFFFFFFFFF8LL);
            v24 = v23;
            if ( v23 )
            {
              if ( (_QWORD *)*v23 != v23 + 2 )
                _libc_free(*v23, v20 * 8);
              v20 = 6;
              j_j___libc_free_0(v24, 48);
            }
          }
        }
      }
      v19 += 2;
    }
    while ( v21 != v19 );
    v18 = *(unsigned int *)(a1 + 936);
  }
  v25 = 16 * v18;
  sub_C7D6A0(*(_QWORD *)(a1 + 920), v25, 8);
  v26 = *(_QWORD *)(a1 + 896);
  if ( a1 + 912 != v26 )
    _libc_free(v26, v25);
  sub_C7D6A0(*(_QWORD *)(a1 + 872), 16LL * *(unsigned int *)(a1 + 888), 8);
  v27 = 16LL * *(unsigned int *)(a1 + 856);
  sub_C7D6A0(*(_QWORD *)(a1 + 840), v27, 8);
  if ( !*(_BYTE *)(a1 + 796) )
    _libc_free(*(_QWORD *)(a1 + 776), v27);
  v28 = 16LL * *(unsigned int *)(a1 + 760);
  sub_C7D6A0(*(_QWORD *)(a1 + 744), v28, 8);
  if ( !*(_BYTE *)(a1 + 476) )
    _libc_free(*(_QWORD *)(a1 + 456), v28);
  if ( !*(_BYTE *)(a1 + 316) )
    _libc_free(*(_QWORD *)(a1 + 296), v28);
  v29 = *(_QWORD *)(a1 + 184);
  v30 = v29 + 8LL * *(unsigned int *)(a1 + 192);
  if ( v29 != v30 )
  {
    do
    {
      v31 = *(_QWORD *)(v30 - 8);
      v30 -= 8;
      if ( v31 )
      {
        v32 = *(_QWORD *)(v31 + 24);
        if ( v32 != v31 + 40 )
          _libc_free(v32, v28);
        v28 = 80;
        j_j___libc_free_0(v31, 80);
      }
    }
    while ( v29 != v30 );
    v30 = *(_QWORD *)(a1 + 184);
  }
  if ( v30 != a1 + 200 )
    _libc_free(v30, v28);
  v33 = *(_QWORD *)(a1 + 160);
  if ( v33 != a1 + 176 )
    _libc_free(v33, v28);
  return sub_A55520((_QWORD *)(a1 + 16), v28);
}
