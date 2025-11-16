// Function: sub_3059A40
// Address: 0x3059a40
//
__int64 __fastcall sub_3059A40(__int64 a1)
{
  _QWORD *v2; // rdi
  __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int64 *v8; // r14
  __int64 *v9; // rbx
  __int64 i; // rdx
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 *v13; // rbx
  unsigned __int64 v14; // r13
  __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi

  v2 = (_QWORD *)(a1 + 538000);
  *(v2 - 67250) = &unk_4A305E8;
  *v2 = &unk_4A3B530;
  nullsub_1681();
  v3 = *(_QWORD *)(a1 + 537992);
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  v4 = *(_QWORD *)(a1 + 526216);
  *(_QWORD *)(a1 + 960) = &unk_4A2CC60;
  sub_3059870(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 992), 8LL * *(unsigned int *)(a1 + 1008), 4);
  v5 = *(_QWORD *)(a1 + 936);
  v6 = 16LL * *(unsigned int *)(a1 + 952);
  *(_QWORD *)(a1 + 376) = &unk_4A3B778;
  *(_QWORD *)(a1 + 456) = &unk_4A2F570;
  sub_C7D6A0(v5, v6, 8);
  v7 = 16LL * *(unsigned int *)(a1 + 920);
  sub_C7D6A0(*(_QWORD *)(a1 + 904), v7, 8);
  v8 = *(__int64 **)(a1 + 808);
  v9 = &v8[*(unsigned int *)(a1 + 816)];
  if ( v8 != v9 )
  {
    for ( i = *(_QWORD *)(a1 + 808); ; i = *(_QWORD *)(a1 + 808) )
    {
      v11 = *v8;
      v12 = (unsigned int)(((__int64)v8 - i) >> 3) >> 7;
      v7 = 4096LL << v12;
      if ( v12 >= 0x1E )
        v7 = 0x40000000000LL;
      ++v8;
      sub_C7D6A0(v11, v7, 16);
      if ( v9 == v8 )
        break;
    }
  }
  v13 = *(__int64 **)(a1 + 856);
  v14 = (unsigned __int64)&v13[2 * *(unsigned int *)(a1 + 864)];
  if ( v13 != (__int64 *)v14 )
  {
    do
    {
      v7 = v13[1];
      v15 = *v13;
      v13 += 2;
      sub_C7D6A0(v15, v7, 16);
    }
    while ( (__int64 *)v14 != v13 );
    v14 = *(_QWORD *)(a1 + 856);
  }
  if ( v14 != a1 + 872 )
    _libc_free(v14);
  v16 = *(_QWORD *)(a1 + 808);
  if ( v16 != a1 + 824 )
    _libc_free(v16);
  *(_QWORD *)(a1 + 456) = &unk_4A2F290;
  sub_2FF61F0(a1 + 456);
  *(_QWORD *)(a1 + 376) = &unk_4A2FD98;
  sub_2FDD4A0((_QWORD *)(a1 + 376));
  v17 = *(_QWORD *)(a1 + 304);
  if ( v17 != a1 + 320 )
  {
    v7 = *(_QWORD *)(a1 + 320) + 1LL;
    j_j___libc_free_0(v17);
  }
  *(_QWORD *)a1 = &unk_4A303E0;
  return sub_35DE000(a1, v7);
}
