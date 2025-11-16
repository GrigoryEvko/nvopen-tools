// Function: sub_3212B10
// Address: 0x3212b10
//
void __fastcall sub_3212B10(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rbx
  unsigned __int64 v4; // rdi
  __int64 v5; // r14
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rdi
  __int64 v8; // rsi
  _QWORD *v9; // rbx
  _QWORD *v10; // r12
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  _QWORD *v14; // rbx
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // rsi

  v2 = *(unsigned int *)(a1 + 520);
  v3 = a1 + 384;
  *(_QWORD *)a1 = &unk_4A35548;
  sub_C7D6A0(*(_QWORD *)(a1 + 504), 16 * v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 472), 16LL * *(unsigned int *)(a1 + 488), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 440), 16LL * *(unsigned int *)(a1 + 456), 8);
  v4 = *(_QWORD *)(a1 + 416);
  if ( v4 != a1 + 432 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 392), 24LL * *(unsigned int *)(a1 + 408), 8);
  v5 = *(_QWORD *)(a1 + 368);
  v6 = v5 + 96LL * *(unsigned int *)(a1 + 376);
  if ( v5 != v6 )
  {
    do
    {
      v6 -= 96LL;
      v7 = *(_QWORD *)(v6 + 16);
      if ( v7 != v6 + 32 )
        _libc_free(v7);
    }
    while ( v5 != v6 );
    v6 = *(_QWORD *)(a1 + 368);
  }
  if ( v6 != v3 )
    _libc_free(v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 344), 24LL * *(unsigned int *)(a1 + 360), 8);
  v8 = *(unsigned int *)(a1 + 328);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD **)(a1 + 312);
    v10 = &v9[2 * v8];
    do
    {
      if ( *v9 != -4096 && *v9 != -8192 )
      {
        v11 = v9[1];
        if ( v11 )
        {
          if ( !*(_BYTE *)(v11 + 28) )
            _libc_free(*(_QWORD *)(v11 + 8));
          j_j___libc_free_0(v11);
        }
      }
      v9 += 2;
    }
    while ( v10 != v9 );
    v8 = *(unsigned int *)(a1 + 328);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 312), 16 * v8, 8);
  v12 = *(_QWORD *)(a1 + 248);
  if ( v12 != a1 + 264 )
    _libc_free(v12);
  sub_2DFA900(a1 + 192);
  v13 = *(_QWORD *)(a1 + 192);
  if ( v13 != a1 + 240 )
    j_j___libc_free_0(v13);
  v14 = *(_QWORD **)(a1 + 152);
  while ( v14 )
  {
    v15 = (unsigned __int64)v14;
    v14 = (_QWORD *)*v14;
    v16 = *(_QWORD *)(v15 + 104);
    if ( v16 != v15 + 120 )
      _libc_free(v16);
    v17 = *(_QWORD *)(v15 + 56);
    if ( v17 != v15 + 72 )
      _libc_free(v17);
    j_j___libc_free_0(v15);
  }
  memset(*(void **)(a1 + 136), 0, 8LL * *(_QWORD *)(a1 + 144));
  v18 = *(_QWORD *)(a1 + 136);
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  if ( v18 != a1 + 184 )
    j_j___libc_free_0(v18);
  sub_2DFA900(a1 + 80);
  v19 = *(_QWORD *)(a1 + 80);
  if ( v19 != a1 + 128 )
    j_j___libc_free_0(v19);
  v20 = *(_QWORD *)(a1 + 24);
  if ( v20 )
    sub_B91220(a1 + 24, v20);
  nullsub_1849();
}
