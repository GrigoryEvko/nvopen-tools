// Function: sub_2E20690
// Address: 0x2e20690
//
__int64 __fastcall sub_2E20690(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6)
{
  unsigned __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rbx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // r12
  __int64 *v14; // r15
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 *i; // rax
  __int64 v18; // rdi
  unsigned int v19; // ecx
  __int64 v20; // rsi
  __int64 *v21; // rbx
  unsigned __int64 v22; // r14
  __int64 v23; // rsi
  __int64 v24; // rdi
  unsigned __int64 v25; // rdi

  *a1 = &unk_4A284E0;
  v7 = a1[34];
  if ( (_QWORD *)v7 != a1 + 36 )
    _libc_free(v7);
  v8 = a1[32];
  if ( v8 )
  {
    v9 = 176LL * *(_QWORD *)(v8 - 8);
    v10 = v8 + v9;
    if ( v8 != v8 + v9 )
    {
      do
      {
        v10 -= 176;
        v11 = *(_QWORD *)(v10 + 112);
        if ( v11 != v10 + 128 )
          _libc_free(v11);
        v12 = *(_QWORD *)(v10 + 32);
        if ( v12 != v10 + 48 )
          _libc_free(v12);
      }
      while ( v8 != v10 );
      v9 = 176LL * *(_QWORD *)(v8 - 8);
    }
    a2 = v9 + 8;
    j_j_j___libc_free_0_0(v8 - 8);
  }
  sub_2E19FF0((__int64)(a1 + 30), a2, a3, a4, a5, a6);
  v13 = a1[29];
  if ( v13 )
  {
    v14 = *(__int64 **)(v13 + 24);
    v15 = *(unsigned int *)(v13 + 32);
    *(_QWORD *)v13 = 0;
    v16 = &v14[v15];
    if ( v14 != v16 )
    {
      for ( i = v14; ; i = *(__int64 **)(v13 + 24) )
      {
        v18 = *v14;
        v19 = (unsigned int)(v14 - i) >> 7;
        v20 = 4096LL << v19;
        if ( v19 >= 0x1E )
          v20 = 0x40000000000LL;
        ++v14;
        sub_C7D6A0(v18, v20, 16);
        if ( v16 == v14 )
          break;
      }
    }
    v21 = *(__int64 **)(v13 + 72);
    v22 = (unsigned __int64)&v21[2 * *(unsigned int *)(v13 + 80)];
    if ( v21 != (__int64 *)v22 )
    {
      do
      {
        v23 = v21[1];
        v24 = *v21;
        v21 += 2;
        sub_C7D6A0(v24, v23, 16);
      }
      while ( (__int64 *)v22 != v21 );
      v22 = *(_QWORD *)(v13 + 72);
    }
    if ( v22 != v13 + 88 )
      _libc_free(v22);
    v25 = *(_QWORD *)(v13 + 24);
    if ( v25 != v13 + 40 )
      _libc_free(v25);
    j_j___libc_free_0(v13);
  }
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
