// Function: sub_307A1F0
// Address: 0x307a1f0
//
__int64 __fastcall sub_307A1F0(__int64 a1, unsigned __int8 *a2, unsigned __int8 **a3, __int64 a4, unsigned int a5)
{
  unsigned __int8 v5; // al
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int64 v9; // rsi
  const __m128i *v10; // r8
  const __m128i *v11; // r12
  const __m128i *v12; // r15
  unsigned int v13; // ebx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  __int64 v18; // r12
  __int64 v19; // rax
  unsigned int v20; // ebx
  __int64 v21; // rax
  unsigned int v22; // ebx
  __m128i v23; // [rsp-98h] [rbp-98h] BYREF
  const __m128i *v24; // [rsp-88h] [rbp-88h] BYREF
  __int64 v25; // [rsp-80h] [rbp-80h]
  _BYTE v26[120]; // [rsp-78h] [rbp-78h] BYREF

  v5 = *a2;
  if ( *a2 <= 0x1Cu )
    return sub_30783B0((__int64 *)(a1 + 8), a2, a3, a4, a5);
  if ( a5 == 1 && v5 == 61 )
  {
    v19 = *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL);
    if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
      v19 = **(_QWORD **)(v19 + 16);
    v20 = *(_DWORD *)(v19 + 8);
    v21 = sub_30783B0((__int64 *)(a1 + 8), a2, a3, a4, 1u);
    v22 = v20 >> 8;
    v18 = v21;
    if ( v22 != 5 )
    {
      if ( v22 > 5 )
      {
        if ( v22 == 101 )
          return v18;
        goto LABEL_38;
      }
      if ( v22 > 1 )
      {
        if ( ((v22 + 16777213) & 0xFFFFFF) <= 1 )
          return v18;
LABEL_38:
        BUG();
      }
    }
    if ( is_mul_ok(2u, v21) )
      return 2 * v21;
    v18 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v21 <= 0 )
      return 0x8000000000000000LL;
    return v18;
  }
  if ( v5 != 85 )
    return sub_30783B0((__int64 *)(a1 + 8), a2, a3, a4, a5);
  v7 = *((_QWORD *)a2 - 4);
  if ( *(_BYTE *)v7 != 25 )
    return sub_30783B0((__int64 *)(a1 + 8), a2, a3, a4, a5);
  v8 = *(_QWORD *)(v7 + 24);
  v9 = *(_QWORD *)(v7 + 32);
  v24 = (const __m128i *)v26;
  v25 = 0x400000000LL;
  sub_C92330(v8, v9, (__int64)&v24, (__int64)";\n", 2);
  v10 = v24;
  v11 = &v24[(unsigned int)v25];
  if ( v11 == v24 )
  {
    v18 = 0;
  }
  else
  {
    v12 = v24;
    v13 = 0;
    do
    {
      v23 = _mm_loadu_si128(v12);
      v14 = sub_C93580(&v23, 32, 0);
      if ( v14 != -1 )
      {
        v15 = v23.m128i_u64[1];
        v16 = v23.m128i_i64[0];
        v17 = 0;
        if ( v14 <= v23.m128i_i64[1] )
        {
          v17 = v23.m128i_i64[1] - v14;
          v15 = v14;
        }
        v23.m128i_i64[1] = v17;
        v23.m128i_i64[0] += v15;
        if ( *(_BYTE *)(v16 + v15) == 64
          || isalpha(*(char *)(v16 + v15))
          || sub_C931B0(v23.m128i_i64, ".pragma", 7u, 0) != -1 )
        {
          ++v13;
        }
      }
      ++v12;
    }
    while ( v11 != v12 );
    v10 = v24;
    v18 = v13;
  }
  if ( v10 != (const __m128i *)v26 )
    _libc_free((unsigned __int64)v10);
  return v18;
}
