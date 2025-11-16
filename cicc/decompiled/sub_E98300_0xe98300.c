// Function: sub_E98300
// Address: 0xe98300
//
__m128i *__fastcall sub_E98300(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r14
  _QWORD *v7; // r13
  _QWORD *v8; // r12
  _QWORD *v9; // rbx
  _QWORD *v10; // r15
  _QWORD *v11; // rdi
  __int64 v12; // rdi
  __int64 *v13; // rcx
  __int64 *v14; // r14
  __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // r15
  __int64 v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rax
  int v27; // ecx
  __m128i v28; // xmm1
  __m128i *result; // rax
  __int64 *v30; // [rsp+0h] [rbp-70h]
  __int64 *v32; // [rsp+18h] [rbp-58h]
  __m128i v33; // [rsp+20h] [rbp-50h] BYREF
  __m128i v34; // [rsp+30h] [rbp-40h] BYREF

  v6 = *(_QWORD **)(a1 + 24);
  v7 = *(_QWORD **)(a1 + 32);
  if ( v6 != v7 )
  {
    v8 = *(_QWORD **)(a1 + 24);
    do
    {
      v9 = (_QWORD *)v8[5];
      v10 = (_QWORD *)v8[4];
      if ( v9 != v10 )
      {
        do
        {
          v11 = (_QWORD *)v10[9];
          if ( v11 != v10 + 11 )
          {
            a2 = v10[11] + 1LL;
            j_j___libc_free_0(v11, a2);
          }
          v12 = v10[6];
          if ( v12 )
          {
            a2 = v10[8] - v12;
            j_j___libc_free_0(v12, a2);
          }
          v10 += 13;
        }
        while ( v9 != v10 );
        v10 = (_QWORD *)v8[4];
      }
      if ( v10 )
      {
        a2 = v8[6] - (_QWORD)v10;
        j_j___libc_free_0(v10, a2);
      }
      v8 += 12;
    }
    while ( v7 != v8 );
    *(_QWORD *)(a1 + 32) = v6;
  }
  v13 = *(__int64 **)(a1 + 80);
  *(_QWORD *)(a1 + 104) = 0;
  v30 = v13;
  v32 = *(__int64 **)(a1 + 88);
  if ( v13 != v32 )
  {
    v14 = v13;
    do
    {
      v15 = *v14;
      if ( *v14 )
      {
        v16 = *(_QWORD *)(v15 + 168);
        v17 = *(_QWORD *)(v15 + 160);
        if ( v16 != v17 )
        {
          do
          {
            v18 = *(_QWORD *)(v17 + 64);
            v19 = v17 + 80;
            if ( v18 != v17 + 80 )
              _libc_free(v18, a2);
            v20 = *(unsigned int *)(v17 + 56);
            v21 = *(_QWORD *)(v17 + 40);
            v17 += 80;
            a2 = 16 * v20;
            sub_C7D6A0(v21, a2, 8);
          }
          while ( v16 != v19 );
          v17 = *(_QWORD *)(v15 + 160);
        }
        if ( v17 )
        {
          a2 = *(_QWORD *)(v15 + 176) - v17;
          j_j___libc_free_0(v17, a2);
        }
        v22 = *(_QWORD *)(v15 + 144);
        v23 = v22 + 48LL * *(unsigned int *)(v15 + 152);
        if ( v22 != v23 )
        {
          do
          {
            v24 = *(_QWORD *)(v23 - 40);
            v23 -= 48;
            if ( v24 )
            {
              a2 = *(_QWORD *)(v23 + 24) - v24;
              j_j___libc_free_0(v24, a2);
            }
          }
          while ( v22 != v23 );
          v23 = *(_QWORD *)(v15 + 144);
        }
        if ( v15 + 160 != v23 )
          _libc_free(v23, a2);
        sub_C7D6A0(*(_QWORD *)(v15 + 120), 16LL * *(unsigned int *)(v15 + 136), 8);
        v25 = *(_QWORD *)(v15 + 88);
        if ( v25 )
          j_j___libc_free_0(v25, *(_QWORD *)(v15 + 104) - v25);
        a2 = 184;
        j_j___libc_free_0(v15, 184);
      }
      ++v14;
    }
    while ( v32 != v14 );
    *(_QWORD *)(a1 + 88) = v30;
  }
  v33 = 0u;
  v26 = 0;
  v27 = *(_DWORD *)(a1 + 132);
  v34.m128i_i64[0] = 0;
  *(_DWORD *)(a1 + 128) = 0;
  v34.m128i_i64[1] = 0;
  if ( !v27 )
  {
    sub_C8D5F0(a1 + 120, (const void *)(a1 + 136), 1u, 0x20u, a5, a6);
    v26 = 32LL * *(unsigned int *)(a1 + 128);
  }
  v28 = _mm_loadu_si128(&v34);
  result = (__m128i *)(*(_QWORD *)(a1 + 120) + v26);
  *result = _mm_loadu_si128(&v33);
  result[1] = v28;
  *(_QWORD *)(a1 + 288) = 0;
  ++*(_DWORD *)(a1 + 128);
  return result;
}
