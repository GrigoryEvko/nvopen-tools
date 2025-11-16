// Function: sub_23BB630
// Address: 0x23bb630
//
__int64 __fastcall sub_23BB630(__int64 a1, const void *a2, size_t a3, int a4, __m128i *a5)
{
  unsigned int v7; // r8d
  __int64 *v8; // rcx
  __int64 v10; // rax
  unsigned int v11; // r8d
  __int64 *v12; // rcx
  __int64 v13; // r13
  __m128i *v14; // rdx
  __int64 v15; // rdx
  __m128i *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rcx
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // [rsp+0h] [rbp-40h]
  unsigned int v25; // [rsp+Ch] [rbp-34h]

  v7 = sub_C92740(a1, a2, a3, a4);
  v8 = (__int64 *)(*(_QWORD *)a1 + 8LL * v7);
  if ( *v8 )
  {
    if ( *v8 != -8 )
      return *(_QWORD *)a1 + 8LL * v7;
    --*(_DWORD *)(a1 + 16);
  }
  v24 = v8;
  v25 = v7;
  v10 = sub_C7D670(a3 + 97, 8);
  v11 = v25;
  v12 = v24;
  v13 = v10;
  if ( a3 )
  {
    memcpy((void *)(v10 + 96), a2, a3);
    v11 = v25;
    v12 = v24;
  }
  v14 = (__m128i *)a5->m128i_i64[0];
  *(_BYTE *)(v13 + a3 + 96) = 0;
  *(_QWORD *)(v13 + 8) = v13 + 24;
  *(_QWORD *)v13 = a3;
  if ( v14 == &a5[1] )
  {
    *(__m128i *)(v13 + 24) = _mm_loadu_si128(a5 + 1);
  }
  else
  {
    *(_QWORD *)(v13 + 8) = v14;
    *(_QWORD *)(v13 + 24) = a5[1].m128i_i64[0];
  }
  v15 = a5->m128i_i64[1];
  a5->m128i_i64[0] = (__int64)a5[1].m128i_i64;
  *(_QWORD *)(v13 + 40) = v13 + 56;
  *(_QWORD *)(v13 + 16) = v15;
  v16 = (__m128i *)a5[2].m128i_i64[0];
  a5->m128i_i64[1] = 0;
  a5[1].m128i_i8[0] = 0;
  if ( v16 == &a5[3] )
  {
    *(__m128i *)(v13 + 56) = _mm_loadu_si128(a5 + 3);
  }
  else
  {
    *(_QWORD *)(v13 + 40) = v16;
    *(_QWORD *)(v13 + 56) = a5[3].m128i_i64[0];
  }
  a5[2].m128i_i64[0] = (__int64)a5[3].m128i_i64;
  v17 = a5[4].m128i_i64[0];
  v18 = a5[2].m128i_i64[1];
  a5[3].m128i_i8[0] = 0;
  *(_QWORD *)(v13 + 72) = v17;
  v19 = a5[4].m128i_i64[1];
  *(_QWORD *)(v13 + 48) = v18;
  *(_QWORD *)(v13 + 80) = v19;
  v20 = a5[5].m128i_i64[0];
  a5[2].m128i_i64[1] = 0;
  a5[4].m128i_i64[0] = 0;
  a5[4].m128i_i64[1] = 0;
  a5[5].m128i_i32[0] = 0;
  *(_QWORD *)(v13 + 88) = v20;
  *v12 = v13;
  ++*(_DWORD *)(a1 + 12);
  v21 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v11));
  if ( !*v21 || *v21 == -8 )
  {
    v22 = v21 + 1;
    do
    {
      do
      {
        v23 = *v22;
        v21 = v22++;
      }
      while ( !v23 );
    }
    while ( v23 == -8 );
  }
  return (__int64)v21;
}
