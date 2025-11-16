// Function: sub_23BBF00
// Address: 0x23bbf00
//
__int64 *__fastcall sub_23BBF00(__int64 a1, const void *a2, size_t a3, int a4, const __m128i *a5)
{
  unsigned int v7; // r8d
  __int64 *v8; // r9
  __int64 *result; // rax
  __int64 v10; // rax
  unsigned int v11; // r8d
  __int64 *v12; // r9
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rax
  const __m128i *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 *v23; // [rsp+0h] [rbp-40h]
  unsigned int v24; // [rsp+Ch] [rbp-34h]

  v7 = sub_C92740(a1, a2, a3, a4);
  v8 = (__int64 *)(*(_QWORD *)a1 + 8LL * v7);
  if ( *v8 )
  {
    if ( *v8 != -8 )
      return (__int64 *)(*(_QWORD *)a1 + 8LL * v7);
    --*(_DWORD *)(a1 + 16);
  }
  v23 = v8;
  v24 = v7;
  v10 = sub_C7D670(a3 + 89, 8);
  v11 = v24;
  v12 = v23;
  v13 = v10;
  if ( a3 )
  {
    memcpy((void *)(v10 + 88), a2, a3);
    v11 = v24;
    v12 = v23;
  }
  v14 = a5[1].m128i_i64[1];
  v15 = a5[1].m128i_i64[0];
  *(_BYTE *)(v13 + a3 + 88) = 0;
  v16 = a5->m128i_i64[1];
  v17 = a5->m128i_i64[0];
  *(_QWORD *)v13 = a3;
  *(_QWORD *)(v13 + 32) = v14;
  v18 = a5[2].m128i_i64[0];
  *(_QWORD *)(v13 + 24) = v15;
  v19 = (const __m128i *)a5[3].m128i_i64[0];
  *(_QWORD *)(v13 + 40) = v18;
  v20 = a5[2].m128i_i64[1];
  *(_QWORD *)(v13 + 8) = v17;
  *(_QWORD *)(v13 + 48) = v20;
  *(_QWORD *)(v13 + 56) = v13 + 72;
  *(_QWORD *)(v13 + 16) = v16;
  a5->m128i_i64[0] = 0;
  a5->m128i_i64[1] = 0;
  a5[1].m128i_i64[0] = 0;
  a5[1].m128i_i64[1] = 0;
  a5[2].m128i_i64[0] = 0;
  a5[2].m128i_i32[2] = 0;
  if ( v19 == &a5[4] )
  {
    *(__m128i *)(v13 + 72) = _mm_loadu_si128(a5 + 4);
  }
  else
  {
    *(_QWORD *)(v13 + 56) = v19;
    *(_QWORD *)(v13 + 72) = a5[4].m128i_i64[0];
  }
  v21 = a5[3].m128i_i64[1];
  a5[3].m128i_i64[0] = (__int64)a5[4].m128i_i64;
  a5[3].m128i_i64[1] = 0;
  *(_QWORD *)(v13 + 64) = v21;
  a5[4].m128i_i8[0] = 0;
  *v12 = v13;
  ++*(_DWORD *)(a1 + 12);
  result = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v11));
  v22 = *result;
  if ( *result )
    goto LABEL_11;
  do
  {
    do
    {
      v22 = result[1];
      ++result;
    }
    while ( !v22 );
LABEL_11:
    ;
  }
  while ( v22 == -8 );
  return result;
}
