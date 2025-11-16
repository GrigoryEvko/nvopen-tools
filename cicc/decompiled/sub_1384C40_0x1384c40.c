// Function: sub_1384C40
// Address: 0x1384c40
//
__m128i *__fastcall sub_1384C40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __m128i *result; // rax
  int v8; // esi
  unsigned int v9; // edi
  __m128i *v10; // rdx
  __int64 v11; // r8
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rdi
  unsigned int v15; // r8d
  __m128i *v16; // rdx
  __int64 v17; // r9
  __int64 v18; // rbx
  __int64 v19; // rsi
  __int64 v20; // rsi
  int v21; // edx
  int v22; // r10d
  int v23; // edx
  int v24; // r9d
  __m128i v25; // [rsp+0h] [rbp-40h] BYREF
  __m128i *v26; // [rsp+10h] [rbp-30h]

  sub_13848E0(*(_QWORD *)(a1 + 24), a2, 1u, 0);
  v5 = *(_QWORD *)(a1 + 24);
  v6 = *(_QWORD *)(v5 + 8);
  result = (__m128i *)*(unsigned int *)(v5 + 24);
  if ( !(_DWORD)result )
  {
    v14 = 0;
    goto LABEL_19;
  }
  v8 = (_DWORD)result - 1;
  v9 = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__m128i *)(v6 + 32LL * v9);
  v11 = v10->m128i_i64[0];
  if ( a2 != v10->m128i_i64[0] )
  {
    v23 = 1;
    while ( v11 != -8 )
    {
      v24 = v23 + 1;
      v9 = v8 & (v23 + v9);
      v10 = (__m128i *)(v6 + 32LL * v9);
      v11 = v10->m128i_i64[0];
      if ( a2 == v10->m128i_i64[0] )
        goto LABEL_3;
      v23 = v24;
    }
    result = (__m128i *)(v6 + 32LL * (_QWORD)result);
    goto LABEL_29;
  }
LABEL_3:
  result = (__m128i *)(v6 + 32LL * (_QWORD)result);
  if ( v10 == result )
  {
LABEL_29:
    v14 = 0;
    goto LABEL_6;
  }
  v12 = v10->m128i_i64[1];
  v13 = v10[1].m128i_i64[0] - v12;
  v14 = v12 + 56;
  if ( -1227133513 * (unsigned int)(v13 >> 3) <= 1 )
    v14 = 0;
LABEL_6:
  v15 = v8 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v16 = (__m128i *)(v6 + 32LL * v15);
  v17 = v16->m128i_i64[0];
  if ( a3 != v16->m128i_i64[0] )
  {
    v21 = 1;
    while ( v17 != -8 )
    {
      v22 = v21 + 1;
      v15 = v8 & (v21 + v15);
      v16 = (__m128i *)(v6 + 32LL * v15);
      v17 = v16->m128i_i64[0];
      if ( a3 == v16->m128i_i64[0] )
        goto LABEL_7;
      v21 = v22;
    }
    goto LABEL_19;
  }
LABEL_7:
  if ( v16 == result )
  {
LABEL_19:
    v18 = 0;
    goto LABEL_10;
  }
  v18 = v16->m128i_i64[1];
  result = 0;
  if ( !(-1227133513 * (unsigned int)((v16[1].m128i_i64[0] - v18) >> 3)) )
    v18 = 0;
LABEL_10:
  v25.m128i_i64[0] = a3;
  v25.m128i_i32[2] = 0;
  v26 = 0;
  v19 = *(_QWORD *)(v14 + 8);
  if ( v19 == *(_QWORD *)(v14 + 16) )
  {
    result = sub_1384280(v14, (_BYTE *)v19, &v25);
  }
  else
  {
    if ( v19 )
    {
      *(__m128i *)v19 = _mm_loadu_si128(&v25);
      result = v26;
      *(_QWORD *)(v19 + 16) = v26;
      v19 = *(_QWORD *)(v14 + 8);
    }
    *(_QWORD *)(v14 + 8) = v19 + 24;
  }
  v25.m128i_i64[0] = a2;
  v25.m128i_i32[2] = 1;
  v26 = 0;
  v20 = *(_QWORD *)(v18 + 32);
  if ( v20 == *(_QWORD *)(v18 + 40) )
    return sub_1384280(v18 + 24, (_BYTE *)v20, &v25);
  if ( v20 )
  {
    *(__m128i *)v20 = _mm_loadu_si128(&v25);
    result = v26;
    *(_QWORD *)(v20 + 16) = v26;
    v20 = *(_QWORD *)(v18 + 32);
  }
  *(_QWORD *)(v18 + 32) = v20 + 24;
  return result;
}
