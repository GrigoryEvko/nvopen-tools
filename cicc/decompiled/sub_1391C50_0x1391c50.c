// Function: sub_1391C50
// Address: 0x1391c50
//
__m128i *__fastcall sub_1391C50(__int64 a1, __int64 a2, __int64 a3, __m128i *a4)
{
  unsigned __int8 v8; // al
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdi
  char v12; // al
  __int64 v13; // rdx
  __m128i *result; // rax
  __int64 v15; // rcx
  int v16; // esi
  unsigned int v17; // edi
  __m128i *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rdi
  unsigned int v21; // r8d
  __m128i *v22; // rdx
  __int64 v23; // r9
  __int64 v24; // r12
  __int64 v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rax
  int v29; // edx
  int v30; // edx
  int v31; // r10d
  int v32; // r9d
  __m128i v33; // [rsp+0h] [rbp-50h] BYREF
  __m128i *v34; // [rsp+10h] [rbp-40h]

  v8 = *(_BYTE *)(a3 + 16);
  v9 = *(_QWORD *)(a1 + 24);
  if ( v8 > 3u )
  {
    if ( v8 == 5 )
    {
      if ( (unsigned int)*(unsigned __int16 *)(a3 + 18) - 51 <= 1 )
        goto LABEL_4;
      if ( (unsigned __int8)sub_13848E0(*(_QWORD *)(a1 + 24), a3, 0, 0) )
      {
        sub_1391610(a1, a3, v27);
        v9 = *(_QWORD *)(a1 + 24);
        goto LABEL_4;
      }
    }
    else
    {
      sub_13848E0(*(_QWORD *)(a1 + 24), a3, 0, 0);
    }
    v9 = *(_QWORD *)(a1 + 24);
  }
  else
  {
    v10 = sub_14C81A0(a3);
    v11 = v9;
    v12 = sub_13848E0(v9, a3, 0, v10);
    v9 = *(_QWORD *)(a1 + 24);
    if ( v12 )
    {
      v28 = sub_14C8160(v11, a3, v13);
      sub_13848E0(v9, a3, 1u, v28);
      v9 = *(_QWORD *)(a1 + 24);
    }
  }
LABEL_4:
  result = (__m128i *)*(unsigned int *)(v9 + 24);
  v15 = *(_QWORD *)(v9 + 8);
  if ( !(_DWORD)result )
  {
    v20 = 0;
    goto LABEL_26;
  }
  v16 = (_DWORD)result - 1;
  v17 = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v18 = (__m128i *)(v15 + 32LL * v17);
  v19 = v18->m128i_i64[0];
  if ( a2 != v18->m128i_i64[0] )
  {
    v29 = 1;
    while ( v19 != -8 )
    {
      v32 = v29 + 1;
      v17 = v16 & (v29 + v17);
      v18 = (__m128i *)(v15 + 32LL * v17);
      v19 = v18->m128i_i64[0];
      if ( a2 == v18->m128i_i64[0] )
        goto LABEL_6;
      v29 = v32;
    }
    result = (__m128i *)(v15 + 32LL * (_QWORD)result);
    goto LABEL_34;
  }
LABEL_6:
  result = (__m128i *)(v15 + 32LL * (_QWORD)result);
  if ( result == v18 )
  {
LABEL_34:
    v20 = 0;
    goto LABEL_9;
  }
  v20 = v18->m128i_i64[1];
  if ( !(-1227133513 * (unsigned int)((v18[1].m128i_i64[0] - v20) >> 3)) )
    v20 = 0;
LABEL_9:
  v21 = v16 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v22 = (__m128i *)(v15 + 32LL * v21);
  v23 = v22->m128i_i64[0];
  if ( a3 != v22->m128i_i64[0] )
  {
    v30 = 1;
    while ( v23 != -8 )
    {
      v31 = v30 + 1;
      v21 = v16 & (v30 + v21);
      v22 = (__m128i *)(v15 + 32LL * v21);
      v23 = v22->m128i_i64[0];
      if ( a3 == v22->m128i_i64[0] )
        goto LABEL_10;
      v30 = v31;
    }
    goto LABEL_26;
  }
LABEL_10:
  if ( result == v22 )
  {
LABEL_26:
    v24 = 0;
    goto LABEL_13;
  }
  v24 = v22->m128i_i64[1];
  result = 0;
  if ( !(-1227133513 * (unsigned int)((v22[1].m128i_i64[0] - v24) >> 3)) )
    v24 = 0;
LABEL_13:
  v33.m128i_i64[0] = a3;
  v33.m128i_i32[2] = 0;
  v34 = a4;
  v25 = *(_QWORD *)(v20 + 8);
  if ( v25 == *(_QWORD *)(v20 + 16) )
  {
    result = sub_1384280(v20, (_BYTE *)v25, &v33);
  }
  else
  {
    if ( v25 )
    {
      *(__m128i *)v25 = _mm_loadu_si128(&v33);
      result = v34;
      *(_QWORD *)(v25 + 16) = v34;
      v25 = *(_QWORD *)(v20 + 8);
    }
    *(_QWORD *)(v20 + 8) = v25 + 24;
  }
  v33.m128i_i64[0] = a2;
  v33.m128i_i32[2] = 0;
  v34 = a4;
  v26 = *(_QWORD *)(v24 + 32);
  if ( v26 == *(_QWORD *)(v24 + 40) )
    return sub_1384280(v24 + 24, (_BYTE *)v26, &v33);
  if ( v26 )
  {
    *(__m128i *)v26 = _mm_loadu_si128(&v33);
    result = v34;
    *(_QWORD *)(v26 + 16) = v34;
    v26 = *(_QWORD *)(v24 + 32);
  }
  *(_QWORD *)(v24 + 32) = v26 + 24;
  return result;
}
