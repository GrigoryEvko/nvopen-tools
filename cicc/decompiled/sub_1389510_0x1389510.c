// Function: sub_1389510
// Address: 0x1389510
//
__m128i *__fastcall sub_1389510(__int64 a1, __int64 a2, __int64 a3, __m128i *a4)
{
  unsigned __int8 v7; // al
  __int64 v8; // r12
  __int64 v9; // rax
  char v10; // al
  __m128i *result; // rax
  __int64 v12; // rcx
  int v13; // esi
  unsigned int v14; // edi
  __m128i *v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rdi
  unsigned int v18; // r8d
  __m128i *v19; // rdx
  __int64 v20; // r9
  __int64 v21; // r12
  __int64 v22; // rsi
  __int64 v23; // rsi
  __int64 v24; // rax
  int v25; // edx
  int v26; // edx
  int v27; // r10d
  int v28; // r9d
  __m128i v29; // [rsp+0h] [rbp-50h] BYREF
  __m128i *v30; // [rsp+10h] [rbp-40h]

  v7 = *(_BYTE *)(a3 + 16);
  v8 = *(_QWORD *)(a1 + 24);
  if ( v7 > 3u )
  {
    if ( v7 == 5 )
    {
      if ( (unsigned int)*(unsigned __int16 *)(a3 + 18) - 51 <= 1 )
        goto LABEL_4;
      if ( (unsigned __int8)sub_13848E0(*(_QWORD *)(a1 + 24), a3, 0, 0) )
      {
        sub_1389140(a1, a3);
        v8 = *(_QWORD *)(a1 + 24);
        goto LABEL_4;
      }
    }
    else
    {
      sub_13848E0(*(_QWORD *)(a1 + 24), a3, 0, 0);
    }
    v8 = *(_QWORD *)(a1 + 24);
  }
  else
  {
    v9 = sub_14C81A0(a3);
    v10 = sub_13848E0(v8, a3, 0, v9);
    v8 = *(_QWORD *)(a1 + 24);
    if ( v10 )
    {
      v24 = sub_14C8160();
      sub_13848E0(v8, a3, 1u, v24);
      v8 = *(_QWORD *)(a1 + 24);
    }
  }
LABEL_4:
  result = (__m128i *)*(unsigned int *)(v8 + 24);
  v12 = *(_QWORD *)(v8 + 8);
  if ( !(_DWORD)result )
  {
    v17 = 0;
    goto LABEL_26;
  }
  v13 = (_DWORD)result - 1;
  v14 = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (__m128i *)(v12 + 32LL * v14);
  v16 = v15->m128i_i64[0];
  if ( a2 != v15->m128i_i64[0] )
  {
    v25 = 1;
    while ( v16 != -8 )
    {
      v28 = v25 + 1;
      v14 = v13 & (v25 + v14);
      v15 = (__m128i *)(v12 + 32LL * v14);
      v16 = v15->m128i_i64[0];
      if ( a2 == v15->m128i_i64[0] )
        goto LABEL_6;
      v25 = v28;
    }
    result = (__m128i *)(v12 + 32LL * (_QWORD)result);
    goto LABEL_34;
  }
LABEL_6:
  result = (__m128i *)(v12 + 32LL * (_QWORD)result);
  if ( result == v15 )
  {
LABEL_34:
    v17 = 0;
    goto LABEL_9;
  }
  v17 = v15->m128i_i64[1];
  if ( !(-1227133513 * (unsigned int)((v15[1].m128i_i64[0] - v17) >> 3)) )
    v17 = 0;
LABEL_9:
  v18 = v13 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v19 = (__m128i *)(v12 + 32LL * v18);
  v20 = v19->m128i_i64[0];
  if ( a3 != v19->m128i_i64[0] )
  {
    v26 = 1;
    while ( v20 != -8 )
    {
      v27 = v26 + 1;
      v18 = v13 & (v26 + v18);
      v19 = (__m128i *)(v12 + 32LL * v18);
      v20 = v19->m128i_i64[0];
      if ( a3 == v19->m128i_i64[0] )
        goto LABEL_10;
      v26 = v27;
    }
    goto LABEL_26;
  }
LABEL_10:
  if ( result == v19 )
  {
LABEL_26:
    v21 = 0;
    goto LABEL_13;
  }
  v21 = v19->m128i_i64[1];
  result = 0;
  if ( !(-1227133513 * (unsigned int)((v19[1].m128i_i64[0] - v21) >> 3)) )
    v21 = 0;
LABEL_13:
  v29.m128i_i64[0] = a3;
  v29.m128i_i32[2] = 0;
  v30 = a4;
  v22 = *(_QWORD *)(v17 + 8);
  if ( v22 == *(_QWORD *)(v17 + 16) )
  {
    result = sub_1384280(v17, (_BYTE *)v22, &v29);
  }
  else
  {
    if ( v22 )
    {
      *(__m128i *)v22 = _mm_loadu_si128(&v29);
      result = v30;
      *(_QWORD *)(v22 + 16) = v30;
      v22 = *(_QWORD *)(v17 + 8);
    }
    *(_QWORD *)(v17 + 8) = v22 + 24;
  }
  v29.m128i_i64[0] = a2;
  v29.m128i_i32[2] = 0;
  v30 = a4;
  v23 = *(_QWORD *)(v21 + 32);
  if ( v23 == *(_QWORD *)(v21 + 40) )
    return sub_1384280(v21 + 24, (_BYTE *)v23, &v29);
  if ( v23 )
  {
    *(__m128i *)v23 = _mm_loadu_si128(&v29);
    result = v30;
    *(_QWORD *)(v23 + 16) = v30;
    v23 = *(_QWORD *)(v21 + 32);
  }
  *(_QWORD *)(v21 + 32) = v23 + 24;
  return result;
}
