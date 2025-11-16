// Function: sub_13911E0
// Address: 0x13911e0
//
__m128i *__fastcall sub_13911E0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __m128i *result; // rax
  unsigned __int8 v6; // al
  __int64 v7; // r15
  __int64 v11; // rax
  __int64 v12; // rdi
  char v13; // al
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  __int64 v16; // rax
  __int64 v17; // rdi
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx
  int v22; // esi
  unsigned int v23; // edi
  __m128i *v24; // rdx
  __int64 v25; // r8
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rdi
  unsigned int v29; // r8d
  __m128i *v30; // rdx
  __int64 v31; // r9
  __int64 v32; // rbx
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // edx
  int v38; // edx
  int v39; // r10d
  int v40; // r9d
  __m128i v41; // [rsp-58h] [rbp-58h] BYREF
  __m128i *v42; // [rsp-48h] [rbp-48h]

  result = *(__m128i **)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 15 )
    return result;
  result = *(__m128i **)a3;
  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) != 15 )
    return result;
  v6 = *(_BYTE *)(a2 + 16);
  v7 = *(_QWORD *)(a1 + 24);
  if ( v6 > 3u )
  {
    if ( v6 == 5 )
    {
      if ( (unsigned int)*(unsigned __int16 *)(a2 + 18) - 51 <= 1 )
        goto LABEL_8;
      if ( (unsigned __int8)sub_13848E0(*(_QWORD *)(a1 + 24), a2, 0, 0) )
      {
        sub_1391610(a1, a2);
        v7 = *(_QWORD *)(a1 + 24);
        goto LABEL_8;
      }
    }
    else
    {
      sub_13848E0(*(_QWORD *)(a1 + 24), a2, 0, 0);
    }
    v7 = *(_QWORD *)(a1 + 24);
    goto LABEL_8;
  }
  v11 = sub_14C81A0(a2);
  v12 = v7;
  v13 = sub_13848E0(v7, a2, 0, v11);
  v7 = *(_QWORD *)(a1 + 24);
  if ( v13 )
  {
    v35 = sub_14C8160(v12, a2, v14);
    sub_13848E0(v7, a2, 1u, v35);
    v7 = *(_QWORD *)(a1 + 24);
  }
LABEL_8:
  v15 = *(_BYTE *)(a3 + 16);
  if ( v15 > 3u )
  {
    if ( v15 == 5 )
    {
      if ( (unsigned int)*(unsigned __int16 *)(a3 + 18) - 51 <= 1 )
        goto LABEL_11;
      if ( (unsigned __int8)sub_13848E0(v7, a3, 0, 0) )
      {
        sub_1391610(a1, a3);
        v7 = *(_QWORD *)(a1 + 24);
        goto LABEL_11;
      }
    }
    else
    {
      sub_13848E0(v7, a3, 0, 0);
    }
    v7 = *(_QWORD *)(a1 + 24);
    goto LABEL_11;
  }
  v16 = sub_14C81A0(a3);
  v17 = v7;
  v18 = sub_13848E0(v7, a3, 0, v16);
  v7 = *(_QWORD *)(a1 + 24);
  if ( v18 )
  {
    v36 = sub_14C8160(v17, a3, v19);
    sub_13848E0(v7, a3, 1u, v36);
    v7 = *(_QWORD *)(a1 + 24);
  }
LABEL_11:
  if ( !a4 )
  {
    sub_13848E0(v7, a3, 1u, 0);
    return sub_1384420(*(_QWORD *)(a1 + 24), a2, 0, a3, 1, 0);
  }
  sub_13848E0(v7, a2, 1u, 0);
  v20 = *(_QWORD *)(a1 + 24);
  v21 = *(_QWORD *)(v20 + 8);
  result = (__m128i *)*(unsigned int *)(v20 + 24);
  if ( !(_DWORD)result )
  {
    v28 = 0;
    goto LABEL_45;
  }
  v22 = (_DWORD)result - 1;
  v23 = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v24 = (__m128i *)(v21 + 32LL * v23);
  v25 = v24->m128i_i64[0];
  if ( a2 != v24->m128i_i64[0] )
  {
    v37 = 1;
    while ( v25 != -8 )
    {
      v40 = v37 + 1;
      v23 = v22 & (v37 + v23);
      v24 = (__m128i *)(v21 + 32LL * v23);
      v25 = v24->m128i_i64[0];
      if ( a2 == v24->m128i_i64[0] )
        goto LABEL_14;
      v37 = v40;
    }
    result = (__m128i *)(v21 + 32LL * (_QWORD)result);
    goto LABEL_41;
  }
LABEL_14:
  result = (__m128i *)(v21 + 32LL * (_QWORD)result);
  if ( result == v24 )
  {
LABEL_41:
    v28 = 0;
    goto LABEL_17;
  }
  v26 = v24->m128i_i64[1];
  v27 = v24[1].m128i_i64[0] - v26;
  v28 = v26 + 56;
  if ( -1227133513 * (unsigned int)(v27 >> 3) <= 1 )
    v28 = 0;
LABEL_17:
  v29 = v22 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v30 = (__m128i *)(v21 + 32LL * v29);
  v31 = v30->m128i_i64[0];
  if ( a3 == v30->m128i_i64[0] )
  {
LABEL_18:
    if ( result != v30 )
    {
      v32 = v30->m128i_i64[1];
      result = 0;
      if ( !(-1227133513 * (unsigned int)((v30[1].m128i_i64[0] - v32) >> 3)) )
        v32 = 0;
      goto LABEL_21;
    }
  }
  else
  {
    v38 = 1;
    while ( v31 != -8 )
    {
      v39 = v38 + 1;
      v29 = v22 & (v38 + v29);
      v30 = (__m128i *)(v21 + 32LL * v29);
      v31 = v30->m128i_i64[0];
      if ( a3 == v30->m128i_i64[0] )
        goto LABEL_18;
      v38 = v39;
    }
  }
LABEL_45:
  v32 = 0;
LABEL_21:
  v41.m128i_i64[0] = a3;
  v41.m128i_i32[2] = 0;
  v42 = 0;
  v33 = *(_QWORD *)(v28 + 8);
  if ( v33 == *(_QWORD *)(v28 + 16) )
  {
    result = sub_1384280(v28, (_BYTE *)v33, &v41);
  }
  else
  {
    if ( v33 )
    {
      *(__m128i *)v33 = _mm_loadu_si128(&v41);
      result = v42;
      *(_QWORD *)(v33 + 16) = v42;
      v33 = *(_QWORD *)(v28 + 8);
    }
    *(_QWORD *)(v28 + 8) = v33 + 24;
  }
  v41.m128i_i64[0] = a2;
  v41.m128i_i32[2] = 1;
  v42 = 0;
  v34 = *(_QWORD *)(v32 + 32);
  if ( v34 == *(_QWORD *)(v32 + 40) )
    return sub_1384280(v32 + 24, (_BYTE *)v34, &v41);
  if ( v34 )
  {
    *(__m128i *)v34 = _mm_loadu_si128(&v41);
    result = v42;
    *(_QWORD *)(v34 + 16) = v42;
    v34 = *(_QWORD *)(v32 + 32);
  }
  *(_QWORD *)(v32 + 32) = v34 + 24;
  return result;
}
