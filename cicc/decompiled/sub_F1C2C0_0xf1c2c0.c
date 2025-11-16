// Function: sub_F1C2C0
// Address: 0xf1c2c0
//
__int64 __fastcall sub_F1C2C0(__int64 a1, __int64 a2, __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v6; // r15
  __int64 v8; // rax
  char v9; // dl
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // r8
  __m128i *v13; // r13
  __int64 v14; // rdx
  __m128i *v15; // rax
  __int8 v16; // si
  const __m128i *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // r10d
  __m128i *v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  __int8 v28; // al
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // rax
  const void *v31; // rsi
  __int8 *v32; // r15
  unsigned int v33; // [rsp+4h] [rbp-4Ch]
  __int64 v34; // [rsp+8h] [rbp-48h]
  _QWORD *v35; // [rsp+10h] [rbp-40h]
  __int64 v36; // [rsp+18h] [rbp-38h]

  v6 = a3;
  if ( *(_QWORD *)(a2 + 216) )
  {
    v8 = sub_F18EE0(a2 + 176, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v8;
    *(_BYTE *)(a1 + 16) = v9;
    return a1;
  }
  v11 = *(unsigned int *)(a2 + 8);
  v12 = *(_QWORD *)a2;
  v13 = (__m128i *)(*(_QWORD *)a2 + 40 * v11);
  if ( *(__m128i **)a2 != v13 )
  {
    v14 = a3->m128i_i64[0];
    v15 = *(__m128i **)a2;
    do
    {
      if ( v15->m128i_i64[0] == v14 )
      {
        v16 = v15[1].m128i_i8[8];
        if ( v16 == v6[1].m128i_i8[8]
          && (!v16 || v15->m128i_i64[1] == v6->m128i_i64[1] && v15[1].m128i_i64[0] == v6[1].m128i_i64[0])
          && v15[2].m128i_i64[0] == v6[2].m128i_i64[0] )
        {
          *(_BYTE *)(a1 + 8) = 1;
          *(_QWORD *)a1 = v15;
          *(_BYTE *)(a1 + 16) = 0;
          return a1;
        }
      }
      v15 = (__m128i *)((char *)v15 + 40);
    }
    while ( v13 != v15 );
    if ( v11 > 3 )
    {
      v17 = (const __m128i *)v12;
      v35 = (_QWORD *)(a2 + 176);
      v36 = a2 + 184;
      while ( 1 )
      {
        v18 = sub_F1BF80(v35, v36, (__int64)v17);
        if ( v19 )
          break;
LABEL_22:
        v17 = (const __m128i *)((char *)v17 + 40);
        if ( v13 == v17 )
          goto LABEL_23;
      }
      if ( v18 )
        goto LABEL_20;
      if ( v19 == v36 )
        goto LABEL_20;
      v27 = *(_QWORD *)(v19 + 32);
      if ( v17->m128i_i64[0] < v27 )
        goto LABEL_20;
      if ( v17->m128i_i64[0] != v27 )
      {
LABEL_28:
        v20 = 0;
        goto LABEL_21;
      }
      v28 = v17[1].m128i_i8[8];
      if ( *(_BYTE *)(v19 + 56) )
      {
        if ( !v28 )
          goto LABEL_20;
        v29 = v17->m128i_u64[1];
        v30 = *(_QWORD *)(v19 + 40);
        if ( v29 < v30 || v29 == v30 && v17[1].m128i_i64[0] < *(_QWORD *)(v19 + 48) )
          goto LABEL_20;
        if ( v29 > v30 || *(_QWORD *)(v19 + 48) < v17[1].m128i_i64[0] )
          goto LABEL_28;
      }
      else if ( v28 )
      {
        goto LABEL_28;
      }
      if ( v17[2].m128i_i64[0] >= *(_QWORD *)(v19 + 64) )
        goto LABEL_28;
LABEL_20:
      v20 = 1;
LABEL_21:
      v33 = v20;
      v34 = v19;
      v21 = (__m128i *)sub_22077B0(72);
      v21[2] = _mm_loadu_si128(v17);
      v21[3] = _mm_loadu_si128(v17 + 1);
      v21[4].m128i_i64[0] = v17[2].m128i_i64[0];
      sub_220F040(v33, v21, v34, v36);
      ++*(_QWORD *)(a2 + 216);
      goto LABEL_22;
    }
    goto LABEL_24;
  }
  if ( v11 <= 3 )
  {
LABEL_24:
    v23 = v11 + 1;
    if ( v11 + 1 > *(unsigned int *)(a2 + 12) )
    {
      v31 = (const void *)(a2 + 16);
      if ( v12 > (unsigned __int64)v6 || v13 <= v6 )
      {
        sub_C8D5F0(a2, v31, v23, 0x28u, v12, a6);
        v13 = (__m128i *)(*(_QWORD *)a2 + 40LL * *(unsigned int *)(a2 + 8));
      }
      else
      {
        v32 = &v6->m128i_i8[-v12];
        sub_C8D5F0(a2, v31, v23, 0x28u, v12, a6);
        v6 = (__m128i *)&v32[*(_QWORD *)a2];
        v13 = (__m128i *)(*(_QWORD *)a2 + 40LL * *(unsigned int *)(a2 + 8));
      }
    }
    *v13 = _mm_loadu_si128(v6);
    v13[1] = _mm_loadu_si128(v6 + 1);
    v13[2].m128i_i64[0] = v6[2].m128i_i64[0];
    v24 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v24;
    v25 = 5 * v24;
    v26 = *(_QWORD *)a2;
    *(_BYTE *)(a1 + 8) = 1;
    *(_BYTE *)(a1 + 16) = 1;
    *(_QWORD *)a1 = v26 + 8 * v25 - 40;
    return a1;
  }
  v35 = (_QWORD *)(a2 + 176);
LABEL_23:
  *(_DWORD *)(a2 + 8) = 0;
  v22 = sub_F18EE0((__int64)v35, v6);
  *(_BYTE *)(a1 + 8) = 0;
  *(_QWORD *)a1 = v22;
  *(_BYTE *)(a1 + 16) = 1;
  return a1;
}
