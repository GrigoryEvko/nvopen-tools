// Function: sub_123CE30
// Address: 0x123ce30
//
__int64 __fastcall sub_123CE30(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  int i; // eax
  _QWORD *v7; // rdi
  unsigned __int8 v8; // al
  __int64 v9; // rdx
  __m128i *v10; // rsi
  _QWORD *v11; // r15
  __int64 v12; // r13
  const __m128i *v13; // r10
  const __m128i *v14; // rcx
  unsigned __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdi
  __m128i *v18; // rdx
  __m128i *v19; // r10
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  const __m128i *v25; // r12
  __int64 *v26; // rcx
  const __m128i *v27; // rbx
  __m128i v28; // xmm0
  __int64 v29; // rax
  unsigned int v30; // edi
  _QWORD *v31; // rax
  __int64 v32; // r15
  int *v33; // rsi
  __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rax
  __m128i *v37; // r8
  __int64 v38; // [rsp+18h] [rbp-B8h]
  __int64 v39; // [rsp+20h] [rbp-B0h]
  __int64 *v40; // [rsp+28h] [rbp-A8h]
  unsigned __int8 v41; // [rsp+28h] [rbp-A8h]
  __int64 v42; // [rsp+38h] [rbp-98h] BYREF
  __m128i v43; // [rsp+40h] [rbp-90h] BYREF
  __m128i v44; // [rsp+50h] [rbp-80h] BYREF
  __int64 v45; // [rsp+60h] [rbp-70h]
  unsigned __int64 v46; // [rsp+68h] [rbp-68h]
  __int64 v47; // [rsp+70h] [rbp-60h] BYREF
  int v48; // [rsp+78h] [rbp-58h] BYREF
  _QWORD *v49; // [rsp+80h] [rbp-50h]
  int *v50; // [rsp+88h] [rbp-48h]
  int *v51; // [rsp+90h] [rbp-40h]
  __int64 v52; // [rsp+98h] [rbp-38h]

  v3 = a1;
  v4 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in typeIdInfo") )
  {
    return 1;
  }
  v48 = 0;
  v49 = 0;
  v52 = 0;
  v50 = &v48;
  v51 = &v48;
  for ( i = *(_DWORD *)(a1 + 240); ; *(_DWORD *)(v3 + 240) = i )
  {
    v43.m128i_i64[0] = 0;
    if ( i == 506 )
    {
      v30 = *(_DWORD *)(v3 + 280);
      v31 = v49;
      v32 = *(_QWORD *)(v3 + 232);
      v33 = &v48;
      LODWORD(v42) = v30;
      if ( !v49 )
        goto LABEL_54;
      do
      {
        while ( 1 )
        {
          v34 = v31[2];
          v35 = v31[3];
          if ( v30 <= *((_DWORD *)v31 + 8) )
            break;
          v31 = (_QWORD *)v31[3];
          if ( !v35 )
            goto LABEL_52;
        }
        v33 = (int *)v31;
        v31 = (_QWORD *)v31[2];
      }
      while ( v34 );
LABEL_52:
      if ( v33 == &v48 || v30 < v33[8] )
      {
LABEL_54:
        v44.m128i_i64[0] = (__int64)&v42;
        v33 = (int *)sub_1239060(&v47, (__int64)v33, (unsigned int **)&v44);
      }
      v36 = a2[1] - *a2;
      v44.m128i_i64[1] = v32;
      v44.m128i_i32[0] = v36 >> 3;
      v37 = (__m128i *)*((_QWORD *)v33 + 6);
      if ( v37 == *((__m128i **)v33 + 7) )
      {
        sub_12171B0((const __m128i **)v33 + 5, *((const __m128i **)v33 + 6), &v44);
      }
      else
      {
        if ( v37 )
        {
          *v37 = _mm_loadu_si128(&v44);
          v37 = (__m128i *)*((_QWORD *)v33 + 6);
        }
        *((_QWORD *)v33 + 6) = v37 + 1;
      }
      v7 = (_QWORD *)v4;
      *(_DWORD *)(v3 + 240) = sub_1205200(v4);
    }
    else
    {
      v7 = (_QWORD *)v3;
      v8 = sub_120C050(v3, v43.m128i_i64);
      if ( v8 )
        goto LABEL_43;
    }
    v10 = (__m128i *)a2[1];
    if ( v10 == (__m128i *)a2[2] )
    {
      v7 = a2;
      sub_9CA200((__int64)a2, v10, &v43);
    }
    else
    {
      if ( v10 )
      {
        v10->m128i_i64[0] = v43.m128i_i64[0];
        v10 = (__m128i *)a2[1];
      }
      v10 = (__m128i *)((char *)v10 + 8);
      a2[1] = v10;
    }
    if ( *(_DWORD *)(v3 + 240) != 4 )
      break;
    i = sub_1205200(v4);
  }
  v39 = v3 + 1656;
  if ( v50 == &v48 )
    goto LABEL_42;
  v38 = v3;
  v11 = a2;
  v12 = (__int64)v50;
  do
  {
    v44.m128i_i32[0] = *(_DWORD *)(v12 + 32);
    v13 = *(const __m128i **)(v12 + 48);
    v14 = *(const __m128i **)(v12 + 40);
    v44.m128i_i64[1] = 0;
    v45 = 0;
    v46 = 0;
    v15 = (char *)v13 - (char *)v14;
    if ( v13 == v14 )
    {
      v17 = 0;
    }
    else
    {
      if ( v15 > 0x7FFFFFFFFFFFFFF0LL )
        sub_4261EA(v7, v10, v9);
      v16 = sub_22077B0((char *)v13 - (char *)v14);
      v13 = *(const __m128i **)(v12 + 48);
      v14 = *(const __m128i **)(v12 + 40);
      v17 = v16;
    }
    v44.m128i_i64[1] = v17;
    v45 = v17;
    v46 = v17 + v15;
    if ( v14 == v13 )
    {
      v19 = (__m128i *)v17;
    }
    else
    {
      v18 = (__m128i *)v17;
      v19 = (__m128i *)(v17 + (char *)v13 - (char *)v14);
      do
      {
        if ( v18 )
          *v18 = _mm_loadu_si128(v14);
        ++v18;
        ++v14;
      }
      while ( v19 != v18 );
    }
    v45 = (__int64)v19;
    v20 = *(_QWORD *)(v38 + 1664);
    if ( !v20 )
    {
      v21 = v39;
LABEL_29:
      v10 = (__m128i *)v21;
      v43.m128i_i64[0] = (__int64)&v44;
      v24 = sub_123CD60((_QWORD *)(v38 + 1648), v21, (unsigned int **)&v43);
      v19 = (__m128i *)v45;
      v17 = v44.m128i_i64[1];
      v21 = v24;
      goto LABEL_30;
    }
    v10 = (__m128i *)v44.m128i_u32[0];
    v21 = v39;
    do
    {
      while ( 1 )
      {
        v22 = *(_QWORD *)(v20 + 16);
        v23 = *(_QWORD *)(v20 + 24);
        if ( *(_DWORD *)(v20 + 32) >= v44.m128i_i32[0] )
          break;
        v20 = *(_QWORD *)(v20 + 24);
        if ( !v23 )
          goto LABEL_27;
      }
      v21 = v20;
      v20 = *(_QWORD *)(v20 + 16);
    }
    while ( v22 );
LABEL_27:
    if ( v21 == v39 || v44.m128i_i32[0] < *(_DWORD *)(v21 + 32) )
      goto LABEL_29;
LABEL_30:
    if ( v19 != (__m128i *)v17 )
    {
      v25 = (const __m128i *)v17;
      v26 = &v43.m128i_i64[1];
      v27 = v19;
      do
      {
        while ( 1 )
        {
          v28 = _mm_loadu_si128(v25);
          v29 = *v11 + 8LL * v25->m128i_u32[0];
          v42 = v29;
          v43 = v28;
          v10 = *(__m128i **)(v21 + 48);
          if ( v10 != *(__m128i **)(v21 + 56) )
            break;
          ++v25;
          v40 = v26;
          sub_12149F0((const __m128i **)(v21 + 40), v10, &v42, v26);
          v26 = v40;
          if ( v27 == v25 )
            goto LABEL_37;
        }
        if ( v10 )
        {
          v10->m128i_i64[0] = v29;
          v10->m128i_i64[1] = v43.m128i_i64[1];
          v10 = *(__m128i **)(v21 + 48);
        }
        ++v10;
        ++v25;
        *(_QWORD *)(v21 + 48) = v10;
      }
      while ( v27 != v25 );
LABEL_37:
      v17 = v44.m128i_i64[1];
    }
    if ( v17 )
    {
      v10 = (__m128i *)(v46 - v17);
      j_j___libc_free_0(v17, v46 - v17);
    }
    v7 = (_QWORD *)v12;
    v12 = sub_220EEE0(v12);
  }
  while ( (int *)v12 != &v48 );
  v3 = v38;
LABEL_42:
  v8 = sub_120AFE0(v3, 13, "expected ')' in typeIdInfo");
LABEL_43:
  v41 = v8;
  sub_1207E40(v49);
  return v41;
}
