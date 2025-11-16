// Function: sub_125CA70
// Address: 0x125ca70
//
void __fastcall sub_125CA70(__m128i **a1, const __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r9
  const __m128i *v5; // r8
  const __m128i *v6; // rcx
  __int64 v7; // r10
  unsigned __int64 v8; // rax
  __int64 v9; // r13
  unsigned __int64 v10; // r14
  const __m128i *v11; // rbx
  __int64 v12; // r11
  __m128i *v13; // rdx
  const __m128i *v14; // rax
  const __m128i *v15; // rsi
  const __m128i *v16; // rdx
  unsigned __int64 v17; // rbx
  bool v18; // cf
  unsigned __int64 v19; // rbx
  __int64 v20; // rbx
  __int64 v21; // rax
  __m128i *v22; // r14
  const __m128i *v23; // rax
  signed __int64 v24; // r12
  __m128i *v25; // rdx
  __m128i *v26; // r12
  const __m128i *v27; // rcx
  __int64 v28; // rax
  _QWORD *v29; // r12
  __m128i *v30; // rbx
  __int64 v31; // rax
  __int64 v32; // r15
  __int64 v33; // rax
  _BYTE *v34; // rax
  _QWORD *v35; // r8
  __m128i *v36; // rdi
  size_t v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rsi
  _BYTE *v40; // rsi
  __int64 v41; // rdx
  const __m128i *v42; // rbx
  const __m128i *v43; // rax
  __m128i *v44; // r13
  __int64 v45; // rdx
  const __m128i *v46; // rdx
  const __m128i *v47; // r12
  __int64 v48; // r14
  __int64 *v49; // r12
  __m128i *v50; // rdx
  __int64 v51; // r12
  __m128i *v52; // rax
  const __m128i *v53; // rdx
  __m128i *v54; // rbx
  const __m128i *v55; // rsi
  __int64 v56; // rsi
  unsigned __int64 v57; // [rsp-60h] [rbp-60h]
  __int64 v58; // [rsp-58h] [rbp-58h]
  __int64 v59; // [rsp-50h] [rbp-50h]
  _QWORD *v60; // [rsp-50h] [rbp-50h]
  const __m128i *v61; // [rsp-50h] [rbp-50h]
  __int64 v62; // [rsp-48h] [rbp-48h]
  const __m128i *v63; // [rsp-48h] [rbp-48h]
  __int64 v64; // [rsp-48h] [rbp-48h]
  const __m128i *v65; // [rsp-48h] [rbp-48h]
  __int64 v66; // [rsp-40h] [rbp-40h]
  const __m128i *v67; // [rsp-40h] [rbp-40h]
  const __m128i *v68; // [rsp-40h] [rbp-40h]
  __int64 v69; // [rsp-40h] [rbp-40h]
  const __m128i *v70; // [rsp-40h] [rbp-40h]

  if ( a3 == a4 )
    return;
  v4 = a4;
  v5 = a2;
  v6 = a2;
  v7 = v4 - a3;
  v8 = (v4 - a3) >> 5;
  v9 = a3;
  v10 = v8;
  v11 = a1[1];
  if ( (char *)a1[2] - (char *)v11 < (unsigned __int64)(v4 - a3) )
  {
    v16 = *a1;
    v17 = ((char *)v11 - (char *)*a1) >> 5;
    if ( v8 > 0x3FFFFFFFFFFFFFFLL - v17 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v8 < v17 )
      v8 = v17;
    v18 = __CFADD__(v8, v17);
    v19 = v8 + v17;
    if ( v18 )
    {
      v20 = 0x7FFFFFFFFFFFFFE0LL;
    }
    else
    {
      if ( !v19 )
      {
        v59 = 0;
        v22 = 0;
        goto LABEL_20;
      }
      if ( v19 > 0x3FFFFFFFFFFFFFFLL )
        v19 = 0x3FFFFFFFFFFFFFFLL;
      v20 = 32 * v19;
    }
    v62 = v4;
    v21 = sub_22077B0(v20);
    v16 = *a1;
    v5 = a2;
    v22 = (__m128i *)v21;
    v4 = v62;
    v59 = v20 + v21;
LABEL_20:
    if ( v5 == v16 )
    {
      v26 = v22;
    }
    else
    {
      v23 = v16 + 1;
      v24 = (char *)v5 - (char *)v16;
      v25 = v22;
      v26 = (__m128i *)((char *)v22 + v24);
      do
      {
        if ( v25 )
        {
          v25->m128i_i64[0] = (__int64)v25[1].m128i_i64;
          v27 = (const __m128i *)v23[-1].m128i_i64[0];
          if ( v23 == v27 )
          {
            v25[1] = _mm_loadu_si128(v23);
          }
          else
          {
            v25->m128i_i64[0] = (__int64)v27;
            v25[1].m128i_i64[0] = v23->m128i_i64[0];
          }
          v25->m128i_i64[1] = v23[-1].m128i_i64[1];
          v23[-1].m128i_i64[0] = (__int64)v23;
          v23[-1].m128i_i64[1] = 0;
          v23->m128i_i8[0] = 0;
        }
        v25 += 2;
        v23 += 2;
      }
      while ( v25 != v26 );
    }
    do
    {
      if ( v26 )
      {
        v40 = *(_BYTE **)v9;
        v41 = *(_QWORD *)(v9 + 8);
        v26->m128i_i64[0] = (__int64)v26[1].m128i_i64;
        v64 = v4;
        v68 = v5;
        sub_125C500(v26->m128i_i64, v40, (__int64)&v40[v41]);
        v4 = v64;
        v5 = v68;
      }
      v9 += 32;
      v26 += 2;
    }
    while ( v4 != v9 );
    v42 = a1[1];
    if ( v5 == v42 )
    {
      v44 = v26;
    }
    else
    {
      v43 = v5 + 1;
      v44 = (__m128i *)((char *)v26 + (char *)v42 - (char *)v5);
      do
      {
        if ( v26 )
        {
          v26->m128i_i64[0] = (__int64)v26[1].m128i_i64;
          v46 = (const __m128i *)v43[-1].m128i_i64[0];
          if ( v46 == v43 )
          {
            v26[1] = _mm_loadu_si128(v43);
          }
          else
          {
            v26->m128i_i64[0] = (__int64)v46;
            v26[1].m128i_i64[0] = v43->m128i_i64[0];
          }
          v45 = v43[-1].m128i_i64[1];
          v43[-1].m128i_i64[0] = (__int64)v43;
          v43[-1].m128i_i64[1] = 0;
          v26->m128i_i64[1] = v45;
          v43->m128i_i8[0] = 0;
        }
        v26 += 2;
        v43 += 2;
      }
      while ( v26 != v44 );
      v42 = a1[1];
    }
    v47 = *a1;
    if ( *a1 != v42 )
    {
      do
      {
        if ( (const __m128i *)v47->m128i_i64[0] != &v47[1] )
          j_j___libc_free_0(v47->m128i_i64[0], v47[1].m128i_i64[0] + 1);
        v47 += 2;
      }
      while ( v47 != v42 );
      v42 = *a1;
    }
    if ( v42 )
      j_j___libc_free_0(v42, (char *)a1[2] - (char *)v42);
    *a1 = v22;
    a1[1] = v44;
    a1[2] = (__m128i *)v59;
    return;
  }
  v12 = (char *)v11 - (char *)a2;
  if ( v7 < (unsigned __int64)((char *)v11 - (char *)a2) )
  {
    v13 = a1[1];
    v14 = (const __m128i *)((char *)v11 - v7 + 16);
    do
    {
      if ( v13 )
      {
        v13->m128i_i64[0] = (__int64)v13[1].m128i_i64;
        v15 = (const __m128i *)v14[-1].m128i_i64[0];
        if ( v14 == v15 )
        {
          v13[1] = _mm_loadu_si128(v14);
        }
        else
        {
          v13->m128i_i64[0] = (__int64)v15;
          v13[1].m128i_i64[0] = v14->m128i_i64[0];
        }
        v13->m128i_i64[1] = v14[-1].m128i_i64[1];
        v14[-1].m128i_i64[0] = (__int64)v14;
        v14[-1].m128i_i64[1] = 0;
        v14->m128i_i8[0] = 0;
      }
      v13 += 2;
      v14 += 2;
    }
    while ( v13 != (__m128i *)&v11->m128i_i8[v7] );
    v28 = (__int64)v11->m128i_i64 - v7;
    a1[1] = (__m128i *)((char *)a1[1] + v7);
    v29 = (__int64 *)((char *)v11[-1].m128i_i64 - v7);
    v30 = (__m128i *)&v11[-1];
    v31 = v28 - (_QWORD)v5;
    v32 = v31 >> 5;
    if ( v31 <= 0 )
    {
LABEL_41:
      if ( v7 > 0 )
      {
        do
        {
          v39 = v9;
          v67 = v6;
          v9 += 32;
          sub_2240AE0(v6, v39);
          v6 = v67 + 2;
          --v10;
        }
        while ( v10 );
      }
      return;
    }
    while ( 1 )
    {
      v35 = (_QWORD *)*(v29 - 2);
      v36 = (__m128i *)v30[-1].m128i_i64[0];
      if ( v29 == v35 )
      {
        v37 = *(v29 - 1);
        if ( v37 )
        {
          if ( v37 == 1 )
          {
            v36->m128i_i8[0] = *(_BYTE *)v29;
            v36 = (__m128i *)v30[-1].m128i_i64[0];
          }
          else
          {
            v60 = (_QWORD *)*(v29 - 2);
            v63 = v6;
            v66 = v7;
            memcpy(v36, v29, v37);
            v36 = (__m128i *)v30[-1].m128i_i64[0];
            v35 = v60;
            v6 = v63;
            v7 = v66;
          }
        }
        v38 = *(v35 - 1);
        v30[-1].m128i_i64[1] = v38;
        v36->m128i_i8[v38] = 0;
        goto LABEL_33;
      }
      if ( v36 == v30 )
        break;
      v30[-1].m128i_i64[0] = (__int64)v35;
      v33 = v30->m128i_i64[0];
      v30[-1].m128i_i64[1] = *(v29 - 1);
      v30->m128i_i64[0] = *v29;
      if ( !v36 )
        goto LABEL_40;
      *(v29 - 2) = v36;
      *v29 = v33;
LABEL_33:
      v34 = (_BYTE *)*(v29 - 2);
      v30 -= 2;
      v29 -= 4;
      v29[3] = 0;
      *v34 = 0;
      if ( !--v32 )
        goto LABEL_41;
    }
    v30[-1].m128i_i64[0] = (__int64)v35;
    v30[-1].m128i_i64[1] = *(v29 - 1);
    v30->m128i_i64[0] = *v29;
LABEL_40:
    *(v29 - 2) = v29;
    goto LABEL_33;
  }
  v48 = a3 + v12;
  if ( v4 == a3 + v12 )
  {
    v50 = a1[1];
  }
  else
  {
    v49 = (__int64 *)a1[1];
    do
    {
      if ( v49 )
      {
        v57 = v8;
        *v49 = (__int64)(v49 + 2);
        v58 = v4;
        v61 = v5;
        v65 = v6;
        v69 = v12;
        sub_125C500(v49, *(_BYTE **)v48, *(_QWORD *)v48 + *(_QWORD *)(v48 + 8));
        v8 = v57;
        v4 = v58;
        v5 = v61;
        v6 = v65;
        v12 = v69;
      }
      v48 += 32;
      v49 += 4;
    }
    while ( v4 != v48 );
    v50 = a1[1];
  }
  v51 = v12 >> 5;
  v52 = &v50[2 * (v8 - (v12 >> 5))];
  a1[1] = v52;
  if ( v5 != v11 )
  {
    v53 = v5 + 1;
    v54 = (__m128i *)((char *)v52 + (char *)v11 - (char *)v5);
    do
    {
      if ( v52 )
      {
        v52->m128i_i64[0] = (__int64)v52[1].m128i_i64;
        v55 = (const __m128i *)v53[-1].m128i_i64[0];
        if ( v55 == v53 )
        {
          v52[1] = _mm_loadu_si128(v53);
        }
        else
        {
          v52->m128i_i64[0] = (__int64)v55;
          v52[1].m128i_i64[0] = v53->m128i_i64[0];
        }
        v52->m128i_i64[1] = v53[-1].m128i_i64[1];
        v53[-1].m128i_i64[0] = (__int64)v53;
        v53[-1].m128i_i64[1] = 0;
        v53->m128i_i8[0] = 0;
      }
      v52 += 2;
      v53 += 2;
    }
    while ( v52 != v54 );
    v52 = a1[1];
  }
  a1[1] = (__m128i *)((char *)v52 + v12);
  if ( v12 > 0 )
  {
    do
    {
      v56 = v9;
      v70 = v6;
      v9 += 32;
      sub_2240AE0(v6, v56);
      v6 = v70 + 2;
      --v51;
    }
    while ( v51 );
  }
}
