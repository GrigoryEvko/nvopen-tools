// Function: sub_13876C0
// Address: 0x13876c0
//
__m128i *__fastcall sub_13876C0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *result; // rax
  const __m128i *v7; // r14
  unsigned __int32 v9; // edi
  unsigned __int32 v10; // ecx
  unsigned __int32 v11; // r9d
  unsigned __int32 v12; // ebx
  __m128i *v13; // rax
  unsigned __int32 v14; // edx
  unsigned __int32 v15; // esi
  unsigned __int32 v16; // r8d
  unsigned __int32 v17; // r10d
  unsigned __int32 v18; // r11d
  unsigned __int32 v19; // r14d
  unsigned __int32 v20; // r13d
  __int64 v21; // r12
  __int32 v22; // edx
  unsigned __int32 v23; // edi
  unsigned __int32 v24; // r8d
  unsigned __int32 v25; // r9d
  __int64 v26; // r10
  unsigned __int32 v27; // r12d
  unsigned __int32 v28; // r13d
  unsigned __int32 v29; // r14d
  unsigned __int32 v30; // r11d
  unsigned __int32 v31; // esi
  unsigned __int32 *v32; // rcx
  __m128i *v33; // rax
  unsigned __int32 v34; // ebx
  unsigned __int32 v35; // edx
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rcx
  unsigned __int32 v39; // r11d
  unsigned __int32 v40; // r14d
  __int64 v41; // r13
  unsigned __int32 v42; // ecx
  __int64 v43; // rsi
  __int64 v44; // rax
  __m128i *v45; // r15
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rax
  __m128i v50; // xmm1
  unsigned __int32 *v51; // [rsp+0h] [rbp-A0h]
  __int64 v52; // [rsp+8h] [rbp-98h]
  __m128i *v53; // [rsp+10h] [rbp-90h]
  __int64 v54; // [rsp+18h] [rbp-88h]
  __int32 v55; // [rsp+24h] [rbp-7Ch]
  unsigned __int32 v56; // [rsp+28h] [rbp-78h]
  __int64 v57; // [rsp+28h] [rbp-78h]
  unsigned __int32 v58; // [rsp+28h] [rbp-78h]
  unsigned __int32 v59; // [rsp+30h] [rbp-70h]
  unsigned __int32 v60; // [rsp+30h] [rbp-70h]
  unsigned __int32 v61; // [rsp+34h] [rbp-6Ch]
  __int64 v62; // [rsp+38h] [rbp-68h]
  __m128i *v63; // [rsp+38h] [rbp-68h]
  __int64 v64; // [rsp+40h] [rbp-60h]
  unsigned __int32 v65; // [rsp+40h] [rbp-60h]

  result = (__m128i *)((char *)a2 - a1);
  v53 = a2;
  v52 = a3;
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  v7 = (const __m128i *)a1;
  if ( !a3 )
  {
    v45 = a2;
    goto LABEL_96;
  }
  v51 = (unsigned __int32 *)(a1 + 48);
  while ( 2 )
  {
    v9 = *(_DWORD *)(a1 + 28);
    v10 = *(_DWORD *)(a1 + 24);
    --v52;
    v11 = *(_DWORD *)(a1 + 32);
    v12 = *(_DWORD *)(a1 + 36);
    v64 = *(_QWORD *)(a1 + 40);
    v13 = (__m128i *)(a1
                    + 8
                    * (((__int64)(0xAAAAAAAAAAAAAAABLL * ((__int64)result >> 3)) >> 1)
                     + ((0xAAAAAAAAAAAAAAABLL * ((__int64)result >> 3)) & 0xFFFFFFFFFFFFFFFELL)));
    v14 = v13->m128i_i32[0];
    v15 = v13->m128i_u32[1];
    v16 = v13->m128i_u32[2];
    v62 = v13[1].m128i_i64[0];
    v17 = v13->m128i_u32[3];
    if ( v10 < v13->m128i_i32[0] || v9 < v15 && v10 == v14 )
      goto LABEL_11;
    if ( v10 > v14 || v9 > v15 && v10 == v14 )
      goto LABEL_67;
    if ( v11 < v16 )
    {
LABEL_11:
      v18 = v53[-2].m128i_u32[2];
      if ( v18 <= v14 )
      {
        v19 = v53[-2].m128i_u32[3];
        if ( v19 <= v15 || v18 != v14 )
        {
          if ( (v20 = v53[-1].m128i_u32[0], v56 = v53[-1].m128i_u32[1], v21 = v53[-1].m128i_i64[1], v18 < v14)
            || v19 < v15 && v18 == v14
            || v20 <= v16 && (v56 <= v17 || v20 != v16) && (v20 < v16 || v56 < v17 && v20 == v16 || v21 <= v62) )
          {
            if ( v10 >= v18
              && (v9 >= v19 || v10 != v18)
              && (v10 > v18
               || v9 > v19 && v10 == v18
               || v11 >= v20 && (v12 >= v56 || v11 != v20) && (v11 > v20 || v12 > v56 && v11 == v20 || v64 >= v21)) )
            {
LABEL_94:
              v42 = *(_DWORD *)(a1 + 12);
              v43 = *(_QWORD *)(a1 + 16);
              v27 = *(_DWORD *)a1;
              v28 = *(_DWORD *)(a1 + 4);
              v29 = *(_DWORD *)(a1 + 8);
              v61 = v42;
              v57 = v43;
              *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 24));
              v44 = *(_QWORD *)(a1 + 40);
              *(_DWORD *)(a1 + 24) = v27;
              *(_DWORD *)(a1 + 28) = v28;
              *(_QWORD *)(a1 + 16) = v44;
              *(_DWORD *)(a1 + 32) = v29;
              *(_DWORD *)(a1 + 36) = v42;
              *(_QWORD *)(a1 + 40) = v43;
              goto LABEL_51;
            }
            goto LABEL_25;
          }
        }
      }
      v36 = *(_QWORD *)(a1 + 16);
      v37 = *(_QWORD *)a1;
      v38 = *(_QWORD *)(a1 + 8);
      *(__m128i *)a1 = _mm_loadu_si128(v13);
LABEL_50:
      *(_QWORD *)(a1 + 16) = v13[1].m128i_i64[0];
      v13->m128i_i64[0] = v37;
      v13->m128i_i64[1] = v38;
      v13[1].m128i_i64[0] = v36;
      v27 = *(_DWORD *)(a1 + 24);
      v28 = *(_DWORD *)(a1 + 28);
      v61 = *(_DWORD *)(a1 + 36);
      v29 = *(_DWORD *)(a1 + 32);
      v57 = *(_QWORD *)(a1 + 40);
LABEL_51:
      v24 = v53[-1].m128i_u32[0];
      v25 = v53[-1].m128i_u32[1];
      v26 = v53[-1].m128i_i64[1];
      v22 = v53[-2].m128i_i32[2];
      v23 = v53[-2].m128i_u32[3];
      goto LABEL_26;
    }
    if ( v12 >= v17 )
    {
      if ( v11 > v16 )
        goto LABEL_67;
      goto LABEL_114;
    }
    if ( v11 == v16 )
      goto LABEL_11;
    if ( v11 <= v16 )
    {
LABEL_114:
      if ( (v12 <= v17 || v11 != v16) && v64 < v62 )
        goto LABEL_11;
    }
LABEL_67:
    v39 = v53[-2].m128i_u32[2];
    if ( v10 < v39 )
      goto LABEL_94;
    v40 = v53[-2].m128i_u32[3];
    if ( v9 < v40 && v10 == v39 )
      goto LABEL_94;
    v60 = v53[-1].m128i_u32[0];
    v58 = v53[-1].m128i_u32[1];
    v41 = v53[-1].m128i_i64[1];
    if ( v10 <= v39
      && (v9 <= v40 || v10 != v39)
      && (v11 < v60 || v12 < v58 && v11 == v60 || v11 <= v60 && (v12 <= v58 || v11 != v60) && v64 < v41) )
    {
      goto LABEL_94;
    }
    if ( v39 <= v14
      && (v40 <= v15 || v39 != v14)
      && (v39 < v14
       || v40 < v15 && v39 == v14
       || v60 <= v16 && (v58 <= v17 || v60 != v16) && (v60 < v16 || v58 < v17 && v60 == v16 || v41 <= v62)) )
    {
      v37 = *(_QWORD *)a1;
      v38 = *(_QWORD *)(a1 + 8);
      v36 = *(_QWORD *)(a1 + 16);
      *(__m128i *)a1 = _mm_loadu_si128(v13);
      goto LABEL_50;
    }
LABEL_25:
    v22 = *(_DWORD *)a1;
    v23 = *(_DWORD *)(a1 + 4);
    v24 = *(_DWORD *)(a1 + 8);
    v25 = *(_DWORD *)(a1 + 12);
    v26 = *(_QWORD *)(a1 + 16);
    *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v53 - 24));
    *(_QWORD *)(a1 + 16) = v53[-1].m128i_i64[1];
    v53[-2].m128i_i32[2] = v22;
    v53[-2].m128i_i32[3] = v23;
    v53[-1].m128i_i32[0] = v24;
    v53[-1].m128i_i32[1] = v25;
    v53[-1].m128i_i64[1] = v26;
    v27 = *(_DWORD *)(a1 + 24);
    v28 = *(_DWORD *)(a1 + 28);
    v61 = *(_DWORD *)(a1 + 36);
    v29 = *(_DWORD *)(a1 + 32);
    v57 = *(_QWORD *)(a1 + 40);
LABEL_26:
    v30 = *(_DWORD *)(a1 + 4);
    v55 = v22;
    v31 = *(_DWORD *)a1;
    v32 = v51;
    v65 = *(_DWORD *)(a1 + 8);
    v59 = *(_DWORD *)(a1 + 12);
    v54 = *(_QWORD *)(a1 + 16);
    v33 = v53;
    while ( 1 )
    {
      v63 = (__m128i *)(v32 - 6);
      if ( v27 >= v31
        && (v28 >= v30 || v27 != v31)
        && (v27 > v31
         || v28 > v30 && v27 == v31
         || v65 <= v29 && (v59 <= v61 || v65 != v29) && (v65 < v29 || v59 < v61 && v65 == v29 || v57 >= v54)) )
      {
        break;
      }
LABEL_32:
      v34 = v32[3];
      v27 = *v32;
      v32 += 6;
      v28 = *(v32 - 5);
      v29 = *(v32 - 4);
      v61 = v34;
      v57 = *((_QWORD *)v32 - 1);
    }
    v35 = v55;
    v33 = (__m128i *)((char *)v33 - 24);
    while ( v31 < v35
         || v30 < v23 && v31 == v35
         || v31 <= v35
         && (v30 <= v23 || v31 != v35)
         && (v24 > v65 || v59 < v25 && v24 == v65 || v24 >= v65 && (v59 <= v25 || v24 != v65) && v54 < v26) )
    {
      v33 = (__m128i *)((char *)v33 - 24);
      v24 = v33->m128i_u32[2];
      v25 = v33->m128i_u32[3];
      v26 = v33[1].m128i_i64[0];
      v35 = v33->m128i_i32[0];
      v23 = v33->m128i_u32[1];
    }
    if ( v33 > v63 )
    {
      *(__m128i *)(v32 - 6) = _mm_loadu_si128(v33);
      *((_QWORD *)v32 - 1) = v33[1].m128i_i64[0];
      v24 = v33[-1].m128i_u32[0];
      v33->m128i_i32[0] = v27;
      v25 = v33[-1].m128i_u32[1];
      v33->m128i_i32[1] = v28;
      v26 = v33[-1].m128i_i64[1];
      v33->m128i_i32[2] = v29;
      v33->m128i_i32[3] = v61;
      v33[1].m128i_i64[0] = v57;
      v65 = *(_DWORD *)(a1 + 8);
      v59 = *(_DWORD *)(a1 + 12);
      v30 = *(_DWORD *)(a1 + 4);
      v55 = v33[-2].m128i_i32[2];
      v23 = v33[-2].m128i_u32[3];
      v54 = *(_QWORD *)(a1 + 16);
      v31 = *(_DWORD *)a1;
      goto LABEL_32;
    }
    sub_13876C0(v63, v53, v52);
    result = (__m128i *)((char *)v63 - a1);
    if ( (__int64)v63->m128i_i64 - a1 > 384 )
    {
      if ( v52 )
      {
        v53 = v63;
        continue;
      }
      v7 = (const __m128i *)a1;
      v45 = v63;
LABEL_96:
      sub_1387510(v7, v45, (unsigned __int64)v45, a4, a5, a6);
      do
      {
        v45 = (__m128i *)((char *)v45 - 24);
        v49 = v45[1].m128i_i64[0];
        v50 = _mm_loadu_si128(v45);
        *v45 = _mm_loadu_si128(v7);
        v45[1].m128i_i64[0] = v7[1].m128i_i64[0];
        result = sub_13821F0(
                   (__int64)v7,
                   0,
                   0xAAAAAAAAAAAAAAABLL * (((char *)v45 - (char *)v7) >> 3),
                   v46,
                   v47,
                   v48,
                   v50.m128i_i64[0],
                   v50.m128i_i64[1],
                   v49);
      }
      while ( (char *)v45 - (char *)v7 > 24 );
    }
    return result;
  }
}
