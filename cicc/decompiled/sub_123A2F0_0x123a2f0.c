// Function: sub_123A2F0
// Address: 0x123a2f0
//
__int64 __fastcall sub_123A2F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __m128i *v3; // r14
  __m128i *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i v10; // xmm4
  __m128i *v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // r12
  unsigned __int64 v14; // rax
  __m128i *v15; // r13
  __m128i *v16; // rsi
  __int64 v17; // r8
  __int64 v18; // r9
  __m128i *v19; // r13
  __int64 v20; // rdx
  unsigned __int64 v21; // r10
  __int64 v22; // rax
  __m128i *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  __m128i *v27; // r10
  __int64 v28; // rdi
  __m128i *v29; // r15
  const __m128i *v30; // r10
  const __m128i *v31; // rcx
  unsigned __int64 v32; // r14
  __int64 v33; // rax
  __m128i *v34; // rdx
  __m128i *v35; // r10
  __int64 v36; // rax
  __int64 v37; // r14
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rax
  const __m128i *v41; // r12
  __int64 *v42; // rcx
  const __m128i *v43; // r13
  __int64 v44; // rax
  signed __int64 v45; // rdx
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rsi
  bool v48; // cf
  unsigned __int64 v49; // rax
  __int64 v50; // rax
  __m128i *v51; // rdx
  __m128i v52; // xmm5
  __m128i *v53; // rdx
  const __m128i *v54; // rax
  __int64 v55; // r8
  __int64 v56; // rax
  __m128i *v57; // [rsp+0h] [rbp-E0h]
  __m128i *src; // [rsp+8h] [rbp-D8h]
  __int64 v59; // [rsp+20h] [rbp-C0h]
  __int64 v60; // [rsp+20h] [rbp-C0h]
  __int64 v61; // [rsp+30h] [rbp-B0h]
  __int64 v62; // [rsp+30h] [rbp-B0h]
  __int64 v63; // [rsp+30h] [rbp-B0h]
  __int64 v64; // [rsp+30h] [rbp-B0h]
  unsigned __int8 v65; // [rsp+38h] [rbp-A8h]
  unsigned __int8 v66; // [rsp+38h] [rbp-A8h]
  __int64 *v67; // [rsp+38h] [rbp-A8h]
  __int64 v68; // [rsp+48h] [rbp-98h] BYREF
  __m128i v69; // [rsp+50h] [rbp-90h] BYREF
  __m128i v70; // [rsp+60h] [rbp-80h] BYREF
  __int64 v71; // [rsp+70h] [rbp-70h]
  unsigned __int64 v72; // [rsp+78h] [rbp-68h]
  __m128i v73; // [rsp+80h] [rbp-60h] BYREF
  __m128i *v74; // [rsp+90h] [rbp-50h]
  __m128i *v75; // [rsp+98h] [rbp-48h]
  __int64 *v76; // [rsp+A0h] [rbp-40h]
  __int64 v77; // [rsp+A8h] [rbp-38h]

  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' in refs")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in refs") )
  {
    return 1;
  }
  v3 = 0;
  v57 = 0;
  src = 0;
  while ( 1 )
  {
    v6 = *(__m128i **)(a1 + 232);
    v73.m128i_i64[0] = 0;
    v74 = v6;
    result = sub_12122D0(a1, &v73, &v73.m128i_i32[2]);
    if ( (_BYTE)result )
      goto LABEL_40;
    if ( v3 != v57 )
    {
      if ( v3 )
      {
        v10 = _mm_loadu_si128(&v73);
        v3[1].m128i_i64[0] = (__int64)v74;
        *v3 = v10;
      }
      v3 = (__m128i *)((char *)v3 + 24);
      goto LABEL_11;
    }
    v45 = (char *)v3 - (char *)src;
    v46 = 0xAAAAAAAAAAAAAAABLL * (((char *)v3 - (char *)src) >> 3);
    if ( v46 == 0x555555555555555LL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v47 = 1;
    if ( v46 )
      v47 = 0xAAAAAAAAAAAAAAABLL * (((char *)v3 - (char *)src) >> 3);
    v48 = __CFADD__(v47, v46);
    v49 = v47 - 0x5555555555555555LL * (((char *)v3 - (char *)src) >> 3);
    if ( v48 )
    {
      v55 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_97:
      v64 = v55;
      v56 = sub_22077B0(v55);
      v45 = (char *)v3 - (char *)src;
      v9 = v56;
      v8 = v56 + v64;
      v50 = v56 + 24;
      goto LABEL_80;
    }
    if ( v49 )
    {
      if ( v49 > 0x555555555555555LL )
        v49 = 0x555555555555555LL;
      v55 = 24 * v49;
      goto LABEL_97;
    }
    v50 = 24;
    v8 = 0;
    v9 = 0;
LABEL_80:
    v51 = (__m128i *)(v9 + v45);
    if ( v51 )
    {
      v52 = _mm_loadu_si128(&v73);
      v51[1].m128i_i64[0] = (__int64)v74;
      *v51 = v52;
    }
    if ( v3 == src )
    {
      v3 = (__m128i *)v50;
    }
    else
    {
      v53 = (__m128i *)v9;
      v54 = src;
      do
      {
        if ( v53 )
        {
          *v53 = _mm_loadu_si128(v54);
          v7 = v54[1].m128i_i64[0];
          v53[1].m128i_i64[0] = v7;
        }
        v54 = (const __m128i *)((char *)v54 + 24);
        v53 = (__m128i *)((char *)v53 + 24);
      }
      while ( v3 != v54 );
      v3 = (__m128i *)(v9
                     + 8
                     * (3
                      * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)&v3[-2].m128i_u64[1] - (char *)src) >> 3))
                       & 0x1FFFFFFFFFFFFFFFLL)
                      + 6));
    }
    if ( src )
    {
      v60 = v8;
      v63 = v9;
      j_j___libc_free_0(src, (char *)v57 - (char *)src);
      v8 = v60;
      v9 = v63;
    }
    v57 = (__m128i *)v8;
    src = (__m128i *)v9;
LABEL_11:
    if ( *(_DWORD *)(a1 + 240) != 4 )
      break;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  v11 = src;
  v12 = a2;
  v13 = a1;
  if ( src == v3 )
  {
    v73.m128i_i32[2] = 0;
    v74 = 0;
    v75 = (__m128i *)&v73.m128i_u64[1];
    v76 = &v73.m128i_i64[1];
    v77 = 0;
  }
  else
  {
    _BitScanReverse64(&v14, 0xAAAAAAAAAAAAAAABLL * (((char *)v3 - (char *)src) >> 3));
    sub_12065D0((__int64)src, v3, 2LL * (int)(63 - (v14 ^ 0x3F)), v7, v8, v9);
    if ( (char *)v3 - (char *)src <= 384 )
    {
      v11 = src;
      v16 = v3;
      sub_1205790(src, v3);
    }
    else
    {
      v15 = src + 24;
      v16 = src + 24;
      sub_1205790(src, src + 24);
      if ( &src[24] != v3 )
      {
        do
        {
          v11 = v15;
          v15 = (__m128i *)((char *)v15 + 24);
          sub_1205720(v11);
        }
        while ( v3 != v15 );
      }
    }
    v73.m128i_i32[2] = 0;
    v74 = 0;
    v75 = (__m128i *)&v73.m128i_u64[1];
    v76 = &v73.m128i_i64[1];
    v19 = src;
    v77 = 0;
    while ( 2 )
    {
      v22 = v19->m128i_i64[0];
      if ( (v19->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) == 0xFFFFFFFFFFFFFFF8LL )
      {
        v23 = v74;
        if ( !v74 )
        {
          v16 = (__m128i *)&v73.m128i_u64[1];
          goto LABEL_27;
        }
        v11 = (__m128i *)v19->m128i_u32[2];
        v16 = (__m128i *)&v73.m128i_u64[1];
        do
        {
          while ( 1 )
          {
            v24 = v23[1].m128i_i64[0];
            v25 = v23[1].m128i_i64[1];
            if ( v23[2].m128i_i32[0] >= (unsigned int)v11 )
              break;
            v23 = (__m128i *)v23[1].m128i_i64[1];
            if ( !v25 )
              goto LABEL_25;
          }
          v16 = v23;
          v23 = (__m128i *)v23[1].m128i_i64[0];
        }
        while ( v24 );
LABEL_25:
        if ( v16 == (__m128i *)&v73.m128i_u64[1] || (unsigned int)v11 < v16[2].m128i_i32[0] )
        {
LABEL_27:
          v11 = &v73;
          v70.m128i_i64[0] = (__int64)&v19->m128i_i64[1];
          v16 = (__m128i *)sub_1239060(&v73, (__int64)v16, (unsigned int **)&v70);
        }
        v26 = v19[1].m128i_i64[0];
        v70.m128i_i32[0] = *(_DWORD *)(v12 + 8);
        v70.m128i_i64[1] = v26;
        v27 = (__m128i *)v16[3].m128i_i64[0];
        if ( v27 == (__m128i *)v16[3].m128i_i64[1] )
        {
          v11 = (__m128i *)((char *)v16 + 40);
          v16 = (__m128i *)v16[3].m128i_i64[0];
          sub_12171B0((const __m128i **)v11, v16, &v70);
        }
        else
        {
          if ( v27 )
          {
            *v27 = _mm_loadu_si128(&v70);
            v27 = (__m128i *)v16[3].m128i_i64[0];
          }
          v16[3].m128i_i64[0] = (__int64)v27[1].m128i_i64;
        }
        v20 = *(unsigned int *)(v12 + 8);
        v22 = v19->m128i_i64[0];
        v21 = v20 + 1;
        if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
        {
LABEL_33:
          v16 = (__m128i *)(v12 + 16);
          v11 = (__m128i *)v12;
          v61 = v22;
          sub_C8D5F0(v12, (const void *)(v12 + 16), v21, 8u, v17, v18);
          v20 = *(unsigned int *)(v12 + 8);
          v22 = v61;
        }
      }
      else
      {
        v20 = *(unsigned int *)(v12 + 8);
        v21 = v20 + 1;
        if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
          goto LABEL_33;
      }
      v19 = (__m128i *)((char *)v19 + 24);
      *(_QWORD *)(*(_QWORD *)v12 + 8 * v20) = v22;
      ++*(_DWORD *)(v12 + 8);
      if ( v3 != v19 )
        continue;
      break;
    }
    if ( v75 != (__m128i *)&v73.m128i_u64[1] )
    {
      v59 = v13;
      v29 = v75;
      v62 = v13 + 1536;
      while ( 1 )
      {
        v70.m128i_i32[0] = v29[2].m128i_i32[0];
        v30 = (const __m128i *)v29[3].m128i_i64[0];
        v31 = (const __m128i *)v29[2].m128i_i64[1];
        v70.m128i_i64[1] = 0;
        v71 = 0;
        v72 = 0;
        v32 = (char *)v30 - (char *)v31;
        if ( v30 == v31 )
        {
          v28 = 0;
        }
        else
        {
          if ( v32 > 0x7FFFFFFFFFFFFFF0LL )
            sub_4261EA(v11, v16, v20);
          v33 = sub_22077B0((char *)v30 - (char *)v31);
          v30 = (const __m128i *)v29[3].m128i_i64[0];
          v31 = (const __m128i *)v29[2].m128i_i64[1];
          v28 = v33;
        }
        v70.m128i_i64[1] = v28;
        v71 = v28;
        v72 = v28 + v32;
        if ( v30 == v31 )
        {
          v35 = (__m128i *)v28;
        }
        else
        {
          v34 = (__m128i *)v28;
          v35 = (__m128i *)(v28 + (char *)v30 - (char *)v31);
          do
          {
            if ( v34 )
              *v34 = _mm_loadu_si128(v31);
            ++v34;
            ++v31;
          }
          while ( v35 != v34 );
        }
        v71 = (__int64)v35;
        v36 = *(_QWORD *)(v59 + 1544);
        if ( v36 )
        {
          v16 = (__m128i *)v70.m128i_u32[0];
          v37 = v62;
          do
          {
            while ( 1 )
            {
              v38 = *(_QWORD *)(v36 + 16);
              v39 = *(_QWORD *)(v36 + 24);
              if ( *(_DWORD *)(v36 + 32) >= v70.m128i_i32[0] )
                break;
              v36 = *(_QWORD *)(v36 + 24);
              if ( !v39 )
                goto LABEL_57;
            }
            v37 = v36;
            v36 = *(_QWORD *)(v36 + 16);
          }
          while ( v38 );
LABEL_57:
          if ( v37 != v62 && v70.m128i_i32[0] >= *(_DWORD *)(v37 + 32) )
            goto LABEL_60;
        }
        else
        {
          v37 = v62;
        }
        v16 = (__m128i *)v37;
        v69.m128i_i64[0] = (__int64)&v70;
        v40 = sub_12395C0((_QWORD *)(v59 + 1528), v37, (unsigned int **)&v69);
        v28 = v70.m128i_i64[1];
        v35 = (__m128i *)v71;
        v37 = v40;
LABEL_60:
        if ( v35 != (__m128i *)v28 )
        {
          v41 = (const __m128i *)v28;
          v42 = &v69.m128i_i64[1];
          v43 = v35;
          do
          {
            v44 = *(_QWORD *)v12 + 8LL * v41->m128i_u32[0];
            v69 = _mm_loadu_si128(v41);
            v16 = *(__m128i **)(v37 + 48);
            v68 = v44;
            if ( v16 == *(__m128i **)(v37 + 56) )
            {
              v67 = v42;
              sub_12135D0((const __m128i **)(v37 + 40), v16, &v68, v42);
              v42 = v67;
            }
            else
            {
              if ( v16 )
              {
                v16->m128i_i64[0] = v44;
                v16->m128i_i64[1] = v69.m128i_i64[1];
                v16 = *(__m128i **)(v37 + 48);
              }
              *(_QWORD *)(v37 + 48) = ++v16;
            }
            ++v41;
          }
          while ( v43 != v41 );
          v28 = v70.m128i_i64[1];
        }
        if ( v28 )
        {
          v16 = (__m128i *)(v72 - v28);
          j_j___libc_free_0(v28, v72 - v28);
        }
        v11 = v29;
        v29 = (__m128i *)sub_220EEE0(v29);
        if ( v29 == (__m128i *)&v73.m128i_u64[1] )
        {
          v13 = v59;
          break;
        }
      }
    }
  }
  v65 = sub_120AFE0(v13, 13, "expected ')' in refs");
  sub_1207E40(v74);
  result = v65;
LABEL_40:
  if ( src )
  {
    v66 = result;
    j_j___libc_free_0(src, (char *)v57 - (char *)src);
    return v66;
  }
  return result;
}
