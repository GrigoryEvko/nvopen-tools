// Function: sub_FE9FC0
// Address: 0xfe9fc0
//
unsigned __int64 __fastcall sub_FE9FC0(__int64 a1)
{
  unsigned __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // r13
  char *v5; // r15
  unsigned __int64 *v6; // r14
  unsigned __int64 v7; // rax
  __m128i *v8; // rdi
  __int64 m128i_i64; // rsi
  __m128i *v10; // rcx
  const __m128i *v11; // rax
  bool v12; // cf
  __int64 v13; // rdx
  __int64 v14; // rdx
  __m128i v15; // xmm1
  __m128i *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdi
  char v19; // r8
  __m128i *v20; // rsi
  __m128i *j; // rdi
  char *v22; // rbx
  __int32 v23; // edi
  unsigned __int32 v24; // ecx
  const __m128i *v25; // rax
  __int64 v26; // rsi
  __m128i v27; // xmm2
  __m128i *v28; // rdx
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned int v31; // esi
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  _DWORD *v35; // rax
  __int64 v36; // r9
  __int64 v37; // r8
  _DWORD *i; // rdx
  const __m128i *v39; // rbx
  __int64 v40; // rax
  __int64 v41; // r13
  __int32 v42; // edx
  int v43; // r11d
  unsigned int v44; // edi
  unsigned int *v45; // rcx
  __int32 *v46; // rax
  __int64 v47; // rax
  __m128i *v48; // rdx
  __int64 v49; // rax
  int v50; // edx
  int v51; // esi
  __int32 v52; // r15d
  __int32 *v53; // rdi
  __int32 v54; // edx
  unsigned __int64 v55; // rdx
  __int64 v56; // rdi
  __int64 v57; // rsi
  int v58; // esi
  __int32 v59; // r15d
  _DWORD *v60; // r13
  _DWORD *v61; // rax
  _DWORD *v62; // rbx
  __int64 v63; // rax
  __m128i v64; // xmm0
  __m128i v65; // [rsp-68h] [rbp-68h] BYREF
  __int64 v66; // [rsp-58h] [rbp-58h] BYREF
  _DWORD *v67; // [rsp-50h] [rbp-50h]
  __int64 v68; // [rsp-48h] [rbp-48h]
  unsigned int v69; // [rsp-40h] [rbp-40h]

  result = *(unsigned int *)(a1 + 8);
  if ( !(_DWORD)result )
    return result;
  if ( (_DWORD)result == 1 )
  {
LABEL_3:
    *(_QWORD *)(a1 + 80) = 1;
    result = *(_QWORD *)a1;
    *(_QWORD *)(*(_QWORD *)a1 + 8LL) = 1;
    return result;
  }
  v3 = (unsigned int)result;
  if ( (unsigned int)result <= 0x80 )
  {
    v4 = 16LL * (unsigned int)result;
    v5 = *(char **)a1;
    v6 = (unsigned __int64 *)(*(_QWORD *)a1 + v4);
    _BitScanReverse64(&v7, v4 >> 4);
    sub_FE83C0(*(__m128i **)a1, v6, 2LL * (int)(63 - (v7 ^ 0x3F)));
    if ( (unsigned __int64)v4 > 0x100 )
    {
      v22 = v5 + 256;
      sub_FE82B0(v5, v5 + 256);
      if ( v6 != (unsigned __int64 *)(v5 + 256) )
      {
        do
        {
          v23 = *(_DWORD *)v22;
          v24 = *((_DWORD *)v22 + 1);
          v25 = (const __m128i *)(v22 - 16);
          v26 = *((_QWORD *)v22 + 1);
          if ( v24 >= *((_DWORD *)v22 - 3) )
          {
            v28 = (__m128i *)v22;
          }
          else
          {
            do
            {
              v27 = _mm_loadu_si128(v25);
              v28 = (__m128i *)v25--;
              v25[2] = v27;
            }
            while ( v24 < v25->m128i_i32[1] );
          }
          v22 += 16;
          v28->m128i_i32[0] = v23;
          v28->m128i_i32[1] = v24;
          v28->m128i_i64[1] = v26;
        }
        while ( v6 != (unsigned __int64 *)v22 );
      }
    }
    else
    {
      sub_FE82B0(v5, (char *)v6);
    }
    v8 = *(__m128i **)a1;
    m128i_i64 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 == m128i_i64 )
    {
      v16 = (__m128i *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
    }
    else
    {
      v10 = *(__m128i **)a1;
      while ( 1 )
      {
        v11 = v8 + 1;
        *v10 = _mm_loadu_si128(v8);
        if ( (__m128i *)m128i_i64 == &v8[1] )
        {
LABEL_18:
          v16 = *(__m128i **)a1;
          m128i_i64 = (__int64)v10[1].m128i_i64;
          goto LABEL_19;
        }
        while ( v8->m128i_i32[1] == v11->m128i_i32[1] )
        {
          v14 = v10->m128i_i64[1];
          if ( v14 )
          {
            v12 = __CFADD__(v11->m128i_i64[1], v14);
            v13 = v11->m128i_i64[1] + v14;
            if ( v12 )
              v10->m128i_i64[1] = -1;
            else
              v10->m128i_i64[1] = v13;
            if ( (const __m128i *)m128i_i64 == ++v11 )
              goto LABEL_18;
          }
          else
          {
            v15 = _mm_loadu_si128(v11++);
            *v10 = v15;
            if ( (const __m128i *)m128i_i64 == v11 )
              goto LABEL_18;
          }
        }
        ++v10;
        if ( (const __m128i *)m128i_i64 == v11 )
          break;
        v8 = (__m128i *)v11;
      }
      v16 = *(__m128i **)a1;
      m128i_i64 = (__int64)v10;
    }
LABEL_19:
    v17 = (m128i_i64 - (__int64)v16) >> 4;
    *(_DWORD *)(a1 + 8) = v17;
    result = (unsigned int)v17;
    goto LABEL_20;
  }
  v66 = 0;
  v29 = (((((((unsigned int)result | (2 * (unsigned __int64)(unsigned int)result)) >> 2)
          | (unsigned int)result
          | (2LL * (unsigned int)result)) >> 4)
        | (((unsigned int)result | (2 * (unsigned __int64)(unsigned int)result)) >> 2)
        | (unsigned int)result
        | (2LL * (unsigned int)result)) >> 8)
      | (((((unsigned int)result | (2 * (unsigned __int64)(unsigned int)result)) >> 2)
        | (unsigned int)result
        | (2LL * (unsigned int)result)) >> 4)
      | (((unsigned int)result | (2 * (unsigned __int64)(unsigned int)result)) >> 2)
      | (unsigned int)result
      | (2LL * (unsigned int)result);
  v30 = (v29
       | ((((((((v3 | (unsigned __int64)(2 * v3)) >> 2) | v3 | (2 * v3)) >> 4)
           | ((v3 | (unsigned __int64)(2 * v3)) >> 2)
           | v3
           | (2 * v3)) >> 8)
         | ((((v3 | (unsigned __int64)(2 * v3)) >> 2) | v3 | (2 * v3)) >> 4)
         | ((v3 | (unsigned __int64)(2 * v3)) >> 2)
         | v3
         | (2 * v3)) >> 16)
       | HIDWORD(v29))
      + 1;
  v31 = v30;
  if ( !(_DWORD)v30 )
  {
    v39 = *(const __m128i **)a1;
    v37 = 0;
    v67 = 0;
    v68 = 0;
    v69 = 0;
    v41 = (__int64)v39[v3].m128i_i64;
    goto LABEL_54;
  }
  v32 = (4 * (int)v30 / 3u + 1) | ((unsigned __int64)(4 * (int)v30 / 3u + 1) >> 1);
  v33 = (((v32 >> 2) | v32) >> 4) | (v32 >> 2) | v32;
  v34 = ((((v33 >> 8) | v33) >> 16) | (v33 >> 8) | v33) + 1;
  v69 = v34;
  v35 = (_DWORD *)sub_C7D670(24 * v34, 8);
  v68 = 0;
  v37 = (__int64)v35;
  v67 = v35;
  v31 = v69;
  for ( i = &v35[6 * v69]; i != v35; v35 += 6 )
  {
    if ( v35 )
      *v35 = -1;
  }
  v39 = *(const __m128i **)a1;
  v40 = *(unsigned int *)(a1 + 8);
  v41 = *(_QWORD *)a1 + 16 * v40;
  if ( *(_QWORD *)a1 == v41 )
  {
    if ( (_DWORD)v40 )
    {
      *(_DWORD *)(a1 + 8) = 0;
      goto LABEL_80;
    }
    goto LABEL_106;
  }
  while ( 1 )
  {
LABEL_54:
    if ( !v31 )
    {
      ++v66;
      goto LABEL_56;
    }
    v42 = v39->m128i_i32[1];
    v43 = 1;
    v44 = (v31 - 1) & (37 * v42);
    v45 = (unsigned int *)(v37 + 24LL * v44);
    v46 = 0;
    v36 = *v45;
    if ( v42 != (_DWORD)v36 )
    {
      while ( (_DWORD)v36 != -1 )
      {
        if ( !v46 && (_DWORD)v36 == -2 )
          v46 = (__int32 *)v45;
        v44 = (v31 - 1) & (v43 + v44);
        v45 = (unsigned int *)(v37 + 24LL * v44);
        v36 = *v45;
        if ( v42 == (_DWORD)v36 )
          goto LABEL_49;
        ++v43;
      }
      if ( !v46 )
        v46 = (__int32 *)v45;
      ++v66;
      v50 = v68 + 1;
      if ( 4 * ((int)v68 + 1) < 3 * v31 )
      {
        if ( v31 - (v50 + HIDWORD(v68)) > v31 >> 3 )
          goto LABEL_73;
        sub_FE9DE0((__int64)&v66, v31);
        if ( !v69 )
        {
LABEL_115:
          LODWORD(v68) = v68 + 1;
          BUG();
        }
        v36 = v39->m128i_u32[1];
        v37 = 1;
        v50 = v68 + 1;
        v53 = 0;
        v58 = (v69 - 1) & (37 * v36);
        v46 = &v67[6 * v58];
        v59 = *v46;
        if ( *v46 == (_DWORD)v36 )
          goto LABEL_73;
        while ( v59 != -1 )
        {
          if ( v59 == -2 && !v53 )
            v53 = v46;
          v58 = (v69 - 1) & (v37 + v58);
          v46 = &v67[6 * v58];
          v59 = *v46;
          if ( (_DWORD)v36 == *v46 )
            goto LABEL_73;
          v37 = (unsigned int)(v37 + 1);
        }
        goto LABEL_60;
      }
LABEL_56:
      sub_FE9DE0((__int64)&v66, 2 * v31);
      if ( !v69 )
        goto LABEL_115;
      v36 = v39->m128i_u32[1];
      v50 = v68 + 1;
      v51 = (v69 - 1) & (37 * v36);
      v46 = &v67[6 * v51];
      v52 = *v46;
      if ( (_DWORD)v36 == *v46 )
        goto LABEL_73;
      v37 = 1;
      v53 = 0;
      while ( v52 != -1 )
      {
        if ( !v53 && v52 == -2 )
          v53 = v46;
        v51 = (v69 - 1) & (v37 + v51);
        v46 = &v67[6 * v51];
        v52 = *v46;
        if ( (_DWORD)v36 == *v46 )
          goto LABEL_73;
        v37 = (unsigned int)(v37 + 1);
      }
LABEL_60:
      if ( v53 )
        v46 = v53;
LABEL_73:
      LODWORD(v68) = v50;
      if ( *v46 != -1 )
        --HIDWORD(v68);
      v54 = v39->m128i_i32[1];
      *((_QWORD *)v46 + 1) = 0xFFFFFFFF00000000LL;
      *((_QWORD *)v46 + 2) = 0;
      *v46 = v54;
      v48 = (__m128i *)(v46 + 2);
LABEL_76:
      *v48 = _mm_loadu_si128(v39);
      goto LABEL_52;
    }
LABEL_49:
    v47 = *((_QWORD *)v45 + 2);
    v48 = (__m128i *)(v45 + 2);
    if ( !v47 )
      goto LABEL_76;
    v12 = __CFADD__(v39->m128i_i64[1], v47);
    v49 = v39->m128i_i64[1] + v47;
    *((_QWORD *)v45 + 2) = v12 ? -1LL : v49;
LABEL_52:
    if ( ++v39 == (const __m128i *)v41 )
      break;
    v37 = (__int64)v67;
    v31 = v69;
  }
  v55 = (unsigned int)v68;
  if ( (_DWORD)v68 == *(_DWORD *)(a1 + 8) )
  {
LABEL_106:
    sub_C7D6A0((__int64)v67, 24LL * v69, 8);
    result = *(unsigned int *)(a1 + 8);
    goto LABEL_20;
  }
  *(_DWORD *)(a1 + 8) = 0;
  if ( v55 > *(unsigned int *)(a1 + 12) )
    sub_C8D5F0(a1, (const void *)(a1 + 16), v55, 0x10u, v37, v36);
LABEL_80:
  v56 = (__int64)v67;
  v57 = 6LL * v69;
  if ( (_DWORD)v68 )
  {
    v60 = &v67[v57];
    if ( &v67[v57] != v67 )
    {
      v61 = v67;
      while ( 1 )
      {
        v62 = v61;
        if ( *v61 <= 0xFFFFFFFD )
          break;
        v61 += 6;
        if ( v60 == v61 )
          goto LABEL_81;
      }
      if ( v60 != v61 )
      {
        v63 = *(unsigned int *)(a1 + 8);
        do
        {
          v64 = _mm_loadu_si128((const __m128i *)(v62 + 2));
          if ( v63 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
          {
            v65 = v64;
            sub_C8D5F0(a1, (const void *)(a1 + 16), v63 + 1, 0x10u, v37, v36);
            v63 = *(unsigned int *)(a1 + 8);
            v64 = _mm_load_si128(&v65);
          }
          v62 += 6;
          *(__m128i *)(*(_QWORD *)a1 + 16 * v63) = v64;
          v63 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
          *(_DWORD *)(a1 + 8) = v63;
          if ( v62 == v60 )
            break;
          while ( *v62 > 0xFFFFFFFD )
          {
            v62 += 6;
            if ( v60 == v62 )
              goto LABEL_102;
          }
        }
        while ( v62 != v60 );
LABEL_102:
        v56 = (__int64)v67;
        v57 = 6LL * v69;
      }
    }
  }
LABEL_81:
  sub_C7D6A0(v56, v57 * 4, 8);
  result = *(unsigned int *)(a1 + 8);
LABEL_20:
  v18 = (unsigned int)result;
  if ( (_DWORD)result == 1 )
    goto LABEL_3;
  if ( *(_BYTE *)(a1 + 88) )
  {
    v19 = 33;
    goto LABEL_24;
  }
  result = *(_QWORD *)(a1 + 80);
  if ( result > 0xFFFFFFFF )
  {
    _BitScanReverse64(&result, result);
    result ^= 0x3Fu;
    v19 = 33 - result;
LABEL_24:
    v20 = *(__m128i **)a1;
    *(_QWORD *)(a1 + 80) = 0;
    for ( j = &v20[v18]; j != v20; *(_QWORD *)(a1 + 80) += result )
    {
      result = (((unsigned __int64)v20->m128i_i64[1] >> (v19 - 1)) & 1) + ((unsigned __int64)v20->m128i_i64[1] >> v19);
      if ( !result )
        result = 1;
      ++v20;
      v20[-1].m128i_i64[1] = result;
    }
  }
  return result;
}
