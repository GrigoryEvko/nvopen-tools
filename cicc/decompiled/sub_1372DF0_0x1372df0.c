// Function: sub_1372DF0
// Address: 0x1372df0
//
unsigned __int64 __fastcall sub_1372DF0(__int64 a1)
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
  __m128i v15; // xmm0
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
  __m128i v27; // xmm1
  __m128i *v28; // rdx
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned int v31; // esi
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  _DWORD *v35; // rax
  _DWORD *v36; // r8
  _DWORD *i; // rdx
  const __m128i *v38; // rbx
  __int64 v39; // rax
  __int64 v40; // r13
  __int32 v41; // eax
  unsigned int v42; // edi
  __int32 *v43; // rcx
  int v44; // r9d
  __int64 v45; // rax
  __int64 v46; // rax
  __int32 v47; // r9d
  unsigned int v48; // esi
  __int32 *v49; // rdx
  int v50; // r15d
  int v51; // eax
  int v52; // r8d
  __int32 *v53; // rdi
  __int64 v54; // rdx
  _DWORD *v55; // rdi
  int v56; // r11d
  __int32 v57; // eax
  __int32 v58; // r9d
  int v59; // r8d
  unsigned int v60; // esi
  int v61; // r15d
  _DWORD *v62; // r13
  _DWORD *v63; // rax
  _DWORD *v64; // rbx
  __int64 v65; // rax
  __m128i v66; // xmm3
  __int64 v67; // [rsp-58h] [rbp-58h] BYREF
  _DWORD *v68; // [rsp-50h] [rbp-50h]
  __int64 v69; // [rsp-48h] [rbp-48h]
  unsigned int v70; // [rsp-40h] [rbp-40h]

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
    sub_1370740(*(__m128i **)a1, v6, 2LL * (int)(63 - (v7 ^ 0x3F)));
    if ( (unsigned __int64)v4 > 0x100 )
    {
      v22 = v5 + 256;
      sub_1370510(v5, v5 + 256);
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
      sub_1370510(v5, (char *)v6);
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
  v67 = 0;
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
    v38 = *(const __m128i **)a1;
    v36 = 0;
    v68 = 0;
    v69 = 0;
    v70 = 0;
    v40 = (__int64)v38[v3].m128i_i64;
    goto LABEL_54;
  }
  v32 = (4 * (int)v30 / 3u + 1) | ((unsigned __int64)(4 * (int)v30 / 3u + 1) >> 1);
  v33 = (((v32 >> 2) | v32) >> 4) | (v32 >> 2) | v32;
  v34 = ((((v33 >> 8) | v33) >> 16) | (v33 >> 8) | v33) + 1;
  v70 = v34;
  v35 = (_DWORD *)sub_22077B0(24 * v34);
  v69 = 0;
  v36 = v35;
  v68 = v35;
  v31 = v70;
  for ( i = &v35[6 * v70]; i != v35; v35 += 6 )
  {
    if ( v35 )
      *v35 = -1;
  }
  v38 = *(const __m128i **)a1;
  v39 = *(unsigned int *)(a1 + 8);
  v40 = *(_QWORD *)a1 + 16 * v39;
  if ( *(_QWORD *)a1 == v40 )
  {
    if ( (_DWORD)v39 )
    {
      *(_DWORD *)(a1 + 8) = 0;
      goto LABEL_66;
    }
    goto LABEL_102;
  }
  while ( 1 )
  {
LABEL_54:
    if ( !v31 )
    {
      ++v67;
      goto LABEL_56;
    }
    v41 = v38->m128i_i32[1];
    v42 = (v31 - 1) & (37 * v41);
    v43 = &v36[6 * v42];
    v44 = *v43;
    if ( v41 != *v43 )
    {
      v56 = 1;
      v49 = 0;
      while ( v44 != -1 )
      {
        if ( !v49 && v44 == -2 )
          v49 = v43;
        v42 = (v31 - 1) & (v56 + v42);
        v43 = &v36[6 * v42];
        v44 = *v43;
        if ( v41 == *v43 )
          goto LABEL_49;
        ++v56;
      }
      if ( !v49 )
        v49 = v43;
      ++v67;
      v51 = v69 + 1;
      if ( 4 * ((int)v69 + 1) < 3 * v31 )
      {
        if ( v31 - (v51 + HIDWORD(v69)) > v31 >> 3 )
          goto LABEL_74;
        sub_1372C20((__int64)&v67, v31);
        if ( !v70 )
        {
LABEL_117:
          LODWORD(v69) = v69 + 1;
          BUG();
        }
        v58 = v38->m128i_i32[1];
        v53 = 0;
        v59 = 1;
        v60 = (v70 - 1) & (37 * v58);
        v49 = &v68[6 * v60];
        v61 = *v49;
        v51 = v69 + 1;
        if ( *v49 == v58 )
          goto LABEL_74;
        while ( v61 != -1 )
        {
          if ( v61 == -2 && !v53 )
            v53 = v49;
          v60 = (v70 - 1) & (v59 + v60);
          v49 = &v68[6 * v60];
          v61 = *v49;
          if ( v58 == *v49 )
            goto LABEL_74;
          ++v59;
        }
        goto LABEL_60;
      }
LABEL_56:
      sub_1372C20((__int64)&v67, 2 * v31);
      if ( !v70 )
        goto LABEL_117;
      v47 = v38->m128i_i32[1];
      v48 = (v70 - 1) & (37 * v47);
      v49 = &v68[6 * v48];
      v50 = *v49;
      v51 = v69 + 1;
      if ( v47 == *v49 )
        goto LABEL_74;
      v52 = 1;
      v53 = 0;
      while ( v50 != -1 )
      {
        if ( !v53 && v50 == -2 )
          v53 = v49;
        v48 = (v70 - 1) & (v52 + v48);
        v49 = &v68[6 * v48];
        v50 = *v49;
        if ( v47 == *v49 )
          goto LABEL_74;
        ++v52;
      }
LABEL_60:
      if ( v53 )
        v49 = v53;
LABEL_74:
      LODWORD(v69) = v51;
      if ( *v49 != -1 )
        --HIDWORD(v69);
      v57 = v38->m128i_i32[1];
      *((_QWORD *)v49 + 1) = 0xFFFFFFFF00000000LL;
      *((_QWORD *)v49 + 2) = 0;
      *v49 = v57;
      goto LABEL_77;
    }
LABEL_49:
    v45 = *((_QWORD *)v43 + 2);
    if ( !v45 )
    {
      v49 = v43;
LABEL_77:
      *(__m128i *)(v49 + 2) = _mm_loadu_si128(v38);
      goto LABEL_52;
    }
    v12 = __CFADD__(v38->m128i_i64[1], v45);
    v46 = v38->m128i_i64[1] + v45;
    *((_QWORD *)v43 + 2) = v12 ? -1LL : v46;
LABEL_52:
    if ( (const __m128i *)v40 == ++v38 )
      break;
    v36 = v68;
    v31 = v70;
  }
  v54 = (unsigned int)v69;
  if ( *(_DWORD *)(a1 + 8) == (_DWORD)v69 )
  {
LABEL_102:
    j___libc_free_0(v68);
    result = *(unsigned int *)(a1 + 8);
    goto LABEL_20;
  }
  *(_DWORD *)(a1 + 8) = 0;
  if ( *(_DWORD *)(a1 + 12) < (unsigned int)v54 )
    sub_16CD150(a1, a1 + 16, v54, 16);
LABEL_66:
  v55 = v68;
  if ( (_DWORD)v69 )
  {
    v62 = &v68[6 * v70];
    if ( v62 != v68 )
    {
      v63 = v68;
      while ( 1 )
      {
        v64 = v63;
        if ( *v63 <= 0xFFFFFFFD )
          break;
        v63 += 6;
        if ( v62 == v63 )
          goto LABEL_67;
      }
      if ( v62 != v63 )
      {
        v65 = *(unsigned int *)(a1 + 8);
        do
        {
          if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v65 )
          {
            sub_16CD150(a1, a1 + 16, 0, 16);
            v65 = *(unsigned int *)(a1 + 8);
          }
          v66 = _mm_loadu_si128((const __m128i *)(v64 + 2));
          v64 += 6;
          *(__m128i *)(*(_QWORD *)a1 + 16 * v65) = v66;
          v65 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
          *(_DWORD *)(a1 + 8) = v65;
          if ( v64 == v62 )
            break;
          while ( *v64 > 0xFFFFFFFD )
          {
            v64 += 6;
            if ( v62 == v64 )
              goto LABEL_98;
          }
        }
        while ( v62 != v64 );
LABEL_98:
        v55 = v68;
      }
    }
  }
LABEL_67:
  j___libc_free_0(v55);
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
