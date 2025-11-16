// Function: sub_201ADC0
// Address: 0x201adc0
//
__int64 __fastcall sub_201ADC0(__int64 a1, unsigned __int64 a2, __int32 a3, unsigned __int64 a4, __int64 a5)
{
  const __m128i *v5; // r11
  char v12; // cl
  int v13; // ecx
  __int64 v14; // r10
  int v15; // esi
  __m128i *v16; // rdi
  __int64 result; // rax
  __m128i *v18; // rdx
  __int64 v19; // r8
  unsigned int v20; // eax
  unsigned int v21; // esi
  unsigned int v22; // eax
  int v23; // edx
  unsigned int v24; // r8d
  __m128i v25; // xmm0
  char v26; // cl
  int v27; // ecx
  __int64 v28; // rdi
  int v29; // esi
  int v30; // r9d
  __m128i *v31; // r8
  __m128i *v32; // rdx
  __int64 v33; // r10
  unsigned int v34; // eax
  unsigned int v35; // esi
  __int32 v36; // r8d
  __int32 v37; // r10d
  unsigned int v38; // eax
  int v39; // edx
  unsigned int v40; // edi
  __m128i v41; // xmm1
  __int64 v42; // rsi
  int v43; // edx
  int v44; // r10d
  __m128i *v45; // r8
  unsigned int i; // eax
  unsigned int v47; // eax
  __int64 v48; // rsi
  int v49; // edx
  int v50; // r10d
  unsigned int j; // eax
  unsigned int v52; // eax
  __int64 v53; // rcx
  int v54; // edx
  int v55; // r9d
  __m128i *v56; // rdi
  unsigned int k; // eax
  unsigned int v58; // eax
  __int64 v59; // rcx
  int v60; // edx
  int v61; // r9d
  unsigned int m; // eax
  unsigned int v63; // eax
  int v64; // edx
  int v65; // edx
  int v66; // edx
  int v67; // edx
  __int32 v68; // ecx
  __int32 v69; // ecx
  __int32 v70; // esi
  __int32 v71; // esi
  int v72; // [rsp+0h] [rbp-60h]
  const __m128i *v73; // [rsp+0h] [rbp-60h]
  const __m128i *v74; // [rsp+0h] [rbp-60h]
  __int32 v75; // [rsp+8h] [rbp-58h]
  __int32 v76; // [rsp+8h] [rbp-58h]
  int v77; // [rsp+Ch] [rbp-54h]
  __m128i v78; // [rsp+20h] [rbp-40h] BYREF

  v5 = (const __m128i *)(a1 + 24);
  v78.m128i_i64[0] = a4;
  v12 = *(_BYTE *)(a1 + 32);
  v77 = a5;
  v78.m128i_i64[1] = a5;
  v13 = v12 & 1;
  if ( v13 )
  {
    v14 = a1 + 40;
    v15 = 63;
  }
  else
  {
    v21 = *(_DWORD *)(a1 + 48);
    v14 = *(_QWORD *)(a1 + 40);
    if ( !v21 )
    {
      v22 = *(_DWORD *)(a1 + 32);
      ++*(_QWORD *)(a1 + 24);
      v16 = 0;
      v23 = (v22 >> 1) + 1;
LABEL_10:
      v24 = 3 * v21;
      goto LABEL_11;
    }
    v15 = v21 - 1;
  }
  v72 = 1;
  v16 = 0;
  for ( result = v15 & (a3 + ((unsigned int)(a2 >> 9) ^ (unsigned int)(a2 >> 4))); ; result = v15 & v20 )
  {
    v18 = (__m128i *)(v14 + 32LL * (unsigned int)result);
    v19 = v18->m128i_i64[0];
    if ( v18->m128i_i64[0] != a2 )
      break;
    if ( v18->m128i_i32[2] == a3 )
      goto LABEL_16;
    if ( !v19 )
      goto LABEL_29;
LABEL_6:
    v20 = v72 + result;
    ++v72;
  }
  if ( v19 )
    goto LABEL_6;
LABEL_29:
  v36 = v18->m128i_i32[2];
  if ( v36 != -1 )
  {
    if ( v36 == -2 && !v16 )
      v16 = (__m128i *)(v14 + 32LL * (unsigned int)result);
    goto LABEL_6;
  }
  v22 = *(_DWORD *)(a1 + 32);
  if ( !v16 )
    v16 = v18;
  ++*(_QWORD *)(a1 + 24);
  v23 = (v22 >> 1) + 1;
  if ( !(_BYTE)v13 )
  {
    v21 = *(_DWORD *)(a1 + 48);
    goto LABEL_10;
  }
  v24 = 192;
  v21 = 64;
LABEL_11:
  if ( 4 * v23 >= v24 )
  {
    v75 = a3;
    v73 = v5;
    sub_201A8D0(v5, 2 * v21);
    v5 = v73;
    if ( (*(_BYTE *)(a1 + 32) & 1) != 0 )
    {
      v42 = a1 + 40;
      v43 = 63;
    }
    else
    {
      v64 = *(_DWORD *)(a1 + 48);
      v42 = *(_QWORD *)(a1 + 40);
      if ( !v64 )
        goto LABEL_126;
      v43 = v64 - 1;
    }
    v44 = 1;
    v45 = 0;
    for ( i = v43 & (a3 + ((a2 >> 9) ^ (a2 >> 4))); ; i = v43 & v47 )
    {
      v16 = (__m128i *)(v42 + 32LL * i);
      if ( a2 == v16->m128i_i64[0] && v75 == v16->m128i_i32[2] )
        break;
      if ( !v16->m128i_i64[0] )
      {
        v68 = v16->m128i_i32[2];
        if ( v68 == -1 )
        {
LABEL_120:
          if ( v45 )
            v16 = v45;
          goto LABEL_89;
        }
        if ( v68 == -2 && !v45 )
          v45 = (__m128i *)(v42 + 32LL * i);
      }
      v47 = v44 + i;
      ++v44;
    }
    goto LABEL_89;
  }
  if ( v21 - *(_DWORD *)(a1 + 36) - v23 > v21 >> 3 )
    goto LABEL_13;
  v76 = a3;
  v74 = v5;
  sub_201A8D0(v5, v21);
  v5 = v74;
  if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
  {
    v65 = *(_DWORD *)(a1 + 48);
    v48 = *(_QWORD *)(a1 + 40);
    if ( v65 )
    {
      v49 = v65 - 1;
      goto LABEL_60;
    }
LABEL_126:
    *(_DWORD *)(a1 + 32) = (2 * (*(_DWORD *)(a1 + 32) >> 1) + 2) | *(_DWORD *)(a1 + 32) & 1;
    BUG();
  }
  v48 = a1 + 40;
  v49 = 63;
LABEL_60:
  v50 = 1;
  v45 = 0;
  for ( j = v49 & (a3 + ((a2 >> 9) ^ (a2 >> 4))); ; j = v49 & v52 )
  {
    v16 = (__m128i *)(v48 + 32LL * j);
    if ( a2 == v16->m128i_i64[0] && v76 == v16->m128i_i32[2] )
      break;
    if ( !v16->m128i_i64[0] )
    {
      v69 = v16->m128i_i32[2];
      if ( v69 == -1 )
        goto LABEL_120;
      if ( !v45 && v69 == -2 )
        v45 = (__m128i *)(v48 + 32LL * j);
    }
    v52 = v50 + j;
    ++v50;
  }
LABEL_89:
  v22 = *(_DWORD *)(a1 + 32);
LABEL_13:
  result = (2 * (v22 >> 1) + 2) | v22 & 1;
  *(_DWORD *)(a1 + 32) = result;
  if ( v16->m128i_i64[0] || v16->m128i_i32[2] != -1 )
    --*(_DWORD *)(a1 + 36);
  v78.m128i_i64[0] = a4;
  v78.m128i_i32[2] = a5;
  v25 = _mm_loadu_si128(&v78);
  v16->m128i_i64[0] = a2;
  v16->m128i_i32[2] = a3;
  v16[1] = v25;
LABEL_16:
  if ( a2 == a4 && a3 == (_DWORD)a5 )
    return result;
  v26 = *(_BYTE *)(a1 + 32);
  v78.m128i_i64[0] = a4;
  v78.m128i_i64[1] = a5;
  v27 = v26 & 1;
  if ( v27 )
  {
    v28 = a1 + 40;
    v29 = 63;
    goto LABEL_19;
  }
  v35 = *(_DWORD *)(a1 + 48);
  v28 = *(_QWORD *)(a1 + 40);
  if ( !v35 )
  {
    v38 = *(_DWORD *)(a1 + 32);
    ++*(_QWORD *)(a1 + 24);
    v31 = 0;
    v39 = (v38 >> 1) + 1;
LABEL_42:
    v40 = 3 * v35;
    goto LABEL_43;
  }
  v29 = v35 - 1;
LABEL_19:
  v30 = 1;
  v31 = 0;
  result = v29 & ((unsigned int)a5 + ((unsigned int)(a4 >> 9) ^ (unsigned int)(a4 >> 4)));
  while ( 2 )
  {
    v32 = (__m128i *)(v28 + 32LL * (unsigned int)result);
    v33 = v32->m128i_i64[0];
    if ( v32->m128i_i64[0] == a4 )
    {
      if ( v32->m128i_i32[2] == v77 )
        return result;
      if ( v33 )
        goto LABEL_22;
    }
    else if ( v33 )
    {
LABEL_22:
      v34 = v30 + result;
      ++v30;
      result = v29 & v34;
      continue;
    }
    break;
  }
  v37 = v32->m128i_i32[2];
  if ( v37 != -1 )
  {
    if ( !v31 && v37 == -2 )
      v31 = (__m128i *)(v28 + 32LL * (unsigned int)result);
    goto LABEL_22;
  }
  v38 = *(_DWORD *)(a1 + 32);
  if ( !v31 )
    v31 = v32;
  ++*(_QWORD *)(a1 + 24);
  v39 = (v38 >> 1) + 1;
  if ( !(_BYTE)v27 )
  {
    v35 = *(_DWORD *)(a1 + 48);
    goto LABEL_42;
  }
  v40 = 192;
  v35 = 64;
LABEL_43:
  if ( v40 <= 4 * v39 )
  {
    sub_201A8D0(v5, 2 * v35);
    if ( (*(_BYTE *)(a1 + 32) & 1) != 0 )
    {
      v53 = a1 + 40;
      v54 = 63;
    }
    else
    {
      v66 = *(_DWORD *)(a1 + 48);
      v53 = *(_QWORD *)(a1 + 40);
      if ( !v66 )
        goto LABEL_127;
      v54 = v66 - 1;
    }
    v55 = 1;
    v56 = 0;
    for ( k = v54 & (a5 + ((a4 >> 9) ^ (a4 >> 4))); ; k = v54 & v58 )
    {
      v31 = (__m128i *)(v53 + 32LL * k);
      if ( v31->m128i_i64[0] == a4 && v77 == v31->m128i_i32[2] )
        break;
      if ( !v31->m128i_i64[0] )
      {
        v70 = v31->m128i_i32[2];
        if ( v70 == -1 )
        {
LABEL_122:
          if ( v56 )
            v31 = v56;
          goto LABEL_95;
        }
        if ( !v56 && v70 == -2 )
          v56 = (__m128i *)(v53 + 32LL * k);
      }
      v58 = v55 + k;
      ++v55;
    }
    goto LABEL_95;
  }
  if ( v35 - *(_DWORD *)(a1 + 36) - v39 <= v35 >> 3 )
  {
    sub_201A8D0(v5, v35);
    if ( (*(_BYTE *)(a1 + 32) & 1) != 0 )
    {
      v59 = a1 + 40;
      v60 = 63;
      goto LABEL_72;
    }
    v67 = *(_DWORD *)(a1 + 48);
    v59 = *(_QWORD *)(a1 + 40);
    if ( v67 )
    {
      v60 = v67 - 1;
LABEL_72:
      v61 = 1;
      v56 = 0;
      for ( m = v60 & (a5 + ((a4 >> 9) ^ (a4 >> 4))); ; m = v60 & v63 )
      {
        v31 = (__m128i *)(v59 + 32LL * m);
        if ( a4 == v31->m128i_i64[0] && v77 == v31->m128i_i32[2] )
          break;
        if ( !v31->m128i_i64[0] )
        {
          v71 = v31->m128i_i32[2];
          if ( v71 == -1 )
            goto LABEL_122;
          if ( v71 == -2 && !v56 )
            v56 = (__m128i *)(v59 + 32LL * m);
        }
        v63 = v61 + m;
        ++v61;
      }
LABEL_95:
      v38 = *(_DWORD *)(a1 + 32);
      goto LABEL_45;
    }
LABEL_127:
    *(_DWORD *)(a1 + 32) = (2 * (*(_DWORD *)(a1 + 32) >> 1) + 2) | *(_DWORD *)(a1 + 32) & 1;
    BUG();
  }
LABEL_45:
  result = (2 * (v38 >> 1) + 2) | v38 & 1;
  *(_DWORD *)(a1 + 32) = result;
  if ( v31->m128i_i64[0] || v31->m128i_i32[2] != -1 )
    --*(_DWORD *)(a1 + 36);
  v78.m128i_i64[0] = a4;
  v78.m128i_i32[2] = a5;
  v41 = _mm_loadu_si128(&v78);
  v31->m128i_i64[0] = a4;
  v31->m128i_i32[2] = a5;
  v31[1] = v41;
  return result;
}
