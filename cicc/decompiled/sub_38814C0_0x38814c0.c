// Function: sub_38814C0
// Address: 0x38814c0
//
__int64 __fastcall sub_38814C0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  _QWORD *v4; // rdi
  __int64 v5; // rdx
  _QWORD *v6; // rdi
  __int64 v7; // rdx
  _QWORD *v8; // rdi
  __int64 v9; // rdx
  unsigned __int64 v10; // rdi
  __int64 v11; // r9
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // r11
  unsigned __int64 v15; // r10
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  const __m128i *v19; // rbx
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rax
  const __m128i *v22; // r10
  __m128i *v23; // r9
  const __m128i *v24; // rdx
  const __m128i *v25; // rax
  __m128i *v26; // rsi
  __m128i v27; // xmm0
  const __m128i *v28; // rdx
  size_t v29; // rdx
  size_t v30; // rdx
  size_t v31; // rdx
  unsigned __int64 v32; // r12
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // r15
  unsigned __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // r12
  unsigned __int64 v38; // rdi
  const __m128i *v39; // r15
  __int64 v40; // r8
  unsigned __int64 v41; // rdi
  __m128i *v42; // rsi
  __m128i *v43; // r8
  __int64 v44; // r12
  __m128i *v45; // r11
  __int64 v46; // rdx
  __m128i *v47; // rdi
  __m128i *v48; // rax
  size_t v49; // rdx
  __m128i *v50; // rsi
  __m128i *v51; // r13
  __int64 v52; // r10
  __m128i *v53; // r12
  __int64 v54; // rdx
  __m128i *v55; // rdi
  __m128i *v56; // rax
  size_t v57; // rdx
  __m128i *v58; // [rsp+0h] [rbp-1D0h]
  __int64 v59; // [rsp+0h] [rbp-1D0h]
  unsigned int v60; // [rsp+Ch] [rbp-1C4h]
  unsigned __int64 v61; // [rsp+10h] [rbp-1C0h]
  __m128i *v62; // [rsp+10h] [rbp-1C0h]
  unsigned __int64 v63; // [rsp+10h] [rbp-1C0h]
  __int64 v64; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v65; // [rsp+18h] [rbp-1B8h]
  _QWORD v66[2]; // [rsp+30h] [rbp-1A0h] BYREF
  _QWORD *v67; // [rsp+40h] [rbp-190h]
  size_t v68; // [rsp+48h] [rbp-188h]
  _QWORD v69[2]; // [rsp+50h] [rbp-180h] BYREF
  int v70; // [rsp+60h] [rbp-170h]
  int v71; // [rsp+64h] [rbp-16Ch]
  int v72; // [rsp+68h] [rbp-168h]
  _QWORD *v73; // [rsp+70h] [rbp-160h]
  size_t n; // [rsp+78h] [rbp-158h]
  _QWORD v75[2]; // [rsp+80h] [rbp-150h] BYREF
  _QWORD *v76; // [rsp+90h] [rbp-140h]
  size_t v77; // [rsp+98h] [rbp-138h]
  _QWORD v78[2]; // [rsp+A0h] [rbp-130h] BYREF
  unsigned __int64 v79; // [rsp+B0h] [rbp-120h]
  __int64 v80; // [rsp+B8h] [rbp-118h]
  __int64 v81; // [rsp+C0h] [rbp-110h]
  const __m128i *v82; // [rsp+C8h] [rbp-108h] BYREF
  unsigned int v83; // [rsp+D0h] [rbp-100h]
  int v84; // [rsp+D4h] [rbp-FCh]
  _BYTE v85[32]; // [rsp+D8h] [rbp-F8h] BYREF
  _BYTE v86[216]; // [rsp+F8h] [rbp-D8h] BYREF

  sub_16D0E30((__int64)v66, *(__int64 **)(a1 + 32), a2, 0, a3, a2, 0, 0, 0, 0);
  v3 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)v3 = v66[0];
  v4 = *(_QWORD **)(v3 + 16);
  *(_QWORD *)(v3 + 8) = v66[1];
  if ( v67 == v69 )
  {
    v30 = v68;
    if ( v68 )
    {
      if ( v68 == 1 )
        *(_BYTE *)v4 = v69[0];
      else
        memcpy(v4, v69, v68);
      v30 = v68;
      v4 = *(_QWORD **)(v3 + 16);
    }
    *(_QWORD *)(v3 + 24) = v30;
    *((_BYTE *)v4 + v30) = 0;
    v4 = v67;
  }
  else
  {
    if ( v4 == (_QWORD *)(v3 + 32) )
    {
      *(_QWORD *)(v3 + 16) = v67;
      *(_QWORD *)(v3 + 24) = v68;
      *(_QWORD *)(v3 + 32) = v69[0];
    }
    else
    {
      *(_QWORD *)(v3 + 16) = v67;
      v5 = *(_QWORD *)(v3 + 32);
      *(_QWORD *)(v3 + 24) = v68;
      *(_QWORD *)(v3 + 32) = v69[0];
      if ( v4 )
      {
        v67 = v4;
        v69[0] = v5;
        goto LABEL_5;
      }
    }
    v67 = v69;
    v4 = v69;
  }
LABEL_5:
  v68 = 0;
  *(_BYTE *)v4 = 0;
  v6 = *(_QWORD **)(v3 + 64);
  *(_DWORD *)(v3 + 48) = v70;
  *(_DWORD *)(v3 + 52) = v71;
  *(_DWORD *)(v3 + 56) = v72;
  if ( v73 == v75 )
  {
    v29 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v6 = v75[0];
      else
        memcpy(v6, v75, n);
      v29 = n;
      v6 = *(_QWORD **)(v3 + 64);
    }
    *(_QWORD *)(v3 + 72) = v29;
    *((_BYTE *)v6 + v29) = 0;
    v6 = v73;
  }
  else
  {
    if ( v6 == (_QWORD *)(v3 + 80) )
    {
      *(_QWORD *)(v3 + 64) = v73;
      *(_QWORD *)(v3 + 72) = n;
      *(_QWORD *)(v3 + 80) = v75[0];
    }
    else
    {
      *(_QWORD *)(v3 + 64) = v73;
      v7 = *(_QWORD *)(v3 + 80);
      *(_QWORD *)(v3 + 72) = n;
      *(_QWORD *)(v3 + 80) = v75[0];
      if ( v6 )
      {
        v73 = v6;
        v75[0] = v7;
        goto LABEL_9;
      }
    }
    v73 = v75;
    v6 = v75;
  }
LABEL_9:
  n = 0;
  *(_BYTE *)v6 = 0;
  v8 = *(_QWORD **)(v3 + 96);
  if ( v76 == v78 )
  {
    v31 = v77;
    if ( v77 )
    {
      if ( v77 == 1 )
        *(_BYTE *)v8 = v78[0];
      else
        memcpy(v8, v78, v77);
      v31 = v77;
      v8 = *(_QWORD **)(v3 + 96);
    }
    *(_QWORD *)(v3 + 104) = v31;
    *((_BYTE *)v8 + v31) = 0;
    v8 = v76;
  }
  else
  {
    if ( v8 == (_QWORD *)(v3 + 112) )
    {
      *(_QWORD *)(v3 + 96) = v76;
      *(_QWORD *)(v3 + 104) = v77;
      *(_QWORD *)(v3 + 112) = v78[0];
    }
    else
    {
      *(_QWORD *)(v3 + 96) = v76;
      v9 = *(_QWORD *)(v3 + 112);
      *(_QWORD *)(v3 + 104) = v77;
      *(_QWORD *)(v3 + 112) = v78[0];
      if ( v8 )
      {
        v76 = v8;
        v78[0] = v9;
        goto LABEL_13;
      }
    }
    v76 = v78;
    v8 = v78;
  }
LABEL_13:
  v77 = 0;
  *(_BYTE *)v8 = 0;
  v10 = *(_QWORD *)(v3 + 128);
  *(_QWORD *)(v3 + 128) = v79;
  v79 = 0;
  *(_QWORD *)(v3 + 136) = v80;
  v80 = 0;
  *(_QWORD *)(v3 + 144) = v81;
  v81 = 0;
  if ( v10 )
    j_j___libc_free_0(v10);
  v11 = v3 + 152;
  v12 = (unsigned __int64)v82;
  if ( (const __m128i **)(v3 + 152) == &v82 )
  {
    v19 = &v82[3 * v83];
    if ( v19 != v82 )
    {
      do
      {
        v19 -= 3;
        v20 = v19[1].m128i_u64[0];
        if ( (const __m128i *)v20 != &v19[2] )
          j_j___libc_free_0(v20);
      }
      while ( v19 != (const __m128i *)v12 );
      v12 = (unsigned __int64)v82;
    }
    goto LABEL_41;
  }
  v13 = *(_QWORD *)(v3 + 152);
  v14 = *(unsigned int *)(v3 + 160);
  v15 = v13;
  if ( v82 != (const __m128i *)v85 )
  {
    v16 = v13 + 48 * v14;
    if ( v16 != v13 )
    {
      do
      {
        v16 -= 48LL;
        v17 = *(_QWORD *)(v16 + 16);
        if ( v17 != v16 + 32 )
          j_j___libc_free_0(v17);
      }
      while ( v16 != v13 );
      v15 = *(_QWORD *)(v3 + 152);
    }
    if ( v15 != v3 + 168 )
      _libc_free(v15);
    *(_QWORD *)(v3 + 152) = v82;
    *(_DWORD *)(v3 + 160) = v83;
    *(_DWORD *)(v3 + 164) = v84;
    goto LABEL_25;
  }
  v21 = v83;
  v60 = v83;
  if ( v83 <= v14 )
  {
    v36 = *(_QWORD *)(v3 + 152);
    if ( !v83 )
    {
LABEL_85:
      v37 = v36 + 48 * v14;
      while ( v15 != v37 )
      {
        v37 -= 48;
        v38 = *(_QWORD *)(v37 + 16);
        if ( v38 != v37 + 32 )
        {
          v65 = v15;
          j_j___libc_free_0(v38);
          v15 = v65;
        }
      }
      *(_DWORD *)(v3 + 160) = v60;
      v39 = v82;
      v40 = 3LL * v83;
      v12 = (unsigned __int64)&v82[v40];
      if ( v82 != &v82[v40] )
      {
        do
        {
          v12 -= 48LL;
          v41 = *(_QWORD *)(v12 + 16);
          if ( v41 != v12 + 32 )
            j_j___libc_free_0(v41);
        }
        while ( v39 != (const __m128i *)v12 );
        v12 = (unsigned __int64)v82;
      }
      goto LABEL_41;
    }
    v42 = (__m128i *)v86;
    v43 = (__m128i *)(v13 + 32);
    v44 = 48LL * v83;
    v45 = (__m128i *)&v86[v44];
    while ( 1 )
    {
      v47 = (__m128i *)v43[-1].m128i_i64[0];
      v43[-2] = _mm_loadu_si128(v42 - 2);
      v48 = (__m128i *)v42[-1].m128i_i64[0];
      if ( v48 == v42 )
      {
        v49 = v42[-1].m128i_u64[1];
        if ( v49 )
        {
          if ( v49 == 1 )
          {
            v47->m128i_i8[0] = v42->m128i_i8[0];
            v49 = v42[-1].m128i_u64[1];
            v47 = (__m128i *)v43[-1].m128i_i64[0];
          }
          else
          {
            v58 = v43;
            v62 = v45;
            memcpy(v47, v42, v49);
            v43 = v58;
            v45 = v62;
            v49 = v42[-1].m128i_u64[1];
            v47 = (__m128i *)v58[-1].m128i_i64[0];
          }
        }
        v43[-1].m128i_i64[1] = v49;
        v47->m128i_i8[v49] = 0;
        v47 = (__m128i *)v42[-1].m128i_i64[0];
        goto LABEL_101;
      }
      if ( v43 == v47 )
        break;
      v43[-1].m128i_i64[0] = (__int64)v48;
      v46 = v43->m128i_i64[0];
      v43[-1].m128i_i64[1] = v42[-1].m128i_i64[1];
      v43->m128i_i64[0] = v42->m128i_i64[0];
      if ( !v47 )
        goto LABEL_108;
      v42[-1].m128i_i64[0] = (__int64)v47;
      v42->m128i_i64[0] = v46;
LABEL_101:
      v42[-1].m128i_i64[1] = 0;
      v42 += 3;
      v43 += 3;
      v47->m128i_i8[0] = 0;
      if ( v45 == v42 )
      {
        v36 = *(_QWORD *)(v3 + 152);
        v14 = *(unsigned int *)(v3 + 160);
        v15 = v13 + v44;
        goto LABEL_85;
      }
    }
    v43[-1].m128i_i64[0] = (__int64)v48;
    v43[-1].m128i_i64[1] = v42[-1].m128i_i64[1];
    v43->m128i_i64[0] = v42->m128i_i64[0];
LABEL_108:
    v42[-1].m128i_i64[0] = (__int64)v42;
    v47 = v42;
    goto LABEL_101;
  }
  if ( v83 > (unsigned __int64)*(unsigned int *)(v3 + 164) )
  {
    v32 = v13 + 48 * v14;
    while ( v32 != v13 )
    {
      while ( 1 )
      {
        v32 -= 48LL;
        v33 = *(_QWORD *)(v32 + 16);
        if ( v33 == v32 + 32 )
          break;
        v61 = v21;
        v64 = v11;
        j_j___libc_free_0(v33);
        v21 = v61;
        v11 = v64;
        if ( v32 == v13 )
          goto LABEL_77;
      }
    }
LABEL_77:
    *(_DWORD *)(v3 + 160) = 0;
    sub_166DC00(v11, v21);
    v12 = (unsigned __int64)v82;
    v21 = v83;
    v14 = 0;
    v13 = *(_QWORD *)(v3 + 152);
    v22 = v82;
    goto LABEL_46;
  }
  v22 = (const __m128i *)v85;
  if ( !*(_DWORD *)(v3 + 160) )
    goto LABEL_46;
  v50 = (__m128i *)v86;
  v51 = (__m128i *)(v13 + 32);
  v52 = 48 * v14;
  v14 = v52;
  v53 = (__m128i *)&v86[v52];
  do
  {
    v55 = (__m128i *)v51[-1].m128i_i64[0];
    v51[-2] = _mm_loadu_si128(v50 - 2);
    v56 = (__m128i *)v50[-1].m128i_i64[0];
    if ( v56 == v50 )
    {
      v57 = v50[-1].m128i_u64[1];
      if ( v57 )
      {
        if ( v57 == 1 )
        {
          v55->m128i_i8[0] = v50->m128i_i8[0];
          v57 = v50[-1].m128i_u64[1];
          v55 = (__m128i *)v51[-1].m128i_i64[0];
        }
        else
        {
          v59 = v52;
          v63 = v14;
          memcpy(v55, v50, v57);
          v55 = (__m128i *)v51[-1].m128i_i64[0];
          v52 = v59;
          v14 = v63;
          v57 = v50[-1].m128i_u64[1];
        }
      }
      v51[-1].m128i_i64[1] = v57;
      v55->m128i_i8[v57] = 0;
      v55 = (__m128i *)v50[-1].m128i_i64[0];
    }
    else
    {
      if ( v55 == v51 )
      {
        v51[-1].m128i_i64[0] = (__int64)v56;
        v51[-1].m128i_i64[1] = v50[-1].m128i_i64[1];
        v51->m128i_i64[0] = v50->m128i_i64[0];
      }
      else
      {
        v51[-1].m128i_i64[0] = (__int64)v56;
        v54 = v51->m128i_i64[0];
        v51[-1].m128i_i64[1] = v50[-1].m128i_i64[1];
        v51->m128i_i64[0] = v50->m128i_i64[0];
        if ( v55 )
        {
          v50[-1].m128i_i64[0] = (__int64)v55;
          v50->m128i_i64[0] = v54;
          goto LABEL_114;
        }
      }
      v50[-1].m128i_i64[0] = (__int64)v50;
      v55 = v50;
    }
LABEL_114:
    v50[-1].m128i_i64[1] = 0;
    v50 += 3;
    v51 += 3;
    v55->m128i_i8[0] = 0;
  }
  while ( v50 != v53 );
  v12 = (unsigned __int64)v82;
  v21 = v83;
  v13 = *(_QWORD *)(v3 + 152);
  v22 = (const __m128i *)((char *)v82 + v52);
LABEL_46:
  v23 = (__m128i *)(v13 + v14);
  v24 = (const __m128i *)(v12 + 48 * v21);
  if ( v24 != v22 )
  {
    v25 = v22 + 2;
    v26 = &v23[3
             * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v24 - (char *)v22 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
             + 3];
    do
    {
      if ( v23 )
      {
        v27 = _mm_loadu_si128(v25 - 2);
        v23[1].m128i_i64[0] = (__int64)v23[2].m128i_i64;
        *v23 = v27;
        v28 = (const __m128i *)v25[-1].m128i_i64[0];
        if ( v28 == v25 )
        {
          v23[2] = _mm_loadu_si128(v25);
        }
        else
        {
          v23[1].m128i_i64[0] = (__int64)v28;
          v23[2].m128i_i64[0] = v25->m128i_i64[0];
        }
        v23[1].m128i_i64[1] = v25[-1].m128i_i64[1];
        v25[-1].m128i_i64[0] = (__int64)v25;
        v25[-1].m128i_i64[1] = 0;
        v25->m128i_i8[0] = 0;
      }
      v23 += 3;
      v25 += 3;
    }
    while ( v26 != v23 );
    v12 = (unsigned __int64)v82;
  }
  *(_DWORD *)(v3 + 160) = v60;
  v34 = v12 + 48LL * v83;
  if ( v34 != v12 )
  {
    do
    {
      v34 -= 48LL;
      v35 = *(_QWORD *)(v34 + 16);
      if ( v35 != v34 + 32 )
        j_j___libc_free_0(v35);
    }
    while ( v34 != v12 );
    v12 = (unsigned __int64)v82;
  }
LABEL_41:
  if ( (_BYTE *)v12 != v85 )
    _libc_free(v12);
LABEL_25:
  if ( v79 )
    j_j___libc_free_0(v79);
  if ( v76 != v78 )
    j_j___libc_free_0((unsigned __int64)v76);
  if ( v73 != v75 )
    j_j___libc_free_0((unsigned __int64)v73);
  if ( v67 != v69 )
    j_j___libc_free_0((unsigned __int64)v67);
  return 1;
}
