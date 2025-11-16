// Function: sub_37CF670
// Address: 0x37cf670
//
void __fastcall sub_37CF670(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int32 a5,
        __int64 a6,
        unsigned int a7,
        int a8,
        int a9,
        int a10,
        unsigned int a11,
        const __m128i a12)
{
  __m128i *v14; // rbx
  __int64 v15; // rcx
  int *v16; // r8
  unsigned int *v17; // r14
  __m128i *v18; // r10
  __m128i *v19; // rdi
  __int64 v20; // rbx
  unsigned int v21; // r11d
  __m128i *v22; // rax
  unsigned __int64 v23; // rsi
  int *v24; // r12
  __int64 v25; // rdx
  __int64 v26; // rdx
  unsigned __int64 v27; // rdx
  const __m128i *v28; // r13
  __m128i *v29; // rdi
  __m128i v30; // xmm3
  __int64 v31; // rcx
  __m128i v32; // xmm1
  __m128i v33; // xmm0
  __m128i *v34; // rsi
  unsigned __int64 v35; // rdx
  __m128i *v36; // rdx
  __m128i v37; // xmm6
  signed __int64 v38; // r13
  __int64 v39; // rdi
  __int64 i; // rdx
  unsigned __int64 v41; // rdx
  const __m128i *v42; // rdx
  __int64 v43; // rdx
  __int8 *v44; // r13
  __int64 v45; // r12
  unsigned __int64 v46; // r15
  signed __int64 v47; // rax
  _BYTE *v48; // r13
  _BYTE *v49; // rbx
  int *v50; // rax
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r11
  __int8 v54; // r10
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // r8
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rdx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r12
  __int64 v68; // r13
  _QWORD *v69; // rax
  __m128i *v70; // rdi
  __int64 v71; // r12
  _QWORD *v72; // [rsp+0h] [rbp-1D0h]
  __m128i *v73; // [rsp+0h] [rbp-1D0h]
  unsigned int na; // [rsp+8h] [rbp-1C8h]
  size_t nb; // [rsp+8h] [rbp-1C8h]
  size_t nc; // [rsp+8h] [rbp-1C8h]
  size_t n; // [rsp+8h] [rbp-1C8h]
  size_t nd; // [rsp+8h] [rbp-1C8h]
  __m128i *v79; // [rsp+10h] [rbp-1C0h]
  __m128i *v80; // [rsp+10h] [rbp-1C0h]
  unsigned int v81; // [rsp+10h] [rbp-1C0h]
  __m128i *v82; // [rsp+10h] [rbp-1C0h]
  __int64 v83; // [rsp+10h] [rbp-1C0h]
  int v84; // [rsp+10h] [rbp-1C0h]
  __m128i *v85; // [rsp+18h] [rbp-1B8h]
  _QWORD *v86; // [rsp+18h] [rbp-1B8h]
  __m128i *v87; // [rsp+18h] [rbp-1B8h]
  _QWORD *v88; // [rsp+18h] [rbp-1B8h]
  __int8 v89; // [rsp+18h] [rbp-1B8h]
  __int64 v90; // [rsp+18h] [rbp-1B8h]
  __m128i *v93; // [rsp+30h] [rbp-1A0h]
  unsigned __int64 v94; // [rsp+30h] [rbp-1A0h]
  __int8 v95; // [rsp+30h] [rbp-1A0h]
  unsigned int v96; // [rsp+38h] [rbp-198h]
  __int8 v97; // [rsp+38h] [rbp-198h]
  unsigned __int8 v98; // [rsp+38h] [rbp-198h]
  __int32 v99; // [rsp+3Ch] [rbp-194h] BYREF
  __m128i v100; // [rsp+40h] [rbp-190h] BYREF
  __m128i v101; // [rsp+50h] [rbp-180h]
  __int64 v102; // [rsp+60h] [rbp-170h]
  __m128i *v103; // [rsp+70h] [rbp-160h] BYREF
  __int64 v104; // [rsp+78h] [rbp-158h]
  _BYTE v105[48]; // [rsp+80h] [rbp-150h] BYREF
  unsigned __int64 v106; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v107; // [rsp+B8h] [rbp-118h]
  _BYTE v108[48]; // [rsp+C0h] [rbp-110h] BYREF
  __m128i v109; // [rsp+F0h] [rbp-E0h] BYREF
  __m128i v110; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v111; // [rsp+110h] [rbp-C0h]
  char v112; // [rsp+118h] [rbp-B8h]
  __m128i v113; // [rsp+130h] [rbp-A0h] BYREF
  __m128i v114; // [rsp+140h] [rbp-90h] BYREF
  __m128i v115; // [rsp+150h] [rbp-80h] BYREF
  __int64 v116; // [rsp+160h] [rbp-70h]
  char v117; // [rsp+168h] [rbp-68h]
  __m128i v118; // [rsp+188h] [rbp-48h]

  v14 = (__m128i *)v105;
  v15 = a11;
  v99 = a5;
  v16 = (int *)(&a7 + a11);
  v106 = (unsigned __int64)v108;
  v103 = (__m128i *)v105;
  v104 = 0x100000000LL;
  v107 = 0x100000000LL;
  if ( &a7 == (unsigned int *)v16 )
  {
    v53 = a12.m128i_i64[0];
    LODWORD(v47) = 0;
    LODWORD(v55) = 0;
    v109.m128i_i64[0] = (__int64)&v110;
    v54 = a12.m128i_i8[8];
    v109.m128i_i64[1] = 0x100000000LL;
    v56 = a12.m128i_u8[9];
    goto LABEL_50;
  }
  v17 = &a7;
  v18 = (__m128i *)v105;
  v19 = (__m128i *)v105;
  v20 = a4;
  v21 = a7;
  v15 = 0;
  v22 = &v109;
  v23 = 1;
  v24 = (int *)(&a7 + a11);
  v96 = 0;
  if ( a7 != dword_5051178[0] )
    goto LABEL_3;
LABEL_12:
  v112 = 0;
  v28 = v22;
  v109.m128i_i64[0] = unk_5051170;
  v27 = v15 + 1;
  if ( v15 + 1 <= v23 )
    goto LABEL_6;
LABEL_13:
  if ( v19 > v22 || v22 >= &v19[3 * v15] )
  {
    v73 = v22;
    nc = (size_t)a3;
    v81 = v21;
    v87 = v18;
    sub_C8D5F0((__int64)&v103, v18, v27, 0x30u, (__int64)v16, (__int64)a3);
    v22 = v73;
    v19 = v103;
    v15 = (unsigned int)v104;
    a3 = (_QWORD *)nc;
    v21 = v81;
    v18 = v87;
    v28 = v73;
  }
  else
  {
    v38 = (char *)v28 - (char *)v19;
    v72 = a3;
    na = v21;
    v79 = v22;
    v85 = v18;
    sub_C8D5F0((__int64)&v103, v18, v27, 0x30u, (__int64)v16, (__int64)a3);
    v19 = v103;
    v15 = (unsigned int)v104;
    v18 = v85;
    v22 = v79;
    v21 = na;
    a3 = v72;
    v28 = (__m128i *)((char *)v103 + v38);
  }
  while ( 1 )
  {
LABEL_6:
    v29 = &v19[3 * v15];
    *v29 = _mm_loadu_si128(v28);
    v30 = _mm_loadu_si128(v28 + 1);
    LODWORD(v104) = v104 + 1;
    v29[1] = v30;
    v29[2] = _mm_loadu_si128(v28 + 2);
    if ( dword_5051178[0] == v21 )
    {
      v14 = v18;
      goto LABEL_29;
    }
    if ( (v21 & 1) != 0 )
    {
      v31 = (unsigned int)v107;
      v117 = 1;
      v32 = _mm_load_si128(&v109);
      v33 = _mm_load_si128(&v110);
      v34 = &v114;
      v102 = v111;
      v116 = v111;
      v35 = v106;
      v100 = v32;
      v101 = v33;
      v114 = v32;
      v115 = v33;
      if ( (unsigned __int64)(unsigned int)v107 + 1 > HIDWORD(v107) )
      {
        n = (size_t)v18;
        v82 = v22;
        v88 = a3;
        if ( v106 > (unsigned __int64)&v114 || (unsigned __int64)&v114 >= v106 + 48LL * (unsigned int)v107 )
        {
          sub_C8D5F0((__int64)&v106, v108, (unsigned int)v107 + 1LL, 0x30u, (__int64)v16, (__int64)a3);
          v35 = v106;
          v34 = &v114;
          v31 = (unsigned int)v107;
          v18 = (__m128i *)n;
          v22 = v82;
          a3 = v88;
        }
        else
        {
          v44 = &v114.m128i_i8[-v106];
          sub_C8D5F0((__int64)&v106, v108, (unsigned int)v107 + 1LL, 0x30u, (__int64)v16, (__int64)a3);
          v35 = v106;
          v31 = (unsigned int)v107;
          a3 = v88;
          v22 = v82;
          v18 = (__m128i *)n;
          v34 = (__m128i *)&v44[v106];
        }
      }
      v15 = 48 * v31;
      v36 = (__m128i *)(v15 + v35);
      *v36 = _mm_loadu_si128(v34);
      v37 = _mm_loadu_si128(v34 + 1);
      LODWORD(v107) = v107 + 1;
      v36[1] = v37;
      v36[2] = _mm_loadu_si128(v34 + 2);
    }
    else
    {
      v39 = *(_QWORD *)v20;
      for ( i = *(unsigned int *)(v20 + 8); i > 0; i >>= 1 )
      {
        while ( 1 )
        {
          v15 = v39 + 16 * (i >> 1);
          if ( v109.m128i_i64[0] <= *(_QWORD *)v15 )
            break;
          v39 = v15 + 16;
          i = i - (i >> 1) - 1;
          if ( i <= 0 )
            goto LABEL_20;
        }
      }
LABEL_20:
      if ( *(_BYTE *)(v39 + 11) )
      {
        nb = (size_t)v18;
        v80 = v22;
        v86 = a3;
        v114.m128i_i32[0] = *(_DWORD *)(v39 + 8) & 0xFFFFFF;
        v117 = 0;
        sub_37BC120((__int64)&v106, &v114, v114.m128i_u32[0], v15, (__int64)v16, (__int64)a3);
        v18 = (__m128i *)nb;
        v22 = v80;
        a3 = v86;
      }
      else
      {
        if ( (v109.m128i_i32[0] & 0xFFFFF) != *(_DWORD *)(a2 + 24) || (v109.m128i_i64[0] & 0xFFFFF00000LL) == 0 )
        {
          v14 = v18;
          sub_37BA020(a1, v99, (__int64)&a12, (__int64)v22);
          goto LABEL_29;
        }
        v41 = ((unsigned __int64)v109.m128i_i64[0] >> 20) & 0xFFFFF;
        if ( v96 >= (unsigned int)v41 )
          LODWORD(v41) = v96;
        v96 = v41;
      }
    }
    if ( v24 == (int *)++v17 )
      break;
    v15 = (unsigned int)v104;
    v23 = HIDWORD(v104);
    v19 = v103;
    v21 = *v17;
    if ( *v17 == dword_5051178[0] )
      goto LABEL_12;
LABEL_3:
    v25 = v21 >> 1;
    if ( (v21 & 1) != 0 )
    {
      v16 = (int *)a3[2];
      v42 = (const __m128i *)&v16[10 * v25];
      v114 = _mm_loadu_si128(v42);
      v115 = _mm_loadu_si128(v42 + 1);
      v43 = v42[2].m128i_i64[0];
      v112 = 1;
      v116 = v43;
      v111 = v43;
      v109 = v114;
      v110 = v115;
    }
    else
    {
      v26 = *(_QWORD *)(*a3 + 8 * v25);
      v112 = 0;
      v109.m128i_i64[0] = v26;
    }
    v27 = v15 + 1;
    v28 = v22;
    if ( v15 + 1 > v23 )
      goto LABEL_13;
  }
  v45 = (__int64)v22;
  v14 = v18;
  if ( v96 )
  {
    sub_37CF130(a1, v99, &a12, (__int64)&v103, v96);
    goto LABEL_29;
  }
  v46 = v106;
  v47 = 48LL * (unsigned int)v107;
  v48 = (_BYTE *)(v106 + v47);
  if ( v106 != v106 + v47 )
  {
    v49 = (_BYTE *)(v106 + v47);
    v93 = v18;
    do
    {
      if ( !*(_BYTE *)(v46 + 40) )
      {
        v50 = sub_37BEF10(a1 + 3408, (int *)v46);
        sub_2B5C0F0((__int64)&v114, (__int64)v50, (unsigned int *)&v99, v51, v52);
      }
      v46 += 48LL;
    }
    while ( (_BYTE *)v46 != v49 );
    v14 = v93;
    v48 = (_BYTE *)v106;
    v47 = 48LL * (unsigned int)v107;
  }
  v53 = a12.m128i_i64[0];
  v109.m128i_i64[0] = (__int64)&v110;
  v54 = a12.m128i_i8[8];
  v109.m128i_i64[1] = 0x100000000LL;
  v55 = 0xAAAAAAAAAAAAAAABLL * (v47 >> 4);
  v56 = a12.m128i_u8[9];
  if ( (unsigned __int64)v47 > 0x30 )
  {
    nd = v47;
    v83 = a12.m128i_i64[0];
    v89 = a12.m128i_i8[9];
    v97 = a12.m128i_i8[8];
    v94 = 0xAAAAAAAAAAAAAAABLL * (v47 >> 4);
    sub_C8D5F0(v45, &v110, v94, 0x30u, v55, a12.m128i_u8[9]);
    LODWORD(v55) = v94;
    v54 = v97;
    LOBYTE(v56) = v89;
    v53 = v83;
    v47 = nd;
    v70 = (__m128i *)(v109.m128i_i64[0] + 48LL * v109.m128i_u32[2]);
LABEL_61:
    v84 = v55;
    v90 = v53;
    v98 = v56;
    v95 = v54;
    memcpy(v70, v48, v47);
    LODWORD(v47) = v109.m128i_i32[2];
    LODWORD(v55) = v84;
    v53 = v90;
    v56 = v98;
    v54 = v95;
    goto LABEL_50;
  }
  if ( v47 )
  {
    v70 = &v110;
    goto LABEL_61;
  }
LABEL_50:
  v109.m128i_i32[2] = v55 + v47;
  v57 = a1 + 3440;
  v113.m128i_i64[0] = v53;
  v113.m128i_i8[8] = v54;
  v113.m128i_i8[9] = v56;
  v114.m128i_i32[0] = v99;
  v114.m128i_i64[1] = (__int64)&v115.m128i_i64[1];
  v115.m128i_i64[0] = 0x100000000LL;
  if ( v109.m128i_i32[2] )
  {
    sub_37B6E70((__int64)&v114.m128i_i64[1], (__int64)&v109, (unsigned int)v99, v15, v57, v56);
    v57 = a1 + 3440;
  }
  v118 = _mm_load_si128(&v113);
  sub_37BF2A0((__int64)&v100, v57, v114.m128i_i32, (__int64)&v114.m128i_i64[1]);
  if ( (unsigned __int64 *)v114.m128i_i64[1] != &v115.m128i_u64[1] )
    _libc_free(v114.m128i_u64[1]);
  if ( !(_BYTE)v102 )
  {
    v71 = v101.m128i_i64[0];
    sub_37B6E70(v101.m128i_i64[0] + 8, (__int64)&v109, v58, v59, v60, v61);
    *(_QWORD *)(v71 + 72) = v113.m128i_i64[0];
    *(_WORD *)(v71 + 80) = v113.m128i_i16[4];
  }
  v62 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL) + 48LL * (unsigned int)v99;
  sub_37BA660(*(_QWORD **)(a1 + 16), (__int64)&v106, v62, *(_QWORD *)(v62 + 40), (__int64)&a12);
  v65 = *(unsigned int *)(a1 + 3480);
  v67 = v66;
  v68 = (unsigned int)v99;
  if ( v65 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 3484) )
  {
    sub_C8D5F0(a1 + 3472, (const void *)(a1 + 3488), v65 + 1, 0x10u, v63, v64);
    v65 = *(unsigned int *)(a1 + 3480);
  }
  v69 = (_QWORD *)(*(_QWORD *)(a1 + 3472) + 16 * v65);
  *v69 = v68;
  v69[1] = v67;
  ++*(_DWORD *)(a1 + 3480);
  if ( (__m128i *)v109.m128i_i64[0] != &v110 )
    _libc_free(v109.m128i_u64[0]);
LABEL_29:
  if ( (_BYTE *)v106 != v108 )
    _libc_free(v106);
  if ( v103 != v14 )
    _libc_free((unsigned __int64)v103);
}
