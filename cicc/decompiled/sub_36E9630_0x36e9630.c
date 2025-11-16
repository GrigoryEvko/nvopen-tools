// Function: sub_36E9630
// Address: 0x36e9630
//
void __fastcall sub_36E9630(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v3; // r13
  __int64 v5; // rsi
  _QWORD *v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // rax
  __int64 v9; // rcx
  _QWORD *v10; // r14
  __int64 v11; // rdx
  _QWORD *v12; // rbx
  __int64 v13; // r9
  unsigned int v14; // r8d
  unsigned int v15; // edx
  __int64 v16; // rdi
  __int64 *v17; // r15
  unsigned __int8 *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  unsigned __int8 *v21; // rax
  __int32 v22; // edx
  __int32 v23; // r8d
  __int64 v24; // rdx
  __m128i si128; // xmm4
  __m128i v26; // xmm2
  __m128i v27; // xmm1
  __m128i v28; // xmm0
  __int64 v29; // r8
  __m128i v30; // xmm3
  __int64 v31; // r9
  __int64 v32; // rax
  int v33; // ebx
  unsigned __int64 **v34; // rdi
  __int64 v35; // rax
  __m128i v36; // xmm0
  __m128i v37; // xmm7
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __m128i v41; // xmm0
  __int64 v42; // rdx
  __int64 v43; // rax
  __m128i v44; // xmm0
  unsigned __int64 v45; // rdx
  __int64 v46; // rax
  __m128i v47; // xmm0
  _QWORD *v48; // r9
  unsigned __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // rbx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rax
  __int64 v56; // rax
  __m128i v57; // xmm0
  __int64 v58; // rdx
  __int64 v59; // rax
  __m128i v60; // xmm0
  __int64 v61; // rdx
  __int64 v62; // rax
  int v63; // ebx
  unsigned __int64 **v64; // rdi
  __int64 v65; // r15
  int v66; // r13d
  __int64 v67; // rbx
  __m128i v68; // xmm0
  __m128i v69; // xmm0
  __int64 v70; // rax
  _QWORD *v71; // rsi
  __int64 v72; // rax
  unsigned int v73; // edx
  unsigned __int64 v74; // rdx
  unsigned __int64 *v75; // rax
  __int64 v76; // rax
  unsigned __int64 **v77; // rdi
  __m128i v78; // xmm0
  __int64 v79; // rax
  int v80; // ebx
  unsigned __int64 **v81; // rdi
  __int64 v82; // rbx
  __m128i v83; // xmm0
  __m128i v84; // xmm6
  __m128i v85; // xmm6
  __m128i v86; // xmm6
  __m128i v87; // xmm7
  __m128i v88; // xmm7
  __m128i v89; // xmm6
  __m128i v90; // xmm7
  __m128i v91; // xmm5
  __m128i v92; // xmm5
  __int64 v93; // [rsp+0h] [rbp-200h]
  __int64 v94; // [rsp+8h] [rbp-1F8h]
  char v95; // [rsp+1Fh] [rbp-1E1h]
  __m128i v96; // [rsp+20h] [rbp-1E0h] BYREF
  __m128i v97; // [rsp+30h] [rbp-1D0h] BYREF
  __m128i v98; // [rsp+40h] [rbp-1C0h] BYREF
  __m128i v99; // [rsp+50h] [rbp-1B0h] BYREF
  __int64 v100; // [rsp+60h] [rbp-1A0h] BYREF
  int v101; // [rsp+68h] [rbp-198h]
  __m128i v102; // [rsp+70h] [rbp-190h] BYREF
  __m128i v103[4]; // [rsp+80h] [rbp-180h] BYREF
  unsigned __int64 *v104; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v105; // [rsp+C8h] [rbp-138h]
  _OWORD v106[5]; // [rsp+D0h] [rbp-130h] BYREF
  __m128i v107; // [rsp+120h] [rbp-E0h]

  v3 = a1;
  v5 = *(_QWORD *)(a2 + 80);
  v100 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v100, v5, 1);
  v6 = *(_QWORD **)(a2 + 40);
  v101 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(v6[10] + 96LL);
  v8 = *(_QWORD **)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    v8 = (_QWORD *)*v8;
  v9 = *(_QWORD *)(v6[15] + 96LL);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v11 = *(_QWORD *)(v6[5] + 96LL);
  v12 = *(_QWORD **)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v12 = (_QWORD *)*v12;
  v13 = *(_QWORD *)(a1 + 1136);
  v14 = *(_DWORD *)(v13 + 340);
  v15 = ((unsigned int)v8 >> 6) & 7;
  if ( __ROR4__(-858993459 * v14 + 1717986918, 1) <= 0x19999999u && *(_DWORD *)(v13 + 336) > 0x57u )
  {
    if ( (_BYTE)v15 == 4 )
      sub_C64ED0("INT8 type is supported only on arch-conditional variants.", 1u);
    if ( (((_BYTE)v15 + 7) & 7u) > 5 && ((unsigned __int8)v8 & 0x20) != 0 )
      sub_C64ED0("MXF4 and MXF4NVF4 types with Sparsity are supported only on arch-conditional variants.", 1u);
    if ( (((unsigned int)v8 >> 2) & 3) - 1 <= 2 )
      sub_C64ED0("Explicit scale vector size is supported only on arch-conditional variants.", 1u);
  }
  v99.m128i_i8[0] = (unsigned __int8)v8 & 0x10;
  if ( ((unsigned __int8)v8 & 0x10) != 0 )
  {
    if ( v14 > 0x3F4 )
    {
      if ( v14 - 1101 > 1 )
      {
LABEL_24:
        if ( (((_BYTE)v15 - 3) & 0xFD) != 0 )
          sub_C64ED0("Scale input accumulator can only be used with f16 and tf32 types", 1u);
        goto LABEL_11;
      }
    }
    else if ( v14 <= 0x3F2 )
    {
      goto LABEL_24;
    }
    sub_C64ED0("Scale input accumulator is not supported on this architecture.", 1u);
  }
LABEL_11:
  if ( (_DWORD)v12 == 10299 || (_DWORD)v12 == 10304 )
  {
    if ( (((_BYTE)v15 + 5) & 7u) <= 2 || v15 == 1 )
      sub_C64ED0("Block scale is not supported for f16, tf32, f8f6f4 and i8 types", 1u);
    if ( ((unsigned __int8)v10 & 4) != 0 )
      sub_C64ED0("ashift is not supported with tcgen05.mma.block_scale variants", 1u);
  }
  if ( ((unsigned __int8)v8 & 3) == 3 )
    sub_C64ED0("cta_group::2 is not supported with weight stationary", 1u);
  v95 = (unsigned __int8)v8 & 1;
  if ( ((unsigned __int8)v8 & 1) != 0 )
  {
    if ( (v15 & 5) == 0 || v15 == 7 )
      sub_C64ED0("Cannot use weight stationary with mxf8f6f4 and fp4 types", 1u);
    if ( ((unsigned __int8)v10 & 6) != 6 )
      goto LABEL_18;
LABEL_129:
    sub_C64ED0("Cannot use collector::a::use or colletor::a::fill with ashift", 1u);
  }
  if ( ((unsigned __int8)v10 & 6) == 6 )
    goto LABEL_129;
  if ( v15 != 7 )
  {
LABEL_18:
    if ( v15 == 2 )
    {
      if ( (((unsigned int)v8 >> 2) & 3) > 1 )
        sub_C64ED0("Cannot use 2X or 4X as scale vector size for mxf8f6f4 type", 1u);
    }
    else if ( !v15 && ((unsigned __int8)v8 & 0xC) == 4 )
    {
      sub_C64ED0("Cannot use 1X as scale vector size for mxf4nvf4 type", 1u);
    }
    goto LABEL_21;
  }
  if ( ((unsigned __int8)v8 & 4) != 0 )
    sub_C64ED0("Cannot use 1X or 4X as scale vector size for mxf4 type", 1u);
LABEL_21:
  v16 = *(_QWORD *)(a1 + 64);
  v17 = &v100;
  v96.m128i_i8[0] = (char)v8;
  v18 = sub_3400BD0(v16, (unsigned int)v8, (__int64)&v100, 7, 0, 1u, a3, 0);
  v19 = *(_QWORD *)(v3 + 64);
  v98.m128i_i64[0] = v20;
  v97.m128i_i64[0] = (__int64)v18;
  v21 = sub_3400BD0(v19, (unsigned int)v10, (__int64)&v100, 7, 0, 1u, a3, 0);
  v104 = (unsigned __int64 *)v106;
  v23 = v22;
  v24 = *(_QWORD *)(a2 + 40);
  v103[0].m128i_i64[0] = (__int64)v21;
  v103[0].m128i_i32[2] = v23;
  si128 = _mm_load_si128(v103);
  v26 = _mm_loadu_si128((const __m128i *)(v24 + 240));
  v27 = _mm_loadu_si128((const __m128i *)(v24 + 160));
  v102.m128i_i64[0] = v97.m128i_i64[0];
  v28 = _mm_loadu_si128((const __m128i *)(v24 + 200));
  v29 = v93;
  v102.m128i_i32[2] = v98.m128i_i32[0];
  v30 = _mm_load_si128(&v102);
  v103[1] = v26;
  v31 = v94;
  v105 = 0x1000000005LL;
  v103[2] = v27;
  v103[3] = v28;
  v106[0] = v30;
  v106[1] = si128;
  v106[2] = v26;
  v106[3] = v27;
  v106[4] = v28;
  switch ( (int)v12 )
  {
    case 10299:
      v55 = 5;
      v33 = 4906;
      if ( (v96.m128i_i8[0] & 0x20) != 0 )
      {
        v91 = _mm_loadu_si128((const __m128i *)(v24 + 320));
        v55 = 6;
        v33 = 4907;
        LODWORD(v105) = 6;
        v107 = v91;
      }
      goto LABEL_69;
    case 10300:
      v39 = 5;
      v33 = 4908;
      if ( (v96.m128i_i8[0] & 0x20) != 0 )
      {
        v92 = _mm_loadu_si128((const __m128i *)(v24 + 320));
        v39 = 6;
        v33 = 4909;
        LODWORD(v105) = 6;
        v107 = v92;
      }
      goto LABEL_56;
    case 10301:
      if ( (v96.m128i_i8[0] & 0x20) != 0 )
      {
        v33 = 4912;
        if ( !v99.m128i_i8[0] )
          v33 = v95 == 0 ? 4911 : 4938;
        v85 = _mm_loadu_si128((const __m128i *)(v24 + 320));
        v38 = 6;
        LODWORD(v105) = 6;
        v107 = v85;
      }
      else
      {
        v38 = 5;
        v33 = 4910;
        if ( !v99.m128i_i8[0] )
          v33 = v95 == 0 ? 4905 : 4937;
      }
      goto LABEL_101;
    case 10302:
      if ( (v96.m128i_i8[0] & 0x20) != 0 )
      {
        v86 = _mm_loadu_si128((const __m128i *)(v24 + 320));
        LODWORD(v105) = 6;
        v33 = 4915 - ((v99.m128i_i8[0] == 0) - 1);
        v76 = 6;
        v107 = v86;
      }
      else
      {
        v76 = 5;
        v33 = 4914 - (v99.m128i_i8[0] == 0);
      }
      v29 = 440;
      v77 = &v104;
      v106[v76] = _mm_loadu_si128((const __m128i *)(v24 + 280));
      v35 = (unsigned int)(v105 + 1);
      LODWORD(v105) = v105 + 1;
      do
      {
        v78 = _mm_loadu_si128((const __m128i *)(v24 + v29));
        if ( v35 + 1 > (unsigned __int64)HIDWORD(v105) )
        {
          v96.m128i_i64[0] = v29;
          v98.m128i_i64[0] = (__int64)v77;
          v97 = v78;
          sub_C8D5F0((__int64)v77, v106, v35 + 1, 0x10u, v29, v31);
          v35 = (unsigned int)v105;
          v29 = v96.m128i_i64[0];
          v78 = _mm_load_si128(&v97);
          v77 = (unsigned __int64 **)v98.m128i_i64[0];
        }
        v29 += 40;
        *(__m128i *)&v104[2 * v35] = v78;
        v24 = *(_QWORD *)(a2 + 40);
        v35 = (unsigned int)(v105 + 1);
        LODWORD(v105) = v105 + 1;
      }
      while ( v29 != 600 );
      goto LABEL_83;
    case 10303:
      if ( (v96.m128i_i8[0] & 0x20) != 0 )
      {
        v87 = _mm_loadu_si128((const __m128i *)(v24 + 320));
        LODWORD(v105) = 6;
        v63 = 4919 - ((v99.m128i_i8[0] == 0) - 1);
        v62 = 6;
        v107 = v87;
      }
      else
      {
        v62 = 5;
        v63 = 4918 - (v99.m128i_i8[0] == 0);
      }
      v29 = 440;
      v98.m128i_i64[0] = (__int64)&v100;
      v64 = &v104;
      v65 = v3;
      v66 = v63;
      v67 = 440;
      v106[v62] = _mm_loadu_si128((const __m128i *)(v24 + 280));
      v35 = (unsigned int)(v105 + 1);
      LODWORD(v105) = v105 + 1;
      while ( 1 )
      {
        v68 = _mm_loadu_si128((const __m128i *)(v24 + v67));
        if ( v35 + 1 > (unsigned __int64)HIDWORD(v105) )
        {
          v97.m128i_i64[0] = (__int64)v64;
          v96 = v68;
          sub_C8D5F0((__int64)v64, v106, v35 + 1, 0x10u, v29, v31);
          v35 = (unsigned int)v105;
          v68 = _mm_load_si128(&v96);
          v64 = (unsigned __int64 **)v97.m128i_i64[0];
        }
        v67 += 40;
        *(__m128i *)&v104[2 * v35] = v68;
        v35 = (unsigned int)(v105 + 1);
        LODWORD(v105) = v105 + 1;
        if ( v67 == 760 )
          break;
        v24 = *(_QWORD *)(a2 + 40);
      }
      goto LABEL_82;
    case 10304:
      v55 = 5;
      v33 = 4922;
      if ( (v96.m128i_i8[0] & 0x20) != 0 )
      {
        v89 = _mm_loadu_si128((const __m128i *)(v24 + 320));
        v55 = 6;
        v33 = 4923;
        LODWORD(v105) = 6;
        v107 = v89;
      }
LABEL_69:
      v106[v55] = _mm_loadu_si128((const __m128i *)(v24 + 280));
      LODWORD(v105) = v105 + 1;
      v56 = (unsigned int)v105;
      v57 = _mm_loadu_si128((const __m128i *)(v24 + 400));
      if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
      {
        v99 = v57;
        sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 0x10u, v93, v94);
        v56 = (unsigned int)v105;
        v57 = _mm_load_si128(&v99);
      }
      *(__m128i *)&v104[2 * v56] = v57;
      v58 = *(_QWORD *)(a2 + 40);
      LODWORD(v105) = v105 + 1;
      v59 = (unsigned int)v105;
      v60 = _mm_loadu_si128((const __m128i *)(v58 + 440));
      if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
      {
        v99 = v60;
        sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 0x10u, v29, v31);
        v59 = (unsigned int)v105;
        v60 = _mm_load_si128(&v99);
      }
      *(__m128i *)&v104[2 * v59] = v60;
      v61 = *(_QWORD *)(a2 + 40);
      LODWORD(v105) = v105 + 1;
      v43 = (unsigned int)v105;
      v44 = _mm_loadu_si128((const __m128i *)(v61 + 360));
      v45 = (unsigned int)v105 + 1LL;
      if ( v45 > HIDWORD(v105) )
        goto LABEL_74;
      goto LABEL_59;
    case 10305:
      v39 = 5;
      v33 = 4924;
      if ( (v96.m128i_i8[0] & 0x20) != 0 )
      {
        v90 = _mm_loadu_si128((const __m128i *)(v24 + 320));
        v39 = 6;
        v33 = 4925;
        LODWORD(v105) = 6;
        v107 = v90;
      }
LABEL_56:
      v106[v39] = _mm_loadu_si128((const __m128i *)(v24 + 280));
      LODWORD(v105) = v105 + 1;
      v40 = (unsigned int)v105;
      v41 = _mm_loadu_si128((const __m128i *)(v24 + 360));
      if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
      {
        v99 = v41;
        sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 0x10u, v93, v94);
        v40 = (unsigned int)v105;
        v41 = _mm_load_si128(&v99);
      }
      *(__m128i *)&v104[2 * v40] = v41;
      v42 = *(_QWORD *)(a2 + 40);
      LODWORD(v105) = v105 + 1;
      v43 = (unsigned int)v105;
      v44 = _mm_loadu_si128((const __m128i *)(v42 + 440));
      v45 = (unsigned int)v105 + 1LL;
      if ( v45 <= HIDWORD(v105) )
        goto LABEL_59;
LABEL_74:
      v99 = v44;
      sub_C8D5F0((__int64)&v104, v106, v45, 0x10u, v29, v31);
      v43 = (unsigned int)v105;
      v44 = _mm_load_si128(&v99);
LABEL_59:
      *(__m128i *)&v104[2 * v43] = v44;
      v46 = (unsigned int)(v105 + 1);
      LODWORD(v105) = v105 + 1;
      goto LABEL_60;
    case 10306:
      if ( (v96.m128i_i8[0] & 0x20) != 0 )
      {
        v33 = 4928;
        if ( !v99.m128i_i8[0] )
          v33 = v95 == 0 ? 4927 : 4940;
        v37 = _mm_loadu_si128((const __m128i *)(v24 + 320));
        v38 = 6;
        LODWORD(v105) = 6;
        v107 = v37;
      }
      else
      {
        v38 = 5;
        v33 = 4926;
        if ( !v99.m128i_i8[0] )
          v33 = v95 == 0 ? 4921 : 4939;
      }
LABEL_101:
      v106[v38] = _mm_loadu_si128((const __m128i *)(v24 + 280));
      v35 = (unsigned int)(v105 + 1);
      LODWORD(v105) = v105 + 1;
      goto LABEL_83;
    case 10307:
      if ( (v96.m128i_i8[0] & 0x20) != 0 )
      {
        v88 = _mm_loadu_si128((const __m128i *)(v24 + 320));
        LODWORD(v105) = 6;
        v33 = 4931 - ((v99.m128i_i8[0] == 0) - 1);
        v32 = 6;
        v107 = v88;
      }
      else
      {
        v32 = 5;
        v33 = 4930 - (v99.m128i_i8[0] == 0);
      }
      v29 = 440;
      v34 = &v104;
      v106[v32] = _mm_loadu_si128((const __m128i *)(v24 + 280));
      v35 = (unsigned int)(v105 + 1);
      LODWORD(v105) = v105 + 1;
      do
      {
        v36 = _mm_loadu_si128((const __m128i *)(v24 + v29));
        if ( v35 + 1 > (unsigned __int64)HIDWORD(v105) )
        {
          v96.m128i_i64[0] = v29;
          v98.m128i_i64[0] = (__int64)v34;
          v97 = v36;
          sub_C8D5F0((__int64)v34, v106, v35 + 1, 0x10u, v29, v31);
          v35 = (unsigned int)v105;
          v29 = v96.m128i_i64[0];
          v36 = _mm_load_si128(&v97);
          v34 = (unsigned __int64 **)v98.m128i_i64[0];
        }
        v29 += 40;
        *(__m128i *)&v104[2 * v35] = v36;
        v24 = *(_QWORD *)(a2 + 40);
        v35 = (unsigned int)(v105 + 1);
        LODWORD(v105) = v105 + 1;
      }
      while ( v29 != 600 );
      goto LABEL_83;
    case 10308:
      if ( (v96.m128i_i8[0] & 0x20) != 0 )
      {
        v84 = _mm_loadu_si128((const __m128i *)(v24 + 320));
        LODWORD(v105) = 6;
        v80 = 4935 - ((v99.m128i_i8[0] == 0) - 1);
        v79 = 6;
        v107 = v84;
      }
      else
      {
        v79 = 5;
        v80 = 4934 - (v99.m128i_i8[0] == 0);
      }
      v29 = 440;
      v98.m128i_i64[0] = (__int64)&v100;
      v81 = &v104;
      v65 = v3;
      v66 = v80;
      v82 = 440;
      v106[v79] = _mm_loadu_si128((const __m128i *)(v24 + 280));
      v35 = (unsigned int)(v105 + 1);
      LODWORD(v105) = v105 + 1;
      while ( 1 )
      {
        v83 = _mm_loadu_si128((const __m128i *)(v24 + v82));
        if ( v35 + 1 > (unsigned __int64)HIDWORD(v105) )
        {
          v97.m128i_i64[0] = (__int64)v81;
          v96 = v83;
          sub_C8D5F0((__int64)v81, v106, v35 + 1, 0x10u, v29, v31);
          v35 = (unsigned int)v105;
          v83 = _mm_load_si128(&v96);
          v81 = (unsigned __int64 **)v97.m128i_i64[0];
        }
        v82 += 40;
        *(__m128i *)&v104[2 * v35] = v83;
        v35 = (unsigned int)(v105 + 1);
        LODWORD(v105) = v105 + 1;
        if ( v82 == 760 )
          break;
        v24 = *(_QWORD *)(a2 + 40);
      }
LABEL_82:
      v33 = v66;
      v24 = *(_QWORD *)(a2 + 40);
      v3 = v65;
      v17 = (__int64 *)v98.m128i_i64[0];
LABEL_83:
      v69 = _mm_loadu_si128((const __m128i *)(v24 + 360));
      if ( v35 + 1 > (unsigned __int64)HIDWORD(v105) )
      {
        v98 = v69;
        sub_C8D5F0((__int64)&v104, v106, v35 + 1, 0x10u, v29, v31);
        v35 = (unsigned int)v105;
        v69 = _mm_load_si128(&v98);
      }
      *(__m128i *)&v104[2 * v35] = v69;
      v46 = (unsigned int)(v105 + 1);
      LODWORD(v105) = v105 + 1;
      if ( v99.m128i_i8[0] )
      {
        v70 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 400LL) + 96LL);
        v71 = *(_QWORD **)(v70 + 24);
        if ( *(_DWORD *)(v70 + 32) > 0x40u )
          v71 = (_QWORD *)*v71;
        v29 = (__int64)sub_3400BD0(*(_QWORD *)(v3 + 64), (unsigned int)v71, (__int64)v17, 7, 0, 1u, v69, 0);
        v72 = (unsigned int)v105;
        v31 = v73;
        v74 = (unsigned int)v105 + 1LL;
        if ( v74 > HIDWORD(v105) )
        {
          v98.m128i_i64[0] = v29;
          v99.m128i_i64[0] = v31;
          sub_C8D5F0((__int64)&v104, v106, v74, 0x10u, v29, v31);
          v72 = (unsigned int)v105;
          v29 = v98.m128i_i64[0];
          v31 = v99.m128i_i64[0];
        }
        v75 = &v104[2 * v72];
        *v75 = v29;
        v75[1] = v31;
        v46 = (unsigned int)(v105 + 1);
        LODWORD(v105) = v105 + 1;
      }
LABEL_60:
      v47 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
      if ( v46 + 1 > (unsigned __int64)HIDWORD(v105) )
      {
        v99 = v47;
        sub_C8D5F0((__int64)&v104, v106, v46 + 1, 0x10u, v29, v31);
        v46 = (unsigned int)v105;
        v47 = _mm_load_si128(&v99);
      }
      *(__m128i *)&v104[2 * v46] = v47;
      v48 = *(_QWORD **)(v3 + 64);
      v49 = *(_QWORD *)(a2 + 48);
      v50 = *(unsigned int *)(a2 + 68);
      LODWORD(v105) = v105 + 1;
      v51 = sub_33E66D0(v48, v33, (__int64)v17, v49, v50, (__int64)v48, v104, (unsigned int)v105);
      sub_34158F0(*(_QWORD *)(v3 + 64), a2, v51, v52, v53, v54);
      sub_3421DB0(v51);
      sub_33ECEA0(*(const __m128i **)(v3 + 64), a2);
      if ( v104 != (unsigned __int64 *)v106 )
        _libc_free((unsigned __int64)v104);
      if ( v100 )
        sub_B91220((__int64)v17, v100);
      return;
    default:
      BUG();
  }
}
