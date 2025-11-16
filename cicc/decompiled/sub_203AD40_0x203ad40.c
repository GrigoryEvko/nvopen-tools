// Function: sub_203AD40
// Address: 0x203ad40
//
__int64 *__fastcall sub_203AD40(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // r13
  __int64 *v3; // rax
  __int64 v4; // r12
  __int64 v5; // r15
  int v7; // eax
  char *v10; // rdx
  char v11; // al
  __int64 v12; // rdx
  __int8 v13; // di
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned int v21; // eax
  __m128 v22; // xmm0
  __int8 v23; // al
  __int64 v24; // r8
  unsigned int v25; // r13d
  char v26; // di
  unsigned int v27; // r14d
  unsigned int v28; // eax
  __int64 v29; // rdx
  unsigned int v30; // r12d
  __int64 v31; // rdx
  __int64 v32; // r13
  __int64 v33; // r14
  __int64 v34; // r12
  unsigned int v35; // r9d
  __int64 v36; // rax
  unsigned __int8 v37; // r10
  __int64 v38; // r8
  unsigned int v39; // eax
  unsigned __int8 v40; // r14
  __int64 v41; // r12
  unsigned int v42; // r9d
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rsi
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rax
  __m128i v51; // xmm1
  __m128i v52; // xmm2
  __int32 v53; // eax
  __int64 v54; // rdx
  bool v55; // al
  int v56; // eax
  _QWORD *v57; // rax
  __int32 v58; // eax
  __int64 v59; // rdx
  __int32 v60; // eax
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // r8
  __int64 v74; // r9
  unsigned int v75; // r13d
  __int64 v76; // rcx
  const __m128i *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // r8
  unsigned int v81; // edx
  int v82; // r9d
  __int64 v83; // rsi
  unsigned int v84; // edx
  __int64 *v85; // r13
  __int64 *v86; // r12
  unsigned int v87; // edx
  unsigned __int64 v88; // r15
  int v89; // r9d
  unsigned __int64 v90; // r10
  unsigned int v91; // edx
  unsigned __int64 v92; // r11
  __int64 *v93; // r12
  __int8 v94; // al
  __int64 v95; // rdx
  unsigned int v96; // edx
  unsigned int v97; // edx
  unsigned int v98; // eax
  const void **v99; // rdx
  int v100; // r9d
  unsigned int v101; // edx
  char v102; // di
  unsigned int v103; // eax
  char v104; // cl
  char v105; // r8
  int v106; // esi
  char v107; // di
  bool v108; // zf
  char v109; // al
  int v110; // [rsp+0h] [rbp-1C0h]
  unsigned int v111; // [rsp+0h] [rbp-1C0h]
  __int64 v112; // [rsp+8h] [rbp-1B8h]
  int v113; // [rsp+10h] [rbp-1B0h]
  unsigned int v114; // [rsp+10h] [rbp-1B0h]
  __int64 v115; // [rsp+18h] [rbp-1A8h]
  __int64 v116; // [rsp+18h] [rbp-1A8h]
  __int64 v117; // [rsp+20h] [rbp-1A0h]
  __int64 v118; // [rsp+20h] [rbp-1A0h]
  unsigned int v119; // [rsp+20h] [rbp-1A0h]
  __int64 v120; // [rsp+30h] [rbp-190h]
  unsigned int v121; // [rsp+30h] [rbp-190h]
  __m128i v122; // [rsp+30h] [rbp-190h]
  unsigned __int8 v123; // [rsp+40h] [rbp-180h]
  __m128i v124; // [rsp+40h] [rbp-180h]
  char v126; // [rsp+40h] [rbp-180h]
  __int64 v127; // [rsp+50h] [rbp-170h]
  __m128i v128; // [rsp+50h] [rbp-170h]
  unsigned __int64 v130; // [rsp+50h] [rbp-170h]
  bool v131; // [rsp+50h] [rbp-170h]
  __int16 *v132; // [rsp+58h] [rbp-168h]
  _QWORD *v133; // [rsp+60h] [rbp-160h]
  unsigned int v134; // [rsp+60h] [rbp-160h]
  __int64 (__fastcall *v135)(__int64, __int64, __int64, __int64, __int64); // [rsp+60h] [rbp-160h]
  __m128i v136; // [rsp+60h] [rbp-160h]
  unsigned int v137; // [rsp+70h] [rbp-150h]
  __int64 v138; // [rsp+70h] [rbp-150h]
  unsigned int v139; // [rsp+70h] [rbp-150h]
  __int64 v140; // [rsp+70h] [rbp-150h]
  __int128 v141; // [rsp+70h] [rbp-150h]
  __int64 *v142; // [rsp+70h] [rbp-150h]
  __int64 v143; // [rsp+80h] [rbp-140h]
  __int64 v144; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v145; // [rsp+108h] [rbp-B8h]
  __m128i v146; // [rsp+110h] [rbp-B0h] BYREF
  __m128 i; // [rsp+120h] [rbp-A0h] BYREF
  __m128i v148; // [rsp+130h] [rbp-90h] BYREF
  __m128i v149; // [rsp+140h] [rbp-80h] BYREF
  __m128i v150; // [rsp+150h] [rbp-70h] BYREF
  __m128i v151; // [rsp+160h] [rbp-60h] BYREF
  __m128i v152; // [rsp+170h] [rbp-50h] BYREF
  __int64 v153; // [rsp+180h] [rbp-40h]

  v2 = *(_QWORD **)(a1[1] + 48);
  v3 = *(__int64 **)(a2 + 32);
  v4 = *v3;
  v5 = v3[1];
  if ( *(_WORD *)(a2 + 24) != 135 )
    return 0;
  v7 = *(unsigned __int16 *)(v4 + 24);
  if ( v7 != 137 && (unsigned int)(v7 - 118) > 2 )
    return 0;
  v10 = *(char **)(v4 + 40);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOBYTE(v144) = v11;
  v145 = v12;
  if ( v11 )
  {
    if ( (unsigned __int8)(v11 - 14) > 0x5Fu )
      goto LABEL_7;
LABEL_25:
    v13 = sub_1F7E0F0((__int64)&v144);
    goto LABEL_8;
  }
  if ( sub_1F58D20((__int64)&v144) )
    goto LABEL_25;
LABEL_7:
  v13 = v144;
  v14 = v145;
LABEL_8:
  v152.m128i_i8[0] = v13;
  v152.m128i_i64[1] = v14;
  if ( v13 )
    v15 = sub_2021900(v13);
  else
    v15 = sub_1F58D40((__int64)&v152);
  if ( v15 != 1 )
    return 0;
  v19 = *(_QWORD *)(a2 + 40);
  v20 = *(_QWORD *)(v19 + 8);
  v123 = *(_BYTE *)v19;
  v146.m128i_i8[0] = *(_BYTE *)v19;
  v127 = v20;
  v146.m128i_i64[1] = v20;
  v21 = sub_1D159A0(v146.m128i_i8, v20, v16, v146.m128i_u8[0], v17, v18, v110, v112, v113, v115);
  if ( !v21 || (v21 & (v21 - 1LL)) != 0 )
    return 0;
  v22 = (__m128)_mm_loadu_si128(&v146);
  v133 = v2;
  v23 = v123;
  v24 = v20;
  v25 = v137;
  v117 = a2;
  v143 = v4;
  for ( i = v22; ; i.m128_u64[1] = v24 )
  {
    LOBYTE(v25) = v23;
    sub_1F40D10((__int64)&v152, *a1, *(_QWORD *)(a1[1] + 48), v25, v24);
    if ( v152.m128i_i8[0] != 6 )
      break;
    LOBYTE(v28) = sub_1F7E0F0((__int64)&i);
    v138 = v29;
    v30 = v28;
    if ( !i.m128_i8[0] )
    {
      v26 = v28;
      v27 = (unsigned int)sub_1F58D30((__int64)&i) >> 1;
LABEL_15:
      v23 = sub_1D15020(v26, v27);
      goto LABEL_16;
    }
    v26 = v28;
    v27 = word_4305480[(unsigned __int8)(i.m128_i8[0] - 14)] >> 1;
    if ( (unsigned __int8)(i.m128_i8[0] - 56) > 0x1Du && (unsigned __int8)(i.m128_i8[0] - 98) > 0xBu )
      goto LABEL_15;
    v23 = sub_1D154A0(v28, v27);
LABEL_16:
    v24 = 0;
    if ( !v23 )
    {
      v23 = sub_1F593D0(v133, v30, v138, v27);
      v24 = v31;
    }
    i.m128_i8[0] = v23;
  }
  v32 = (__int64)v133;
  v33 = v117;
  v34 = v143;
  if ( (unsigned int)sub_1D15970(&i) == 1 )
    return 0;
  if ( *(_WORD *)(v143 + 24) == 137 )
  {
    v35 = v111;
    v36 = *(_QWORD *)(**(_QWORD **)(v143 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v143 + 32) + 8LL);
    v37 = *(_BYTE *)v36;
    v38 = *(_QWORD *)(v36 + 8);
    v39 = v114;
    v40 = v37;
    v41 = v38;
    while ( 1 )
    {
      LOBYTE(v39) = v40;
      v134 = v35;
      v139 = sub_1F40D10((__int64)&v152, *a1, v32, v39, v41);
      if ( !v152.m128i_i8[0] )
        break;
      v42 = v134;
      LOBYTE(v42) = v40;
      sub_1F40D10((__int64)&v152, *a1, v32, v42, v41);
      v40 = v152.m128i_u8[8];
      v41 = v153;
      v39 = v139;
    }
    v116 = v41;
    v43 = a1[1];
    v44 = v40;
    v33 = v117;
    v34 = v143;
    v118 = v44;
    v120 = *a1;
    v135 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 264LL);
    v140 = *(_QWORD *)(v43 + 48);
    v45 = sub_1E0A0C0(*(_QWORD *)(v43 + 32));
    v152.m128i_i8[0] = v135(v120, v45, v140, v118, v116);
    v152.m128i_i64[1] = v46;
    if ( (unsigned int)sub_1D159C0((__int64)&v152, v45, v46, v47, v48, v49) != 1 )
      goto LABEL_33;
    return 0;
  }
  else
  {
    if ( sub_1D15870((char *)&v144) == 2 )
    {
      while ( 1 )
      {
        sub_1F40D10((__int64)&v152, *a1, (__int64)v133, v144, v145);
        if ( !v152.m128i_i8[0] )
          break;
        sub_1F40D10((__int64)&v152, *a1, (__int64)v133, v144, v145);
        LOBYTE(v144) = v152.m128i_i8[8];
        v145 = v153;
      }
      v34 = v143;
      if ( sub_1D15870((char *)&v144) == 2 )
        return 0;
    }
LABEL_33:
    v50 = *(_QWORD *)(v33 + 32);
    v51 = _mm_loadu_si128((const __m128i *)(v50 + 40));
    v52 = _mm_loadu_si128((const __m128i *)(v50 + 80));
    v141 = (__int128)v51;
    v136 = v52;
    sub_1F40D10((__int64)&v152, *a1, *(_QWORD *)(a1[1] + 48), v123, v127);
    if ( v152.m128i_i8[0] == 7 )
    {
      sub_1F40D10((__int64)&v152, *a1, v32, v146.m128i_i64[0], v146.m128i_i64[1]);
      v146.m128i_i8[0] = v152.m128i_i8[8];
      v146.m128i_i64[1] = v153;
      *(_QWORD *)&v141 = sub_20363F0((__int64)a1, v51.m128i_u64[0], v51.m128i_i64[1]);
      *((_QWORD *)&v141 + 1) = v96 | v51.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v136.m128i_i64[0] = sub_20363F0((__int64)a1, v52.m128i_u64[0], v52.m128i_i64[1]);
      v136.m128i_i64[1] = v97 | v52.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    }
    v148 = _mm_loadu_si128(&v146);
    LOBYTE(v53) = sub_1D15870(v148.m128i_i8);
    v152.m128i_i32[0] = v53;
    v152.m128i_i64[1] = v54;
    if ( (_BYTE)v53 )
      v55 = (unsigned __int8)(v53 - 14) <= 0x47u || (unsigned __int8)(v53 - 2) <= 5u;
    else
      v55 = sub_1F58CF0((__int64)&v152);
    if ( !v55 )
    {
      if ( v148.m128i_i8[0] )
      {
        switch ( v148.m128i_i8[0] )
        {
          case 0xE:
          case 0xF:
          case 0x10:
          case 0x11:
          case 0x12:
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
          case 0x17:
          case 0x38:
          case 0x39:
          case 0x3A:
          case 0x3B:
          case 0x3C:
          case 0x3D:
            v102 = 2;
            goto LABEL_70;
          case 0x18:
          case 0x19:
          case 0x1A:
          case 0x1B:
          case 0x1C:
          case 0x1D:
          case 0x1E:
          case 0x1F:
          case 0x20:
          case 0x3E:
          case 0x3F:
          case 0x40:
          case 0x41:
          case 0x42:
          case 0x43:
            v102 = 3;
            goto LABEL_70;
          case 0x21:
          case 0x22:
          case 0x23:
          case 0x24:
          case 0x25:
          case 0x26:
          case 0x27:
          case 0x28:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x47:
          case 0x48:
          case 0x49:
            v102 = 4;
            goto LABEL_70;
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x2C:
          case 0x2D:
          case 0x2E:
          case 0x2F:
          case 0x30:
          case 0x4A:
          case 0x4B:
          case 0x4C:
          case 0x4D:
          case 0x4E:
          case 0x4F:
            v102 = 5;
            goto LABEL_70;
          case 0x31:
          case 0x32:
          case 0x33:
          case 0x34:
          case 0x35:
          case 0x36:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x53:
          case 0x54:
          case 0x55:
            v102 = 6;
            goto LABEL_70;
          case 0x37:
            v103 = sub_2021900(7);
            if ( v103 != 32 )
              goto LABEL_71;
            v107 = 5;
            v106 = sub_1D15970(&v148);
            goto LABEL_90;
          case 0x56:
          case 0x57:
          case 0x58:
          case 0x62:
          case 0x63:
          case 0x64:
            v102 = 8;
            goto LABEL_70;
          case 0x59:
          case 0x5A:
          case 0x5B:
          case 0x5C:
          case 0x5D:
          case 0x65:
          case 0x66:
          case 0x67:
          case 0x68:
          case 0x69:
            v102 = 9;
            goto LABEL_70;
          case 0x5E:
          case 0x5F:
          case 0x60:
          case 0x61:
          case 0x6A:
          case 0x6B:
          case 0x6C:
          case 0x6D:
            v102 = 10;
LABEL_70:
            v103 = sub_2021900(v102);
            if ( v103 == 32 )
            {
              v105 = 5;
            }
            else
            {
LABEL_71:
              if ( v103 > 0x20 )
              {
                v105 = 6;
                if ( v103 != 64 )
                {
                  v108 = v103 == 128;
                  v109 = 7;
                  if ( !v108 )
                    v109 = 0;
                  v105 = v109;
                }
              }
              else
              {
                v105 = 3;
                if ( v103 != 8 )
                {
                  v105 = 4;
                  if ( v103 != 16 )
                    v105 = 2 * (v103 == 1);
                }
              }
            }
            v126 = v105;
            v131 = (unsigned __int8)(v104 - 98) <= 0xBu || (unsigned __int8)(v104 - 56) <= 0x1Du;
            v106 = sub_1D15970(&v148);
            v107 = v126;
            if ( v131 )
            {
              v94 = sub_1D154A0(v126, v106);
              v95 = 0;
            }
            else
            {
LABEL_90:
              v94 = sub_1D15020(v107, v106);
              v95 = 0;
            }
            break;
        }
      }
      else
      {
        v94 = sub_1F5A910((__int64)&v148);
      }
      v148.m128i_i8[0] = v94;
      v148.m128i_i64[1] = v95;
    }
    v56 = *(unsigned __int16 *)(v34 + 24);
    if ( v56 == 137 )
    {
      v98 = sub_202EF30(a1, v34);
      v90 = sub_202F990(
              a1,
              v34,
              (__m128i)v22,
              *(double *)v51.m128i_i64,
              v52,
              v5,
              v98,
              v99,
              v100,
              v148.m128i_i64[0],
              (const void **)v148.m128i_i64[1]);
      v92 = v101;
    }
    else
    {
      if ( (unsigned int)(v56 - 118) > 2 )
        return 0;
      v57 = *(_QWORD **)(v34 + 32);
      if ( *(_WORD *)(*v57 + 24LL) != 137 || *(_WORD *)(v57[5] + 24LL) != 137 )
        return 0;
      v128 = _mm_loadu_si128((const __m128i *)v57);
      v124 = _mm_loadu_si128((const __m128i *)(v57 + 5));
      v58 = sub_202EF30(a1, v128.m128i_i64[0]);
      v149.m128i_i64[1] = v59;
      v149.m128i_i32[0] = v58;
      v60 = sub_202EF30(a1, v124.m128i_i64[0]);
      v150.m128i_i64[1] = v61;
      v150.m128i_i32[0] = v60;
      v121 = sub_1D159C0((__int64)&v149, v124.m128i_i64[0], (__int64)&v149, v62, v63, v64);
      v119 = sub_1D159C0((__int64)&v150, (__int64)&v150, v65, v66, v67, v68);
      v75 = sub_1D159C0((__int64)&v148, (__int64)&v150, v69, v70, v71, v72);
      if ( v121 == v119 )
      {
        v122 = _mm_loadu_si128(&v149);
      }
      else
      {
        if ( v121 >= v119 )
          v151 = _mm_loadu_si128(&v150);
        else
          v151 = _mm_loadu_si128(&v149);
        v76 = v151.m128i_u8[0];
        v77 = &v149;
        if ( v149.m128i_i8[0] == v151.m128i_i8[0] )
        {
          v77 = &v150;
          if ( !v151.m128i_i8[0] )
          {
            v76 = v151.m128i_i64[1];
            if ( v149.m128i_i64[1] != v151.m128i_i64[1] )
              v77 = &v149;
          }
        }
        v152 = _mm_loadu_si128(v77);
        if ( (unsigned int)sub_1D159C0((__int64)&v152, (__int64)&v150, (__int64)&v149, v76, v73, v74) > v75 )
        {
          if ( (unsigned int)sub_1D159C0((__int64)&v151, (__int64)&v150, v78, v79, v80, v74) < v75 )
            v122 = _mm_loadu_si128(&v148);
          else
            v122 = _mm_loadu_si128(&v151);
        }
        else
        {
          v122 = _mm_loadu_si128(&v152);
        }
      }
      v128.m128i_i64[0] = sub_202F990(
                            a1,
                            v128.m128i_i64[0],
                            (__m128i)v22,
                            *(double *)v51.m128i_i64,
                            v52,
                            v128.m128i_i64[1],
                            v149.m128i_u32[0],
                            (const void **)v149.m128i_i64[1],
                            v74,
                            v122.m128i_i64[0],
                            (const void **)v122.m128i_i64[1]);
      v128.m128i_i64[1] = v81 | v128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v124.m128i_i64[0] = sub_202F990(
                            a1,
                            v124.m128i_i64[0],
                            (__m128i)v22,
                            *(double *)v51.m128i_i64,
                            v52,
                            v124.m128i_i64[1],
                            v150.m128i_u32[0],
                            (const void **)v150.m128i_i64[1],
                            v82,
                            v122.m128i_i64[0],
                            (const void **)v122.m128i_i64[1]);
      v83 = *(_QWORD *)(v34 + 72);
      v85 = (__int64 *)a1[1];
      v152.m128i_i64[0] = v83;
      v124.m128i_i64[1] = v84 | v124.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      if ( v83 )
        sub_1623A60((__int64)&v152, v83, 2);
      v152.m128i_i32[2] = *(_DWORD *)(v34 + 64);
      v86 = sub_1D332F0(
              v85,
              *(unsigned __int16 *)(v34 + 24),
              (__int64)&v152,
              v122.m128i_i64[0],
              (const void **)v122.m128i_i64[1],
              0,
              *(double *)v22.m128_u64,
              *(double *)v51.m128i_i64,
              v52,
              v128.m128i_i64[0],
              v128.m128i_u64[1],
              *(_OWORD *)&v124);
      v88 = v87 | v5 & 0xFFFFFFFF00000000LL;
      sub_17CD270(v152.m128i_i64);
      v90 = sub_202F990(
              a1,
              (__int64)v86,
              (__m128i)v22,
              *(double *)v51.m128i_i64,
              v52,
              v88,
              v122.m128i_u32[0],
              (const void **)v122.m128i_i64[1],
              v89,
              v148.m128i_i64[0],
              (const void **)v148.m128i_i64[1]);
      v92 = v91;
    }
    v93 = (__int64 *)a1[1];
    v152.m128i_i64[0] = *(_QWORD *)(v33 + 72);
    if ( v152.m128i_i64[0] )
    {
      v130 = v90;
      v132 = (__int16 *)v92;
      sub_20219D0(v152.m128i_i64);
      v92 = (unsigned __int64)v132;
      v90 = v130;
    }
    v152.m128i_i32[2] = *(_DWORD *)(v33 + 64);
    v142 = sub_1D3A900(
             v93,
             0x87u,
             (__int64)&v152,
             v146.m128i_u32[0],
             (const void **)v146.m128i_i64[1],
             0,
             v22,
             *(double *)v51.m128i_i64,
             v52,
             v90,
             (__int16 *)v92,
             v141,
             v136.m128i_i64[0],
             v136.m128i_i64[1]);
    sub_17CD270(v152.m128i_i64);
    return v142;
  }
}
