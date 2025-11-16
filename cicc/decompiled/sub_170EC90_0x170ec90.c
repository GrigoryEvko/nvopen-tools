// Function: sub_170EC90
// Address: 0x170ec90
//
__int64 __fastcall sub_170EC90(
        __m128i *a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r15
  __int64 *v11; // r13
  __int64 v13; // rbx
  __m128 v14; // xmm0
  __m128i v15; // xmm1
  unsigned int v16; // edx
  const void *v17; // rsi
  __int64 v18; // rax
  double v19; // xmm4_8
  double v20; // xmm5_8
  __int64 v21; // rbx
  __int64 v22; // r13
  __int64 v23; // r14
  _QWORD *v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  unsigned __int8 v27; // al
  unsigned int *v29; // r10
  _DWORD *v30; // r14
  unsigned int *v31; // rcx
  unsigned int *v32; // r11
  _DWORD *v33; // r15
  bool v34; // dl
  bool v35; // si
  __int64 *v36; // r12
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned int *v43; // r15
  __int64 v44; // rax
  unsigned int *v45; // rax
  unsigned int *v46; // r14
  __int64 v47; // r13
  __int64 v48; // rax
  int v49; // r8d
  int v50; // r9d
  __int64 v51; // r13
  __int64 v52; // rax
  __int64 *v53; // r14
  __int64 v54; // rsi
  __int64 v55; // rsi
  unsigned __int8 *v56; // rsi
  __int64 v57; // rdi
  __int64 v58; // rax
  _QWORD *v59; // rax
  _QWORD *v60; // r14
  __int64 v61; // rax
  unsigned __int64 v62; // rsi
  __int64 v63; // rax
  __int64 *v64; // rsi
  _QWORD *v65; // rdi
  __int64 v66; // rdx
  __int64 v67; // rsi
  __int64 v68; // rsi
  __int64 v69; // rdx
  unsigned __int8 *v70; // rsi
  double v71; // xmm4_8
  double v72; // xmm5_8
  __int64 *v73; // r12
  __int64 v74; // r14
  __int64 v75; // rax
  double v76; // xmm4_8
  double v77; // xmm5_8
  __int64 *v78; // r12
  __int64 v79; // r14
  __int64 v80; // rax
  double v81; // xmm4_8
  double v82; // xmm5_8
  __int64 *v83; // r12
  __int64 v84; // r14
  __int64 v85; // rax
  double v86; // xmm4_8
  double v87; // xmm5_8
  __int64 *v88; // rbx
  __int64 v89; // r12
  _QWORD *v90; // rax
  _QWORD *v91; // r13
  __int64 v92; // rax
  __int64 v93; // rdx
  unsigned __int64 v94; // rcx
  __int64 v95; // rdx
  __int64 v96; // rax
  __int64 v97; // rdi
  __int64 v98; // r12
  __int64 v99; // rbx
  _QWORD **v100; // rax
  _QWORD *v101; // r13
  __int64 *v102; // rax
  __int64 v103; // rsi
  __int64 v104; // r13
  __int64 *v105; // r12
  __int64 v106; // rax
  __int64 v107; // rbx
  _QWORD *v108; // rax
  __int64 v109; // r13
  __int64 *v110; // rax
  unsigned int *v111; // rsi
  __int64 v112; // rax
  __int64 v113; // rdx
  unsigned __int64 v114; // rsi
  __int64 v115; // rdx
  __int64 v116; // rax
  unsigned __int64 *v117; // r13
  __int64 v118; // rax
  unsigned __int64 v119; // rcx
  unsigned int *v120; // [rsp+0h] [rbp-100h]
  unsigned int *v121; // [rsp+10h] [rbp-F0h]
  __int64 v122; // [rsp+10h] [rbp-F0h]
  unsigned int *v123; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v124; // [rsp+18h] [rbp-E8h]
  unsigned __int64 *v125; // [rsp+18h] [rbp-E8h]
  __int64 v126; // [rsp+18h] [rbp-E8h]
  __int64 v127; // [rsp+20h] [rbp-E0h]
  unsigned int *v128; // [rsp+20h] [rbp-E0h]
  __int64 v129; // [rsp+28h] [rbp-D8h]
  __int64 *v130; // [rsp+30h] [rbp-D0h] BYREF
  _QWORD *v131; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v132[2]; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v133; // [rsp+50h] [rbp-B0h]
  __int64 v134[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v135; // [rsp+70h] [rbp-90h]
  __int64 v136[2]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v137; // [rsp+90h] [rbp-70h]
  __m128 v138; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v139; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v140; // [rsp+C0h] [rbp-40h]

  v10 = a2;
  v11 = (__int64 *)a1;
  v13 = *(_QWORD *)(a2 - 24);
  v14 = (__m128)_mm_loadu_si128(a1 + 167);
  v140 = a2;
  v15 = _mm_loadu_si128(a1 + 168);
  v16 = *(_DWORD *)(a2 + 64);
  v17 = *(const void **)(a2 + 56);
  v138 = v14;
  v139 = v15;
  v18 = sub_13D16C0(v13, v17, v16);
  if ( v18 )
  {
    v21 = *(_QWORD *)(a2 + 8);
    if ( v21 )
    {
      v22 = a1->m128i_i64[0];
      v23 = v18;
      do
      {
        v24 = sub_1648700(v21);
        sub_170B990(v22, (__int64)v24);
        v21 = *(_QWORD *)(v21 + 8);
      }
      while ( v21 );
      if ( a2 == v23 )
        v23 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v23, v14, *(double *)v15.m128i_i64, a5, a6, v25, v26, a9, a10);
      return v10;
    }
    return 0;
  }
  v27 = *(_BYTE *)(v13 + 16);
  if ( v27 <= 0x17u )
    return 0;
  if ( v27 != 87 )
  {
    if ( v27 == 78 )
    {
      v38 = *(_QWORD *)(v13 - 24);
      if ( !*(_BYTE *)(v38 + 16) && (*(_BYTE *)(v38 + 33) & 0x20) != 0 )
      {
        v39 = *(_QWORD *)(v13 + 8);
        if ( v39 )
        {
          v10 = *(_QWORD *)(v39 + 8);
          if ( !v10 )
          {
            v40 = *(_DWORD *)(v38 + 36);
            switch ( v40 )
            {
              case 189:
              case 209:
                if ( **(_DWORD **)(a2 + 56) )
                {
                  if ( v40 != 209 )
                    return 0;
                  v96 = *(_DWORD *)(v13 + 20) & 0xFFFFFFF;
                  v97 = *(_QWORD *)(v13 + 24 * (1 - v96));
                  if ( *(_BYTE *)(v97 + 16) != 13 )
                    return 0;
                  v98 = *(_QWORD *)(v13 - 24 * v96);
                  v139.m128i_i16[0] = 257;
                  v99 = sub_15A2B00((__int64 *)v97, *(double *)v14.m128_u64, *(double *)v15.m128i_i64, a5);
                  v10 = (__int64)sub_1648A60(56, 2u);
                  if ( v10 )
                  {
                    v100 = *(_QWORD ***)v98;
                    if ( *(_BYTE *)(*(_QWORD *)v98 + 8LL) == 16 )
                    {
                      v101 = v100[4];
                      v102 = (__int64 *)sub_1643320(*v100);
                      v103 = (__int64)sub_16463B0(v102, (unsigned int)v101);
                    }
                    else
                    {
                      v103 = sub_1643320(*v100);
                    }
                    sub_15FEC10(v10, v103, 51, 34, v98, v99, (__int64)&v138, 0);
                  }
                }
                else
                {
                  v83 = *(__int64 **)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
                  v84 = *(_QWORD *)(v13 + 24 * (1LL - (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)));
                  v85 = sub_1599EF0(*(__int64 ***)v13);
                  sub_170E100(a1->m128i_i64, v13, v85, v14, *(double *)v15.m128i_i64, a5, a6, v86, v87, a9, a10);
                  sub_170BC50((__int64)a1, v13);
                  v139.m128i_i16[0] = 257;
                  v10 = sub_15FB440(11, v83, v84, (__int64)&v138, 0);
                }
                break;
              case 195:
              case 210:
                if ( **(_DWORD **)(a2 + 56) )
                  return 0;
                v78 = *(__int64 **)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
                v79 = *(_QWORD *)(v13 + 24 * (1LL - (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)));
                v80 = sub_1599EF0(*(__int64 ***)v13);
                sub_170E100(a1->m128i_i64, v13, v80, v14, *(double *)v15.m128i_i64, a5, a6, v81, v82, a9, a10);
                sub_170BC50((__int64)a1, v13);
                v139.m128i_i16[0] = 257;
                v10 = sub_15FB440(15, v78, v79, (__int64)&v138, 0);
                break;
              case 198:
              case 211:
                if ( **(_DWORD **)(a2 + 56) )
                  return 0;
                v73 = *(__int64 **)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
                v74 = *(_QWORD *)(v13 + 24 * (1LL - (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)));
                v75 = sub_1599EF0(*(__int64 ***)v13);
                sub_170E100(a1->m128i_i64, v13, v75, v14, *(double *)v15.m128i_i64, a5, a6, v76, v77, a9, a10);
                sub_170BC50((__int64)a1, v13);
                v139.m128i_i16[0] = 257;
                v10 = sub_15FB440(13, v73, v74, (__int64)&v138, 0);
                break;
              default:
                return v10;
            }
            return v10;
          }
        }
      }
      return 0;
    }
    if ( v27 != 54 )
      return 0;
    if ( sub_15F32D0(v13) )
      return 0;
    if ( (*(_BYTE *)(v13 + 18) & 1) != 0 )
      return 0;
    v41 = *(_QWORD *)(v13 + 8);
    if ( !v41 || *(_QWORD *)(v41 + 8) )
      return 0;
    v138.m128_u64[0] = (unsigned __int64)&v139;
    v138.m128_u64[1] = 0x400000000LL;
    v42 = sub_1643350(*(_QWORD **)(a1->m128i_i64[1] + 24));
    v139.m128i_i64[0] = sub_159C470(v42, 0, 0);
    v43 = *(unsigned int **)(a2 + 56);
    v44 = *(unsigned int *)(a2 + 64);
    v138.m128_i32[2] = 1;
    v45 = &v43[v44];
    if ( v45 != v43 )
    {
      v46 = v45;
      do
      {
        v47 = *v43;
        v48 = sub_1643350(*(_QWORD **)(a1->m128i_i64[1] + 24));
        v51 = sub_159C470(v48, v47, 0);
        v52 = v138.m128_u32[2];
        if ( v138.m128_i32[2] >= (unsigned __int32)v138.m128_i32[3] )
        {
          sub_16CD150((__int64)&v138, &v139, 0, 8, v49, v50);
          v52 = v138.m128_u32[2];
        }
        ++v43;
        *(_QWORD *)(v138.m128_u64[0] + 8 * v52) = v51;
        ++v138.m128_i32[2];
      }
      while ( v43 != v46 );
      v11 = (__int64 *)a1;
    }
    v53 = (__int64 *)v11[1];
    v53[1] = *(_QWORD *)(v13 + 40);
    v53[2] = v13 + 24;
    v54 = *(_QWORD *)(v13 + 48);
    v136[0] = v54;
    if ( v54 )
    {
      sub_1623A60((__int64)v136, v54, 2);
      v55 = *v53;
      if ( !*v53 )
        goto LABEL_40;
    }
    else
    {
      v55 = *v53;
      if ( !*v53 )
        goto LABEL_42;
    }
    sub_161E7C0((__int64)v53, v55);
LABEL_40:
    v56 = (unsigned __int8 *)v136[0];
    *v53 = v136[0];
    if ( v56 )
    {
      sub_1623210((__int64)v136, v56, (__int64)v53);
    }
    else if ( v136[0] )
    {
      sub_161E7C0((__int64)v136, v136[0]);
    }
LABEL_42:
    v57 = v11[1];
    LOWORD(v137) = 257;
    v124 = sub_1709730(v57, *(_QWORD *)v13, *(_BYTE **)(v13 - 24), (__int64 **)v138.m128_u64[0], v138.m128_u32[2], v136);
    v58 = v11[1];
    v135 = 257;
    v129 = v58;
    v59 = sub_1648A60(64, 1u);
    v60 = v59;
    if ( v59 )
      sub_15F9210((__int64)v59, *(_QWORD *)(*(_QWORD *)v124 + 24LL), (__int64)v124, 0, 0, 0);
    v61 = *(_QWORD *)(v129 + 8);
    if ( v61 )
    {
      v125 = *(unsigned __int64 **)(v129 + 16);
      sub_157E9D0(v61 + 40, (__int64)v60);
      v62 = *v125;
      v63 = v60[3] & 7LL;
      v60[4] = v125;
      v62 &= 0xFFFFFFFFFFFFFFF8LL;
      v60[3] = v62 | v63;
      *(_QWORD *)(v62 + 8) = v60 + 3;
      *v125 = *v125 & 7 | (unsigned __int64)(v60 + 3);
    }
    v64 = v134;
    v65 = v60;
    sub_164B780((__int64)v60, v134);
    v131 = v60;
    if ( *(_QWORD *)(v129 + 80) )
    {
      (*(void (__fastcall **)(__int64, _QWORD **))(v129 + 88))(v129 + 64, &v131);
      v67 = *(_QWORD *)v129;
      if ( *(_QWORD *)v129 )
      {
        v136[0] = *(_QWORD *)v129;
        sub_1623A60((__int64)v136, v67, 2);
        v68 = v60[6];
        v69 = (__int64)(v60 + 6);
        if ( v68 )
        {
          sub_161E7C0((__int64)(v60 + 6), v68);
          v69 = (__int64)(v60 + 6);
        }
        v70 = (unsigned __int8 *)v136[0];
        v60[6] = v136[0];
        if ( v70 )
          sub_1623210((__int64)v136, v70, v69);
      }
      v136[0] = 0;
      v136[1] = 0;
      v137 = 0;
      sub_14A8180(v13, v136, 0);
      sub_1626170((__int64)v60, v136);
      v10 = sub_170E100(v11, a2, (__int64)v60, v14, *(double *)v15.m128i_i64, a5, a6, v71, v72, a9, a10);
      if ( (__m128i *)v138.m128_u64[0] != &v139 )
        _libc_free(v138.m128_u64[0]);
      return v10;
    }
LABEL_100:
    sub_4263D6(v65, v64, v66);
  }
  v29 = *(unsigned int **)(a2 + 56);
  v30 = *(_DWORD **)(v13 + 56);
  v31 = &v29[*(unsigned int *)(a2 + 64)];
  v127 = *(unsigned int *)(a2 + 64);
  v32 = v29;
  v33 = &v30[*(unsigned int *)(v13 + 64)];
  v34 = v31 == v29;
  v35 = v33 == v30;
  if ( v33 != v30 && v31 != v29 )
  {
    while ( *v30 == *v32 )
    {
      ++v32;
      ++v30;
      v34 = v32 == v31;
      v35 = v30 == v33;
      if ( v32 == v31 )
        goto LABEL_60;
      if ( v30 == v33 )
        goto LABEL_61;
    }
    v123 = *(unsigned int **)(a2 + 56);
    v139.m128i_i16[0] = 257;
    v36 = *(__int64 **)(v13 - 48);
    v10 = (__int64)sub_1648A60(88, 1u);
    if ( v10 )
    {
      v37 = sub_15FB2A0(*v36, v123, v127);
      sub_15F1EA0(v10, v37, 62, v10 - 24, 1, 0);
      sub_1593B40((_QWORD *)(v10 - 24), (__int64)v36);
      *(_QWORD *)(v10 + 56) = v10 + 72;
      *(_QWORD *)(v10 + 64) = 0x400000000LL;
      sub_15FB110(v10, v123, v127, (__int64)&v138);
    }
    return v10;
  }
LABEL_60:
  if ( !v35 || !v34 )
  {
LABEL_61:
    if ( v32 != v31 )
    {
      if ( v30 == v33 )
      {
        v139.m128i_i16[0] = 257;
        v88 = *(__int64 **)(v13 - 24);
        v128 = v32;
        v89 = v31 - v32;
        v90 = sub_1648A60(88, 1u);
        v10 = (__int64)v90;
        if ( v90 )
        {
          v91 = v90 - 3;
          v92 = sub_15FB2A0(*v88, v128, v89);
          sub_15F1EA0(v10, v92, 62, v10 - 24, 1, 0);
          if ( *(_QWORD *)(v10 - 24) )
          {
            v93 = *(_QWORD *)(v10 - 16);
            v94 = *(_QWORD *)(v10 - 8) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v94 = v93;
            if ( v93 )
              *(_QWORD *)(v93 + 16) = v94 | *(_QWORD *)(v93 + 16) & 3LL;
          }
          *(_QWORD *)(v10 - 24) = v88;
          v95 = v88[1];
          *(_QWORD *)(v10 - 16) = v95;
          if ( v95 )
            *(_QWORD *)(v95 + 16) = (v10 - 16) | *(_QWORD *)(v95 + 16) & 3LL;
          *(_QWORD *)(v10 - 8) = *(_QWORD *)(v10 - 8) & 3LL | (unsigned __int64)(v88 + 1);
          v88[1] = (__int64)v91;
          *(_QWORD *)(v10 + 56) = v10 + 72;
          *(_QWORD *)(v10 + 64) = 0x400000000LL;
          sub_15FB110(v10, v128, v89, (__int64)&v138);
        }
        return v10;
      }
      return 0;
    }
    v126 = a1->m128i_i64[1];
    v133 = 257;
    v104 = *(_QWORD *)(v13 - 48);
    if ( *(_BYTE *)(v104 + 16) > 0x10u )
    {
      v121 = v29;
      LOWORD(v137) = 257;
      v110 = sub_1648A60(88, 1u);
      v105 = v110;
      if ( v110 )
      {
        v111 = v121;
        v120 = v121;
        v122 = (__int64)v110;
        v112 = sub_15FB2A0(*(_QWORD *)v104, v111, v127);
        sub_15F1EA0((__int64)v105, v112, 62, (__int64)(v105 - 3), 1, 0);
        if ( *(v105 - 3) )
        {
          v113 = *(v105 - 2);
          v114 = *(v105 - 1) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v114 = v113;
          if ( v113 )
            *(_QWORD *)(v113 + 16) = v114 | *(_QWORD *)(v113 + 16) & 3LL;
        }
        *(v105 - 3) = v104;
        v115 = *(_QWORD *)(v104 + 8);
        *(v105 - 2) = v115;
        if ( v115 )
          *(_QWORD *)(v115 + 16) = (unsigned __int64)(v105 - 2) | *(_QWORD *)(v115 + 16) & 3LL;
        *(v105 - 1) = *(v105 - 1) & 3 | (v104 + 8);
        *(_QWORD *)(v104 + 8) = v105 - 3;
        v105[7] = (__int64)(v105 + 9);
        v105[8] = 0x400000000LL;
        sub_15FB110((__int64)v105, v120, v127, (__int64)v136);
      }
      else
      {
        v122 = 0;
      }
      v116 = *(_QWORD *)(v126 + 8);
      if ( v116 )
      {
        v117 = *(unsigned __int64 **)(v126 + 16);
        sub_157E9D0(v116 + 40, (__int64)v105);
        v118 = v105[3];
        v119 = *v117;
        v105[4] = (__int64)v117;
        v119 &= 0xFFFFFFFFFFFFFFF8LL;
        v105[3] = v119 | v118 & 7;
        *(_QWORD *)(v119 + 8) = v105 + 3;
        *v117 = *v117 & 7 | (unsigned __int64)(v105 + 3);
      }
      v65 = (_QWORD *)v122;
      v64 = v132;
      sub_164B780(v122, v132);
      v130 = v105;
      if ( !*(_QWORD *)(v126 + 80) )
        goto LABEL_100;
      (*(void (__fastcall **)(__int64, __int64 **))(v126 + 88))(v126 + 64, &v130);
      sub_12A86E0((__int64 *)v126, (__int64)v105);
    }
    else
    {
      v105 = (__int64 *)sub_15A3AE0(*(_QWORD **)(v13 - 48), v29, v127, 0);
      v106 = sub_14DBA30((__int64)v105, *(_QWORD *)(v126 + 96), 0);
      if ( v106 )
        v105 = (__int64 *)v106;
    }
    v139.m128i_i16[0] = 257;
    v107 = *(_QWORD *)(v13 - 24);
    v108 = sub_1648A60(88, 2u);
    v109 = v33 - v30;
    v10 = (__int64)v108;
    if ( v108 )
    {
      sub_15F1EA0((__int64)v108, *v105, 63, (__int64)(v108 - 6), 2, 0);
      *(_QWORD *)(v10 + 56) = v10 + 72;
      *(_QWORD *)(v10 + 64) = 0x400000000LL;
      sub_15FAD90(v10, (__int64)v105, v107, v30, v109, (__int64)&v138);
    }
    return v10;
  }
  return sub_170E100(a1->m128i_i64, a2, *(_QWORD *)(v13 - 24), v14, *(double *)v15.m128i_i64, a5, a6, v19, v20, a9, a10);
}
