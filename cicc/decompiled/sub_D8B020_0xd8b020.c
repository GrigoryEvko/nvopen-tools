// Function: sub_D8B020
// Address: 0xd8b020
//
__int64 __fastcall sub_D8B020(__int64 a1, __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 *v10; // rbx
  __m128i **v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __m128i *v15; // r14
  __int8 v16; // r15
  __m128i **v17; // rax
  __int64 v18; // rax
  __m128i **v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __m128i *v23; // r14
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r15
  _BOOL4 v27; // r8d
  __int64 v28; // rax
  __int64 v29; // rdx
  char v30; // r14
  unsigned __int64 v31; // r15
  char v32; // al
  __m128i *v33; // r15
  __int64 v34; // r15
  _QWORD *v35; // rax
  __int64 v36; // rdx
  _BOOL4 v37; // r8d
  __int64 v38; // rax
  char v39; // dl
  __int64 v40; // rax
  __m128i *v41; // r15
  unsigned __int64 v42; // rdx
  _QWORD *v43; // rax
  __int64 v44; // rdx
  _BOOL4 v45; // r14d
  __int64 v46; // rax
  __m128i *v47; // r14
  _QWORD *v48; // rax
  __int64 v49; // rdx
  _BOOL4 v50; // r8d
  __int64 v51; // rax
  int v52; // edx
  __int64 v53; // rcx
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r15
  __int64 v57; // rax
  int v58; // r15d
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r15
  __int64 v62; // rax
  __int64 v63; // r8
  __int64 v64; // rsi
  __int64 v65; // rdx
  char v66; // r14
  unsigned __int64 v67; // r15
  char v68; // al
  __int64 v69; // rax
  int v70; // edx
  __int64 v71; // rcx
  __int64 v72; // rax
  __int64 v73; // rax
  int v74; // eax
  __int64 v75; // rcx
  __int64 v76; // rdx
  unsigned __int8 *v77; // rax
  __int64 v78; // r14
  __m128i **v79; // rax
  __int64 v80; // rax
  __m128i *v81; // r15
  unsigned __int64 v82; // rdx
  __int64 v83; // rax
  __int64 v84; // rax
  __m128i *v85; // rax
  char v86; // dl
  char v87; // dl
  _QWORD *v88; // rax
  __int64 v89; // r15
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // r15
  unsigned int v93; // eax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // r14
  unsigned int v97; // r15d
  bool v98; // al
  unsigned int v99; // eax
  __int64 *m128i_i64; // r15
  __int64 v101; // [rsp+28h] [rbp-228h]
  _BOOL4 v102; // [rsp+30h] [rbp-220h]
  _BOOL4 v103; // [rsp+30h] [rbp-220h]
  __int64 v104; // [rsp+30h] [rbp-220h]
  __int64 v105; // [rsp+38h] [rbp-218h]
  __int64 v106; // [rsp+38h] [rbp-218h]
  _BOOL4 v107; // [rsp+38h] [rbp-218h]
  __int64 v108; // [rsp+38h] [rbp-218h]
  __int64 v109; // [rsp+38h] [rbp-218h]
  unsigned int v110; // [rsp+38h] [rbp-218h]
  __int64 v111; // [rsp+38h] [rbp-218h]
  __int64 v112; // [rsp+38h] [rbp-218h]
  __m128i *v114; // [rsp+48h] [rbp-208h] BYREF
  __int64 v115; // [rsp+58h] [rbp-1F8h] BYREF
  __int64 v116; // [rsp+60h] [rbp-1F0h] BYREF
  __m128i *v117; // [rsp+68h] [rbp-1E8h] BYREF
  __int64 v118; // [rsp+70h] [rbp-1E0h] BYREF
  unsigned int v119; // [rsp+78h] [rbp-1D8h]
  unsigned __int64 v120; // [rsp+80h] [rbp-1D0h] BYREF
  unsigned int v121; // [rsp+88h] [rbp-1C8h]
  unsigned __int64 v122; // [rsp+90h] [rbp-1C0h] BYREF
  unsigned int v123; // [rsp+98h] [rbp-1B8h]
  __int64 v124; // [rsp+A0h] [rbp-1B0h] BYREF
  __m128i *v125; // [rsp+B0h] [rbp-1A0h] BYREF
  unsigned int v126; // [rsp+B8h] [rbp-198h]
  unsigned __int64 v127; // [rsp+C0h] [rbp-190h] BYREF
  unsigned int v128; // [rsp+C8h] [rbp-188h]
  __m128i v129; // [rsp+D0h] [rbp-180h] BYREF
  __int64 v130[2]; // [rsp+E0h] [rbp-170h] BYREF
  _QWORD v131[8]; // [rsp+F0h] [rbp-160h] BYREF
  _QWORD *v132; // [rsp+130h] [rbp-120h] BYREF
  __int64 v133; // [rsp+138h] [rbp-118h]
  _QWORD v134[8]; // [rsp+140h] [rbp-110h] BYREF
  __int64 v135; // [rsp+180h] [rbp-D0h] BYREF
  __m128i **v136; // [rsp+188h] [rbp-C8h]
  __int64 v137; // [rsp+190h] [rbp-C0h]
  int v138; // [rsp+198h] [rbp-B8h]
  char v139; // [rsp+19Ch] [rbp-B4h]
  char v140; // [rsp+1A0h] [rbp-B0h] BYREF

  result = (__int64)a2;
  v7 = v134;
  v136 = (__m128i **)&v140;
  v133 = 0x800000001LL;
  v114 = a2;
  v135 = 0;
  v137 = 16;
  v138 = 0;
  v139 = 1;
  v132 = v134;
  v134[0] = a2;
  if ( a2->m128i_i8[0] != 60 )
    result = 0;
  v115 = result;
  LODWORD(result) = 1;
LABEL_4:
  v8 = (unsigned int)result;
  result = (unsigned int)(result - 1);
  v9 = v7[v8 - 1];
  LODWORD(v133) = result;
  v10 = *(__int64 **)(v9 + 16);
  v116 = v9;
  if ( !v10 )
    goto LABEL_14;
  while ( 2 )
  {
    a2 = (__m128i *)v10[3];
    v117 = a2;
    if ( !(unsigned __int8)sub_104D2E0(a4) )
      goto LABEL_12;
    v15 = v117;
    v131[3] = a1;
    v131[0] = &v116;
    v131[5] = a4;
    v131[1] = a3;
    v131[2] = &v117;
    v131[4] = &v115;
    v131[6] = v10;
    v131[7] = &v114;
    v16 = v117->m128i_i8[0];
    switch ( v117->m128i_i8[0] )
    {
      case 0x1E:
        v34 = a1 + 32;
        goto LABEL_44;
      case 0x22:
      case 0x55:
        if ( sub_B46A10((__int64)v117) )
          goto LABEL_12;
        if ( !v115 )
          goto LABEL_25;
        if ( !(unsigned __int8)sub_104D360(a4, v115, v15) )
        {
          v15 = v117;
          v34 = a1 + 32;
LABEL_44:
          v129.m128i_i64[0] = (__int64)v15;
          v35 = sub_D8ACC0(a3 + 32, (unsigned __int64 *)&v129);
          if ( v36 )
          {
            v37 = 1;
            if ( !v35 && v36 != a3 + 40 )
              v37 = (unsigned __int64)v15 < *(_QWORD *)(v36 + 32);
            v103 = v37;
            v105 = v36;
            v38 = sub_22077B0(40);
            *(_QWORD *)(v38 + 32) = v129.m128i_i64[0];
            sub_220F040(v103, v38, v105, a3 + 40);
            ++*(_QWORD *)(a3 + 72);
          }
LABEL_47:
          a2 = (__m128i *)v34;
          sub_D87370(a3, v34);
          v10 = (__int64 *)v10[1];
          if ( !v10 )
            goto LABEL_13;
          continue;
        }
        v15 = v117;
        v16 = v117->m128i_i8[0];
LABEL_25:
        if ( v16 == 85 )
        {
          v69 = v15[-2].m128i_i64[0];
          if ( v69 )
          {
            if ( !*(_BYTE *)v69 && *(_QWORD *)(v69 + 24) == v15[5].m128i_i64[0] && (*(_BYTE *)(v69 + 33) & 0x20) != 0 )
            {
              v70 = *(_DWORD *)(v69 + 36);
              if ( (unsigned int)(v70 - 238) <= 7 && ((1LL << ((unsigned __int8)v70 + 18)) & 0xAD) != 0 )
              {
                v71 = *v10;
                v72 = v15->m128i_i32[1] & 0x7FFFFFF;
                if ( (v70 != 238 && (unsigned int)(v70 - 240) > 1 || v71 != v15[2 * (1 - v72)].m128i_i64[0])
                  && v71 != v15[-2 * v72].m128i_i64[0] )
                {
                  sub_AADB10((__int64)&v122, *(_DWORD *)(a1 + 24), 0);
LABEL_95:
                  v73 = v15[-2].m128i_i64[0];
                  if ( !v73 || *(_BYTE *)v73 || *(_QWORD *)(v73 + 24) != v15[5].m128i_i64[0] )
                    BUG();
                  v74 = *(_DWORD *)(v73 + 36);
                  v75 = *v10;
                  v76 = v15->m128i_i32[1] & 0x7FFFFFF;
                  if ( ((v74 == 238 || (unsigned int)(v74 - 240) <= 1) && v15[2 * (1 - v76)].m128i_i64[0] == v75
                     || v15[-2 * v76].m128i_i64[0] == v75)
                    && (v112 = v115,
                        v95 = sub_DD8400(*(_QWORD *)(a1 + 16), v15[2 * (2 - v76)].m128i_i64[0]),
                        v96 = v95,
                        v112)
                    && ((unsigned __int8)sub_D96A50(v95) || !(unsigned __int8)sub_D88C60(a1, *v10, v10[3], v112, v96)) )
                  {
                    v129.m128i_i64[0] = (__int64)v117;
                    sub_D8AD60(a3 + 32, (unsigned __int64 *)&v129);
                  }
                  else
                  {
                    v129.m128i_i64[0] = (__int64)v117;
                  }
                  a2 = (__m128i *)&v122;
                  sub_D87370(a3, (__int64)&v122);
                  sub_969240(&v124);
                  sub_969240((__int64 *)&v122);
                  v10 = (__int64 *)v10[1];
                  if ( !v10 )
                    goto LABEL_13;
                  continue;
                }
                v110 = *(_DWORD *)(a1 + 24);
                v101 = (__int64)v114;
                v88 = (_QWORD *)sub_B2BE50(**(_QWORD **)(a1 + 16));
                v89 = sub_BCD140(v88, v110);
                if ( !(unsigned __int8)sub_D97040(
                                         *(_QWORD *)(a1 + 16),
                                         *(_QWORD *)(v15[2 * (2LL - (v15->m128i_i32[1] & 0x7FFFFFF))].m128i_i64[0] + 8)) )
                {
                  sub_AAF450((__int64)&v122, a1 + 32);
                  goto LABEL_95;
                }
                v111 = *(_QWORD *)(a1 + 16);
                v90 = sub_DD8400(v111, v15[2 * (2LL - (v15->m128i_i32[1] & 0x7FFFFFF))].m128i_i64[0]);
                v91 = sub_DC5760(v111, v90, v89, 0);
                v92 = sub_DBB9F0(*(_QWORD *)(a1 + 16), v91, 1, 0);
                v126 = *(_DWORD *)(v92 + 8);
                if ( v126 > 0x40 )
                  sub_C43780((__int64)&v125, (const void **)v92);
                else
                  v125 = *(__m128i **)v92;
                v93 = *(_DWORD *)(v92 + 24);
                v128 = v93;
                if ( v93 <= 0x40 )
                {
                  v94 = 1LL << ((unsigned __int8)v93 - 1);
                  v127 = *(_QWORD *)(v92 + 16);
                  goto LABEL_140;
                }
                sub_C43780((__int64)&v127, (const void **)(v92 + 16));
                v97 = v128;
                v94 = 1LL << ((unsigned __int8)v128 - 1);
                if ( v128 <= 0x40 )
                {
LABEL_140:
                  if ( (v94 & v127) == 0 )
                  {
                    v98 = v127 == 0;
                    goto LABEL_150;
                  }
                }
                else if ( (*(_QWORD *)(v127 + 8LL * ((v128 - 1) >> 6)) & v94) == 0 )
                {
                  v98 = v97 == (unsigned int)sub_C444A0((__int64)&v127);
LABEL_150:
                  if ( !v98 && !sub_D85770((__int64)&v125) )
                  {
                    sub_AB4E00((__int64)&v129, (__int64)&v125, *(_DWORD *)(a1 + 24));
                    sub_D859E0((__int64 *)&v125, v129.m128i_i64);
                    sub_969240(v130);
                    sub_969240(v129.m128i_i64);
                    v121 = v128;
                    if ( v128 > 0x40 )
                      sub_C43780((__int64)&v120, (const void **)&v127);
                    else
                      v120 = v127;
                    sub_C46F20((__int64)&v120, 1u);
                    v99 = v121;
                    v121 = 0;
                    v123 = v99;
                    v122 = v120;
                    v119 = *(_DWORD *)(a1 + 24);
                    if ( v119 > 0x40 )
                      sub_C43690((__int64)&v118, 0, 0);
                    else
                      v118 = 0;
                    sub_AADC30((__int64)&v129, (__int64)&v118, (__int64 *)&v122);
                    sub_969240(&v118);
                    sub_969240((__int64 *)&v122);
                    sub_969240((__int64 *)&v120);
                    sub_D89240((__int64)&v122, a1, *v10, v101, (__int64)&v129);
                    sub_969240(v130);
                    sub_969240(v129.m128i_i64);
                    goto LABEL_142;
                  }
                }
                sub_AAF450((__int64)&v122, a1 + 32);
LABEL_142:
                sub_969240((__int64 *)&v127);
                sub_969240((__int64 *)&v125);
                goto LABEL_95;
              }
            }
          }
        }
        v18 = sub_B494D0((__int64)v15, 52);
        if ( v116 != v18 )
          goto LABEL_27;
        if ( !v139 )
          goto LABEL_133;
        v79 = v136;
        v20 = HIDWORD(v137);
        v19 = &v136[HIDWORD(v137)];
        if ( v136 == v19 )
        {
LABEL_114:
          if ( HIDWORD(v137) >= (unsigned int)v137 )
          {
LABEL_133:
            sub_C8CC70((__int64)&v135, (__int64)v117, (__int64)v19, v20, v21, v22);
            if ( !v87 )
              goto LABEL_27;
          }
          else
          {
            ++HIDWORD(v137);
            *v19 = v117;
            ++v135;
          }
          v80 = (unsigned int)v133;
          v81 = v117;
          v82 = (unsigned int)v133 + 1LL;
          if ( v82 > HIDWORD(v133) )
          {
            sub_C8D5F0((__int64)&v132, v134, v82, 8u, v21, v22);
            v80 = (unsigned int)v133;
          }
          v132[v80] = v81;
          LODWORD(v133) = v133 + 1;
        }
        else
        {
          while ( v117 != *v79 )
          {
            if ( v19 == ++v79 )
              goto LABEL_114;
          }
        }
LABEL_27:
        if ( v10 < v15[-2 * (v15->m128i_i32[1] & 0x7FFFFFF)].m128i_i64 )
          goto LABEL_28;
        v52 = v15->m128i_u8[0];
        if ( v52 == 40 )
        {
          v53 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v15);
        }
        else
        {
          v53 = -32;
          if ( v52 != 85 )
          {
            if ( v52 != 34 )
              BUG();
            v53 = -96;
          }
        }
        if ( v15->m128i_i8[7] < 0 )
        {
          v108 = v53;
          v54 = sub_BD2BC0((__int64)v15);
          v53 = v108;
          v56 = v54 + v55;
          if ( v15->m128i_i8[7] >= 0 )
          {
            if ( (unsigned int)(v56 >> 4) )
LABEL_165:
              BUG();
          }
          else
          {
            v57 = sub_BD2BC0((__int64)v15);
            v53 = v108;
            if ( (unsigned int)((v56 - v57) >> 4) )
            {
              if ( v15->m128i_i8[7] >= 0 )
                goto LABEL_165;
              v58 = *(_DWORD *)(sub_BD2BC0((__int64)v15) + 8);
              if ( v15->m128i_i8[7] >= 0 )
                BUG();
              v59 = sub_BD2BC0((__int64)v15);
              v53 = v108 - 32LL * (unsigned int)(*(_DWORD *)(v59 + v60 - 4) - v58);
            }
          }
        }
        if ( v10 >= (__int64 *)((char *)v15->m128i_i64 + v53) )
        {
LABEL_28:
          v23 = v117;
          v129.m128i_i64[0] = (__int64)v117;
          v24 = sub_D8ACC0(a3 + 32, (unsigned __int64 *)&v129);
          v26 = v25;
          if ( v25 )
          {
            v27 = 1;
            if ( !v24 && v25 != a3 + 40 )
              v27 = (unsigned __int64)v23 < *(_QWORD *)(v25 + 32);
            v102 = v27;
            v28 = sub_22077B0(40);
            *(_QWORD *)(v28 + 32) = v129.m128i_i64[0];
            sub_220F040(v102, v28, v26, a3 + 40);
            ++*(_QWORD *)(a3 + 72);
          }
          a2 = (__m128i *)(a1 + 32);
          sub_D87370(a3, a1 + 32);
          v10 = (__int64 *)v10[1];
          if ( !v10 )
            goto LABEL_13;
        }
        else
        {
          v61 = ((char *)v10 - (char *)&v15[-2 * (v15->m128i_i32[1] & 0x7FFFFFF)]) >> 5;
          if ( (unsigned __int8)sub_B49B80((__int64)v15, v61, 81) )
          {
            v109 = *(_QWORD *)(a1 + 8);
            v62 = sub_A748A0(&v15[4].m128i_i64[1], v61);
            v63 = v109;
            v64 = v62;
            if ( !v62 )
            {
              v83 = v15[-2].m128i_i64[0];
              if ( v83 )
              {
                if ( !*(_BYTE *)v83 && *(_QWORD *)(v83 + 24) == v15[5].m128i_i64[0] )
                {
                  v129.m128i_i64[0] = *(_QWORD *)(v83 + 120);
                  v84 = sub_A748A0(&v129, v61);
                  v63 = v109;
                  v64 = v84;
                }
              }
            }
            v129.m128i_i64[0] = sub_9208B0(v63, v64);
            v129.m128i_i64[1] = v65;
            v66 = v65;
            v67 = (unsigned __int64)(v129.m128i_i64[0] + 7) >> 3;
            sub_D89430((__int64)&v129, a1, *v10, (__int64)v114, v67, v65);
            v68 = sub_D88E20(a1, v10, v115, v67, v66);
            v125 = v117;
            if ( !v68 )
              sub_D8AD60(a3 + 32, (unsigned __int64 *)&v125);
            a2 = &v129;
            sub_D87370(a3, (__int64)&v129);
            sub_969240(v130);
            sub_969240(v129.m128i_i64);
            v10 = (__int64 *)v10[1];
            if ( !v10 )
              goto LABEL_13;
          }
          else
          {
            v77 = sub_BD3990((unsigned __int8 *)v15[-2].m128i_i64[0], (unsigned int)v61);
            v78 = (__int64)v77;
            if ( *v77 > 3u || *v77 == 2 )
            {
              v129.m128i_i64[0] = (__int64)v117;
              sub_D8AD60(a3 + 32, (unsigned __int64 *)&v129);
              a2 = (__m128i *)(a1 + 32);
              sub_D87370(a3, a1 + 32);
              v10 = (__int64 *)v10[1];
              if ( !v10 )
                goto LABEL_13;
            }
            else
            {
              sub_D890C0((__int64)&v125, a1, *v10, (__int64)v114);
              a2 = &v129;
              v129.m128i_i64[1] = (unsigned int)v61;
              v129.m128i_i64[0] = v78;
              v85 = sub_D87C70((_QWORD *)(a3 + 80), &v129, (__int64)&v125);
              if ( !v86 )
              {
                m128i_i64 = v85[3].m128i_i64;
                sub_AB3510((__int64)&v129, (__int64)v85[3].m128i_i64, (__int64)&v125, 0);
                a2 = &v129;
                sub_D859E0(m128i_i64, v129.m128i_i64);
                sub_969240(v130);
                sub_969240(v129.m128i_i64);
              }
              sub_969240((__int64 *)&v127);
              sub_969240((__int64 *)&v125);
              v10 = (__int64 *)v10[1];
              if ( !v10 )
                goto LABEL_13;
            }
          }
        }
        continue;
      case 0x3D:
        if ( !v115 )
          goto LABEL_38;
        if ( !(unsigned __int8)sub_104D360(a4, v115, v117) )
        {
          v47 = v117;
          v34 = a1 + 32;
          v129.m128i_i64[0] = (__int64)v117;
          v48 = sub_D8ACC0(a3 + 32, (unsigned __int64 *)&v129);
          if ( v49 )
          {
            v50 = 1;
            if ( !v48 && v49 != a3 + 40 )
              v50 = (unsigned __int64)v47 < *(_QWORD *)(v49 + 32);
            v104 = v49;
            v107 = v50;
            v51 = sub_22077B0(40);
            *(_QWORD *)(v51 + 32) = v129.m128i_i64[0];
            sub_220F040(v107, v51, v104, a3 + 40);
            ++*(_QWORD *)(a3 + 72);
          }
          goto LABEL_47;
        }
        v15 = v117;
LABEL_38:
        v129.m128i_i64[0] = sub_9208B0(*(_QWORD *)(a1 + 8), v15->m128i_i64[1]);
        v129.m128i_i64[1] = v29;
        v30 = v29;
        v31 = (unsigned __int64)(v129.m128i_i64[0] + 7) >> 3;
        sub_D89430((__int64)&v129, a1, *v10, (__int64)v114, v31, v29);
        v32 = sub_D88E20(a1, v10, v115, v31, v30);
        v33 = v117;
        v125 = v117;
        if ( !v32 )
        {
          v43 = sub_D8ACC0(a3 + 32, (unsigned __int64 *)&v125);
          if ( v44 )
          {
            v45 = 1;
            if ( !v43 && v44 != a3 + 40 )
              v45 = (unsigned __int64)v33 < *(_QWORD *)(v44 + 32);
            v106 = v44;
            v46 = sub_22077B0(40);
            *(_QWORD *)(v46 + 32) = v125;
            sub_220F040(v45, v46, v106, a3 + 40);
            ++*(_QWORD *)(a3 + 72);
          }
        }
        a2 = &v129;
        sub_D87370(a3, (__int64)&v129);
        sub_969240(v130);
        if ( v129.m128i_i32[2] <= 0x40u || !v129.m128i_i64[0] )
          goto LABEL_12;
        j_j___libc_free_0_0(v129.m128i_i64[0]);
        v10 = (__int64 *)v10[1];
        if ( !v10 )
          goto LABEL_13;
        continue;
      case 0x3E:
        a2 = (__m128i *)v117[-4].m128i_i64[0];
        sub_D8AE10((__int64)v131, (__int64)a2);
        v10 = (__int64 *)v10[1];
        if ( !v10 )
          goto LABEL_13;
        continue;
      case 0x41:
      case 0x42:
        a2 = (__m128i *)v117[-2].m128i_i64[0];
        sub_D8AE10((__int64)v131, (__int64)a2);
        v10 = (__int64 *)v10[1];
        if ( !v10 )
          goto LABEL_13;
        continue;
      case 0x59:
        goto LABEL_12;
      default:
        if ( !v139 )
          goto LABEL_49;
        v17 = v136;
        v12 = HIDWORD(v137);
        v11 = &v136[HIDWORD(v137)];
        if ( v136 == v11 )
          goto LABEL_54;
        do
        {
          if ( v117 == *v17 )
            goto LABEL_12;
          ++v17;
        }
        while ( v11 != v17 );
LABEL_54:
        if ( HIDWORD(v137) >= (unsigned int)v137 )
        {
LABEL_49:
          a2 = v117;
          sub_C8CC70((__int64)&v135, (__int64)v117, (__int64)v11, v12, v13, v14);
          if ( v39 )
            goto LABEL_50;
LABEL_12:
          v10 = (__int64 *)v10[1];
          if ( !v10 )
            goto LABEL_13;
          continue;
        }
        ++HIDWORD(v137);
        *v11 = v117;
        ++v135;
LABEL_50:
        v40 = (unsigned int)v133;
        v41 = v117;
        v42 = (unsigned int)v133 + 1LL;
        if ( v42 > HIDWORD(v133) )
        {
          a2 = (__m128i *)v134;
          sub_C8D5F0((__int64)&v132, v134, v42, 8u, v13, v14);
          v40 = (unsigned int)v133;
        }
        v132[v40] = v41;
        LODWORD(v133) = v133 + 1;
        v10 = (__int64 *)v10[1];
        if ( v10 )
          continue;
LABEL_13:
        result = (unsigned int)v133;
        v7 = v132;
LABEL_14:
        if ( (_DWORD)result )
          goto LABEL_4;
        if ( v7 != v134 )
          result = _libc_free(v7, a2);
        if ( !v139 )
          return _libc_free(v136, a2);
        return result;
    }
  }
}
