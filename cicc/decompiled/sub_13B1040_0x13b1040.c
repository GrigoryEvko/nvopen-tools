// Function: sub_13B1040
// Address: 0x13b1040
//
__m128i **__fastcall sub_13B1040(__m128i **a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v5; // rbx
  char v6; // al
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // r13
  __m128i *v10; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __m128i v15; // xmm5
  __m128i v16; // xmm6
  int v17; // edx
  __m128i v18; // xmm0
  __int64 v19; // rax
  __int64 v20; // rdx
  __m128i v21; // xmm7
  __m128i v22; // xmm6
  __int64 v23; // rax
  __int64 v24; // rax
  __m128i v25; // xmm3
  __int64 v26; // rdi
  __m128i v27; // xmm4
  __int64 v28; // rbx
  __m128i v29; // xmm5
  __m128i v30; // xmm6
  __m128i v31; // xmm6
  __m128i v32; // xmm7
  __m128i v33; // xmm7
  __m128i v34; // xmm5
  __int64 v35; // rbx
  __int64 v36; // rbx
  __int64 v37; // rbx
  __int64 v38; // rbx
  __int64 v39; // rax
  int v40; // r9d
  __int64 v41; // r8
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // rbx
  __int64 v45; // rcx
  int v46; // r8d
  int v47; // r9d
  __int64 v48; // rcx
  int v49; // r8d
  int v50; // r9d
  __int64 v51; // r15
  _QWORD *v52; // rax
  unsigned int v53; // esi
  int v54; // r8d
  int v55; // r9d
  unsigned int v56; // r15d
  __int64 v57; // rax
  __int64 v58; // r12
  __int64 v59; // rbx
  int v60; // edx
  __int64 v61; // rbx
  __int64 v62; // rdx
  __int64 v63; // rcx
  unsigned __int64 v64; // rcx
  __int64 v65; // rdx
  __int64 v66; // r14
  int v67; // r8d
  int v68; // r9d
  _QWORD *v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rcx
  int v72; // r8d
  int v73; // r9d
  _QWORD *v74; // rax
  __int64 v75; // r15
  int v76; // eax
  __int64 v77; // rax
  size_t v78; // r10
  size_t v79; // r8
  void *v80; // r9
  const void *v81; // rsi
  _QWORD **v82; // rax
  __int64 v83; // r12
  _QWORD **v84; // rax
  int v85; // r12d
  __int64 v87; // rax
  unsigned int v88; // edx
  unsigned __int64 v89; // r10
  int v91; // eax
  unsigned int v92; // r9d
  unsigned int v93; // esi
  __int64 v94; // r11
  int v95; // ecx
  unsigned __int64 v96; // r8
  __int64 v97; // rdx
  unsigned __int64 v98; // r8
  unsigned __int64 v99; // r10
  int v102; // eax
  __int64 v103; // r12
  __m128i *v104; // rax
  __m128i *v105; // rdx
  unsigned int v106; // ebx
  __int64 v107; // rdx
  unsigned int v108; // eax
  const __m128i *v109; // r13
  _QWORD *v110; // rbx
  unsigned __int64 *v111; // rax
  unsigned int v112; // esi
  __int64 v113; // r8
  __int64 v114; // rax
  __int64 v115; // r8
  __int64 v116; // r14
  __m128i *v117; // r12
  unsigned __int64 *v118; // r13
  unsigned __int64 *v119; // r13
  unsigned __int64 *v120; // r13
  unsigned int v121; // eax
  __int64 v122; // r12
  __int64 v123; // rcx
  int v124; // r8d
  int v125; // r9d
  __int64 v126; // rdx
  __int64 v127; // r12
  __int64 v128; // rsi
  unsigned int v129; // edx
  unsigned __int64 v130; // rax
  unsigned int v131; // eax
  int v132; // ebx
  __m128i *v133; // rax
  __int64 v134; // rdx
  unsigned __int64 v135; // r11
  unsigned int v136; // eax
  unsigned int v137; // r12d
  _QWORD *v138; // rax
  unsigned int v139; // ebx
  __int64 v140; // r12
  char v141; // al
  unsigned int v142; // r12d
  unsigned int v143; // r13d
  __int64 v144; // rbx
  __int64 v145; // r12
  _QWORD *v146; // rax
  __int64 v147; // rbx
  unsigned __int64 *v148; // r10
  unsigned int v149; // eax
  unsigned int v150; // edx
  _QWORD *v151; // rax
  __int64 v152; // rdx
  unsigned int v153; // r12d
  __int64 v154; // rax
  int v155; // ebx
  _QWORD *v156; // rax
  _QWORD *v157; // r12
  int v158; // eax
  void *v159; // r8
  __int64 **v160; // rsi
  __int64 v161; // rdx
  __int64 v162; // rax
  __int64 v163; // rax
  __int64 v164; // [rsp+8h] [rbp-318h]
  _QWORD *v165; // [rsp+10h] [rbp-310h]
  void *v166; // [rsp+10h] [rbp-310h]
  void *v167; // [rsp+10h] [rbp-310h]
  size_t v168; // [rsp+28h] [rbp-2F8h]
  __int64 v169; // [rsp+28h] [rbp-2F8h]
  size_t v170; // [rsp+30h] [rbp-2F0h]
  __int64 v171; // [rsp+38h] [rbp-2E8h]
  unsigned int v172; // [rsp+48h] [rbp-2D8h]
  size_t v173; // [rsp+58h] [rbp-2C8h]
  size_t v174; // [rsp+58h] [rbp-2C8h]
  size_t n; // [rsp+60h] [rbp-2C0h]
  size_t na; // [rsp+60h] [rbp-2C0h]
  bool v177; // [rsp+70h] [rbp-2B0h]
  const __m128i *v178; // [rsp+70h] [rbp-2B0h]
  __int64 v179; // [rsp+70h] [rbp-2B0h]
  size_t v180; // [rsp+78h] [rbp-2A8h]
  bool v181; // [rsp+78h] [rbp-2A8h]
  __int64 v182; // [rsp+78h] [rbp-2A8h]
  __int64 v183; // [rsp+80h] [rbp-2A0h]
  unsigned int v184; // [rsp+80h] [rbp-2A0h]
  char v185; // [rsp+8Fh] [rbp-291h]
  __int64 v186; // [rsp+90h] [rbp-290h]
  __int64 v187; // [rsp+90h] [rbp-290h]
  unsigned int v188; // [rsp+90h] [rbp-290h]
  __int64 v189; // [rsp+98h] [rbp-288h]
  _QWORD *v190; // [rsp+98h] [rbp-288h]
  __int64 v191; // [rsp+98h] [rbp-288h]
  unsigned __int64 *v192; // [rsp+98h] [rbp-288h]
  char v193; // [rsp+98h] [rbp-288h]
  __int64 v194; // [rsp+98h] [rbp-288h]
  __int64 v195; // [rsp+98h] [rbp-288h]
  __int64 v196; // [rsp+A0h] [rbp-280h]
  __int64 v197; // [rsp+A0h] [rbp-280h]
  unsigned int v199; // [rsp+B4h] [rbp-26Ch] BYREF
  unsigned __int64 v200; // [rsp+B8h] [rbp-268h] BYREF
  unsigned __int64 *v201; // [rsp+C0h] [rbp-260h] BYREF
  unsigned __int64 v202; // [rsp+C8h] [rbp-258h] BYREF
  unsigned __int64 *v203; // [rsp+D0h] [rbp-250h] BYREF
  unsigned __int64 *v204; // [rsp+D8h] [rbp-248h] BYREF
  unsigned __int64 *v205; // [rsp+E0h] [rbp-240h] BYREF
  __int64 v206; // [rsp+E8h] [rbp-238h] BYREF
  unsigned __int64 **v207; // [rsp+F0h] [rbp-230h] BYREF
  unsigned int v208; // [rsp+F8h] [rbp-228h]
  unsigned __int64 **i; // [rsp+100h] [rbp-220h] BYREF
  unsigned int v210; // [rsp+108h] [rbp-218h]
  _BYTE v211[48]; // [rsp+110h] [rbp-210h] BYREF
  __m128i v212; // [rsp+140h] [rbp-1E0h] BYREF
  __m128i v213; // [rsp+150h] [rbp-1D0h] BYREF
  __int64 v214; // [rsp+160h] [rbp-1C0h]
  __m128i v215; // [rsp+170h] [rbp-1B0h] BYREF
  __m128i v216; // [rsp+180h] [rbp-1A0h] BYREF
  __int64 v217; // [rsp+190h] [rbp-190h]
  __int32 v218; // [rsp+198h] [rbp-188h] BYREF
  __int64 v219; // [rsp+1A0h] [rbp-180h]
  __m128i v220; // [rsp+1B0h] [rbp-170h] BYREF
  __m128i v221; // [rsp+1C0h] [rbp-160h] BYREF
  __int64 v222; // [rsp+1D0h] [rbp-150h]
  __m128i v223; // [rsp+220h] [rbp-100h] BYREF
  __m128i v224; // [rsp+230h] [rbp-F0h] BYREF
  __int64 v225; // [rsp+240h] [rbp-E0h]
  __int64 v226; // [rsp+248h] [rbp-D8h]

  v6 = 0;
  v7 = a3;
  v8 = a2;
  v9 = a4;
  if ( a3 != a4 )
    v6 = a5;
  v185 = v6;
  if ( !(unsigned __int8)sub_15F2ED0(a3) && !(unsigned __int8)sub_15F3040(v7)
    || !(unsigned __int8)sub_15F2ED0(v9) && !(unsigned __int8)sub_15F3040(v9) )
  {
    goto LABEL_13;
  }
  if ( !sub_13A3460(v7) || !(v177 = sub_13A3460(v9)) )
  {
LABEL_8:
    v10 = (__m128i *)sub_22077B0(40);
    if ( v10 )
    {
      v10->m128i_i64[1] = v7;
      v10[1].m128i_i64[0] = v9;
      v10[1].m128i_i64[1] = 0;
      v10->m128i_i64[0] = (__int64)&unk_49E9758;
      v10[2].m128i_i64[0] = 0;
    }
    *a1 = v10;
    return a1;
  }
  v186 = sub_13A4950(v7);
  v183 = sub_13A4950(v9);
  switch ( *(_BYTE *)(v7 + 16) )
  {
    case '6':
      sub_141EB40(&v220, v7, v12, v13, v14);
      goto LABEL_16;
    case '7':
      sub_141EDF0(&v220, v7, v12, v13, v14);
      goto LABEL_32;
    case ':':
      sub_141F110(&v220);
LABEL_16:
      v15 = _mm_loadu_si128(&v220);
      v16 = _mm_loadu_si128(&v221);
      v225 = v222;
      v223 = v15;
      v196 = v220.m128i_i64[0];
      v224 = v16;
      break;
    case ';':
      sub_141F3C0(&v220);
LABEL_32:
      v33 = _mm_loadu_si128(&v220);
      v34 = _mm_loadu_si128(&v221);
      v225 = v222;
      v223 = v33;
      v196 = v220.m128i_i64[0];
      v224 = v34;
      break;
    case 'R':
      sub_141F0A0(&v220);
      v31 = _mm_loadu_si128(&v220);
      v32 = _mm_loadu_si128(&v221);
      v225 = v222;
      v223 = v31;
      v196 = v220.m128i_i64[0];
      v224 = v32;
      break;
    default:
      break;
  }
  v17 = *(unsigned __int8 *)(v9 + 16);
  v18 = _mm_loadu_si128(&v224);
  v223.m128i_i64[0] = v196;
  v19 = v225;
  v20 = (unsigned int)(v17 - 54);
  v216 = v18;
  v217 = v225;
  v215 = _mm_loadu_si128(&v223);
  switch ( (int)v20 )
  {
    case 0:
      sub_141EB40(&v220, v9, v20, v13, v14);
      goto LABEL_19;
    case 1:
      sub_141EDF0(&v220, v9, v20, v13, v14);
      goto LABEL_27;
    case 4:
      sub_141F110(&v220);
LABEL_19:
      v19 = v222;
      v21 = _mm_loadu_si128(&v221);
      v22 = _mm_loadu_si128(&v220);
      v5 = v220.m128i_i64[0];
      v225 = v222;
      v18 = v21;
      v223 = v22;
      v224 = v21;
      break;
    case 5:
      sub_141F3C0(&v220);
LABEL_27:
      v30 = _mm_loadu_si128(&v221);
      v19 = v222;
      v5 = v220.m128i_i64[0];
      v223 = _mm_loadu_si128(&v220);
      v18 = v30;
      v225 = v222;
      v224 = v30;
      break;
    case 28:
      sub_141F0A0(&v220);
      v29 = _mm_loadu_si128(&v221);
      v19 = v222;
      v5 = v220.m128i_i64[0];
      v223 = _mm_loadu_si128(&v220);
      v18 = v29;
      v225 = v222;
      v224 = v29;
      break;
    default:
      break;
  }
  v223.m128i_i64[0] = v5;
  v214 = v19;
  v23 = *(_QWORD *)(a2 + 24);
  v212 = _mm_loadu_si128(&v223);
  v213 = v18;
  v24 = sub_1632FA0(*(_QWORD *)(v23 + 40));
  v25 = _mm_loadu_si128(&v213);
  v26 = *(_QWORD *)a2;
  v189 = v24;
  v27 = _mm_loadu_si128(&v216);
  v220.m128i_i64[0] = v5;
  v222 = v214;
  v220.m128i_i64[1] = -1;
  v223.m128i_i64[0] = v196;
  v223.m128i_i64[1] = -1;
  v225 = v217;
  v221 = v25;
  v224 = v27;
  if ( !(unsigned __int8)sub_134CB50(v26, (__int64)&v220, (__int64)&v223) )
  {
LABEL_13:
    *a1 = 0;
    return a1;
  }
  v28 = sub_14AD280(v5, v189, 6);
  v197 = sub_14AD280(v196, v189, 6);
  if ( v28 != v197 )
  {
    if ( !(unsigned __int8)sub_134E860(v28) || !(unsigned __int8)sub_134E860(v197) )
      goto LABEL_8;
    goto LABEL_13;
  }
  sub_13A68E0(a2, v7, v9);
  sub_13A61A0((__int64)&v215, v7, v9, v185, *(_DWORD *)(a2 + 32));
  v220.m128i_i64[0] = (__int64)&v221;
  v223 = 0u;
  v224.m128i_i32[0] = 0;
  v224.m128i_i64[1] = 1;
  v225 = 1;
  v226 = 1;
  v220.m128i_i64[1] = 0x200000000LL;
  sub_13B0BD0((__int64)&v220, 1u, (__int64)&v223);
  v35 = v226;
  if ( (v226 & 1) == 0 && v226 )
  {
    _libc_free(*(_QWORD *)v226);
    j_j___libc_free_0(v35, 24);
  }
  v36 = v225;
  if ( (v225 & 1) == 0 && v225 )
  {
    _libc_free(*(_QWORD *)v225);
    j_j___libc_free_0(v36, 24);
  }
  v37 = v224.m128i_i64[1];
  if ( (v224.m128i_i8[8] & 1) == 0 && v224.m128i_i64[1] )
  {
    _libc_free(*(_QWORD *)v224.m128i_i64[1]);
    j_j___libc_free_0(v37, 24);
  }
  v38 = sub_146F1B0(*(_QWORD *)(a2 + 8), v186);
  v39 = sub_146F1B0(*(_QWORD *)(a2 + 8), v183);
  v184 = 1;
  v41 = v39;
  *(_QWORD *)v220.m128i_i64[0] = v38;
  v42 = v220.m128i_i64[0];
  *(_QWORD *)(v220.m128i_i64[0] + 8) = v41;
  if ( byte_4F98DE0 )
  {
    if ( (unsigned __int8)sub_13AE9A0(a2, v7, v9, &v220) )
    {
      v184 = v220.m128i_u32[2];
      if ( !v220.m128i_i32[2] )
      {
        sub_13A4F10(&v200, 0, 0);
        sub_13A4F10((unsigned __int64 *)&v201, 0, 0);
        goto LABEL_78;
      }
    }
    v42 = v220.m128i_i64[0];
  }
  n = v9;
  v180 = v7;
  v43 = 0;
  while ( 1 )
  {
    v44 = 48 * v43;
    sub_13A5100(
      (unsigned __int64 *)(v42 + 48 * v43 + 24),
      *(_DWORD *)(v8 + 40) + 1,
      0,
      *(unsigned int *)(v8 + 40),
      v41,
      v40);
    sub_13A5100((unsigned __int64 *)(48 * v43 + v220.m128i_i64[0] + 32), *(_DWORD *)(v8 + 40) + 1, 0, v45, v46, v47);
    sub_13A5100((unsigned __int64 *)(48 * v43 + v220.m128i_i64[0] + 40), v184, 0, v48, v49, v50);
    sub_13A6E90(v8, (_QWORD *)(48 * v43 + v220.m128i_i64[0]));
    v187 = *(_QWORD *)(v8 + 16);
    v51 = 48 * v43 + v220.m128i_i64[0];
    v190 = (_QWORD *)sub_13AE450(v187, *(_QWORD *)(n + 40));
    v52 = (_QWORD *)sub_13AE450(v187, *(_QWORD *)(v180 + 40));
    *(_DWORD *)(v51 + 16) = sub_13A71E0(v8, *(_QWORD *)v51, v52, *(_QWORD *)(v51 + 8), v190, (__int64 *)(v51 + 24));
    sub_13A4D00((__int64 *)(v44 + v220.m128i_i64[0] + 32), (__int64 *)(v44 + v220.m128i_i64[0] + 24));
    v53 = v43++;
    sub_13A34B0((__int64 *)(v220.m128i_i64[0] + v44 + 40), v53);
    if ( v184 <= (unsigned int)v43 )
      break;
    v42 = v220.m128i_i64[0];
  }
  sub_13A4F10(&v200, v184, 0);
  sub_13A4F10((unsigned __int64 *)&v201, v184, 0);
  v170 = v180;
  v168 = n;
  v56 = 0;
  v171 = v8;
LABEL_52:
  v57 = v220.m128i_i64[0];
  v188 = v56 + 1;
  v58 = 48LL * v56;
  v59 = v220.m128i_i64[0] + v58;
  v60 = *(_DWORD *)(v220.m128i_i64[0] + v58 + 16);
  if ( v60 != 4 )
  {
    if ( !v60 )
      goto LABEL_49;
    if ( v184 <= v188 )
      goto LABEL_74;
    v172 = v56;
    v61 = 48LL * v188;
    v62 = v188 + (unsigned __int64)(v184 - 1 - v188);
    v63 = v177;
    v181 = v177;
    v191 = 48 * v62;
    while ( 1 )
    {
      v223.m128i_i64[0] = 1;
      v66 = *(_QWORD *)(v57 + v58 + 32);
      if ( (v66 & 1) != 0 )
      {
        v223.m128i_i64[0] = *(_QWORD *)(v57 + v58 + 32);
      }
      else
      {
        v74 = (_QWORD *)sub_22077B0(24);
        v75 = (__int64)v74;
        if ( v74 )
        {
          *v74 = 0;
          v74[1] = 0;
          v76 = *(_DWORD *)(v66 + 16);
          *(_DWORD *)(v75 + 16) = v76;
          if ( v76 )
          {
            v173 = (unsigned int)(v76 + 63) >> 6;
            v77 = malloc(8 * v173);
            v78 = 8 * v173;
            v79 = v173;
            v80 = (void *)v77;
            if ( !v77 )
            {
              if ( 8 * v173 || (v163 = malloc(1u), v79 = v173, v78 = 0, v80 = 0, !v163) )
              {
                v167 = v80;
                v174 = v78;
                na = v79;
                sub_16BD1C0("Allocation failed");
                v79 = na;
                v78 = v174;
                v80 = v167;
              }
              else
              {
                v80 = (void *)v163;
              }
            }
            *(_QWORD *)v75 = v80;
            v81 = *(const void **)v66;
            *(_QWORD *)(v75 + 8) = v79;
            memcpy(v80, v81, v78);
          }
        }
        v223.m128i_i64[0] = v75;
        v57 = v220.m128i_i64[0];
      }
      sub_13A5720((unsigned __int64 *)&v223, (unsigned __int64 *)(v57 + v61 + 32), v62, v63, v54, v55);
      if ( (v223.m128i_i8[0] & 1) != 0 )
      {
        v64 = (unsigned __int64)v223.m128i_i64[0] >> 58;
        v65 = ~(-1LL << ((unsigned __int64)v223.m128i_i64[0] >> 58));
        if ( (((unsigned __int64)v223.m128i_i64[0] >> 1) & v65) == 0 )
          goto LABEL_57;
      }
      else
      {
        v64 = *(unsigned int *)(v223.m128i_i64[0] + 16);
        if ( !((unsigned int)(v64 + 63) >> 6) )
          goto LABEL_57;
        v69 = *(_QWORD **)v223.m128i_i64[0];
        v65 = *(_QWORD *)v223.m128i_i64[0] + 8LL * (((unsigned int)(v64 + 63) >> 6) - 1) + 8;
        while ( !*v69 )
        {
          if ( (_QWORD *)v65 == ++v69 )
            goto LABEL_57;
        }
      }
      sub_13A5430(
        (unsigned __int64 *)(v220.m128i_i64[0] + v61 + 32),
        (unsigned __int64 *)(v220.m128i_i64[0] + v58 + 32),
        v65,
        v64,
        v67,
        v68);
      sub_13A5430(
        (unsigned __int64 *)(v220.m128i_i64[0] + v61 + 40),
        (unsigned __int64 *)(v220.m128i_i64[0] + v58 + 40),
        v70,
        v71,
        v72,
        v73);
      v181 = 0;
LABEL_57:
      sub_13A50C0((unsigned __int64 **)&v223);
      if ( v191 == v61 )
      {
        v56 = v172;
        if ( v181 )
        {
          v59 = v58 + v220.m128i_i64[0];
LABEL_74:
          if ( (unsigned int)sub_13A3510(*(_QWORD *)(v59 + 40)) != 1 )
          {
            sub_13A34B0((__int64 *)&v201, v56);
            goto LABEL_50;
          }
LABEL_49:
          sub_13A34B0((__int64 *)&v200, v56);
          goto LABEL_50;
        }
LABEL_51:
        v56 = v188;
        goto LABEL_52;
      }
      v57 = v220.m128i_i64[0];
      v61 += 48;
    }
  }
  v82 = (_QWORD **)sub_13AE450(*(_QWORD *)(v171 + 16), *(_QWORD *)(v170 + 40));
  sub_13A6C30(v171, *(_QWORD *)v59, v82, (unsigned __int64 *)(v59 + 24));
  v83 = v220.m128i_i64[0] + v58;
  v84 = (_QWORD **)sub_13AE450(*(_QWORD *)(v171 + 16), *(_QWORD *)(v168 + 40));
  sub_13A6C30(v171, *(_QWORD *)(v83 + 8), v84, (unsigned __int64 *)(v83 + 24));
  HIBYTE(v218) = 0;
LABEL_50:
  if ( v184 > v188 )
    goto LABEL_51;
  v8 = v171;
  v7 = v170;
  v9 = v168;
LABEL_78:
  sub_13A6380((__int64)v211, *(_QWORD *)(v8 + 8));
  v85 = sub_13A3700(v200);
  if ( v85 != -1 )
  {
    LODWORD(_RBX) = v85;
    do
    {
      while ( 1 )
      {
        v87 = v220.m128i_i64[0] + 48LL * (unsigned int)_RBX;
        v88 = *(_DWORD *)(v87 + 16);
        if ( v88 == 2 )
        {
          if ( (unsigned __int8)sub_13AC390(v8, *(_QWORD *)v87, *(_QWORD *)(v87 + 8), (__int64)&v215) )
          {
LABEL_122:
            *a1 = 0;
            goto LABEL_123;
          }
        }
        else if ( v88 > 2 )
        {
          if ( (unsigned __int8)sub_13AD4D0(
                                  v8,
                                  *(_QWORD *)v87,
                                  *(_QWORD *)(v87 + 8),
                                  (unsigned __int64 *)(v87 + 24),
                                  (__int64)&v215) )
            goto LABEL_122;
        }
        else if ( v88 )
        {
          v223.m128i_i64[0] = 0;
          if ( (unsigned __int8)sub_13AC120(
                                  v8,
                                  *(_QWORD *)v87,
                                  *(_QWORD *)(v87 + 8),
                                  v212.m128i_i32,
                                  (__int64)&v215,
                                  (__int64)v211,
                                  &v223) )
            goto LABEL_122;
        }
        else if ( (unsigned __int8)sub_13A8380(v8, *(_QWORD *)v87, *(_QWORD *)(v87 + 8), (__int64)&v215) )
        {
          goto LABEL_122;
        }
        v89 = (unsigned int)(_RBX + 1);
        if ( (v200 & 1) == 0 )
          break;
        _RDX = (-1LL << ((unsigned __int8)_RBX + 1)) & (v200 >> 1) & ~(-1LL << (v200 >> 58));
        if ( !_RDX || v200 >> 58 <= v89 )
          goto LABEL_100;
        __asm { tzcnt   rbx, rdx }
      }
      v91 = *(_DWORD *)(v200 + 16);
      if ( v91 == (_DWORD)v89 )
        break;
      v92 = (unsigned int)v89 >> 6;
      v93 = (unsigned int)(v91 - 1) >> 6;
      if ( (unsigned int)v89 >> 6 > v93 )
        break;
      v94 = *(_QWORD *)v200;
      v95 = 64 - (((_BYTE)_RBX + 1) & 0x3F);
      v96 = 0xFFFFFFFFFFFFFFFFLL >> v95;
      if ( v95 == 64 )
        v96 = 0;
      v97 = v92;
      v98 = ~v96;
      v99 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v91;
      while ( 1 )
      {
        _RAX = *(_QWORD *)(v94 + 8 * v97);
        if ( v92 == (_DWORD)v97 )
          _RAX = v98 & *(_QWORD *)(v94 + 8 * v97);
        if ( v93 == (_DWORD)v97 )
          _RAX &= v99;
        if ( _RAX )
          break;
        if ( v93 < (unsigned int)++v97 )
          goto LABEL_100;
      }
      __asm { tzcnt   rax, rax }
      LODWORD(_RBX) = ((_DWORD)v97 << 6) + _RAX;
    }
    while ( (_DWORD)_RBX != -1 );
  }
LABEL_100:
  if ( !(unsigned int)sub_13A3510((unsigned __int64)v201) )
    goto LABEL_158;
  v102 = *(_DWORD *)(v8 + 40);
  v223.m128i_i64[1] = 0x400000000LL;
  v103 = (unsigned int)(v102 + 1);
  v104 = &v224;
  v223.m128i_i64[0] = (__int64)&v224;
  if ( (unsigned int)v103 > 4 )
  {
    sub_16CD150(&v223, &v224, v103, 48);
    v104 = (__m128i *)v223.m128i_i64[0];
  }
  v223.m128i_i32[2] = v103;
  v105 = &v104[3 * v103];
  if ( v105 != v104 )
  {
    do
    {
      v104->m128i_i32[0] = 0;
      v104 += 3;
      v104[-3].m128i_i64[1] = 0;
      v104[-2].m128i_i64[0] = 0;
      v104[-2].m128i_i64[1] = 0;
      v104[-1].m128i_i64[0] = 0;
      v104[-1].m128i_i64[1] = 0;
    }
    while ( v105 != v104 );
    v104 = (__m128i *)v223.m128i_i64[0];
  }
  v106 = 0;
  while ( 1 )
  {
    v107 = v106++;
    sub_13A6380((__int64)v104[3 * v107].m128i_i64, *(_QWORD *)(v8 + 8));
    if ( *(_DWORD *)(v8 + 40) < v106 )
      break;
    v104 = (__m128i *)v223.m128i_i64[0];
  }
  v108 = sub_13A3700((unsigned __int64)v201);
  v208 = v108;
  v207 = &v201;
  if ( v108 == -1 )
    goto LABEL_156;
  v169 = v9;
  v109 = (const __m128i *)v211;
  v110 = &i;
  do
  {
    v202 = 1;
    v111 = *(unsigned __int64 **)(v220.m128i_i64[0] + 48LL * v108 + 40);
    v192 = v111;
    if ( ((unsigned __int8)v111 & 1) != 0 )
    {
      v202 = (unsigned __int64)v111;
    }
    else
    {
      v156 = (_QWORD *)sub_22077B0(24);
      v157 = v156;
      if ( v156 )
      {
        *v156 = 0;
        v156[1] = 0;
        v158 = *((_DWORD *)v192 + 4);
        *((_DWORD *)v157 + 4) = v158;
        if ( v158 )
        {
          v179 = (unsigned int)(v158 + 63) >> 6;
          v182 = 8 * v179;
          v159 = (void *)malloc(8 * v179);
          if ( !v159 )
          {
            if ( v182 || (v162 = malloc(1u), v159 = 0, !v162) )
            {
              v166 = v159;
              sub_16BD1C0("Allocation failed");
              v159 = v166;
            }
            else
            {
              v159 = (void *)v162;
            }
          }
          *v157 = v159;
          v157[1] = v179;
          memcpy(v159, (const void *)*v192, v182);
        }
      }
      v202 = (unsigned __int64)v157;
    }
    sub_13A4F10((unsigned __int64 *)&v203, v184, 0);
    sub_13A4F10((unsigned __int64 *)&v204, v184, 0);
    sub_13A4F10((unsigned __int64 *)&v205, *(_DWORD *)(v8 + 40) + 1, 0);
    v212.m128i_i64[0] = (__int64)&v213;
    v212.m128i_i64[1] = 0x400000000LL;
    v112 = sub_13A3700(v202);
    v210 = v112;
    i = (unsigned __int64 **)&v202;
    if ( v112 == -1 )
    {
      v160 = (__int64 **)&v213;
      v161 = 0;
    }
    else
    {
      do
      {
        if ( *(_DWORD *)(v220.m128i_i64[0] + 48LL * v112 + 16) == 1 )
          sub_13A34B0((__int64 *)&v203, v112);
        else
          sub_13A34B0((__int64 *)&v204, v112);
        v114 = v212.m128i_u32[2];
        v115 = v220.m128i_i64[0] + v113;
        if ( v212.m128i_i32[2] >= (unsigned __int32)v212.m128i_i32[3] )
        {
          v195 = v115;
          sub_16CD150(&v212, &v213, 0, 8);
          v114 = v212.m128i_u32[2];
          v115 = v195;
        }
        *(_QWORD *)(v212.m128i_i64[0] + 8 * v114) = v115;
        ++v212.m128i_i32[2];
        sub_13AE4C0((__int64)v110);
        v112 = v210;
      }
      while ( v210 != -1 );
      v160 = (__int64 **)v212.m128i_i64[0];
      v161 = v212.m128i_u32[2];
    }
    sub_13A6D30(v8, v160, v161);
LABEL_181:
    v135 = (unsigned __int64)v203;
    while ( (v135 & 1) != 0 )
    {
      if ( ((v135 >> 1) & ~(-1LL << (v135 >> 58))) == 0 )
        goto LABEL_217;
LABEL_184:
      v136 = sub_13A3700(v135);
      i = &v203;
      v210 = v136;
      v137 = v136;
      if ( v136 != -1 )
      {
        v193 = 0;
        v138 = v110;
        v139 = v137;
        v140 = (__int64)v138;
        while ( 1 )
        {
          while ( 1 )
          {
            v206 = 0;
            if ( (unsigned __int8)sub_13AC120(
                                    v8,
                                    *(_QWORD *)(v220.m128i_i64[0] + 48LL * v139),
                                    *(_QWORD *)(v220.m128i_i64[0] + 48LL * v139 + 8),
                                    (int *)&v199,
                                    (__int64)&v215,
                                    (__int64)v109,
                                    &v206) )
              goto LABEL_207;
            sub_13A34B0((__int64 *)&v205, v199);
            v141 = sub_13A7B70(v8, (__m128i *)(v223.m128i_i64[0] + 48LL * v199), v109);
            if ( !v141 )
              break;
            if ( !*(_DWORD *)(v223.m128i_i64[0] + 48LL * v199) )
              goto LABEL_207;
            v193 = v141;
            sub_13A35B0((__int64 *)&v203, v139);
            sub_13AE4C0(v140);
            v139 = v210;
            if ( v210 == -1 )
            {
              v110 = (_QWORD *)v140;
LABEL_193:
              v142 = sub_13A3700((unsigned __int64)v204);
              v210 = v142;
              i = &v204;
              if ( v142 != -1 )
              {
                v178 = v109;
                v143 = v142;
                v194 = (__int64)v110;
                do
                {
                  v144 = 48LL * v143;
                  if ( (unsigned __int8)sub_13AE060(
                                          v8,
                                          (__int64 *)(v144 + v220.m128i_i64[0]),
                                          (__int64 *)(v144 + v220.m128i_i64[0] + 8),
                                          (unsigned __int64 *)(v144 + v220.m128i_i64[0] + 24),
                                          &v223,
                                          (_BYTE *)&v218 + 3) )
                  {
                    v164 = *(_QWORD *)(v8 + 16);
                    v145 = v144 + v220.m128i_i64[0];
                    v165 = (_QWORD *)sub_13AE450(v164, *(_QWORD *)(v169 + 40));
                    v146 = (_QWORD *)sub_13AE450(v164, *(_QWORD *)(v7 + 40));
                    *(_DWORD *)(v145 + 16) = sub_13A71E0(
                                               v8,
                                               *(_QWORD *)v145,
                                               v146,
                                               *(_QWORD *)(v145 + 8),
                                               v165,
                                               (__int64 *)(v145 + 24));
                    v147 = v220.m128i_i64[0] + v144;
                    if ( *(_DWORD *)(v147 + 16) == 1 )
                    {
                      sub_13A34B0((__int64 *)&v203, v143);
                      sub_13A35B0((__int64 *)&v204, v143);
                    }
                    else if ( *(_DWORD *)(v147 + 16) <= 1u )
                    {
                      if ( (unsigned __int8)sub_13A8380(v8, *(_QWORD *)v147, *(_QWORD *)(v147 + 8), (__int64)&v215) )
                        goto LABEL_207;
                      sub_13A35B0((__int64 *)&v204, v143);
                    }
                  }
                  sub_13AE4C0(v194);
                  v143 = v210;
                }
                while ( v210 != -1 );
                v109 = v178;
                v110 = (_QWORD *)v194;
              }
              goto LABEL_181;
            }
          }
          sub_13A35B0((__int64 *)&v203, v139);
          sub_13AE4C0(v140);
          v139 = v210;
          if ( v210 == -1 )
          {
            v110 = (_QWORD *)v140;
            if ( !v193 )
              goto LABEL_181;
            goto LABEL_193;
          }
        }
      }
    }
    v150 = (unsigned int)(*(_DWORD *)(v135 + 16) + 63) >> 6;
    if ( v150 )
    {
      v151 = *(_QWORD **)v135;
      v152 = *(_QWORD *)v135 + 8LL * (v150 - 1) + 8;
      while ( !*v151 )
      {
        if ( ++v151 == (_QWORD *)v152 )
          goto LABEL_217;
      }
      goto LABEL_184;
    }
LABEL_217:
    v153 = sub_13A3700((unsigned __int64)v204);
    v210 = v153;
    i = &v204;
    if ( v153 != -1 )
    {
      do
      {
        v154 = v220.m128i_i64[0] + 48LL * v153;
        if ( *(_DWORD *)(v154 + 16) == 2 )
        {
          if ( (unsigned __int8)sub_13AC390(v8, *(_QWORD *)v154, *(_QWORD *)(v154 + 8), (__int64)&v215) )
            goto LABEL_207;
          sub_13A35B0((__int64 *)&v204, v153);
        }
        sub_13AE4C0((__int64)v110);
        v153 = v210;
      }
      while ( v210 != -1 );
      v148 = v204;
    }
    v149 = sub_13A3700((unsigned __int64)v148);
    v210 = v149;
    i = &v204;
    if ( v149 != -1 )
    {
      while ( !(unsigned __int8)sub_13AD4D0(
                                  v8,
                                  *(_QWORD *)(v220.m128i_i64[0] + 48LL * v149),
                                  *(_QWORD *)(v220.m128i_i64[0] + 48LL * v149 + 8),
                                  (unsigned __int64 *)(v220.m128i_i64[0] + 48LL * v149 + 24),
                                  (__int64)&v215) )
      {
        sub_13AE4C0((__int64)v110);
        v149 = v210;
        if ( v210 == -1 )
          goto LABEL_148;
      }
LABEL_207:
      *a1 = 0;
      if ( (__m128i *)v212.m128i_i64[0] != &v213 )
        _libc_free(v212.m128i_u64[0]);
      sub_13A50C0(&v205);
      sub_13A50C0(&v204);
      sub_13A50C0(&v203);
      sub_13A50C0((unsigned __int64 **)&v202);
      if ( (__m128i *)v223.m128i_i64[0] != &v224 )
        _libc_free(v223.m128i_u64[0]);
      goto LABEL_123;
    }
LABEL_148:
    v121 = sub_13A3700((unsigned __int64)v205);
    v210 = v121;
    for ( i = &v205; v210 != -1; v121 = v210 )
    {
      if ( *(_DWORD *)(v8 + 32) < v121 )
        break;
      v122 = 16LL * (v121 - 1);
      sub_13AE2E0(v8, v122 + v219, (_DWORD *)(v223.m128i_i64[0] + 48LL * v121));
      if ( (*(_BYTE *)(v219 + v122) & 7) == 0 )
        goto LABEL_207;
      sub_13AE4C0((__int64)v110);
    }
    if ( (__m128i *)v212.m128i_i64[0] != &v213 )
      _libc_free(v212.m128i_u64[0]);
    sub_13A50C0(&v205);
    sub_13A50C0(&v204);
    sub_13A50C0(&v203);
    sub_13A50C0((unsigned __int64 **)&v202);
    sub_13AE4C0((__int64)&v207);
    v108 = v208;
  }
  while ( v208 != -1 );
LABEL_156:
  if ( (__m128i *)v223.m128i_i64[0] != &v224 )
    _libc_free(v223.m128i_u64[0]);
LABEL_158:
  sub_13A4F10((unsigned __int64 *)&v223, *(_DWORD *)(v8 + 40) + 1, 0);
  v126 = v184;
  if ( v184 )
  {
    v127 = 0;
    do
    {
      v128 = v127 + v220.m128i_i64[0];
      v127 += 48;
      sub_13A5430((unsigned __int64 *)&v223, (unsigned __int64 *)(v128 + 24), v126, v123, v124, v125);
    }
    while ( v127 != 48LL * v184 );
  }
  if ( !*(_DWORD *)(v8 + 32) )
  {
    if ( v185 )
      goto LABEL_175;
LABEL_227:
    *a1 = 0;
    goto LABEL_178;
  }
  v129 = 1;
  do
  {
    if ( (v223.m128i_i8[0] & 1) != 0 )
      v130 = ((((unsigned __int64)v223.m128i_i64[0] >> 1) & ~(-1LL << ((unsigned __int64)v223.m128i_i64[0] >> 58))) >> v129)
           & 1;
    else
      v130 = (*(_QWORD *)(*(_QWORD *)v223.m128i_i64[0] + 8LL * (v129 >> 6)) >> v129) & 1LL;
    if ( (_BYTE)v130 )
      *(_BYTE *)(v219 + 16LL * (v129 - 1)) &= ~8u;
    v131 = *(_DWORD *)(v8 + 32);
    ++v129;
  }
  while ( v131 >= v129 );
  if ( v185 )
  {
    if ( v131 )
    {
      v132 = 1;
      while ( (sub_13A3110((__int64)&v215, v132) & 2) != 0 )
      {
        if ( *(_DWORD *)(v8 + 32) < (unsigned int)++v132 )
          goto LABEL_175;
      }
      BYTE2(v218) = 0;
    }
    goto LABEL_175;
  }
  if ( !v131 )
    goto LABEL_227;
  v155 = 1;
  while ( (unsigned int)sub_13A3110((__int64)&v215, v155) == 2 )
  {
    if ( *(_DWORD *)(v8 + 32) < (unsigned int)++v155 )
      goto LABEL_227;
  }
LABEL_175:
  v133 = (__m128i *)sub_22077B0(56);
  if ( v133 )
  {
    v133->m128i_i64[0] = (__int64)&unk_49E97C8;
    v133->m128i_i64[1] = v215.m128i_i64[1];
    v133[1] = v216;
    v133[2].m128i_i64[0] = v217;
    v133[2].m128i_i32[2] = v218;
    v134 = v219;
    v219 = 0;
    v133[3].m128i_i64[0] = v134;
  }
  *a1 = v133;
LABEL_178:
  sub_13A50C0((unsigned __int64 **)&v223);
LABEL_123:
  sub_13A50C0(&v201);
  sub_13A50C0((unsigned __int64 **)&v200);
  v116 = v220.m128i_i64[0];
  v117 = (__m128i *)(v220.m128i_i64[0] + 48LL * v220.m128i_u32[2]);
  if ( (__m128i *)v220.m128i_i64[0] != v117 )
  {
    do
    {
      v120 = (unsigned __int64 *)v117[-1].m128i_i64[1];
      v117 -= 3;
      if ( ((unsigned __int8)v120 & 1) == 0 && v120 )
      {
        _libc_free(*v120);
        j_j___libc_free_0(v120, 24);
      }
      v118 = (unsigned __int64 *)v117[2].m128i_i64[0];
      if ( ((unsigned __int8)v118 & 1) == 0 && v118 )
      {
        _libc_free(*v118);
        j_j___libc_free_0(v118, 24);
      }
      v119 = (unsigned __int64 *)v117[1].m128i_i64[1];
      if ( ((unsigned __int8)v119 & 1) == 0 && v119 )
      {
        _libc_free(*v119);
        j_j___libc_free_0(v119, 24);
      }
    }
    while ( (__m128i *)v116 != v117 );
    v117 = (__m128i *)v220.m128i_i64[0];
  }
  if ( v117 != &v221 )
    _libc_free((unsigned __int64)v117);
  v215.m128i_i64[0] = (__int64)&unk_49E97C8;
  if ( v219 )
    j_j___libc_free_0_0(v219);
  return a1;
}
