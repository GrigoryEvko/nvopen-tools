// Function: sub_125FB30
// Address: 0x125fb30
//
__int64 __fastcall sub_125FB30(
        const char *a1,
        const char *a2,
        const char *a3,
        __int64 *a4,
        char a5,
        __int64 a6,
        __int64 a7,
        _BYTE *a8,
        _BYTE *a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  size_t v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // rax
  __m128i si128; // xmm0
  __int64 v19; // rdx
  __int64 v20; // rax
  void *v21; // r13
  size_t v22; // r15
  __int64 v23; // r14
  int *v24; // r12
  size_t v25; // rbx
  size_t v26; // rdx
  int v27; // eax
  __int64 v28; // rbx
  size_t v29; // r14
  int *v30; // rax
  __int64 v31; // r15
  size_t v32; // r12
  size_t v33; // rdx
  signed __int64 v34; // rax
  __int64 *v35; // r12
  size_t v36; // rbx
  size_t v37; // rdx
  int v38; // eax
  __int64 v39; // rbx
  size_t v40; // rbx
  size_t v41; // rdx
  int v42; // eax
  __int64 v43; // rbx
  void *v44; // r15
  size_t v45; // r14
  __int64 *v46; // r12
  size_t v47; // r13
  size_t v48; // rdx
  int v49; // eax
  __int64 v50; // r13
  size_t v51; // rbx
  size_t v52; // rdx
  int v53; // eax
  const char *v54; // rsi
  __int64 v55; // rdi
  const __m128i *v56; // rax
  __m128i *v57; // rdi
  size_t v58; // rax
  _BYTE *v59; // rbx
  size_t v60; // rdx
  size_t v61; // rdx
  __int64 v62; // rsi
  __int64 v63; // rax
  size_t v64; // rdx
  _BYTE *v65; // rdi
  __int64 v66; // rsi
  __int64 v67; // r8
  const char *v68; // rbx
  size_t v69; // rax
  signed __int64 v70; // rbx
  unsigned __int64 v71; // rax
  __int64 v72; // rdi
  __int64 v73; // rax
  _QWORD *v74; // r12
  size_t v75; // rax
  __int64 v76; // r12
  __int64 v77; // r13
  __int64 v78; // r14
  _QWORD *v79; // rbx
  __int64 v80; // rax
  signed __int64 v81; // rax
  __int64 v82; // rdi
  __int64 v83; // rax
  int v84; // edx
  __int64 v85; // rbx
  __int64 v86; // r13
  __int64 v87; // rax
  __int64 i; // r12
  __int64 v89; // rdi
  unsigned int v90; // r13d
  const __m128i *v91; // rbx
  __m128i *v92; // r12
  const __m128i *v93; // rbx
  __m128i *v94; // r12
  const __m128i *v96; // rax
  __m128i *v97; // rdi
  size_t v98; // rdx
  __int64 v99; // rax
  size_t v100; // rdx
  _BYTE *v101; // rdi
  __int64 v102; // rcx
  __int64 v103; // rdx
  __int64 v104; // rsi
  _BYTE *v105; // rdi
  __int64 v106; // rcx
  __int64 v107; // rdx
  __int64 v108; // rsi
  const __m128i *v109; // rax
  __m128i *v110; // rdi
  size_t v111; // rax
  size_t v112; // r12
  _QWORD *v113; // rdx
  char *v114; // rsi
  _BYTE *v115; // rdi
  size_t v116; // rdx
  __int64 v117; // r8
  char *v118; // rsi
  _BYTE *v119; // rdi
  size_t v120; // rdx
  __int64 v121; // r8
  __int64 v122; // rdx
  _QWORD *v123; // rdi
  __int64 v124; // rdx
  __int64 v125; // rcx
  const __m128i *v126; // rax
  __m128i *v127; // rdi
  const char *v128; // rsi
  __m128i *v129; // rbx
  char *v130; // rsi
  __int64 v131; // rax
  char v135; // [rsp+3Eh] [rbp-332h]
  char v136; // [rsp+3Fh] [rbp-331h]
  unsigned int v139; // [rsp+68h] [rbp-308h]
  __int64 v141; // [rsp+88h] [rbp-2E8h]
  __int64 v142; // [rsp+98h] [rbp-2D8h]
  int v143; // [rsp+A8h] [rbp-2C8h] BYREF
  int v144; // [rsp+ACh] [rbp-2C4h] BYREF
  _BYTE *v145; // [rsp+B0h] [rbp-2C0h] BYREF
  size_t v146; // [rsp+B8h] [rbp-2B8h]
  __m128i *v147; // [rsp+C0h] [rbp-2B0h] BYREF
  __m128i *v148; // [rsp+C8h] [rbp-2A8h]
  const __m128i *v149; // [rsp+D0h] [rbp-2A0h]
  __m128i *v150; // [rsp+E0h] [rbp-290h] BYREF
  __m128i *v151; // [rsp+E8h] [rbp-288h]
  const __m128i *v152; // [rsp+F0h] [rbp-280h]
  char *name; // [rsp+100h] [rbp-270h] BYREF
  __int64 v154; // [rsp+108h] [rbp-268h]
  _QWORD v155[2]; // [rsp+110h] [rbp-260h] BYREF
  void *dest; // [rsp+120h] [rbp-250h] BYREF
  size_t v157; // [rsp+128h] [rbp-248h]
  char v158[16]; // [rsp+130h] [rbp-240h] BYREF
  void *v159; // [rsp+140h] [rbp-230h] BYREF
  size_t v160; // [rsp+148h] [rbp-228h]
  _QWORD v161[2]; // [rsp+150h] [rbp-220h] BYREF
  void *s2; // [rsp+160h] [rbp-210h] BYREF
  size_t n; // [rsp+168h] [rbp-208h]
  _QWORD v164[2]; // [rsp+170h] [rbp-200h] BYREF
  _BYTE *v165; // [rsp+180h] [rbp-1F0h] BYREF
  size_t v166; // [rsp+188h] [rbp-1E8h]
  _QWORD v167[2]; // [rsp+190h] [rbp-1E0h] BYREF
  __int64 v168; // [rsp+1A0h] [rbp-1D0h] BYREF
  size_t v169; // [rsp+1A8h] [rbp-1C8h]
  _QWORD v170[2]; // [rsp+1B0h] [rbp-1C0h] BYREF
  __m128i v171; // [rsp+1C0h] [rbp-1B0h] BYREF
  _QWORD v172[2]; // [rsp+1D0h] [rbp-1A0h] BYREF
  __int64 v173; // [rsp+1E0h] [rbp-190h] BYREF
  __int64 v174; // [rsp+1E8h] [rbp-188h]
  __int64 v175; // [rsp+1F0h] [rbp-180h]
  __int64 v176; // [rsp+1F8h] [rbp-178h]
  __int64 v177; // [rsp+200h] [rbp-170h] BYREF
  char v178[8]; // [rsp+208h] [rbp-168h] BYREF
  int v179; // [rsp+210h] [rbp-160h]
  _QWORD *v180; // [rsp+218h] [rbp-158h] BYREF
  __int64 v181; // [rsp+220h] [rbp-150h] BYREF
  _QWORD v182[2]; // [rsp+228h] [rbp-148h] BYREF
  __int64 (__fastcall **v183)(); // [rsp+238h] [rbp-138h] BYREF
  __int64 v184[3]; // [rsp+240h] [rbp-130h] BYREF
  char v185; // [rsp+258h] [rbp-118h]
  __int64 v186[4]; // [rsp+260h] [rbp-110h] BYREF
  _BYTE v187[144]; // [rsp+280h] [rbp-F0h] BYREF
  __int64 v188; // [rsp+310h] [rbp-60h]
  __int16 v189; // [rsp+318h] [rbp-58h]
  __int64 v190; // [rsp+320h] [rbp-50h]
  __int64 v191; // [rsp+328h] [rbp-48h]
  __int64 v192; // [rsp+330h] [rbp-40h]
  __int64 v193; // [rsp+338h] [rbp-38h]

  v12 = 0;
  v145 = a2;
  if ( a2 )
    v12 = strlen(a2);
  v146 = v12;
  sub_125DC90();
  v147 = 0;
  name = (char *)v155;
  dest = v158;
  v159 = v161;
  strcpy((char *)v161, "compute_75");
  v148 = 0;
  v13 = *a4;
  v14 = a4[1];
  v149 = 0;
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v15 = (v14 - v13) >> 5;
  v154 = 0;
  LOBYTE(v155[0]) = 0;
  strcpy(v158, "sm_75");
  v157 = 5;
  v160 = 10;
  v144 = 75;
  v143 = 0;
  if ( v15 )
  {
    v16 = 0;
    while ( 1 )
    {
      v171.m128i_i64[0] = (__int64)v172;
      sub_125C500(
        v171.m128i_i64,
        *(_BYTE **)(v13 + 32 * v16),
        *(_QWORD *)(v13 + 32 * v16) + *(_QWORD *)(v13 + 32 * v16 + 8));
      if ( (unsigned __int8)sub_125D010(a4, &v143, "-arch", &name) )
        break;
      if ( (_QWORD *)v171.m128i_i64[0] != v172 )
        j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
      v16 = (unsigned int)(v143 + 1);
      v143 = v16;
      if ( v15 <= (unsigned int)v16 )
        goto LABEL_12;
      v13 = *a4;
    }
    sub_2240AE0(&v159, &name);
    sub_2240AE0(&dest, &v159);
    if ( (_QWORD *)v171.m128i_i64[0] != v172 )
      j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
  }
LABEL_12:
  if ( !(unsigned int)sub_2241B30(&dest, 0, 8, "compute_") )
  {
    if ( v157 <= 7 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
    v165 = v167;
    sub_125C5B0((__int64 *)&v165, (_BYTE *)dest + 8, (__int64)dest + v157);
    sub_222DF20(&v183);
    v189 = 0;
    v188 = 0;
    v183 = off_4A06798;
    v190 = 0;
    v191 = 0;
    v192 = 0;
    v171.m128i_i64[0] = (__int64)qword_4A07108;
    v193 = 0;
    *(__int64 *)((char *)v171.m128i_i64 + qword_4A07108[-3]) = (__int64)&unk_4A07130;
    v171.m128i_i64[1] = 0;
    sub_222DD70(&v171.m128i_i8[*(_QWORD *)(v171.m128i_i64[0] - 24)], 0);
    v172[1] = 0;
    v173 = 0;
    v174 = 0;
    v171.m128i_i64[0] = (__int64)off_4A07178;
    v183 = off_4A071A0;
    v175 = 0;
    v172[0] = off_4A07480;
    v176 = 0;
    v177 = 0;
    sub_220A990(v178);
    v179 = 0;
    v172[0] = off_4A07080;
    v180 = v182;
    sub_125C5B0((__int64 *)&v180, v165, (__int64)&v165[v166]);
    v179 = 8;
    sub_223FD50(v172, v180, 0, 0);
    sub_222DD70(&v183, v172);
    sub_222E620(&v171, &v144);
    v118 = "sm_";
    sub_8FD6D0((__int64)&v168, "sm_", &v165);
    v119 = dest;
    if ( (_QWORD *)v168 == v170 )
    {
      v120 = v169;
      if ( v169 )
      {
        if ( v169 == 1 )
        {
          *(_BYTE *)dest = v170[0];
        }
        else
        {
          v118 = (char *)v170;
          memcpy(dest, v170, v169);
        }
        v120 = v169;
        v119 = dest;
      }
      v157 = v120;
      v119[v120] = 0;
      v119 = (_BYTE *)v168;
      goto LABEL_207;
    }
    v120 = v170[0];
    v118 = (char *)v169;
    if ( dest == v158 )
    {
      dest = (void *)v168;
      v157 = v169;
      *(_QWORD *)v158 = v170[0];
    }
    else
    {
      v121 = *(_QWORD *)v158;
      dest = (void *)v168;
      v157 = v169;
      *(_QWORD *)v158 = v170[0];
      if ( v119 )
      {
        v168 = (__int64)v119;
        v170[0] = v121;
LABEL_207:
        v169 = 0;
        *v119 = 0;
        if ( (_QWORD *)v168 != v170 )
        {
          v118 = (char *)(v170[0] + 1LL);
          j_j___libc_free_0(v168, v170[0] + 1LL);
        }
        v171.m128i_i64[0] = (__int64)off_4A07178;
        v183 = off_4A071A0;
        v172[0] = off_4A07080;
        if ( v180 != v182 )
        {
          v118 = (char *)(v182[0] + 1LL);
          j_j___libc_free_0(v180, v182[0] + 1LL);
        }
        v172[0] = off_4A07480;
        sub_2209150(v178, v118, v120);
        v171.m128i_i64[0] = (__int64)qword_4A07108;
        *(__int64 *)((char *)v171.m128i_i64 + qword_4A07108[-3]) = (__int64)&unk_4A07130;
        v171.m128i_i64[1] = 0;
        v183 = off_4A06798;
        sub_222E050(&v183);
        if ( v165 != (_BYTE *)v167 )
          j_j___libc_free_0(v165, v167[0] + 1LL);
        goto LABEL_13;
      }
    }
    v168 = (__int64)v170;
    v119 = v170;
    goto LABEL_207;
  }
LABEL_13:
  if ( !(unsigned int)sub_2241B30(&v159, 0, 3, "sm_") )
  {
    if ( v160 <= 2 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
    v165 = v167;
    sub_125C5B0((__int64 *)&v165, (_BYTE *)v159 + 3, (__int64)v159 + v160);
    sub_222DF20(&v183);
    v189 = 0;
    v188 = 0;
    v183 = off_4A06798;
    v190 = 0;
    v191 = 0;
    v192 = 0;
    v171.m128i_i64[0] = (__int64)qword_4A07108;
    v193 = 0;
    *(__int64 *)((char *)v171.m128i_i64 + qword_4A07108[-3]) = (__int64)&unk_4A07130;
    v171.m128i_i64[1] = 0;
    sub_222DD70(&v171.m128i_i8[*(_QWORD *)(v171.m128i_i64[0] - 24)], 0);
    v172[1] = 0;
    v173 = 0;
    v174 = 0;
    v171.m128i_i64[0] = (__int64)off_4A07178;
    v183 = off_4A071A0;
    v175 = 0;
    v172[0] = off_4A07480;
    v176 = 0;
    v177 = 0;
    sub_220A990(v178);
    v179 = 0;
    v172[0] = off_4A07080;
    v180 = v182;
    sub_125C5B0((__int64 *)&v180, v165, (__int64)&v165[v166]);
    v179 = 8;
    sub_223FD50(v172, v180, 0, 0);
    sub_222DD70(&v183, v172);
    sub_222E620(&v171, &v144);
    v114 = "compute_";
    sub_8FD6D0((__int64)&v168, "compute_", &v165);
    v115 = v159;
    if ( (_QWORD *)v168 == v170 )
    {
      v116 = v169;
      if ( v169 )
      {
        if ( v169 == 1 )
        {
          *(_BYTE *)v159 = v170[0];
        }
        else
        {
          v114 = (char *)v170;
          memcpy(v159, v170, v169);
        }
        v116 = v169;
        v115 = v159;
      }
      v160 = v116;
      v115[v116] = 0;
      v115 = (_BYTE *)v168;
      goto LABEL_196;
    }
    v114 = (char *)v169;
    v116 = v170[0];
    if ( v159 == v161 )
    {
      v159 = (void *)v168;
      v160 = v169;
      v161[0] = v170[0];
    }
    else
    {
      v117 = v161[0];
      v159 = (void *)v168;
      v160 = v169;
      v161[0] = v170[0];
      if ( v115 )
      {
        v168 = (__int64)v115;
        v170[0] = v117;
LABEL_196:
        v169 = 0;
        *v115 = 0;
        if ( (_QWORD *)v168 != v170 )
        {
          v114 = (char *)(v170[0] + 1LL);
          j_j___libc_free_0(v168, v170[0] + 1LL);
        }
        v171.m128i_i64[0] = (__int64)off_4A07178;
        v183 = off_4A071A0;
        v172[0] = off_4A07080;
        if ( v180 != v182 )
        {
          v114 = (char *)(v182[0] + 1LL);
          j_j___libc_free_0(v180, v182[0] + 1LL);
        }
        v172[0] = off_4A07480;
        sub_2209150(v178, v114, v116);
        v171.m128i_i64[0] = (__int64)qword_4A07108;
        *(__int64 *)((char *)v171.m128i_i64 + qword_4A07108[-3]) = (__int64)&unk_4A07130;
        v171.m128i_i64[1] = 0;
        v183 = off_4A06798;
        sub_222E050(&v183);
        if ( v165 != (_BYTE *)v167 )
          j_j___libc_free_0(v165, v167[0] + 1LL);
        goto LABEL_14;
      }
    }
    v168 = (__int64)v170;
    v115 = v170;
    goto LABEL_196;
  }
LABEL_14:
  strcpy((char *)v172, "--emit-llvm-bc");
  v171.m128i_i64[0] = (__int64)v172;
  v171.m128i_i64[1] = 14;
  sub_8F9C20(&v147, &v171);
  if ( (_QWORD *)v171.m128i_i64[0] != v172 )
    j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
  v168 = 20;
  v171.m128i_i64[0] = (__int64)v172;
  v17 = sub_22409D0(&v171, &v168, 0);
  si128 = _mm_load_si128((const __m128i *)&xmmword_3C23BC0);
  v171.m128i_i64[0] = v17;
  v172[0] = v168;
  *(_DWORD *)(v17 + 16) = 808938870;
  *(__m128i *)v17 = si128;
  v171.m128i_i64[1] = v168;
  *(_BYTE *)(v171.m128i_i64[0] + v168) = 0;
  sub_8F9C20(&v150, &v171);
  if ( (_QWORD *)v171.m128i_i64[0] != v172 )
    j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
  v143 = 0;
  *a8 = 0;
  v19 = *a4;
  v139 = (a4[1] - *a4) >> 5;
  if ( !v139 )
    goto LABEL_101;
  v136 = 0;
  v20 = 0;
  v135 = 0;
  while ( 2 )
  {
    s2 = v164;
    sub_125C500(
      (__int64 *)&s2,
      *(_BYTE **)(v19 + 32 * v20),
      *(_QWORD *)(v19 + 32 * v20) + *(_QWORD *)(v19 + 32 * v20 + 8));
    if ( !qword_4F92C30 )
      goto LABEL_91;
    v141 = qword_4F92C30;
    v21 = s2;
    v22 = n;
    v23 = qword_4F92C30;
    v24 = &dword_4F92C28;
    do
    {
      while ( 1 )
      {
        v25 = *(_QWORD *)(v23 + 40);
        v26 = v22;
        if ( v25 <= v22 )
          v26 = *(_QWORD *)(v23 + 40);
        if ( v26 )
        {
          v27 = memcmp(*(const void **)(v23 + 32), v21, v26);
          if ( v27 )
            break;
        }
        v28 = v25 - v22;
        if ( v28 >= 0x80000000LL )
          goto LABEL_31;
        if ( v28 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v27 = v28;
          break;
        }
LABEL_22:
        v23 = *(_QWORD *)(v23 + 24);
        if ( !v23 )
          goto LABEL_32;
      }
      if ( v27 < 0 )
        goto LABEL_22;
LABEL_31:
      v24 = (int *)v23;
      v23 = *(_QWORD *)(v23 + 16);
    }
    while ( v23 );
LABEL_32:
    v29 = v22;
    v30 = v24;
    v31 = v141;
    if ( v24 == &dword_4F92C28 )
      goto LABEL_91;
    v32 = *((_QWORD *)v24 + 5);
    v33 = v29;
    if ( v32 <= v29 )
      v33 = v32;
    if ( v33 && (LODWORD(v34) = memcmp(v21, *((const void **)v30 + 4), v33), (_DWORD)v34) )
    {
LABEL_39:
      if ( (int)v34 < 0 )
        goto LABEL_91;
    }
    else
    {
      v34 = v29 - v32;
      if ( (__int64)(v29 - v32) <= 0x7FFFFFFF )
      {
        if ( v34 >= (__int64)0xFFFFFFFF80000000LL )
          goto LABEL_39;
LABEL_91:
        if ( !(unsigned int)sub_2241AC0(&s2, "-extra-device-vectorization") )
        {
          ++v143;
          goto LABEL_87;
        }
        v54 = "-maxreg=";
        if ( (unsigned __int8)sub_125D010(a4, &v143, "-maxreg", &name) )
        {
LABEL_96:
          sub_8FD6D0((__int64)&v171, v54, &name);
          sub_8F9C20(&v150, &v171);
          v55 = v171.m128i_i64[0];
          if ( (_QWORD *)v171.m128i_i64[0] == v172 )
            goto LABEL_87;
          goto LABEL_97;
        }
        if ( (unsigned __int8)sub_125D010(a4, &v143, "-split-compile", &name) )
        {
          v54 = "-split-compile=";
          goto LABEL_96;
        }
        if ( (unsigned __int8)sub_125D010(a4, &v143, "--Xlgenfe", &name) )
        {
          v96 = v148;
          if ( v148 == v149 )
          {
            sub_8FD760(&v147, v148, (__int64)&name);
          }
          else
          {
            if ( v148 )
            {
              v97 = v148;
              v148->m128i_i64[0] = (__int64)v148[1].m128i_i64;
              sub_125C500(v97->m128i_i64, name, (__int64)&name[v154]);
              v96 = v148;
            }
            v148 = (__m128i *)&v96[2];
          }
          goto LABEL_87;
        }
        if ( (unsigned __int8)sub_125D010(a4, &v143, "--Xlibnvvm", &name) )
          goto LABEL_180;
        if ( (unsigned __int8)sub_125D010(a4, &v143, "--Xlnk", &name) )
        {
          v171.m128i_i64[0] = (__int64)v172;
          qmemcpy(v172, "-Xlnk", 5);
          goto LABEL_178;
        }
        if ( (unsigned __int8)sub_125D010(a4, &v143, "--Xopt", &name) )
        {
          v171.m128i_i64[0] = (__int64)v172;
          qmemcpy(v172, "-Xopt", 5);
LABEL_178:
          v171.m128i_i64[1] = 5;
          BYTE5(v172[0]) = 0;
          sub_8F9C20(&v150, &v171);
          if ( (_QWORD *)v171.m128i_i64[0] != v172 )
            j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
LABEL_180:
          v109 = v151;
          if ( v151 == v152 )
          {
            sub_8FD760(&v150, v151, (__int64)&name);
          }
          else
          {
            if ( v151 )
            {
              v110 = v151;
              v151->m128i_i64[0] = (__int64)v151[1].m128i_i64;
              sub_125C500(v110->m128i_i64, name, (__int64)&name[v154]);
              v109 = v151;
            }
            v151 = (__m128i *)&v109[2];
          }
          goto LABEL_87;
        }
        if ( (unsigned __int8)sub_125D010(a4, &v143, "--Xllc", &name) )
        {
          v171.m128i_i64[0] = (__int64)v172;
          qmemcpy(v172, "-Xllc", 5);
          goto LABEL_178;
        }
        if ( (unsigned __int8)sub_125D010(a4, &v143, "-covinfo", &name) )
        {
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-Xopt");
          v171.m128i_i64[1] = 5;
          sub_8F9C20(&v150, &v171);
          if ( (_QWORD *)v171.m128i_i64[0] != v172 )
            j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-coverage=true");
          v171.m128i_i64[1] = 14;
          sub_8F9C20(&v150, &v171);
          if ( (_QWORD *)v171.m128i_i64[0] != v172 )
            j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-Xopt");
          v171.m128i_i64[1] = 5;
          sub_8F9C20(&v150, &v171);
          if ( (_QWORD *)v171.m128i_i64[0] != v172 )
            j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
          v125 = 0x6F666E69766F632DLL;
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-covinfofile=");
          v171.m128i_i64[1] = 13;
LABEL_267:
          sub_2241490(&v171, name, v154, v125);
          v126 = v151;
          if ( v151 == v152 )
          {
            sub_8FD760(&v150, v151, (__int64)&v171);
          }
          else
          {
            if ( v151 )
            {
              v127 = v151;
              v151->m128i_i64[0] = (__int64)v151[1].m128i_i64;
              sub_125C500(v127->m128i_i64, v171.m128i_i64[0], v171.m128i_i64[0] + v171.m128i_i64[1]);
              v126 = v151;
            }
            v151 = (__m128i *)&v126[2];
          }
          v55 = v171.m128i_i64[0];
          if ( (_QWORD *)v171.m128i_i64[0] == v172 )
            goto LABEL_87;
LABEL_97:
          j_j___libc_free_0(v55, v172[0] + 1LL);
          goto LABEL_87;
        }
        if ( (unsigned __int8)sub_125D010(a4, &v143, "-profinfo", &name) )
        {
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-Xopt");
          v171.m128i_i64[1] = 5;
          sub_8F9C20(&v150, &v171);
          if ( (_QWORD *)v171.m128i_i64[0] != v172 )
            j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-profgen=true");
          v171.m128i_i64[1] = 13;
          sub_8F9C20(&v150, &v171);
          if ( (_QWORD *)v171.m128i_i64[0] != v172 )
            j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-Xopt");
          v171.m128i_i64[1] = 5;
          sub_8F9C20(&v150, &v171);
          if ( (_QWORD *)v171.m128i_i64[0] != v172 )
            j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
          v125 = 0x666E69666F72702DLL;
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-profinfofile=");
          v171.m128i_i64[1] = 14;
          goto LABEL_267;
        }
        if ( (unsigned __int8)sub_125D010(a4, &v143, "-profile-instr-use", &name) )
        {
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-Xopt");
          v171.m128i_i64[1] = 5;
          sub_8F9C20(&v150, &v171);
          if ( (_QWORD *)v171.m128i_i64[0] != v172 )
            j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-profuse=true");
          v171.m128i_i64[1] = 13;
          sub_8F9C20(&v150, &v171);
          if ( (_QWORD *)v171.m128i_i64[0] != v172 )
            j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-Xopt");
          v171.m128i_i64[1] = 5;
          sub_8F9C20(&v150, &v171);
          if ( (_QWORD *)v171.m128i_i64[0] != v172 )
            j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
          v125 = 0x6C6966666F72702DLL;
          v171.m128i_i64[0] = (__int64)v172;
          strcpy((char *)v172, "-proffile=");
          v171.m128i_i64[1] = 10;
          goto LABEL_267;
        }
        if ( (unsigned __int8)sub_125D010(a4, &v143, "-arch", &name) )
          goto LABEL_87;
        if ( (unsigned __int8)sub_125D010(a4, &v143, "--orig_src_file_name", &name) )
        {
          sub_125C9C0(v171.m128i_i64, "--orig_src_file_name");
          sub_8F9C20(&v147, &v171);
          sub_2240A30(&v171);
          sub_8FD9B0(&v147, (__int64)&name);
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "--force-llp64") )
        {
          sub_8FD9B0(&v147, (__int64)&s2);
          byte_4F92C5C = 1;
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "--partial-link") )
          goto LABEL_310;
        sub_125C410(&v168, byte_3F9D97C, 4u);
        sub_125C9C0((__int64 *)&v165, "--");
        sub_8FD5D0(&v171, (__int64)&v165, &v168);
        if ( n == v171.m128i_i64[1] && (!n || !memcmp(s2, (const void *)v171.m128i_i64[0], n)) )
        {
          sub_2240A30(&v171);
          sub_2240A30(&v165);
          sub_2240A30(&v168);
          sub_8FD9B0(&v147, (__int64)&s2);
          sub_125C410(&v168, byte_3F9D977, 4u);
          sub_125C9C0((__int64 *)&v165, "-");
          sub_8FD5D0(&v171, (__int64)&v165, &v168);
          sub_8F9C20(&v150, &v171);
          sub_2240A30(&v171);
          sub_2240A30(&v165);
          sub_2240A30(&v168);
          sub_125C9C0(v171.m128i_i64, "-Xopt");
          sub_8F9C20(&v150, &v171);
          sub_2240A30(&v171);
          sub_125C9C0(v171.m128i_i64, "-memdep-cache-byval-loads=false");
          sub_8F9C20(&v150, &v171);
          sub_2240A30(&v171);
          sub_125C9C0(v171.m128i_i64, "-Xllc");
          sub_8F9C20(&v150, &v171);
          sub_2240A30(&v171);
          sub_125C9C0(v171.m128i_i64, "-memdep-cache-byval-loads=false");
          sub_8F9C20(&v150, &v171);
          sub_2240A30(&v171);
          byte_4F92C50 = 1;
          goto LABEL_87;
        }
        sub_2240A30(&v171);
        sub_2240A30(&v165);
        sub_2240A30(&v168);
        if ( !(unsigned int)sub_2241AC0(&s2, "--tile-only") )
        {
          sub_8FD9B0(&v147, (__int64)&s2);
          sub_125C9C0(v171.m128i_i64, "--tile_bc_file_name");
          sub_8F9C20(&v147, &v171);
          sub_2240A30(&v171);
          sub_125C9C0(v171.m128i_i64, a3);
          sub_8F9C20(&v147, &v171);
          sub_2240A30(&v171);
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "-lto") )
        {
          sub_125C9C0(v171.m128i_i64, "-gen-lto");
          sub_8F9C20(&v150, &v171);
          sub_2240A30(&v171);
          v135 = 1;
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "-olto") )
        {
          sub_125C9C0(v171.m128i_i64, "-gen-lto-and-llc");
          sub_8F9C20(&v150, &v171);
          sub_2240A30(&v171);
          sub_8FD9B0(&v150, (__int64)&s2);
          sub_8FD9B0(&v150, *a4 + 32LL * (unsigned int)++v143);
          v136 = 1;
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "-w") )
        {
LABEL_310:
          sub_8FD9B0(&v150, (__int64)&s2);
          sub_8FD9B0(&v147, (__int64)&s2);
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "--promote_warnings") )
        {
          sub_125C9C0(v171.m128i_i64, "-Werror");
          sub_8F9C20(&v150, &v171);
          sub_2240A30(&v171);
          sub_8FD9B0(&v147, (__int64)&s2);
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "-inline-info") )
        {
          v129 = (__m128i *)v187;
          sub_125C9C0(v171.m128i_i64, "-Xopt");
          sub_125C9C0(&v173, "-pass-remarks=inline");
          sub_125C9C0(&v177, "-Xopt");
          sub_125C9C0(&v181, "-pass-remarks-missed=inline");
          sub_125C9C0(v184, "-Xopt");
          sub_125C9C0(v186, "-pass-remarks-analysis=inline");
          sub_125CA70(&v150, v151, (__int64)&v171, (__int64)v187);
          do
          {
            v129 -= 2;
            sub_2240A30(v129);
          }
          while ( v129 != &v171 );
          goto LABEL_87;
        }
        v128 = "-jump-table-density=";
        if ( (unsigned __int8)sub_125D010(a4, &v143, "-jump-table-density", &name) )
        {
LABEL_325:
          sub_125C9C0(&v168, v128);
          sub_125C3A0(&v171, (__int64)&v168, (__int64)name, v154);
          sub_8F9C20(&v150, &v171);
          sub_2240A30(&v171);
          sub_2240A30(&v168);
          goto LABEL_87;
        }
        if ( (unsigned __int8)sub_125D010(a4, &v143, "-opt-passes", &name) )
        {
          v128 = "-opt-passes=";
          goto LABEL_325;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "--device-time-trace") )
        {
          sub_8FD9B0(&v147, (__int64)&s2);
          sub_2240AE0(a10, *a4 + 32LL * (unsigned int)++v143);
          goto LABEL_87;
        }
        if ( (unsigned int)sub_2241AC0(&s2, "--use-trace-pid") )
        {
          if ( !(unsigned __int8)sub_125D010(a4, &v143, "--trace-env", &name) )
          {
            if ( (unsigned int)sub_2241AC0(&s2, "-jobserver") )
            {
              sub_8FD9B0(&v147, (__int64)&s2);
            }
            else
            {
              sub_125C9C0(v171.m128i_i64, "-jobserver");
              sub_8F9C20(&v150, &v171);
              sub_2240A30(&v171);
            }
            goto LABEL_87;
          }
          v130 = getenv(name);
          if ( v130 )
          {
            sub_125C9C0(&v168, v130);
            sub_22400C0(&v171, &v168, 8);
            sub_2240A30(&v168);
            sub_222E4D0(&v171, a12);
            if ( (v185 & 5) == 0 )
              goto LABEL_333;
            sub_125C370(a7, "--trace-env");
            sub_223F4B0(&v171);
          }
          else
          {
            v131 = sub_223E4D0(qword_4FD4BE0, "\n Could not find environment variable: ");
            sub_223E0D0(v131, name, v154);
            sub_125C370(a7, "--trace-env");
          }
        }
        else
        {
          sub_22400C0(&v171, *a4 + 32LL * (unsigned int)++v143, 8);
          sub_222E4D0(&v171, a11);
          if ( (v185 & 5) == 0 )
          {
LABEL_333:
            sub_223F4B0(&v171);
            goto LABEL_87;
          }
          sub_125C370(a7, "--use-trace-pid");
          sub_223F4B0(&v171);
        }
        v90 = 1;
        sub_2240A30(&s2);
        goto LABEL_136;
      }
    }
    v35 = (__int64 *)&dword_4F92C28;
    while ( 2 )
    {
      while ( 2 )
      {
        v36 = *(_QWORD *)(v31 + 40);
        v37 = v36;
        if ( v29 <= v36 )
          v37 = v29;
        if ( !v37 || (v38 = memcmp(*(const void **)(v31 + 32), v21, v37)) == 0 )
        {
          v39 = v36 - v29;
          if ( v39 >= 0x80000000LL )
            goto LABEL_50;
          if ( v39 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          {
            v38 = v39;
            break;
          }
LABEL_41:
          v31 = *(_QWORD *)(v31 + 24);
          if ( !v31 )
            goto LABEL_51;
          continue;
        }
        break;
      }
      if ( v38 < 0 )
        goto LABEL_41;
LABEL_50:
      v35 = (__int64 *)v31;
      v31 = *(_QWORD *)(v31 + 16);
      if ( v31 )
        continue;
      break;
    }
LABEL_51:
    if ( v35 == (__int64 *)&dword_4F92C28 )
      goto LABEL_60;
    v40 = v35[5];
    v41 = v40;
    if ( v29 <= v40 )
      v41 = v29;
    if ( v41 && (v42 = memcmp(v21, (const void *)v35[4], v41)) != 0 )
    {
LABEL_59:
      if ( v42 < 0 )
        goto LABEL_60;
    }
    else if ( (__int64)(v29 - v40) <= 0x7FFFFFFF )
    {
      if ( (__int64)(v29 - v40) >= (__int64)0xFFFFFFFF80000000LL )
      {
        v42 = v29 - v40;
        goto LABEL_59;
      }
LABEL_60:
      v171.m128i_i64[0] = (__int64)&s2;
      v35 = sub_125D890(&qword_4F92C20, v35, v171.m128i_i64);
    }
    sub_1263F80(&v147, *(_QWORD *)v35[8]);
    v43 = qword_4F92C30;
    if ( !qword_4F92C30 )
    {
      v46 = (__int64 *)&dword_4F92C28;
      goto LABEL_82;
    }
    v44 = s2;
    v45 = n;
    v46 = (__int64 *)&dword_4F92C28;
    while ( 2 )
    {
      while ( 2 )
      {
        v47 = *(_QWORD *)(v43 + 40);
        v48 = v45;
        if ( v47 <= v45 )
          v48 = *(_QWORD *)(v43 + 40);
        if ( !v48 || (v49 = memcmp(*(const void **)(v43 + 32), v44, v48)) == 0 )
        {
          v50 = v47 - v45;
          if ( v50 >= 0x80000000LL )
            goto LABEL_72;
          if ( v50 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          {
            v49 = v50;
            break;
          }
LABEL_63:
          v43 = *(_QWORD *)(v43 + 24);
          if ( !v43 )
            goto LABEL_73;
          continue;
        }
        break;
      }
      if ( v49 < 0 )
        goto LABEL_63;
LABEL_72:
      v46 = (__int64 *)v43;
      v43 = *(_QWORD *)(v43 + 16);
      if ( v43 )
        continue;
      break;
    }
LABEL_73:
    if ( v46 == (__int64 *)&dword_4F92C28 )
      goto LABEL_82;
    v51 = v46[5];
    v52 = v45;
    if ( v51 <= v45 )
      v52 = v46[5];
    if ( v52 && (v53 = memcmp(v44, (const void *)v46[4], v52)) != 0 )
    {
LABEL_81:
      if ( v53 < 0 )
        goto LABEL_82;
    }
    else if ( (__int64)(v45 - v51) <= 0x7FFFFFFF )
    {
      if ( (__int64)(v45 - v51) >= (__int64)0xFFFFFFFF80000000LL )
      {
        v53 = v45 - v51;
        goto LABEL_81;
      }
LABEL_82:
      v171.m128i_i64[0] = (__int64)&s2;
      v46 = sub_125D890(&qword_4F92C20, v46, v171.m128i_i64);
    }
    sub_1263F80(&v150, *(_QWORD *)(v46[8] + 8));
    if ( !(unsigned int)sub_2241AC0(&s2, "-m64") )
      *a8 = 1;
    if ( !(unsigned int)sub_2241AC0(&s2, "-discard-value-names") )
      *a9 = 1;
LABEL_87:
    if ( s2 != v164 )
      j_j___libc_free_0(s2, v164[0] + 1LL);
    v20 = (unsigned int)(v143 + 1);
    v143 = v20;
    if ( v139 > (unsigned int)v20 )
    {
      v19 = *a4;
      continue;
    }
    break;
  }
  if ( !v136 && v135 )
  {
    strcpy((char *)v172, "-olto");
    v171.m128i_i64[0] = (__int64)v172;
    v171.m128i_i64[1] = 5;
    sub_8F9C20(&v150, &v171);
    if ( (_QWORD *)v171.m128i_i64[0] != v172 )
      j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
    v171.m128i_i64[0] = (__int64)v172;
    if ( !a3 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v111 = strlen(a3);
    v168 = v111;
    v112 = v111;
    if ( v111 > 0xF )
    {
      v171.m128i_i64[0] = sub_22409D0(&v171, &v168, 0);
      v123 = (_QWORD *)v171.m128i_i64[0];
      v172[0] = v168;
    }
    else
    {
      if ( v111 == 1 )
      {
        LOBYTE(v172[0]) = *a3;
        v113 = v172;
        goto LABEL_217;
      }
      if ( !v111 )
      {
        v113 = v172;
        goto LABEL_217;
      }
      v123 = v172;
    }
    memcpy(v123, a3, v112);
    v111 = v168;
    v113 = (_QWORD *)v171.m128i_i64[0];
LABEL_217:
    v171.m128i_i64[1] = v111;
    *((_BYTE *)v113 + v111) = 0;
    sub_8F9C20(&v150, &v171);
    if ( (_QWORD *)v171.m128i_i64[0] != v172 )
      j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
  }
LABEL_101:
  strcpy((char *)v172, "--nv_arch");
  v171.m128i_i64[0] = (__int64)v172;
  v171.m128i_i64[1] = 9;
  sub_8F9C20(&v147, &v171);
  if ( (_QWORD *)v171.m128i_i64[0] != v172 )
    j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
  v56 = v148;
  if ( v148 == v149 )
  {
    sub_8FD760(&v147, v148, (__int64)&v159);
  }
  else
  {
    if ( v148 )
    {
      v57 = v148;
      v148->m128i_i64[0] = (__int64)v148[1].m128i_i64;
      sub_125C500(v57->m128i_i64, v159, (__int64)v159 + v160);
      v56 = v148;
    }
    v148 = (__m128i *)&v56[2];
  }
  sub_8FD6D0((__int64)&v171, "-arch=", &v159);
  sub_8F9C20(&v150, &v171);
  if ( (_QWORD *)v171.m128i_i64[0] != v172 )
    j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
  *(_BYTE *)(a6 + 72) = 1;
  v58 = strlen(a2);
  v59 = (_BYTE *)(a6 + 120);
  sub_2241130(a6 + 8, 0, *(_QWORD *)(a6 + 16), a2, v58);
  *(_BYTE *)(a6 + 170) = 1;
  if ( a5 )
  {
    v145 = (_BYTE *)sub_1263BC0(v145, v146);
    v146 = v98;
    v99 = sub_1263BC0(v145, v98);
    *(_BYTE *)(a6 + 74) = 1;
    v145 = (_BYTE *)v99;
    LOWORD(v170[0]) = 773;
    v168 = (__int64)&v145;
    v169 = (size_t)".lgenfe.bc";
    v146 = v100;
    sub_16E2FC0(&v171, &v168);
    v101 = *(_BYTE **)(a6 + 104);
    if ( (_QWORD *)v171.m128i_i64[0] == v172 )
    {
      v122 = v171.m128i_i64[1];
      if ( v171.m128i_i64[1] )
      {
        if ( v171.m128i_i64[1] == 1 )
          *v101 = v172[0];
        else
          memcpy(v101, v172, v171.m128i_u64[1]);
        v122 = v171.m128i_i64[1];
        v101 = *(_BYTE **)(a6 + 104);
      }
      *(_QWORD *)(a6 + 112) = v122;
      v101[v122] = 0;
      v101 = (_BYTE *)v171.m128i_i64[0];
      goto LABEL_166;
    }
    v102 = v171.m128i_i64[1];
    v103 = v172[0];
    if ( v59 == v101 )
    {
      *(_QWORD *)(a6 + 104) = v171.m128i_i64[0];
      *(_QWORD *)(a6 + 112) = v102;
      *(_QWORD *)(a6 + 120) = v103;
    }
    else
    {
      v104 = *(_QWORD *)(a6 + 120);
      *(_QWORD *)(a6 + 104) = v171.m128i_i64[0];
      *(_QWORD *)(a6 + 112) = v102;
      *(_QWORD *)(a6 + 120) = v103;
      if ( v101 )
      {
        v171.m128i_i64[0] = (__int64)v101;
        v172[0] = v104;
LABEL_166:
        v171.m128i_i64[1] = 0;
        *v101 = 0;
        sub_2240AE0(a6 + 40, a6 + 104);
        if ( (_QWORD *)v171.m128i_i64[0] == v172 )
          goto LABEL_118;
        j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
        v68 = a3;
        if ( !a3 )
          goto LABEL_168;
LABEL_119:
        v69 = strlen(v68);
        sub_2241130(a6 + 136, 0, *(_QWORD *)(a6 + 144), v68, v69);
        goto LABEL_120;
      }
    }
    v171.m128i_i64[0] = (__int64)v172;
    v101 = v172;
    goto LABEL_166;
  }
  v145 = (_BYTE *)sub_16C40D0(v145, v146, 2);
  v146 = v60;
  v145 = (_BYTE *)sub_16C40D0(v145, v60, 2);
  v165 = v145;
  v146 = v61;
  v171.m128i_i64[0] = (__int64)v172;
  v166 = v61;
  v171.m128i_i64[1] = 0x10000000000LL;
  LOWORD(v170[0]) = 261;
  v168 = (__int64)&v165;
  sub_16C64C0(&v168, ".lgenfe.bc", 10, &v171);
  v62 = v171.m128i_u32[2];
  sub_16CC820(v171.m128i_i64[0], v171.m128i_u32[2], 0);
  v63 = v171.m128i_i64[0];
  v168 = (__int64)v170;
  if ( v171.m128i_i64[0] )
  {
    v62 = v171.m128i_i64[0];
    sub_125C5B0(&v168, v171.m128i_i64[0], v171.m128i_i64[0] + v171.m128i_u32[2]);
    v63 = v171.m128i_i64[0];
    if ( (_QWORD *)v171.m128i_i64[0] == v172 )
      goto LABEL_112;
  }
  else
  {
    v169 = 0;
    LOBYTE(v170[0]) = 0;
  }
  _libc_free(v63, v62);
LABEL_112:
  v64 = v169;
  v65 = *(_BYTE **)(a6 + 104);
  if ( (_QWORD *)v168 != v170 )
  {
    v66 = v170[0];
    if ( v59 == v65 )
    {
      *(_QWORD *)(a6 + 104) = v168;
      *(_QWORD *)(a6 + 112) = v64;
      *(_QWORD *)(a6 + 120) = v66;
    }
    else
    {
      v67 = *(_QWORD *)(a6 + 120);
      *(_QWORD *)(a6 + 104) = v168;
      *(_QWORD *)(a6 + 112) = v64;
      *(_QWORD *)(a6 + 120) = v66;
      if ( v65 )
      {
        v168 = (__int64)v65;
        v170[0] = v67;
        goto LABEL_116;
      }
    }
    v168 = (__int64)v170;
    v65 = v170;
    goto LABEL_116;
  }
  if ( v169 )
  {
    if ( v169 == 1 )
      *v65 = v170[0];
    else
      memcpy(v65, v170, v169);
    v64 = v169;
    v65 = *(_BYTE **)(a6 + 104);
  }
  *(_QWORD *)(a6 + 112) = v64;
  v65[v64] = 0;
  v65 = (_BYTE *)v168;
LABEL_116:
  v169 = 0;
  *v65 = 0;
  sub_2240AE0(a6 + 40, a6 + 104);
  if ( (_QWORD *)v168 != v170 )
    j_j___libc_free_0(v168, v170[0] + 1LL);
LABEL_118:
  v68 = a3;
  if ( a3 )
    goto LABEL_119;
LABEL_168:
  v168 = (__int64)&v145;
  v169 = (size_t)".ptx";
  LOWORD(v170[0]) = 773;
  sub_16E2FC0(&v171, &v168);
  v105 = *(_BYTE **)(a6 + 136);
  if ( (_QWORD *)v171.m128i_i64[0] != v172 )
  {
    v106 = v171.m128i_i64[1];
    v107 = v172[0];
    if ( v105 == (_BYTE *)(a6 + 152) )
    {
      *(_QWORD *)(a6 + 136) = v171.m128i_i64[0];
      *(_QWORD *)(a6 + 144) = v106;
      *(_QWORD *)(a6 + 152) = v107;
    }
    else
    {
      v108 = *(_QWORD *)(a6 + 152);
      *(_QWORD *)(a6 + 136) = v171.m128i_i64[0];
      *(_QWORD *)(a6 + 144) = v106;
      *(_QWORD *)(a6 + 152) = v107;
      if ( v105 )
      {
        v171.m128i_i64[0] = (__int64)v105;
        v172[0] = v108;
        goto LABEL_172;
      }
    }
    v171.m128i_i64[0] = (__int64)v172;
    v105 = v172;
    goto LABEL_172;
  }
  v124 = v171.m128i_i64[1];
  if ( v171.m128i_i64[1] )
  {
    if ( v171.m128i_i64[1] == 1 )
      *v105 = v172[0];
    else
      memcpy(v105, v172, v171.m128i_u64[1]);
    v124 = v171.m128i_i64[1];
    v105 = *(_BYTE **)(a6 + 136);
  }
  *(_QWORD *)(a6 + 144) = v124;
  v105[v124] = 0;
  v105 = (_BYTE *)v171.m128i_i64[0];
LABEL_172:
  v171.m128i_i64[1] = 0;
  *v105 = 0;
  if ( (_QWORD *)v171.m128i_i64[0] != v172 )
    j_j___libc_free_0(v171.m128i_i64[0], v172[0] + 1LL);
LABEL_120:
  v70 = ((char *)v148 - (char *)v147) >> 5;
  *(_DWORD *)(a6 + 76) = v70 + 1;
  v71 = (int)v70 + 1;
  v72 = 8 * v71;
  if ( v71 > 0xFFFFFFFFFFFFFFFLL )
    v72 = -1;
  v73 = sub_2207820(v72);
  *(_QWORD *)(a6 + 80) = v73;
  v74 = (_QWORD *)v73;
  v75 = strlen(a1);
  *v74 = sub_2207820(v75 + 1);
  strcpy(**(char ***)(a6 + 80), a1);
  if ( (int)v70 > 0 )
  {
    v76 = 8;
    v142 = 8LL * (unsigned int)(v70 - 1) + 16;
    do
    {
      v77 = 4 * v76 - 32;
      v78 = *(__int64 *)((char *)&v147->m128i_i64[1] + v77);
      v79 = (_QWORD *)(v76 + *(_QWORD *)(a6 + 80));
      *v79 = sub_2207820(v78 + 1);
      sub_2241570(&v147->m128i_i8[v77], *(_QWORD *)(*(_QWORD *)(a6 + 80) + v76), v78, 0);
      v80 = *(_QWORD *)(*(_QWORD *)(a6 + 80) + v76);
      v76 += 8;
      *(_BYTE *)(v80 + v78) = 0;
    }
    while ( v142 != v76 );
  }
  v81 = ((char *)v151 - (char *)v150) >> 5;
  *(_DWORD *)(a6 + 172) = v81;
  v82 = 8LL * (int)v81;
  if ( (unsigned __int64)(int)v81 > 0xFFFFFFFFFFFFFFFLL )
    v82 = -1;
  v83 = sub_2207820(v82);
  v84 = *(_DWORD *)(a6 + 172);
  *(_QWORD *)(a6 + 176) = v83;
  v85 = 0;
  if ( v84 > 0 )
  {
    while ( 1 )
    {
      v86 = v150[2 * v85].m128i_i64[1];
      *(_QWORD *)(v83 + 8 * v85) = sub_2207820(v86 + 1);
      sub_2241570(&v150[2 * v85], *(_QWORD *)(*(_QWORD *)(a6 + 176) + 8 * v85), v86, 0);
      v87 = *(_QWORD *)(*(_QWORD *)(a6 + 176) + 8 * v85++);
      *(_BYTE *)(v87 + v86) = 0;
      if ( *(_DWORD *)(a6 + 172) <= (int)v85 )
        break;
      v83 = *(_QWORD *)(a6 + 176);
    }
  }
  *(_DWORD *)a6 = v144;
  for ( i = qword_4F92C38; (int *)i != &dword_4F92C28; i = sub_220EEE0(i) )
  {
    v89 = *(_QWORD *)(i + 64);
    if ( v89 )
      j_j___libc_free_0(v89, 16);
  }
  v90 = 0;
LABEL_136:
  if ( v159 != v161 )
    j_j___libc_free_0(v159, v161[0] + 1LL);
  if ( dest != v158 )
    j_j___libc_free_0(dest, *(_QWORD *)v158 + 1LL);
  if ( name != (char *)v155 )
    j_j___libc_free_0(name, v155[0] + 1LL);
  v91 = v151;
  v92 = v150;
  if ( v151 != v150 )
  {
    do
    {
      if ( (__m128i *)v92->m128i_i64[0] != &v92[1] )
        j_j___libc_free_0(v92->m128i_i64[0], v92[1].m128i_i64[0] + 1);
      v92 += 2;
    }
    while ( v91 != v92 );
    v92 = v150;
  }
  if ( v92 )
    j_j___libc_free_0(v92, (char *)v152 - (char *)v92);
  v93 = v148;
  v94 = v147;
  if ( v148 != v147 )
  {
    do
    {
      if ( (__m128i *)v94->m128i_i64[0] != &v94[1] )
        j_j___libc_free_0(v94->m128i_i64[0], v94[1].m128i_i64[0] + 1);
      v94 += 2;
    }
    while ( v93 != v94 );
    v94 = v147;
  }
  if ( v94 )
    j_j___libc_free_0(v94, (char *)v149 - (char *)v94);
  return v90;
}
