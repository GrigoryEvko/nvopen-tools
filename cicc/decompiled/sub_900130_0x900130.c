// Function: sub_900130
// Address: 0x900130
//
__int64 __fastcall sub_900130(
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
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned int v14; // r15d
  __int64 v15; // rax
  __int64 v16; // rax
  __m128i si128; // xmm0
  __int64 v18; // rdx
  __int64 v19; // rax
  void *v20; // r13
  size_t v21; // r15
  __int64 v22; // r14
  int *v23; // r12
  size_t v24; // rbx
  size_t v25; // rdx
  int v26; // eax
  __int64 v27; // rbx
  size_t v28; // r14
  int *v29; // rax
  __int64 v30; // r15
  size_t v31; // r12
  size_t v32; // rdx
  signed __int64 v33; // rax
  __int64 *v34; // r12
  size_t v35; // rbx
  size_t v36; // rdx
  int v37; // eax
  __int64 v38; // rbx
  size_t v39; // rbx
  size_t v40; // rdx
  int v41; // eax
  __int64 v42; // rbx
  void *v43; // r15
  size_t v44; // r14
  __int64 *v45; // r12
  size_t v46; // r13
  size_t v47; // rdx
  int v48; // eax
  __int64 v49; // r13
  size_t v50; // rbx
  size_t v51; // rdx
  int v52; // eax
  const char *v53; // rsi
  __int64 v54; // rdi
  const __m128i *v55; // rax
  __m128i *v56; // rdi
  _BYTE *v57; // r14
  size_t v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  size_t v62; // rdx
  size_t v63; // r13
  __int64 v64; // r12
  __int64 v65; // rsi
  unsigned __int128 v66; // kr00_16
  size_t v67; // rdx
  _BYTE *v68; // rdi
  __int64 v69; // rsi
  __int64 v70; // r8
  size_t v71; // rax
  signed __int64 v72; // rbx
  unsigned __int64 v73; // rax
  __int64 v74; // rdi
  __int64 v75; // rax
  _QWORD *v76; // r12
  size_t v77; // rax
  __int64 v78; // r12
  __int64 v79; // r13
  __int64 v80; // r14
  _QWORD *v81; // rbx
  __int64 v82; // rax
  signed __int64 v83; // rax
  __int64 v84; // rdi
  __int64 v85; // rax
  int v86; // edx
  __int64 v87; // rbx
  __int64 v88; // r13
  __int64 v89; // rax
  __int64 i; // r12
  __int64 v91; // rdi
  unsigned int v92; // r13d
  const __m128i *v93; // rbx
  __m128i *v94; // r12
  const __m128i *v95; // rbx
  __m128i *v96; // r12
  const __m128i *v98; // rax
  __m128i *v99; // rdi
  __int64 v100; // rax
  __int64 v101; // rdx
  __m128i v102; // rax
  size_t v103; // rsi
  __int64 v104; // rdx
  const __m128i *v105; // rax
  __m128i *v106; // rdi
  size_t v107; // rax
  size_t v108; // r12
  _QWORD *v109; // rdx
  char *v110; // rsi
  _BYTE *v111; // rdi
  size_t v112; // rdx
  __int64 v113; // r8
  char *v114; // rsi
  _BYTE *v115; // rdi
  size_t v116; // rdx
  __int64 v117; // r8
  _BYTE *v118; // rdi
  size_t v119; // rsi
  __int64 v120; // rdx
  __int64 v121; // r8
  size_t v122; // rdx
  _QWORD *v123; // rdi
  __int64 v124; // rcx
  const __m128i *v125; // rax
  __m128i *v126; // rdi
  const char *v127; // rsi
  __m128i *v128; // rbx
  char *v129; // rsi
  __int64 v130; // rax
  char v134; // [rsp+36h] [rbp-33Ah]
  char v135; // [rsp+37h] [rbp-339h]
  size_t v136; // [rsp+38h] [rbp-338h]
  unsigned int v139; // [rsp+68h] [rbp-308h]
  __int64 v141; // [rsp+88h] [rbp-2E8h]
  __int64 v142; // [rsp+98h] [rbp-2D8h]
  int v143; // [rsp+A8h] [rbp-2C8h] BYREF
  int v144; // [rsp+ACh] [rbp-2C4h] BYREF
  __m128i *v145; // [rsp+B0h] [rbp-2C0h] BYREF
  __m128i *v146; // [rsp+B8h] [rbp-2B8h]
  const __m128i *v147; // [rsp+C0h] [rbp-2B0h]
  __m128i *v148; // [rsp+D0h] [rbp-2A0h] BYREF
  __m128i *v149; // [rsp+D8h] [rbp-298h]
  const __m128i *v150; // [rsp+E0h] [rbp-290h]
  char *name; // [rsp+F0h] [rbp-280h] BYREF
  __int64 v152; // [rsp+F8h] [rbp-278h]
  _QWORD v153[2]; // [rsp+100h] [rbp-270h] BYREF
  void *v154; // [rsp+110h] [rbp-260h] BYREF
  size_t v155; // [rsp+118h] [rbp-258h]
  char v156[16]; // [rsp+120h] [rbp-250h] BYREF
  void *dest; // [rsp+130h] [rbp-240h] BYREF
  size_t v158; // [rsp+138h] [rbp-238h]
  _QWORD v159[2]; // [rsp+140h] [rbp-230h] BYREF
  void *s2; // [rsp+150h] [rbp-220h] BYREF
  size_t n; // [rsp+158h] [rbp-218h]
  _QWORD v162[2]; // [rsp+160h] [rbp-210h] BYREF
  _BYTE *v163; // [rsp+170h] [rbp-200h] BYREF
  __int64 v164; // [rsp+178h] [rbp-1F8h]
  _QWORD v165[2]; // [rsp+180h] [rbp-1F0h] BYREF
  size_t v166; // [rsp+190h] [rbp-1E0h] BYREF
  size_t v167; // [rsp+198h] [rbp-1D8h]
  _QWORD v168[2]; // [rsp+1A0h] [rbp-1D0h] BYREF
  __int16 v169; // [rsp+1B0h] [rbp-1C0h]
  unsigned __int128 v170; // [rsp+1C0h] [rbp-1B0h] BYREF
  _QWORD v171[2]; // [rsp+1D0h] [rbp-1A0h] BYREF
  __int64 v172; // [rsp+1E0h] [rbp-190h] BYREF
  __int64 v173; // [rsp+1E8h] [rbp-188h]
  __int64 v174; // [rsp+1F0h] [rbp-180h]
  __int64 v175; // [rsp+1F8h] [rbp-178h]
  __int64 v176; // [rsp+200h] [rbp-170h] BYREF
  char v177[8]; // [rsp+208h] [rbp-168h] BYREF
  int v178; // [rsp+210h] [rbp-160h]
  _QWORD *v179; // [rsp+218h] [rbp-158h] BYREF
  __int64 v180; // [rsp+220h] [rbp-150h] BYREF
  _QWORD v181[2]; // [rsp+228h] [rbp-148h] BYREF
  __int64 (__fastcall **v182)(); // [rsp+238h] [rbp-138h] BYREF
  __int64 v183[3]; // [rsp+240h] [rbp-130h] BYREF
  char v184; // [rsp+258h] [rbp-118h]
  __int64 v185[4]; // [rsp+260h] [rbp-110h] BYREF
  _BYTE v186[144]; // [rsp+280h] [rbp-F0h] BYREF
  __int64 v187; // [rsp+310h] [rbp-60h]
  __int16 v188; // [rsp+318h] [rbp-58h]
  __int64 v189; // [rsp+320h] [rbp-50h]
  __int64 v190; // [rsp+328h] [rbp-48h]
  __int64 v191; // [rsp+330h] [rbp-40h]
  __int64 v192; // [rsp+338h] [rbp-38h]

  v136 = 0;
  if ( a2 )
    v136 = strlen(a2);
  sub_8FE280();
  v145 = 0;
  name = (char *)v153;
  v154 = v156;
  dest = v159;
  strcpy((char *)v159, "compute_75");
  v146 = 0;
  v12 = *a4;
  v13 = a4[1];
  v147 = 0;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v14 = (v13 - v12) >> 5;
  v152 = 0;
  LOBYTE(v153[0]) = 0;
  strcpy(v156, "sm_75");
  v155 = 5;
  v158 = 10;
  v144 = 75;
  v143 = 0;
  if ( v14 )
  {
    v15 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v170 = v171;
      sub_8FC5C0(
        (__int64 *)&v170,
        *(_BYTE **)(v12 + 32 * v15),
        *(_QWORD *)(v12 + 32 * v15) + *(_QWORD *)(v12 + 32 * v15 + 8));
      if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "-arch", &name) )
        break;
      if ( (_QWORD *)v170 != v171 )
        j_j___libc_free_0(v170, v171[0] + 1LL);
      v15 = (unsigned int)(v143 + 1);
      v143 = v15;
      if ( v14 <= (unsigned int)v15 )
        goto LABEL_12;
      v12 = *a4;
    }
    sub_2240AE0(&dest, &name);
    sub_2240AE0(&v154, &dest);
    if ( (_QWORD *)v170 != v171 )
      j_j___libc_free_0(v170, v171[0] + 1LL);
  }
LABEL_12:
  if ( !(unsigned int)sub_2241B30(&v154, 0, 8, "compute_") )
  {
    if ( v155 <= 7 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
    v163 = v165;
    sub_8FC670((__int64 *)&v163, (_BYTE *)v154 + 8, (__int64)v154 + v155);
    sub_222DF20(&v182);
    v188 = 0;
    v187 = 0;
    v182 = off_4A06798;
    v189 = 0;
    v190 = 0;
    v191 = 0;
    *(_QWORD *)&v170 = qword_4A07108;
    v192 = 0;
    *(_QWORD *)((char *)&v171[-2] + qword_4A07108[-3]) = &unk_4A07130;
    *((_QWORD *)&v170 + 1) = 0;
    sub_222DD70((char *)&v171[-2] + *(_QWORD *)(v170 - 24), 0);
    v171[1] = 0;
    v172 = 0;
    v173 = 0;
    *(_QWORD *)&v170 = off_4A07178;
    v182 = off_4A071A0;
    v174 = 0;
    v171[0] = off_4A07480;
    v175 = 0;
    v176 = 0;
    sub_220A990(v177);
    v178 = 0;
    v171[0] = off_4A07080;
    v179 = v181;
    sub_8FC670((__int64 *)&v179, v163, (__int64)&v163[v164]);
    v178 = 8;
    sub_223FD50(v171, v179, 0, 0);
    sub_222DD70(&v182, v171);
    sub_222E620(&v170, &v144);
    v114 = "sm_";
    sub_8FD6D0((__int64)&v166, "sm_", &v163);
    v115 = v154;
    if ( (_QWORD *)v166 == v168 )
    {
      v116 = v167;
      if ( v167 )
      {
        if ( v167 == 1 )
        {
          *(_BYTE *)v154 = v168[0];
        }
        else
        {
          v114 = (char *)v168;
          memcpy(v154, v168, v167);
        }
        v116 = v167;
        v115 = v154;
      }
      v155 = v116;
      v115[v116] = 0;
      v115 = (_BYTE *)v166;
      goto LABEL_201;
    }
    v116 = v168[0];
    v114 = (char *)v167;
    if ( v154 == v156 )
    {
      v154 = (void *)v166;
      v155 = v167;
      *(_QWORD *)v156 = v168[0];
    }
    else
    {
      v117 = *(_QWORD *)v156;
      v154 = (void *)v166;
      v155 = v167;
      *(_QWORD *)v156 = v168[0];
      if ( v115 )
      {
        v166 = (size_t)v115;
        v168[0] = v117;
LABEL_201:
        v167 = 0;
        *v115 = 0;
        if ( (_QWORD *)v166 != v168 )
        {
          v114 = (char *)(v168[0] + 1LL);
          j_j___libc_free_0(v166, v168[0] + 1LL);
        }
        *(_QWORD *)&v170 = off_4A07178;
        v182 = off_4A071A0;
        v171[0] = off_4A07080;
        if ( v179 != v181 )
        {
          v114 = (char *)(v181[0] + 1LL);
          j_j___libc_free_0(v179, v181[0] + 1LL);
        }
        v171[0] = off_4A07480;
        sub_2209150(v177, v114, v116);
        *(_QWORD *)&v170 = qword_4A07108;
        *(_QWORD *)((char *)&v171[-2] + qword_4A07108[-3]) = &unk_4A07130;
        *((_QWORD *)&v170 + 1) = 0;
        v182 = off_4A06798;
        sub_222E050(&v182);
        if ( v163 != (_BYTE *)v165 )
          j_j___libc_free_0(v163, v165[0] + 1LL);
        goto LABEL_13;
      }
    }
    v166 = (size_t)v168;
    v115 = v168;
    goto LABEL_201;
  }
LABEL_13:
  if ( !(unsigned int)sub_2241B30(&dest, 0, 3, "sm_") )
  {
    if ( v158 <= 2 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
    v163 = v165;
    sub_8FC670((__int64 *)&v163, (_BYTE *)dest + 3, (__int64)dest + v158);
    sub_222DF20(&v182);
    v188 = 0;
    v187 = 0;
    v182 = off_4A06798;
    v189 = 0;
    v190 = 0;
    v191 = 0;
    *(_QWORD *)&v170 = qword_4A07108;
    v192 = 0;
    *(_QWORD *)((char *)&v171[-2] + qword_4A07108[-3]) = &unk_4A07130;
    *((_QWORD *)&v170 + 1) = 0;
    sub_222DD70((char *)&v171[-2] + *(_QWORD *)(v170 - 24), 0);
    v171[1] = 0;
    v172 = 0;
    v173 = 0;
    *(_QWORD *)&v170 = off_4A07178;
    v182 = off_4A071A0;
    v174 = 0;
    v171[0] = off_4A07480;
    v175 = 0;
    v176 = 0;
    sub_220A990(v177);
    v178 = 0;
    v171[0] = off_4A07080;
    v179 = v181;
    sub_8FC670((__int64 *)&v179, v163, (__int64)&v163[v164]);
    v178 = 8;
    sub_223FD50(v171, v179, 0, 0);
    sub_222DD70(&v182, v171);
    sub_222E620(&v170, &v144);
    v110 = "compute_";
    sub_8FD6D0((__int64)&v166, "compute_", &v163);
    v111 = dest;
    if ( (_QWORD *)v166 == v168 )
    {
      v112 = v167;
      if ( v167 )
      {
        if ( v167 == 1 )
        {
          *(_BYTE *)dest = v168[0];
        }
        else
        {
          v110 = (char *)v168;
          memcpy(dest, v168, v167);
        }
        v112 = v167;
        v111 = dest;
      }
      v158 = v112;
      v111[v112] = 0;
      v111 = (_BYTE *)v166;
      goto LABEL_190;
    }
    v110 = (char *)v167;
    v112 = v168[0];
    if ( dest == v159 )
    {
      dest = (void *)v166;
      v158 = v167;
      v159[0] = v168[0];
    }
    else
    {
      v113 = v159[0];
      dest = (void *)v166;
      v158 = v167;
      v159[0] = v168[0];
      if ( v111 )
      {
        v166 = (size_t)v111;
        v168[0] = v113;
LABEL_190:
        v167 = 0;
        *v111 = 0;
        if ( (_QWORD *)v166 != v168 )
        {
          v110 = (char *)(v168[0] + 1LL);
          j_j___libc_free_0(v166, v168[0] + 1LL);
        }
        *(_QWORD *)&v170 = off_4A07178;
        v182 = off_4A071A0;
        v171[0] = off_4A07080;
        if ( v179 != v181 )
        {
          v110 = (char *)(v181[0] + 1LL);
          j_j___libc_free_0(v179, v181[0] + 1LL);
        }
        v171[0] = off_4A07480;
        sub_2209150(v177, v110, v112);
        *(_QWORD *)&v170 = qword_4A07108;
        *(_QWORD *)((char *)&v171[-2] + qword_4A07108[-3]) = &unk_4A07130;
        *((_QWORD *)&v170 + 1) = 0;
        v182 = off_4A06798;
        sub_222E050(&v182);
        if ( v163 != (_BYTE *)v165 )
          j_j___libc_free_0(v163, v165[0] + 1LL);
        goto LABEL_14;
      }
    }
    v166 = (size_t)v168;
    v111 = v168;
    goto LABEL_190;
  }
LABEL_14:
  strcpy((char *)v171, "--emit-llvm-bc");
  *(_QWORD *)&v170 = v171;
  *((_QWORD *)&v170 + 1) = 14;
  sub_8FDA10(&v145, (__m128i *)&v170);
  if ( (_QWORD *)v170 != v171 )
    j_j___libc_free_0(v170, v171[0] + 1LL);
  v166 = 25;
  *(_QWORD *)&v170 = v171;
  v16 = sub_22409D0(&v170, &v166, 0);
  si128 = _mm_load_si128((const __m128i *)&xmmword_3C23BC0);
  *(_QWORD *)&v170 = v16;
  v171[0] = v166;
  *(_QWORD *)(v16 + 16) = 0x736574616C2D6D76LL;
  *(_BYTE *)(v16 + 24) = 116;
  *(__m128i *)v16 = si128;
  *((_QWORD *)&v170 + 1) = v166;
  *(_BYTE *)(v170 + v166) = 0;
  sub_8FDA10(&v148, (__m128i *)&v170);
  if ( (_QWORD *)v170 != v171 )
    j_j___libc_free_0(v170, v171[0] + 1LL);
  v143 = 0;
  *a8 = 0;
  v18 = *a4;
  v139 = (a4[1] - *a4) >> 5;
  if ( !v139 )
    goto LABEL_101;
  v135 = 0;
  v19 = 0;
  v134 = 0;
  while ( 2 )
  {
    s2 = v162;
    sub_8FC5C0(
      (__int64 *)&s2,
      *(_BYTE **)(v18 + 32 * v19),
      *(_QWORD *)(v18 + 32 * v19) + *(_QWORD *)(v18 + 32 * v19 + 8));
    if ( !qword_4F6D2B0 )
      goto LABEL_91;
    v141 = qword_4F6D2B0;
    v20 = s2;
    v21 = n;
    v22 = qword_4F6D2B0;
    v23 = &dword_4F6D2A8;
    do
    {
      while ( 1 )
      {
        v24 = *(_QWORD *)(v22 + 40);
        v25 = v21;
        if ( v24 <= v21 )
          v25 = *(_QWORD *)(v22 + 40);
        if ( v25 )
        {
          v26 = memcmp(*(const void **)(v22 + 32), v20, v25);
          if ( v26 )
            break;
        }
        v27 = v24 - v21;
        if ( v27 >= 0x80000000LL )
          goto LABEL_31;
        if ( v27 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v26 = v27;
          break;
        }
LABEL_22:
        v22 = *(_QWORD *)(v22 + 24);
        if ( !v22 )
          goto LABEL_32;
      }
      if ( v26 < 0 )
        goto LABEL_22;
LABEL_31:
      v23 = (int *)v22;
      v22 = *(_QWORD *)(v22 + 16);
    }
    while ( v22 );
LABEL_32:
    v28 = v21;
    v29 = v23;
    v30 = v141;
    if ( v23 == &dword_4F6D2A8 )
      goto LABEL_91;
    v31 = *((_QWORD *)v23 + 5);
    v32 = v28;
    if ( v31 <= v28 )
      v32 = v31;
    if ( v32 && (LODWORD(v33) = memcmp(v20, *((const void **)v29 + 4), v32), (_DWORD)v33) )
    {
LABEL_39:
      if ( (int)v33 < 0 )
        goto LABEL_91;
    }
    else
    {
      v33 = v28 - v31;
      if ( (__int64)(v28 - v31) <= 0x7FFFFFFF )
      {
        if ( v33 >= (__int64)0xFFFFFFFF80000000LL )
          goto LABEL_39;
LABEL_91:
        if ( !(unsigned int)sub_2241AC0(&s2, "-extra-device-vectorization") )
        {
          ++v143;
          goto LABEL_87;
        }
        v53 = "-maxreg=";
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "-maxreg", &name) )
        {
LABEL_96:
          sub_8FD6D0((__int64)&v170, v53, &name);
          sub_8FDA10(&v148, (__m128i *)&v170);
          v54 = v170;
          if ( (_QWORD *)v170 == v171 )
            goto LABEL_87;
          goto LABEL_97;
        }
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "-split-compile", &name) )
        {
          v53 = "-split-compile=";
          goto LABEL_96;
        }
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "--Xlgenfe", &name) )
        {
          v98 = v146;
          if ( v146 == v147 )
          {
            sub_8FD760(&v145, v146, (__int64)&name);
          }
          else
          {
            if ( v146 )
            {
              v99 = v146;
              v146->m128i_i64[0] = (__int64)v146[1].m128i_i64;
              sub_8FC5C0(v99->m128i_i64, name, (__int64)&name[v152]);
              v98 = v146;
            }
            v146 = (__m128i *)&v98[2];
          }
          goto LABEL_87;
        }
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "--Xlibnvvm", &name) )
          goto LABEL_172;
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "--Xlnk", &name) )
        {
          *(_QWORD *)&v170 = v171;
          qmemcpy(v171, "-Xlnk", 5);
          goto LABEL_170;
        }
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "--Xopt", &name) )
        {
          *(_QWORD *)&v170 = v171;
          qmemcpy(v171, "-Xopt", 5);
LABEL_170:
          *((_QWORD *)&v170 + 1) = 5;
          BYTE5(v171[0]) = 0;
          sub_8FDA10(&v148, (__m128i *)&v170);
          if ( (_QWORD *)v170 != v171 )
            j_j___libc_free_0(v170, v171[0] + 1LL);
LABEL_172:
          v105 = v149;
          if ( v149 == v150 )
          {
            sub_8FD760(&v148, v149, (__int64)&name);
          }
          else
          {
            if ( v149 )
            {
              v106 = v149;
              v149->m128i_i64[0] = (__int64)v149[1].m128i_i64;
              sub_8FC5C0(v106->m128i_i64, name, (__int64)&name[v152]);
              v105 = v149;
            }
            v149 = (__m128i *)&v105[2];
          }
          goto LABEL_87;
        }
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "--Xllc", &name) )
        {
          *(_QWORD *)&v170 = v171;
          qmemcpy(v171, "-Xllc", 5);
          goto LABEL_170;
        }
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "-covinfo", &name) )
        {
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-Xopt");
          *((_QWORD *)&v170 + 1) = 5;
          sub_8FDA10(&v148, (__m128i *)&v170);
          if ( (_QWORD *)v170 != v171 )
            j_j___libc_free_0(v170, v171[0] + 1LL);
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-coverage=true");
          *((_QWORD *)&v170 + 1) = 14;
          sub_8FDA10(&v148, (__m128i *)&v170);
          if ( (_QWORD *)v170 != v171 )
            j_j___libc_free_0(v170, v171[0] + 1LL);
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-Xopt");
          *((_QWORD *)&v170 + 1) = 5;
          sub_8FDA10(&v148, (__m128i *)&v170);
          if ( (_QWORD *)v170 != v171 )
            j_j___libc_free_0(v170, v171[0] + 1LL);
          v124 = 0x6F666E69766F632DLL;
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-covinfofile=");
          *((_QWORD *)&v170 + 1) = 13;
LABEL_261:
          sub_2241490(&v170, name, v152, v124);
          v125 = v149;
          if ( v149 == v150 )
          {
            sub_8FD760(&v148, v149, (__int64)&v170);
          }
          else
          {
            if ( v149 )
            {
              v126 = v149;
              v149->m128i_i64[0] = (__int64)v149[1].m128i_i64;
              sub_8FC5C0(v126->m128i_i64, (_BYTE *)v170, v170 + *((_QWORD *)&v170 + 1));
              v125 = v149;
            }
            v149 = (__m128i *)&v125[2];
          }
          v54 = v170;
          if ( (_QWORD *)v170 == v171 )
            goto LABEL_87;
LABEL_97:
          j_j___libc_free_0(v54, v171[0] + 1LL);
          goto LABEL_87;
        }
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "-profinfo", &name) )
        {
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-Xopt");
          *((_QWORD *)&v170 + 1) = 5;
          sub_8FDA10(&v148, (__m128i *)&v170);
          if ( (_QWORD *)v170 != v171 )
            j_j___libc_free_0(v170, v171[0] + 1LL);
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-profgen=true");
          *((_QWORD *)&v170 + 1) = 13;
          sub_8FDA10(&v148, (__m128i *)&v170);
          if ( (_QWORD *)v170 != v171 )
            j_j___libc_free_0(v170, v171[0] + 1LL);
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-Xopt");
          *((_QWORD *)&v170 + 1) = 5;
          sub_8FDA10(&v148, (__m128i *)&v170);
          if ( (_QWORD *)v170 != v171 )
            j_j___libc_free_0(v170, v171[0] + 1LL);
          v124 = 0x666E69666F72702DLL;
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-profinfofile=");
          *((_QWORD *)&v170 + 1) = 14;
          goto LABEL_261;
        }
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "-profile-instr-use", &name) )
        {
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-Xopt");
          *((_QWORD *)&v170 + 1) = 5;
          sub_8FDA10(&v148, (__m128i *)&v170);
          if ( (_QWORD *)v170 != v171 )
            j_j___libc_free_0(v170, v171[0] + 1LL);
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-profuse=true");
          *((_QWORD *)&v170 + 1) = 13;
          sub_8FDA10(&v148, (__m128i *)&v170);
          if ( (_QWORD *)v170 != v171 )
            j_j___libc_free_0(v170, v171[0] + 1LL);
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-Xopt");
          *((_QWORD *)&v170 + 1) = 5;
          sub_8FDA10(&v148, (__m128i *)&v170);
          if ( (_QWORD *)v170 != v171 )
            j_j___libc_free_0(v170, v171[0] + 1LL);
          v124 = 0x6C6966666F72702DLL;
          *(_QWORD *)&v170 = v171;
          strcpy((char *)v171, "-proffile=");
          *((_QWORD *)&v170 + 1) = 10;
          goto LABEL_261;
        }
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "-arch", &name) )
          goto LABEL_87;
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "--orig_src_file_name", &name) )
        {
          sub_8FCA80((__int64 *)&v170, "--orig_src_file_name");
          sub_8FDA10(&v145, (__m128i *)&v170);
          sub_2240A30(&v170);
          sub_8FD9B0(&v145, (__int64)&name);
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "--force-llp64") )
        {
          sub_8FD9B0(&v145, (__int64)&s2);
          byte_4F6D2DC = 1;
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "--partial-link") )
          goto LABEL_300;
        sub_8FC4D0((__int64 *)&v166, byte_3C24169, 4u);
        sub_8FCA80((__int64 *)&v163, "--");
        sub_8FD5D0((__m128i *)&v170, (__int64)&v163, &v166);
        if ( n == *((_QWORD *)&v170 + 1) && (!n || !memcmp(s2, (const void *)v170, n)) )
        {
          sub_2240A30(&v170);
          sub_2240A30(&v163);
          sub_2240A30(&v166);
          sub_8FD9B0(&v145, (__int64)&s2);
          sub_8FC4D0((__int64 *)&v166, byte_3C24164, 4u);
          sub_8FCA80((__int64 *)&v163, "-");
          sub_8FD5D0((__m128i *)&v170, (__int64)&v163, &v166);
          sub_8FDA10(&v148, (__m128i *)&v170);
          sub_2240A30(&v170);
          sub_2240A30(&v163);
          sub_2240A30(&v166);
          sub_8FCA80((__int64 *)&v170, "-Xopt");
          sub_8FDA10(&v148, (__m128i *)&v170);
          sub_2240A30(&v170);
          sub_8FCA80((__int64 *)&v170, "-memdep-cache-byval-loads=false");
          sub_8FDA10(&v148, (__m128i *)&v170);
          sub_2240A30(&v170);
          sub_8FCA80((__int64 *)&v170, "-Xllc");
          sub_8FDA10(&v148, (__m128i *)&v170);
          sub_2240A30(&v170);
          sub_8FCA80((__int64 *)&v170, "-memdep-cache-byval-loads=false");
          sub_8FDA10(&v148, (__m128i *)&v170);
          sub_2240A30(&v170);
          byte_4F6D2D0 = 1;
          goto LABEL_87;
        }
        sub_2240A30(&v170);
        sub_2240A30(&v163);
        sub_2240A30(&v166);
        if ( !(unsigned int)sub_2241AC0(&s2, "--tile-only") )
        {
          sub_8FD9B0(&v145, (__int64)&s2);
          sub_8FCA80((__int64 *)&v170, "--tile_bc_file_name");
          sub_8FDA10(&v145, (__m128i *)&v170);
          sub_2240A30(&v170);
          sub_8FCA80((__int64 *)&v170, a3);
          sub_8FDA10(&v145, (__m128i *)&v170);
          sub_2240A30(&v170);
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "-lto") )
        {
          sub_8FCA80((__int64 *)&v170, "-gen-lto");
          sub_8FDA10(&v148, (__m128i *)&v170);
          sub_2240A30(&v170);
          v134 = 1;
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "-olto") )
        {
          sub_8FCA80((__int64 *)&v170, "-gen-lto-and-llc");
          sub_8FDA10(&v148, (__m128i *)&v170);
          sub_2240A30(&v170);
          sub_8FD9B0(&v148, (__int64)&s2);
          sub_8FD9B0(&v148, *a4 + 32LL * (unsigned int)++v143);
          v135 = 1;
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "-w") )
        {
LABEL_300:
          sub_8FD9B0(&v148, (__int64)&s2);
          sub_8FD9B0(&v145, (__int64)&s2);
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "--promote_warnings") )
        {
          sub_8FCA80((__int64 *)&v170, "-Werror");
          sub_8FDA10(&v148, (__m128i *)&v170);
          sub_2240A30(&v170);
          sub_8FD9B0(&v145, (__int64)&s2);
          goto LABEL_87;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "-inline-info") )
        {
          v128 = (__m128i *)v186;
          sub_8FCA80((__int64 *)&v170, "-Xopt");
          sub_8FCA80(&v172, "-pass-remarks=inline");
          sub_8FCA80(&v176, "-Xopt");
          sub_8FCA80(&v180, "-pass-remarks-missed=inline");
          sub_8FCA80(v183, "-Xopt");
          sub_8FCA80(v185, "-pass-remarks-analysis=inline");
          sub_8FCB30(&v148, v149, (__int64)&v170, (__int64)v186);
          do
          {
            v128 -= 2;
            sub_2240A30(v128);
          }
          while ( v128 != (__m128i *)&v170 );
          goto LABEL_87;
        }
        v127 = "-jump-table-density=";
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "-jump-table-density", &name) )
        {
LABEL_315:
          sub_8FCA80((__int64 *)&v166, v127);
          sub_8FC460((__m128i *)&v170, (__int64)&v166, (__int64)name, v152);
          sub_8FDA10(&v148, (__m128i *)&v170);
          sub_2240A30(&v170);
          sub_2240A30(&v166);
          goto LABEL_87;
        }
        if ( (unsigned __int8)sub_8FD0D0(a4, &v143, "-opt-passes", &name) )
        {
          v127 = "-opt-passes=";
          goto LABEL_315;
        }
        if ( !(unsigned int)sub_2241AC0(&s2, "--device-time-trace") )
        {
          sub_8FD9B0(&v145, (__int64)&s2);
          sub_2240AE0(a10, *a4 + 32LL * (unsigned int)++v143);
          goto LABEL_87;
        }
        if ( (unsigned int)sub_2241AC0(&s2, "--use-trace-pid") )
        {
          if ( !(unsigned __int8)sub_8FD0D0(a4, &v143, "--trace-env", &name) )
          {
            if ( (unsigned int)sub_2241AC0(&s2, "-jobserver") )
            {
              sub_8FD9B0(&v145, (__int64)&s2);
            }
            else
            {
              sub_8FCA80((__int64 *)&v170, "-jobserver");
              sub_8FDA10(&v148, (__m128i *)&v170);
              sub_2240A30(&v170);
            }
            goto LABEL_87;
          }
          v129 = getenv(name);
          if ( v129 )
          {
            sub_8FCA80((__int64 *)&v166, v129);
            sub_22400C0(&v170, &v166, 8);
            sub_2240A30(&v166);
            sub_222E4D0(&v170, a12);
            if ( (v184 & 5) == 0 )
              goto LABEL_323;
            sub_8FC430(a7, "--trace-env");
            sub_223F4B0(&v170);
          }
          else
          {
            v130 = sub_223E4D0(&unk_4FD4BE0, "\n Could not find environment variable: ");
            sub_223E0D0(v130, name, v152);
            sub_8FC430(a7, "--trace-env");
          }
        }
        else
        {
          sub_22400C0(&v170, *a4 + 32LL * (unsigned int)++v143, 8);
          sub_222E4D0(&v170, a11);
          if ( (v184 & 5) == 0 )
          {
LABEL_323:
            sub_223F4B0(&v170);
            goto LABEL_87;
          }
          sub_8FC430(a7, "--use-trace-pid");
          sub_223F4B0(&v170);
        }
        v92 = 1;
        sub_2240A30(&s2);
        goto LABEL_136;
      }
    }
    v34 = (__int64 *)&dword_4F6D2A8;
    while ( 2 )
    {
      while ( 2 )
      {
        v35 = *(_QWORD *)(v30 + 40);
        v36 = v35;
        if ( v28 <= v35 )
          v36 = v28;
        if ( !v36 || (v37 = memcmp(*(const void **)(v30 + 32), v20, v36)) == 0 )
        {
          v38 = v35 - v28;
          if ( v38 >= 0x80000000LL )
            goto LABEL_50;
          if ( v38 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          {
            v37 = v38;
            break;
          }
LABEL_41:
          v30 = *(_QWORD *)(v30 + 24);
          if ( !v30 )
            goto LABEL_51;
          continue;
        }
        break;
      }
      if ( v37 < 0 )
        goto LABEL_41;
LABEL_50:
      v34 = (__int64 *)v30;
      v30 = *(_QWORD *)(v30 + 16);
      if ( v30 )
        continue;
      break;
    }
LABEL_51:
    if ( v34 == (__int64 *)&dword_4F6D2A8 )
      goto LABEL_60;
    v39 = v34[5];
    v40 = v39;
    if ( v28 <= v39 )
      v40 = v28;
    if ( v40 && (v41 = memcmp(v20, (const void *)v34[4], v40)) != 0 )
    {
LABEL_59:
      if ( v41 < 0 )
        goto LABEL_60;
    }
    else if ( (__int64)(v28 - v39) <= 0x7FFFFFFF )
    {
      if ( (__int64)(v28 - v39) >= (__int64)0xFFFFFFFF80000000LL )
      {
        v41 = v28 - v39;
        goto LABEL_59;
      }
LABEL_60:
      *(_QWORD *)&v170 = &s2;
      v34 = sub_8FDE80(&qword_4F6D2A0, v34, (__int64 *)&v170);
    }
    sub_904450(&v145, *(_QWORD *)v34[8]);
    v42 = qword_4F6D2B0;
    if ( !qword_4F6D2B0 )
    {
      v45 = (__int64 *)&dword_4F6D2A8;
      goto LABEL_82;
    }
    v43 = s2;
    v44 = n;
    v45 = (__int64 *)&dword_4F6D2A8;
    while ( 2 )
    {
      while ( 2 )
      {
        v46 = *(_QWORD *)(v42 + 40);
        v47 = v44;
        if ( v46 <= v44 )
          v47 = *(_QWORD *)(v42 + 40);
        if ( !v47 || (v48 = memcmp(*(const void **)(v42 + 32), v43, v47)) == 0 )
        {
          v49 = v46 - v44;
          if ( v49 >= 0x80000000LL )
            goto LABEL_72;
          if ( v49 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          {
            v48 = v49;
            break;
          }
LABEL_63:
          v42 = *(_QWORD *)(v42 + 24);
          if ( !v42 )
            goto LABEL_73;
          continue;
        }
        break;
      }
      if ( v48 < 0 )
        goto LABEL_63;
LABEL_72:
      v45 = (__int64 *)v42;
      v42 = *(_QWORD *)(v42 + 16);
      if ( v42 )
        continue;
      break;
    }
LABEL_73:
    if ( v45 == (__int64 *)&dword_4F6D2A8 )
      goto LABEL_82;
    v50 = v45[5];
    v51 = v44;
    if ( v50 <= v44 )
      v51 = v45[5];
    if ( v51 && (v52 = memcmp(v43, (const void *)v45[4], v51)) != 0 )
    {
LABEL_81:
      if ( v52 < 0 )
        goto LABEL_82;
    }
    else if ( (__int64)(v44 - v50) <= 0x7FFFFFFF )
    {
      if ( (__int64)(v44 - v50) >= (__int64)0xFFFFFFFF80000000LL )
      {
        v52 = v44 - v50;
        goto LABEL_81;
      }
LABEL_82:
      *(_QWORD *)&v170 = &s2;
      v45 = sub_8FDE80(&qword_4F6D2A0, v45, (__int64 *)&v170);
    }
    sub_904450(&v148, *(_QWORD *)(v45[8] + 8));
    if ( !(unsigned int)sub_2241AC0(&s2, "-m64") )
      *a8 = 1;
    if ( !(unsigned int)sub_2241AC0(&s2, "-discard-value-names") )
      *a9 = 1;
LABEL_87:
    if ( s2 != v162 )
      j_j___libc_free_0(s2, v162[0] + 1LL);
    v19 = (unsigned int)(v143 + 1);
    v143 = v19;
    if ( v139 > (unsigned int)v19 )
    {
      v18 = *a4;
      continue;
    }
    break;
  }
  if ( !v135 && v134 )
  {
    strcpy((char *)v171, "-olto");
    *(_QWORD *)&v170 = v171;
    *((_QWORD *)&v170 + 1) = 5;
    sub_8FDA10(&v148, (__m128i *)&v170);
    if ( (_QWORD *)v170 != v171 )
      j_j___libc_free_0(v170, v171[0] + 1LL);
    *(_QWORD *)&v170 = v171;
    if ( !a3 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v107 = strlen(a3);
    v166 = v107;
    v108 = v107;
    if ( v107 > 0xF )
    {
      *(_QWORD *)&v170 = sub_22409D0(&v170, &v166, 0);
      v123 = (_QWORD *)v170;
      v171[0] = v166;
    }
    else
    {
      if ( v107 == 1 )
      {
        LOBYTE(v171[0]) = *a3;
        v109 = v171;
        goto LABEL_183;
      }
      if ( !v107 )
      {
        v109 = v171;
        goto LABEL_183;
      }
      v123 = v171;
    }
    memcpy(v123, a3, v108);
    v107 = v166;
    v109 = (_QWORD *)v170;
LABEL_183:
    *((_QWORD *)&v170 + 1) = v107;
    *((_BYTE *)v109 + v107) = 0;
    sub_8FDA10(&v148, (__m128i *)&v170);
    if ( (_QWORD *)v170 != v171 )
      j_j___libc_free_0(v170, v171[0] + 1LL);
  }
LABEL_101:
  strcpy((char *)v171, "--nv_arch");
  *(_QWORD *)&v170 = v171;
  *((_QWORD *)&v170 + 1) = 9;
  sub_8FDA10(&v145, (__m128i *)&v170);
  if ( (_QWORD *)v170 != v171 )
    j_j___libc_free_0(v170, v171[0] + 1LL);
  v55 = v146;
  if ( v146 == v147 )
  {
    sub_8FD760(&v145, v146, (__int64)&dest);
  }
  else
  {
    if ( v146 )
    {
      v56 = v146;
      v146->m128i_i64[0] = (__int64)v146[1].m128i_i64;
      sub_8FC5C0(v56->m128i_i64, dest, (__int64)dest + v158);
      v55 = v146;
    }
    v146 = (__m128i *)&v55[2];
  }
  sub_8FD6D0((__int64)&v170, "-arch=", &dest);
  sub_8FDA10(&v148, (__m128i *)&v170);
  if ( (_QWORD *)v170 != v171 )
    j_j___libc_free_0(v170, v171[0] + 1LL);
  *(_BYTE *)(a6 + 72) = 1;
  v57 = (_BYTE *)(a6 + 120);
  v58 = strlen(a2);
  sub_2241130(a6 + 8, 0, *(_QWORD *)(a6 + 16), a2, v58);
  *(_BYTE *)(a6 + 170) = 1;
  if ( !a5 )
  {
    v59 = sub_C80C90(a2, v136, 0);
    v61 = sub_C80C90(v59, v60, 0);
    v63 = v62;
    v64 = v61;
    v169 = 261;
    v167 = v62;
    *(_QWORD *)&v170 = &v171[1];
    v166 = v61;
    *((_QWORD *)&v170 + 1) = 0;
    v171[0] = 256;
    sub_C85AC0(&v166, ".lgenfe.bc", 10, &v170, 0);
    sub_C8C750(v170, *((_QWORD *)&v170 + 1), 0);
    v65 = v170;
    v166 = (size_t)v168;
    sub_8FC670((__int64 *)&v166, (_BYTE *)v170, v170 + *((_QWORD *)&v170 + 1));
    v66 = __PAIR128__(v63, v64);
    if ( (_QWORD *)v170 != &v171[1] )
      _libc_free(v170, v65);
    v67 = v167;
    v68 = *(_BYTE **)(a6 + 104);
    if ( (_QWORD *)v166 != v168 )
    {
      v69 = v168[0];
      if ( v57 == v68 )
      {
        *(_QWORD *)(a6 + 104) = v166;
        *(_QWORD *)(a6 + 112) = v67;
        *(_QWORD *)(a6 + 120) = v69;
      }
      else
      {
        v70 = *(_QWORD *)(a6 + 120);
        *(_QWORD *)(a6 + 104) = v166;
        *(_QWORD *)(a6 + 112) = v67;
        *(_QWORD *)(a6 + 120) = v69;
        if ( v68 )
          goto LABEL_115;
      }
LABEL_165:
      v166 = (size_t)v168;
      v68 = v168;
      goto LABEL_116;
    }
    if ( v167 )
    {
LABEL_224:
      if ( v67 == 1 )
        *v68 = v168[0];
      else
        memcpy(v68, v168, v67);
      v67 = v167;
      v68 = *(_BYTE **)(a6 + 104);
    }
LABEL_227:
    *(_QWORD *)(a6 + 112) = v67;
    v68[v67] = 0;
    v68 = (_BYTE *)v166;
    goto LABEL_116;
  }
  v100 = sub_904090(a2, v136);
  v102.m128i_i64[0] = sub_904090(v100, v101);
  *(_BYTE *)(a6 + 74) = 1;
  LOWORD(v172) = 773;
  v66 = (unsigned __int128)v102;
  v170 = (unsigned __int128)v102;
  v171[0] = ".lgenfe.bc";
  sub_CA0F50(&v166, &v170);
  v68 = *(_BYTE **)(a6 + 104);
  if ( (_QWORD *)v166 == v168 )
  {
    v67 = v167;
    if ( v167 )
      goto LABEL_224;
    goto LABEL_227;
  }
  v103 = v167;
  v104 = v168[0];
  if ( v57 == v68 )
  {
    *(_QWORD *)(a6 + 104) = v166;
    *(_QWORD *)(a6 + 112) = v103;
    *(_QWORD *)(a6 + 120) = v104;
    goto LABEL_165;
  }
  v70 = *(_QWORD *)(a6 + 120);
  *(_QWORD *)(a6 + 104) = v166;
  *(_QWORD *)(a6 + 112) = v103;
  *(_QWORD *)(a6 + 120) = v104;
  if ( !v68 )
    goto LABEL_165;
LABEL_115:
  v166 = (size_t)v68;
  v168[0] = v70;
LABEL_116:
  v167 = 0;
  *v68 = 0;
  sub_2240AE0(a6 + 40, a6 + 104);
  if ( (_QWORD *)v166 != v168 )
    j_j___libc_free_0(v166, v168[0] + 1LL);
  if ( a3 )
  {
    v71 = strlen(a3);
    sub_2241130(a6 + 136, 0, *(_QWORD *)(a6 + 144), a3, v71);
    goto LABEL_120;
  }
  v171[0] = ".ptx";
  LOWORD(v172) = 773;
  v170 = v66;
  sub_CA0F50(&v166, &v170);
  v118 = *(_BYTE **)(a6 + 136);
  if ( (_QWORD *)v166 == v168 )
  {
    v122 = v167;
    if ( v167 )
    {
      if ( v167 == 1 )
        *v118 = v168[0];
      else
        memcpy(v118, v168, v167);
      v122 = v167;
      v118 = *(_BYTE **)(a6 + 136);
    }
    *(_QWORD *)(a6 + 144) = v122;
    v118[v122] = 0;
    v118 = (_BYTE *)v166;
    goto LABEL_211;
  }
  v119 = v167;
  v120 = v168[0];
  if ( v118 == (_BYTE *)(a6 + 152) )
  {
    *(_QWORD *)(a6 + 136) = v166;
    *(_QWORD *)(a6 + 144) = v119;
    *(_QWORD *)(a6 + 152) = v120;
  }
  else
  {
    v121 = *(_QWORD *)(a6 + 152);
    *(_QWORD *)(a6 + 136) = v166;
    *(_QWORD *)(a6 + 144) = v119;
    *(_QWORD *)(a6 + 152) = v120;
    if ( v118 )
    {
      v166 = (size_t)v118;
      v168[0] = v121;
      goto LABEL_211;
    }
  }
  v166 = (size_t)v168;
  v118 = v168;
LABEL_211:
  v167 = 0;
  *v118 = 0;
  if ( (_QWORD *)v166 != v168 )
    j_j___libc_free_0(v166, v168[0] + 1LL);
LABEL_120:
  v72 = ((char *)v146 - (char *)v145) >> 5;
  *(_DWORD *)(a6 + 76) = v72 + 1;
  v73 = (int)v72 + 1;
  v74 = 8 * v73;
  if ( v73 > 0xFFFFFFFFFFFFFFFLL )
    v74 = -1;
  v75 = sub_2207820(v74);
  *(_QWORD *)(a6 + 80) = v75;
  v76 = (_QWORD *)v75;
  v77 = strlen(a1);
  *v76 = sub_2207820(v77 + 1);
  strcpy(**(char ***)(a6 + 80), a1);
  if ( (int)v72 > 0 )
  {
    v78 = 8;
    v142 = 8LL * (unsigned int)(v72 - 1) + 16;
    do
    {
      v79 = 4 * v78 - 32;
      v80 = *(__int64 *)((char *)&v145->m128i_i64[1] + v79);
      v81 = (_QWORD *)(v78 + *(_QWORD *)(a6 + 80));
      *v81 = sub_2207820(v80 + 1);
      sub_2241570(&v145->m128i_i8[v79], *(_QWORD *)(*(_QWORD *)(a6 + 80) + v78), v80, 0);
      v82 = *(_QWORD *)(*(_QWORD *)(a6 + 80) + v78);
      v78 += 8;
      *(_BYTE *)(v82 + v80) = 0;
    }
    while ( v142 != v78 );
  }
  v83 = ((char *)v149 - (char *)v148) >> 5;
  *(_DWORD *)(a6 + 172) = v83;
  v84 = 8LL * (int)v83;
  if ( (unsigned __int64)(int)v83 > 0xFFFFFFFFFFFFFFFLL )
    v84 = -1;
  v85 = sub_2207820(v84);
  v86 = *(_DWORD *)(a6 + 172);
  *(_QWORD *)(a6 + 176) = v85;
  v87 = 0;
  if ( v86 > 0 )
  {
    while ( 1 )
    {
      v88 = v148[2 * v87].m128i_i64[1];
      *(_QWORD *)(v85 + 8 * v87) = sub_2207820(v88 + 1);
      sub_2241570(&v148[2 * v87], *(_QWORD *)(*(_QWORD *)(a6 + 176) + 8 * v87), v88, 0);
      v89 = *(_QWORD *)(*(_QWORD *)(a6 + 176) + 8 * v87++);
      *(_BYTE *)(v89 + v88) = 0;
      if ( *(_DWORD *)(a6 + 172) <= (int)v87 )
        break;
      v85 = *(_QWORD *)(a6 + 176);
    }
  }
  *(_DWORD *)a6 = v144;
  for ( i = qword_4F6D2B8; (int *)i != &dword_4F6D2A8; i = sub_220EEE0(i) )
  {
    v91 = *(_QWORD *)(i + 64);
    if ( v91 )
      j_j___libc_free_0(v91, 16);
  }
  v92 = 0;
LABEL_136:
  if ( dest != v159 )
    j_j___libc_free_0(dest, v159[0] + 1LL);
  if ( v154 != v156 )
    j_j___libc_free_0(v154, *(_QWORD *)v156 + 1LL);
  if ( name != (char *)v153 )
    j_j___libc_free_0(name, v153[0] + 1LL);
  v93 = v149;
  v94 = v148;
  if ( v149 != v148 )
  {
    do
    {
      if ( (__m128i *)v94->m128i_i64[0] != &v94[1] )
        j_j___libc_free_0(v94->m128i_i64[0], v94[1].m128i_i64[0] + 1);
      v94 += 2;
    }
    while ( v93 != v94 );
    v94 = v148;
  }
  if ( v94 )
    j_j___libc_free_0(v94, (char *)v150 - (char *)v94);
  v95 = v146;
  v96 = v145;
  if ( v146 != v145 )
  {
    do
    {
      if ( (__m128i *)v96->m128i_i64[0] != &v96[1] )
        j_j___libc_free_0(v96->m128i_i64[0], v96[1].m128i_i64[0] + 1);
      v96 += 2;
    }
    while ( v95 != v96 );
    v96 = v145;
  }
  if ( v96 )
    j_j___libc_free_0(v96, (char *)v147 - (char *)v96);
  return v92;
}
