// Function: sub_1221570
// Address: 0x1221570
//
__int64 __fastcall sub_1221570(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  unsigned __int64 v7; // rsi
  unsigned int v8; // eax
  unsigned int v10; // r13d
  char v12; // al
  __int64 *v13; // r13
  _QWORD *v14; // rcx
  _QWORD *v15; // rax
  int v16; // eax
  int v17; // eax
  int v18; // r14d
  unsigned __int64 v19; // rsi
  int v20; // r13d
  int v21; // eax
  int v22; // r13d
  unsigned __int64 p_src; // rsi
  const char *v24; // rax
  int v25; // r14d
  int v26; // eax
  __int64 v27; // rax
  char *v28; // rax
  int v29; // eax
  char *v30; // rsi
  signed __int64 v31; // rdx
  int v32; // r14d
  int v33; // eax
  __int64 v34; // rax
  __int64 v35; // rax
  int v36; // eax
  _QWORD *v37; // rax
  __int64 v38; // r8
  unsigned __int64 v39; // rax
  __int64 v40; // rdx
  unsigned int v41; // r15d
  __int64 v42; // r12
  size_t v43; // r15
  void *v44; // rdi
  __int64 v45; // r8
  __int64 v46; // rax
  __int64 v47; // r8
  __m128i *v48; // rax
  _DWORD *v49; // r13
  int v50; // eax
  _QWORD *v51; // rdi
  __int64 *v52; // r14
  __int64 v53; // rax
  void *v54; // r12
  _DWORD *v55; // r13
  __int32 v56; // esi
  __int64 v57; // r9
  __int64 v58; // rax
  _QWORD *v59; // rax
  _QWORD *v60; // r10
  __int64 v61; // rax
  void *v62; // rbx
  void *v63; // rax
  __int64 v64; // r15
  size_t v65; // r12
  void *v66; // rax
  void *v67; // rcx
  __int64 v68; // r8
  void *v69; // rdi
  int v70; // eax
  void **v71; // rbx
  void **v72; // r12
  void **v73; // r12
  int v74; // eax
  char v75; // al
  const char *v76; // rax
  _QWORD *v77; // rax
  _QWORD *v78; // r14
  _QWORD *v79; // r13
  __int64 v80; // r12
  __int64 *v81; // r14
  _QWORD *v82; // rbx
  int v83; // edx
  _QWORD *v84; // r14
  int v85; // r9d
  unsigned int v86; // r8d
  unsigned int *v87; // rdx
  __int64 v88; // rsi
  __int64 v89; // rdi
  __int64 **v90; // rax
  __int64 v91; // rdx
  __int64 **v92; // rdi
  __int64 v93; // r15
  __int64 i; // rax
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // rax
  __int64 *v105; // rax
  unsigned int v106; // edx
  char *v107; // rsi
  __int64 v108; // rcx
  __int64 v109; // rdx
  __int64 v110; // rax
  __int64 v111; // rax
  _QWORD *v112; // rax
  _QWORD *v113; // r12
  __int64 v114; // r14
  __int64 v115; // rax
  char v116; // cl
  __m128i *v117; // rdi
  __int64 j; // rcx
  int v119; // ecx
  __int64 v120; // rax
  char *v121; // r8
  char *k; // rdx
  __int64 v123; // rsi
  int v124; // r9d
  int v125; // esi
  __int64 *v126; // rdi
  __int64 v127; // r15
  int v128; // edx
  __int64 *v129; // r12
  __int64 v130; // rax
  __int64 v131; // r12
  bool v132; // zf
  const char *v133; // rax
  __int32 v134; // eax
  __int32 v135; // eax
  __int64 v136; // rdx
  __int64 *v137; // rdx
  int v138; // ecx
  int v139; // ecx
  unsigned int m; // eax
  __int64 v141; // rdx
  _BYTE *v142; // r8
  unsigned int v143; // eax
  __int64 *v144; // rdi
  const char *v145; // rax
  __int64 v146; // rax
  __int64 *v147; // rdi
  unsigned __int64 v148; // rsi
  __int64 v149; // rax
  __int64 v150; // rdx
  __int32 v151; // r14d
  __int64 v152; // rax
  _QWORD *v153; // r12
  _QWORD *v154; // rax
  _QWORD *v155; // r14
  _BYTE *v156; // rax
  __int64 v157; // r8
  __int64 v158; // rax
  const char *v159; // rax
  __int64 v160; // rax
  __int64 v161; // rcx
  unsigned int v162; // r8d
  bool v163; // al
  __int32 v164; // eax
  unsigned int v165; // eax
  __int64 **v166; // rax
  __int64 v167; // rax
  __int64 v168; // rax
  __int32 v169; // eax
  unsigned int v170; // eax
  __int32 v171; // eax
  unsigned __int32 v172; // eax
  __int64 v174; // [rsp+8h] [rbp-308h]
  unsigned int v175; // [rsp+20h] [rbp-2F0h]
  __int64 v176; // [rsp+28h] [rbp-2E8h]
  char v177; // [rsp+38h] [rbp-2D8h]
  unsigned __int8 v178; // [rsp+38h] [rbp-2D8h]
  unsigned __int8 *v179; // [rsp+38h] [rbp-2D8h]
  __int32 v180; // [rsp+38h] [rbp-2D8h]
  int v181; // [rsp+40h] [rbp-2D0h]
  bool v182; // [rsp+40h] [rbp-2D0h]
  unsigned __int8 v183; // [rsp+40h] [rbp-2D0h]
  bool v184; // [rsp+40h] [rbp-2D0h]
  _QWORD *v185; // [rsp+40h] [rbp-2D0h]
  _QWORD *v186; // [rsp+40h] [rbp-2D0h]
  __int64 *v187; // [rsp+40h] [rbp-2D0h]
  __int64 *v188; // [rsp+48h] [rbp-2C8h]
  unsigned __int8 v189; // [rsp+48h] [rbp-2C8h]
  bool v190; // [rsp+48h] [rbp-2C8h]
  unsigned __int8 v191; // [rsp+48h] [rbp-2C8h]
  unsigned __int64 v192; // [rsp+48h] [rbp-2C8h]
  unsigned __int64 v193; // [rsp+48h] [rbp-2C8h]
  _QWORD *v194; // [rsp+48h] [rbp-2C8h]
  _BYTE *v195; // [rsp+48h] [rbp-2C8h]
  _QWORD *v196; // [rsp+48h] [rbp-2C8h]
  __int64 v197; // [rsp+48h] [rbp-2C8h]
  __int64 *v198; // [rsp+58h] [rbp-2B8h] BYREF
  __int64 **v199; // [rsp+60h] [rbp-2B0h] BYREF
  unsigned int v200; // [rsp+68h] [rbp-2A8h]
  __int8 v201; // [rsp+6Ch] [rbp-2A4h]
  __m128i v202[2]; // [rsp+80h] [rbp-290h] BYREF
  __int16 v203; // [rsp+A0h] [rbp-270h]
  __m128i v204[2]; // [rsp+B0h] [rbp-260h] BYREF
  char v205; // [rsp+D0h] [rbp-240h]
  char v206; // [rsp+D1h] [rbp-23Fh]
  __m128i v207[3]; // [rsp+E0h] [rbp-230h] BYREF
  __m128i v208; // [rsp+110h] [rbp-200h] BYREF
  __m128i *v209; // [rsp+120h] [rbp-1F0h] BYREF
  unsigned int v210; // [rsp+128h] [rbp-1E8h]
  char v211; // [rsp+130h] [rbp-1E0h]
  char v212; // [rsp+131h] [rbp-1DFh]
  __m128i v213; // [rsp+140h] [rbp-1D0h] BYREF
  __m128i *v214; // [rsp+150h] [rbp-1C0h] BYREF
  unsigned int v215; // [rsp+158h] [rbp-1B8h]
  __int16 v216; // [rsp+160h] [rbp-1B0h]
  __m128i v217; // [rsp+170h] [rbp-1A0h] BYREF
  __m128i v218; // [rsp+180h] [rbp-190h] BYREF
  __int16 v219; // [rsp+190h] [rbp-180h]
  __m128i v220; // [rsp+1A0h] [rbp-170h] BYREF
  __m128i *v221; // [rsp+1B0h] [rbp-160h] BYREF
  __int64 v222; // [rsp+1B8h] [rbp-158h]
  _QWORD *v223; // [rsp+1C0h] [rbp-150h] BYREF
  unsigned __int64 v224; // [rsp+1C8h] [rbp-148h]
  _QWORD v225[2]; // [rsp+1D0h] [rbp-140h] BYREF
  _QWORD *v226; // [rsp+1E0h] [rbp-130h]
  __int64 v227; // [rsp+1E8h] [rbp-128h]
  _QWORD v228[2]; // [rsp+1F0h] [rbp-120h] BYREF
  __int64 v229; // [rsp+200h] [rbp-110h]
  unsigned int v230; // [rsp+208h] [rbp-108h]
  char v231; // [rsp+20Ch] [rbp-104h]
  void *v232; // [rsp+210h] [rbp-100h] BYREF
  void **v233; // [rsp+218h] [rbp-F8h]
  __int64 v234; // [rsp+230h] [rbp-E0h]
  char v235; // [rsp+238h] [rbp-D8h]
  void *src; // [rsp+240h] [rbp-D0h] BYREF
  unsigned __int64 v237; // [rsp+248h] [rbp-C8h]
  unsigned int v238; // [rsp+250h] [rbp-C0h] BYREF
  __int64 v239; // [rsp+258h] [rbp-B8h]
  _QWORD *v240; // [rsp+260h] [rbp-B0h] BYREF
  unsigned __int64 v241; // [rsp+268h] [rbp-A8h]
  _QWORD v242[2]; // [rsp+270h] [rbp-A0h] BYREF
  _QWORD *v243; // [rsp+280h] [rbp-90h]
  __int64 v244; // [rsp+288h] [rbp-88h]
  _QWORD v245[2]; // [rsp+290h] [rbp-80h] BYREF
  __int64 v246; // [rsp+2A0h] [rbp-70h]
  unsigned int v247; // [rsp+2A8h] [rbp-68h]
  char v248; // [rsp+2ACh] [rbp-64h]
  void *v249; // [rsp+2B0h] [rbp-60h] BYREF
  void **v250; // [rsp+2B8h] [rbp-58h]
  __int64 v251; // [rsp+2D0h] [rbp-40h]
  char v252; // [rsp+2D8h] [rbp-38h]

  v5 = (__int64)(a1 + 22);
  v7 = (unsigned __int64)a1[29];
  *(_QWORD *)(a2 + 8) = v7;
  v8 = *((_DWORD *)a1 + 60);
  if ( v8 > 0x194 )
  {
    switch ( v8 )
    {
      case 0x1F7u:
        v17 = *((_DWORD *)a1 + 70);
        *(_DWORD *)a2 = 1;
        *(_DWORD *)(a2 + 16) = v17;
        goto LABEL_13;
      case 0x1F8u:
        v16 = *((_DWORD *)a1 + 70);
        *(_DWORD *)a2 = 0;
        *(_DWORD *)(a2 + 16) = v16;
        goto LABEL_13;
      case 0x1FCu:
        sub_2240AE0(a2 + 32, a1 + 31);
        *(_DWORD *)a2 = 3;
        goto LABEL_13;
      case 0x1FEu:
        sub_2240AE0(a2 + 32, a1 + 31);
        *(_DWORD *)a2 = 2;
        goto LABEL_13;
      case 0x210u:
        v13 = (__int64 *)(a2 + 112);
        v188 = (__int64 *)(a1 + 37);
        v14 = sub_C33340();
        v15 = a1[37];
        if ( *(_QWORD **)(a2 + 112) == v14 )
        {
          if ( v14 == v15 )
          {
            sub_C3C9E0((__int64 *)(a2 + 112), v188);
            goto LABEL_17;
          }
          if ( v13 == v188 )
            goto LABEL_17;
          v77 = *(_QWORD **)(a2 + 120);
          if ( v77 )
          {
            v78 = &v77[3 * *(v77 - 1)];
            if ( v77 != v78 )
            {
              v79 = &v77[3 * *(v77 - 1)];
              v80 = a2;
              v81 = (__int64 *)(a2 + 112);
              v82 = v14;
              do
              {
                v79 -= 3;
                if ( v82 == (_QWORD *)*v79 )
                  sub_969EE0((__int64)v79);
                else
                  sub_C338F0((__int64)v79);
              }
              while ( *(_QWORD **)(v80 + 120) != v79 );
              v105 = v81;
              v14 = v82;
              a2 = v80;
              v5 = (__int64)(a1 + 22);
              v78 = v79;
              v13 = v105;
            }
            v186 = v14;
            j_j_j___libc_free_0_0(v78 - 1);
            v14 = v186;
          }
        }
        else
        {
          if ( v14 != v15 )
          {
            sub_C33E70((__int64 *)(a2 + 112), (__int64 *)a1 + 37);
            goto LABEL_17;
          }
          if ( v13 == v188 )
          {
LABEL_17:
            *(_DWORD *)a2 = 5;
LABEL_13:
            v10 = 0;
            *((_DWORD *)a1 + 60) = sub_1205200(v5);
            return v10;
          }
          v185 = v14;
          sub_C338F0(a2 + 112);
          v14 = v185;
        }
        if ( v14 == a1[37] )
          sub_C3C790(v13, (_QWORD **)v188);
        else
          sub_C33EB0(v13, v188);
        goto LABEL_17;
      case 0x211u:
        if ( *(_DWORD *)(a2 + 104) <= 0x40u && *((_DWORD *)a1 + 82) <= 0x40u )
        {
          *(_QWORD *)(a2 + 96) = a1[40];
          *(_DWORD *)(a2 + 104) = *((_DWORD *)a1 + 82);
        }
        else
        {
          sub_C43990(a2 + 96, (__int64)(a1 + 40));
        }
        v12 = *((_BYTE *)a1 + 332);
        *(_DWORD *)a2 = 4;
        *(_BYTE *)(a2 + 108) = v12;
        goto LABEL_13;
      default:
        goto LABEL_5;
    }
  }
  if ( v8 > 5 )
  {
    switch ( v8 )
    {
      case 6u:
        p_src = (unsigned __int64)&src;
        src = &v238;
        *((_DWORD *)a1 + 60) = sub_1205200((__int64)(a1 + 22));
        v237 = 0x1000000000LL;
        v193 = (unsigned __int64)a1[29];
        if ( (unsigned __int8)sub_1224AB0(a1, &src) )
          goto LABEL_128;
        p_src = 7;
        v10 = sub_120AFE0((__int64)a1, 7, "expected end of array constant");
        if ( (_BYTE)v10 )
          goto LABEL_128;
        if ( !(_DWORD)v237 )
        {
          *(_DWORD *)a2 = 11;
          goto LABEL_54;
        }
        v47 = *(_QWORD *)(*(_QWORD *)src + 8LL);
        v184 = *(_BYTE *)(v47 + 8) != 7 && *(_BYTE *)(v47 + 8) != 13;
        if ( !v184 )
        {
          sub_1207630(v213.m128i_i64, v47);
          v48 = (__m128i *)sub_2241130(&v213, 0, 0, "invalid array element type: ", 28);
          v217.m128i_i64[0] = (__int64)&v218;
          if ( (__m128i *)v48->m128i_i64[0] == &v48[1] )
          {
            v218 = _mm_loadu_si128(v48 + 1);
          }
          else
          {
            v217.m128i_i64[0] = v48->m128i_i64[0];
            v218.m128i_i64[0] = v48[1].m128i_i64[0];
          }
          p_src = v193;
          v217.m128i_i64[1] = v48->m128i_i64[1];
          v48->m128i_i64[0] = (__int64)v48[1].m128i_i64;
          v48->m128i_i64[1] = 0;
          v48[1].m128i_i8[0] = 0;
          LOWORD(v223) = 260;
          v220.m128i_i64[0] = (__int64)&v217;
          sub_11FD800(v5, v193, (__int64)&v220, 1);
          if ( (__m128i *)v217.m128i_i64[0] != &v218 )
          {
            p_src = v218.m128i_i64[0] + 1;
            j_j___libc_free_0(v217.m128i_i64[0], v218.m128i_i64[0] + 1);
          }
          if ( (__m128i **)v213.m128i_i64[0] != &v214 )
          {
            p_src = (unsigned __int64)v214->m128i_u64 + 1;
            j_j___libc_free_0(v213.m128i_i64[0], &v214->m128i_i8[1]);
          }
          goto LABEL_128;
        }
        v90 = (__int64 **)sub_BCD420((__int64 *)v47, (unsigned int)v237);
        v91 = (unsigned int)v237;
        p_src = (unsigned __int64)src;
        v92 = v90;
        if ( (_DWORD)v237 )
        {
          v93 = 0;
          for ( i = *(_QWORD *)(*(_QWORD *)src + 8LL); ; i = *(_QWORD *)(*((_QWORD *)src + v93) + 8LL) )
          {
            if ( *(_QWORD *)(*(_QWORD *)src + 8LL) != i )
            {
              sub_1207630((__int64 *)&v199, *(_QWORD *)(*(_QWORD *)src + 8LL));
              v202[0].m128i_i32[0] = v93;
              v208.m128i_i64[0] = (__int64)" is not of type '";
              v204[0].m128i_i64[0] = (__int64)"array element #";
              v217.m128i_i64[0] = (__int64)&v199;
              v219 = 260;
              v212 = 1;
              v211 = 3;
              v203 = 265;
              v206 = 1;
              v205 = 3;
              sub_9C6370(v207, v204, v202, v95, v96, v97);
              sub_9C6370(&v213, v207, &v208, v98, v99, v100);
              sub_9C6370(&v220, &v213, &v217, v101, v102, v103);
              p_src = v193;
              sub_11FD800(v5, v193, (__int64)&v220, 1);
              sub_2240A30(&v199);
              v10 = v184;
              goto LABEL_54;
            }
            if ( ++v93 == (unsigned int)v237 )
              break;
          }
        }
        else
        {
          v91 = 0;
        }
        v110 = sub_AD1300(v92, (__int64 *)src, v91);
        *(_DWORD *)a2 = 12;
        *(_QWORD *)(a2 + 136) = v110;
        goto LABEL_54;
      case 8u:
        p_src = (unsigned __int64)&src;
        src = &v238;
        *((_DWORD *)a1 + 60) = sub_1205200((__int64)(a1 + 22));
        v237 = 0x1000000000LL;
        if ( (unsigned __int8)sub_1224AB0(a1, &src) )
          goto LABEL_128;
        p_src = 9;
        v10 = sub_120AFE0((__int64)a1, 9, "expected end of struct constant");
        if ( (_BYTE)v10 )
          goto LABEL_128;
        v42 = (unsigned int)v237;
        v43 = 8LL * (unsigned int)v237;
        v44 = (void *)sub_2207820(v43);
        if ( v44 && v42 )
          v44 = memset(v44, 0, v43);
        v45 = *(_QWORD *)(a2 + 144);
        *(_QWORD *)(a2 + 144) = v44;
        if ( v45 )
        {
          j_j___libc_free_0_0(v45);
          v44 = *(void **)(a2 + 144);
        }
        v46 = (unsigned int)v237;
        p_src = (unsigned __int64)src;
        *(_DWORD *)(a2 + 16) = v237;
        memcpy(v44, (const void *)p_src, 8 * v46);
        *(_DWORD *)a2 = 15;
        goto LABEL_54;
      case 0xAu:
        v36 = sub_1205200((__int64)(a1 + 22));
        *((_DWORD *)a1 + 60) = v36;
        if ( v36 == 8 )
        {
          p_src = (unsigned __int64)&src;
          src = &v238;
          *((_DWORD *)a1 + 60) = sub_1205200(v5);
          v237 = 0x1000000000LL;
          if ( !(unsigned __int8)sub_1224AB0(a1, &src) )
          {
            p_src = 9;
            if ( !(unsigned __int8)sub_120AFE0((__int64)a1, 9, "expected end of packed struct") )
            {
              p_src = 11;
              v10 = sub_120AFE0((__int64)a1, 11, "expected end of constant");
              if ( !(_BYTE)v10 )
              {
                v64 = (unsigned int)v237;
                v65 = 8LL * (unsigned int)v237;
                v66 = (void *)sub_2207820(v65);
                v67 = v66;
                if ( v66 && v64 )
                  v67 = memset(v66, 0, v65);
                v68 = *(_QWORD *)(a2 + 144);
                v69 = v67;
                *(_QWORD *)(a2 + 144) = v67;
                if ( v68 )
                {
                  j_j___libc_free_0_0(v68);
                  v69 = *(void **)(a2 + 144);
                }
                p_src = (unsigned __int64)src;
                memcpy(v69, src, 8LL * (unsigned int)v237);
                v70 = v237;
                *(_DWORD *)a2 = 16;
                *(_DWORD *)(a2 + 16) = v70;
                goto LABEL_54;
              }
            }
          }
        }
        else
        {
          p_src = (unsigned __int64)&src;
          v237 = 0x1000000000LL;
          v37 = a1[29];
          src = &v238;
          v192 = (unsigned __int64)v37;
          if ( !(unsigned __int8)sub_1224AB0(a1, &src) )
          {
            p_src = 11;
            v10 = sub_120AFE0((__int64)a1, 11, "expected end of constant");
            if ( !(_BYTE)v10 )
            {
              p_src = (unsigned int)v237;
              if ( (_DWORD)v237 )
              {
                v38 = *(_QWORD *)(*(_QWORD *)src + 8LL);
                v39 = *(unsigned __int8 *)(v38 + 8);
                if ( (unsigned __int8)v39 > 0xCu || (v40 = 4143, !_bittest64(&v40, v39)) )
                {
                  v106 = v39 & 0xFFFFFFFD;
                  LOBYTE(v106) = (_BYTE)v39 != 14 && (v39 & 0xFD) != 4;
                  if ( (_BYTE)v106 )
                  {
                    p_src = v192;
                    v10 = v106;
                    v220.m128i_i64[0] = (__int64)"vector elements must have integer, pointer or floating point type";
                    LOWORD(v223) = 259;
                    sub_11FD800(v5, v192, (__int64)&v220, 1);
LABEL_54:
                    if ( src != &v238 )
                      _libc_free(src, p_src);
                    return v10;
                  }
                }
                if ( (_DWORD)v237 == 1 )
                {
LABEL_297:
                  v104 = sub_AD3730((__int64 *)src, (unsigned int)v237);
                  *(_DWORD *)a2 = 12;
                  *(_QWORD *)(a2 + 136) = v104;
                  goto LABEL_54;
                }
                v41 = 1;
                while ( v38 == *(_QWORD *)(*((_QWORD *)src + v41) + 8LL) )
                {
                  if ( (_DWORD)v237 == ++v41 )
                    goto LABEL_297;
                }
                sub_1207630(v208.m128i_i64, *(_QWORD *)(*(_QWORD *)src + 8LL));
                p_src = v192;
                v213.m128i_i64[0] = (__int64)"vector element #";
                v217.m128i_i64[0] = (__int64)&v213;
                v218.m128i_i64[0] = (__int64)" is not of type '";
                v221 = &v208;
                v220.m128i_i64[0] = (__int64)&v217;
                LODWORD(v214) = v41;
                v216 = 2307;
                v219 = 770;
                LOWORD(v223) = 1026;
                sub_11FD800(v5, v192, (__int64)&v220, 1);
                if ( (__m128i **)v208.m128i_i64[0] != &v209 )
                {
                  p_src = (unsigned __int64)v209->m128i_u64 + 1;
                  j_j___libc_free_0(v208.m128i_i64[0], &v209->m128i_i8[1]);
                }
              }
              else
              {
                p_src = *(_QWORD *)(a2 + 8);
                v220.m128i_i64[0] = (__int64)"constant vector must not be empty";
                LOWORD(v223) = 259;
                sub_11FD800(v5, p_src, (__int64)&v220, 1);
              }
            }
          }
        }
LABEL_128:
        v10 = 1;
        goto LABEL_54;
      case 0x14u:
        v35 = sub_ACD6D0(*a1);
        *(_DWORD *)a2 = 12;
        *(_QWORD *)(a2 + 136) = v35;
        goto LABEL_13;
      case 0x15u:
        v34 = sub_ACD720(*a1);
        *(_DWORD *)a2 = 12;
        *(_QWORD *)(a2 + 136) = v34;
        goto LABEL_13;
      case 0x33u:
        *(_DWORD *)a2 = 8;
        goto LABEL_13;
      case 0x34u:
        *(_DWORD *)a2 = 7;
        goto LABEL_13;
      case 0x35u:
        *(_DWORD *)a2 = 10;
        goto LABEL_13;
      case 0x36u:
        *(_DWORD *)a2 = 6;
        goto LABEL_13;
      case 0x37u:
        *(_DWORD *)a2 = 9;
        goto LABEL_13;
      case 0x65u:
        v32 = 0;
        v33 = sub_1205200((__int64)(a1 + 22));
        *((_DWORD *)a1 + 60) = v33;
        if ( v33 == 102 )
        {
          v32 = 1;
          v33 = sub_1205200(v5);
          *((_DWORD *)a1 + 60) = v33;
        }
        v191 = 0;
        if ( v33 == 259 )
        {
          v33 = sub_1205200(v5);
          v191 = 1;
          *((_DWORD *)a1 + 60) = v33;
        }
        v183 = 0;
        if ( v33 == 103 )
        {
          v33 = sub_1205200(v5);
          v183 = 1;
          *((_DWORD *)a1 + 60) = v33;
        }
        v178 = 0;
        if ( v33 == 66 )
        {
          v178 = 1;
          *((_DWORD *)a1 + 60) = sub_1205200(v5);
        }
        if ( (unsigned __int8)sub_120B3D0((__int64)a1, a2 + 32) )
          return 1;
        if ( (unsigned __int8)sub_120AFE0((__int64)a1, 4, "expected comma in inline asm expression") )
          return 1;
        v10 = sub_120AFE0((__int64)a1, 512, "expected constraint string");
        if ( (_BYTE)v10 )
          return 1;
        sub_2240AE0(a2 + 64, a1 + 31);
        *(_DWORD *)a2 = 14;
        *(_DWORD *)(a2 + 16) = (8 * v178) | (4 * v183) | v32 | (2 * v191);
        return v10;
      case 0x6Bu:
        v29 = sub_1205200((__int64)(a1 + 22));
        v30 = (char *)a1[31];
        v31 = (signed __int64)a1[32];
        *((_DWORD *)a1 + 60) = v29;
        *(_QWORD *)(a2 + 136) = sub_AC9B20((__int64)*a1, v30, v31, 0);
        v10 = sub_120AFE0((__int64)a1, 512, "expected string");
        if ( (_BYTE)v10 )
          return 1;
        goto LABEL_78;
      case 0x14Cu:
        src = "fneg constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x14Du:
      case 0x14Fu:
      case 0x15Eu:
        v25 = *((_DWORD *)a1 + 70);
        v26 = sub_1205200((__int64)(a1 + 22));
        v182 = 0;
        *((_DWORD *)a1 + 60) = v26;
        v190 = v25 == 17 || (v25 & 0xFFFFFFFD) == 13;
        if ( v190 )
        {
          if ( v26 == 85 )
          {
            v26 = sub_1205200(v5);
            *((_DWORD *)a1 + 60) = v26;
            v182 = v25 == 17 || (v25 & 0xFFFFFFFD) == 13;
          }
          if ( v26 == 86 )
          {
            v74 = sub_1205200(v5);
            *((_DWORD *)a1 + 60) = v74;
            if ( v74 == 85 )
            {
              v75 = sub_1205540((__int64)a1);
              if ( v75 )
              {
                v190 = v75;
                v182 = v75;
              }
            }
          }
          else
          {
            v190 = 0;
          }
        }
        if ( (unsigned __int8)sub_120AFE0((__int64)a1, 12, "expected '(' in binary constantexpr") )
          return 1;
        if ( (unsigned __int8)sub_1224A40(a1, &v217) )
          return 1;
        if ( (unsigned __int8)sub_120AFE0((__int64)a1, 4, "expected comma in binary constantexpr") )
          return 1;
        if ( (unsigned __int8)sub_1224A40(a1, &v220) )
          return 1;
        v10 = sub_120AFE0((__int64)a1, 13, "expected ')' in binary constantexpr");
        if ( (_BYTE)v10 )
          return 1;
        v27 = *(_QWORD *)(v217.m128i_i64[0] + 8);
        if ( *(_QWORD *)(v220.m128i_i64[0] + 8) != v27 )
        {
          BYTE1(v240) = 1;
          v28 = "operands of constexpr must have same type";
          goto LABEL_68;
        }
        if ( (unsigned int)*(unsigned __int8 *)(v27 + 8) - 17 <= 1 )
          v27 = **(_QWORD **)(v27 + 16);
        if ( *(_BYTE *)(v27 + 8) != 12 )
        {
          BYTE1(v240) = 1;
          v28 = "constexpr requires integer or integer vector operands";
          goto LABEL_68;
        }
        v116 = v182;
        if ( v190 )
          v116 = v182 | 2;
        *(_QWORD *)(a2 + 136) = sub_AD5570(v25, v217.m128i_i64[0], (unsigned __int8 *)v220.m128i_i64[0], v116, 0);
LABEL_78:
        *(_DWORD *)a2 = 12;
        return v10;
      case 0x14Eu:
        src = "fadd constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x150u:
        src = "fsub constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x151u:
        src = "mul constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x152u:
        src = "fmul constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x153u:
        src = "udiv constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x154u:
        src = "sdiv constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x155u:
        src = "fdiv constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x156u:
        src = "urem constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x157u:
        src = "srem constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x158u:
        src = "frem constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x159u:
        src = "shl constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x15Au:
        src = "lshr constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x15Bu:
        src = "ashr constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x15Cu:
        src = "and constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x15Du:
        src = "or constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x15Fu:
        src = "icmp constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x160u:
        src = "fcmp constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x163u:
      case 0x16Cu:
      case 0x16Du:
      case 0x16Eu:
      case 0x16Fu:
        v18 = *((_DWORD *)a1 + 70);
        v199 = 0;
        *((_DWORD *)a1 + 60) = sub_1205200((__int64)(a1 + 22));
        if ( (unsigned __int8)sub_120AFE0((__int64)a1, 12, "expected '(' after constantexpr cast") )
          return 1;
        if ( (unsigned __int8)sub_1224A40(a1, v202) )
          return 1;
        if ( (unsigned __int8)sub_120AFE0((__int64)a1, 56, "expected 'to' in constantexpr cast") )
          return 1;
        src = "expected type";
        LOWORD(v240) = 259;
        if ( (unsigned __int8)sub_12190A0((__int64)a1, (__int64 **)&v199, (int *)&src, 0) )
          return 1;
        v10 = sub_120AFE0((__int64)a1, 13, "expected ')' at end of constantexpr cast");
        if ( (_BYTE)v10 )
          return 1;
        if ( !(unsigned __int8)sub_B50F30(v18, *(_QWORD *)(v202[0].m128i_i64[0] + 8), (__int64)v199) )
        {
          sub_1207630(v213.m128i_i64, (__int64)v199);
          sub_1207630(v204[0].m128i_i64, *(_QWORD *)(v202[0].m128i_i64[0] + 8));
          sub_95D570(v207, "invalid cast opcode for cast from '", (__int64)v204);
          sub_94F930(&v208, (__int64)v207, "' to '");
          sub_8FD5D0(&v217, (__int64)&v208, &v213);
          sub_94F930(&v220, (__int64)&v217, "'");
          v19 = *(_QWORD *)(a2 + 8);
          LOWORD(v240) = 260;
          src = &v220;
          sub_11FD800(v5, v19, (__int64)&src, 1);
          sub_2240A30(&v220);
          sub_2240A30(&v217);
          sub_2240A30(&v208);
          sub_2240A30(v207);
          sub_2240A30(v204);
          sub_2240A30(&v213);
          return 1;
        }
        v115 = sub_ADAB70(v18, v202[0].m128i_u64[0], v199, 0);
        *(_DWORD *)a2 = 12;
        *(_QWORD *)(a2 + 136) = v115;
        return v10;
      case 0x164u:
        src = "zext constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x165u:
        src = "sext constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x166u:
        src = "fptrunc constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x167u:
        src = "fpext constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x168u:
        src = "uitofp constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x169u:
        src = "sitofp constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x16Au:
        src = "fptoui constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x16Bu:
        src = "fptosi constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x170u:
        src = "select constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x18Au:
      case 0x18Bu:
      case 0x18Cu:
      case 0x18Du:
        v20 = *((_DWORD *)a1 + 70);
        src = &v238;
        v181 = v20;
        v237 = 0x1000000000LL;
        v200 = 1;
        v199 = 0;
        v201 = 0;
        v202[0].m128i_i32[2] = 1;
        v202[0].m128i_i64[0] = 0;
        v202[0].m128i_i8[12] = 0;
        v21 = sub_1205200((__int64)(a1 + 22));
        v189 = 0;
        *((_DWORD *)a1 + 60) = v21;
        if ( v20 == 34 )
        {
          v22 = 0;
          while ( 1 )
          {
            switch ( v21 )
            {
              case 'Z':
                v22 |= 3u;
                *((_DWORD *)a1 + 60) = sub_1205200(v5);
                break;
              case 'W':
                v22 |= 2u;
                *((_DWORD *)a1 + 60) = sub_1205200(v5);
                break;
              case 'U':
                v22 |= 4u;
                *((_DWORD *)a1 + 60) = sub_1205200(v5);
                break;
              default:
                v189 = v22;
                v177 = 0;
                if ( v21 != 93 )
                {
LABEL_35:
                  p_src = 12;
                  if ( (unsigned __int8)sub_120AFE0((__int64)a1, 12, "expected '(' in constantexpr") )
                    goto LABEL_47;
                  p_src = (unsigned __int64)&v198;
                  v220.m128i_i64[0] = (__int64)"expected type";
                  LOWORD(v223) = 259;
                  if ( (unsigned __int8)sub_12190A0((__int64)a1, &v198, v220.m128i_i32, 0) )
                    goto LABEL_47;
                  p_src = 4;
                  if ( (unsigned __int8)sub_120AFE0((__int64)a1, 4, "expected comma after getelementptr's type") )
                    goto LABEL_47;
                  goto LABEL_38;
                }
                p_src = 12;
                *((_DWORD *)a1 + 60) = sub_1205200(v5);
                if ( (unsigned __int8)sub_120AFE0((__int64)a1, 12, "expected '('") )
                  goto LABEL_47;
                if ( *((_DWORD *)a1 + 60) == 529 )
                {
                  if ( v200 <= 0x40 && *((_DWORD *)a1 + 82) <= 0x40u )
                  {
                    v137 = a1[40];
                    v200 = *((_DWORD *)a1 + 82);
                    v199 = (__int64 **)v137;
                  }
                  else
                  {
                    sub_C43990((__int64)&v199, (__int64)(a1 + 40));
                  }
                  v201 = *((_BYTE *)a1 + 332);
                  p_src = 4;
                  *((_DWORD *)a1 + 60) = sub_1205200(v5);
                  if ( (unsigned __int8)sub_120AFE0((__int64)a1, 4, "expected ','") )
                    goto LABEL_47;
                  if ( *((_DWORD *)a1 + 60) == 529 )
                  {
                    if ( v202[0].m128i_i32[2] <= 0x40u && *((_DWORD *)a1 + 82) <= 0x40u )
                    {
                      v150 = (__int64)a1[40];
                      v202[0].m128i_i32[2] = *((_DWORD *)a1 + 82);
                      v202[0].m128i_i64[0] = v150;
                    }
                    else
                    {
                      sub_C43990((__int64)v202, (__int64)(a1 + 40));
                    }
                    v202[0].m128i_i8[12] = *((_BYTE *)a1 + 332);
                    p_src = 13;
                    *((_DWORD *)a1 + 60) = sub_1205200(v5);
                    if ( (unsigned __int8)sub_120AFE0((__int64)a1, 13, "expected ')'") )
                      goto LABEL_47;
                    v177 = 1;
                    goto LABEL_35;
                  }
                }
                p_src = (unsigned __int64)a1[29];
                v220.m128i_i64[0] = (__int64)"expected integer";
                LOWORD(v223) = 259;
LABEL_46:
                sub_11FD800(v5, p_src, (__int64)&v220, 1);
                goto LABEL_47;
            }
            v21 = *((_DWORD *)a1 + 60);
          }
        }
        p_src = 12;
        v177 = sub_120AFE0((__int64)a1, 12, "expected '(' in constantexpr");
        if ( v177 )
          goto LABEL_47;
LABEL_38:
        p_src = (unsigned __int64)&src;
        if ( (unsigned __int8)sub_1224AB0(a1, &src) )
          goto LABEL_47;
        p_src = 13;
        v10 = sub_120AFE0((__int64)a1, 13, "expected ')' in constantexpr");
        if ( (_BYTE)v10 )
          goto LABEL_47;
        if ( v181 != 34 )
        {
          if ( v181 == 63 )
          {
            if ( (_DWORD)v237 != 3 )
            {
              BYTE1(v223) = 1;
              v24 = "expected three operands to shufflevector";
              goto LABEL_45;
            }
            if ( !(unsigned __int8)sub_B4E200(*(_QWORD *)src, *((_QWORD *)src + 1), *((_QWORD *)src + 2)) )
            {
              BYTE1(v223) = 1;
              v24 = "invalid operands to shufflevector";
              goto LABEL_45;
            }
            v220.m128i_i64[0] = (__int64)&v221;
            v220.m128i_i64[1] = 0x1000000000LL;
            sub_B4E3E0(*((unsigned __int8 **)src + 2), v220.m128i_i64);
            p_src = *((_QWORD *)src + 1);
            v146 = sub_AD5CE0(*(_QWORD *)src, p_src, v220.m128i_i64[0], v220.m128i_u32[2], 0);
            v147 = (__int64 *)v220.m128i_i64[0];
            *(_QWORD *)(a2 + 136) = v146;
            if ( v147 != (__int64 *)&v221 )
              _libc_free(v147, p_src);
          }
          else if ( v181 == 61 )
          {
            if ( (_DWORD)v237 != 2 )
            {
              BYTE1(v223) = 1;
              v24 = "expected two operands to extractelement";
              goto LABEL_45;
            }
            if ( !(unsigned __int8)sub_B4DF70(*(_QWORD *)src, *((_QWORD *)src + 1)) )
            {
              BYTE1(v223) = 1;
              v24 = "invalid extractelement operands";
              goto LABEL_45;
            }
            p_src = *((_QWORD *)src + 1);
            *(_QWORD *)(a2 + 136) = sub_AD5840(*(_QWORD *)src, (unsigned __int8 *)p_src, 0);
          }
          else
          {
            if ( (_DWORD)v237 != 3 )
            {
              BYTE1(v223) = 1;
              v24 = "expected three operands to insertelement";
LABEL_45:
              v220.m128i_i64[0] = (__int64)v24;
              p_src = *(_QWORD *)(a2 + 8);
              LOBYTE(v223) = 3;
              goto LABEL_46;
            }
            if ( !(unsigned __int8)sub_B4E100(*(_QWORD *)src, *((_QWORD *)src + 1), *((_QWORD *)src + 2)) )
            {
              BYTE1(v223) = 1;
              v24 = "invalid insertelement operands";
              goto LABEL_45;
            }
            p_src = *((_QWORD *)src + 1);
            *(_QWORD *)(a2 + 136) = sub_AD5A90(*(_QWORD *)src, (_BYTE *)p_src, *((unsigned __int8 **)src + 2), 0);
          }
LABEL_368:
          *(_DWORD *)a2 = 12;
LABEL_48:
          if ( v202[0].m128i_i32[2] > 0x40u && v202[0].m128i_i64[0] )
            j_j___libc_free_0_0(v202[0].m128i_i64[0]);
          if ( v200 > 0x40 && v199 )
            j_j___libc_free_0_0(v199);
          goto LABEL_54;
        }
        if ( !(_DWORD)v237 )
          goto LABEL_307;
        v107 = (char *)src;
        v108 = *(_QWORD *)(*(_QWORD *)src + 8LL);
        v176 = v108;
        v109 = v108;
        if ( (unsigned int)*(unsigned __int8 *)(v108 + 8) - 17 <= 1 )
          v109 = **(_QWORD **)(v108 + 16);
        if ( *(_BYTE *)(v109 + 8) != 14 )
        {
LABEL_307:
          BYTE1(v223) = 1;
          v24 = "base of getelementptr must be a pointer";
          goto LABEL_45;
        }
        v117 = &v208;
        for ( j = 10; j; --j )
        {
          v117->m128i_i32[0] = 0;
          v117 = (__m128i *)((char *)v117 + 4);
        }
        if ( !v177 )
          goto LABEL_342;
        v175 = sub_AE43F0((__int64)(a1[43] + 39), v176);
        sub_1208CE0((__int64)&v220, (__int64)&v199, v175);
        if ( v200 > 0x40 && v199 )
          j_j___libc_free_0_0(v199);
        v199 = (__int64 **)v220.m128i_i64[0];
        v134 = v220.m128i_i32[2];
        v220.m128i_i32[2] = 0;
        v200 = v134;
        v201 = v220.m128i_i8[12];
        sub_969240(v220.m128i_i64);
        sub_1208CE0((__int64)&v220, (__int64)v202, v175);
        if ( v202[0].m128i_i32[2] > 0x40u && v202[0].m128i_i64[0] )
          j_j___libc_free_0_0(v202[0].m128i_i64[0]);
        v202[0].m128i_i64[0] = v220.m128i_i64[0];
        v135 = v220.m128i_i32[2];
        v220.m128i_i32[2] = 0;
        v202[0].m128i_i32[2] = v135;
        v202[0].m128i_i8[12] = v220.m128i_i8[12];
        sub_969240(v220.m128i_i64);
        if ( (int)sub_C4C880((__int64)&v199, (__int64)v202) >= 0 )
        {
          p_src = *(_QWORD *)(a2 + 8);
          v220.m128i_i64[0] = (__int64)"expected end to be larger than start";
          LOWORD(v223) = 259;
          sub_11FD800(v5, p_src, (__int64)&v220, 1);
          goto LABEL_372;
        }
        sub_9865C0((__int64)v207, (__int64)v202);
        sub_9865C0((__int64)v204, (__int64)&v199);
        v162 = v204[0].m128i_u32[2];
        if ( v204[0].m128i_i32[2] <= 0x40u )
        {
          if ( v204[0].m128i_i64[0] == v207[0].m128i_i64[0] )
            goto LABEL_450;
        }
        else
        {
          v180 = v204[0].m128i_i32[2];
          v163 = sub_C43C50((__int64)v204, (const void **)v207);
          v162 = v180;
          if ( v163 )
          {
LABEL_450:
            sub_AADB10((__int64)&v220, v162, 1);
LABEL_451:
            if ( v211 )
            {
              if ( v208.m128i_i32[2] > 0x40u && v208.m128i_i64[0] )
                j_j___libc_free_0_0(v208.m128i_i64[0]);
              v208.m128i_i64[0] = v220.m128i_i64[0];
              v169 = v220.m128i_i32[2];
              v220.m128i_i32[2] = 0;
              v208.m128i_i32[2] = v169;
              if ( v210 > 0x40 && v209 )
                j_j___libc_free_0_0(v209);
              v209 = v221;
              v170 = v222;
              LODWORD(v222) = 0;
              v210 = v170;
            }
            else
            {
              v164 = v220.m128i_i32[2];
              v211 = 1;
              v220.m128i_i32[2] = 0;
              v208.m128i_i32[2] = v164;
              v208.m128i_i64[0] = v220.m128i_i64[0];
              v165 = v222;
              LODWORD(v222) = 0;
              v210 = v165;
              v209 = v221;
            }
            sub_969240((__int64 *)&v221);
            sub_969240(v220.m128i_i64);
            sub_969240(v204[0].m128i_i64);
            sub_969240(v207[0].m128i_i64);
            v107 = (char *)src;
LABEL_342:
            v119 = 0;
            if ( (unsigned int)*(unsigned __int8 *)(v176 + 8) - 17 <= 1 )
              v119 = *(_DWORD *)(v176 + 32);
            v187 = (__int64 *)(v107 + 8);
            v120 = 8LL * (unsigned int)v237;
            v121 = &v107[v120];
            if ( v107 + 8 != &v107[v120] )
            {
              for ( k = v107 + 8; v121 != k; k += 8 )
              {
                v123 = *(_QWORD *)(*(_QWORD *)k + 8LL);
                v124 = *(unsigned __int8 *)(v123 + 8);
                if ( (unsigned int)(v124 - 17) > 1 )
                {
                  if ( (_BYTE)v124 != 12 )
                  {
LABEL_370:
                    BYTE1(v223) = 1;
                    v133 = "getelementptr index must be an integer";
                    goto LABEL_371;
                  }
                }
                else
                {
                  if ( *(_BYTE *)(**(_QWORD **)(v123 + 16) + 8LL) != 12 )
                    goto LABEL_370;
                  v125 = *(_DWORD *)(v123 + 32);
                  if ( v119 && v119 != v125 )
                  {
                    BYTE1(v223) = 1;
                    v133 = "getelementptr vector index has a wrong number of elements";
LABEL_371:
                    p_src = *(_QWORD *)(a2 + 8);
                    v220.m128i_i64[0] = (__int64)v133;
                    LOBYTE(v223) = 3;
                    sub_11FD800(v5, p_src, (__int64)&v220, 1);
LABEL_372:
                    if ( v211 )
                      sub_9963D0((__int64)&v208);
LABEL_47:
                    v10 = 1;
                    goto LABEL_48;
                  }
                  v119 = v125;
                }
              }
            }
            v220.m128i_i64[0] = 0;
            v126 = v198;
            v220.m128i_i64[1] = (__int64)&v223;
            v221 = (__m128i *)4;
            v127 = (v120 - 8) >> 3;
            LODWORD(v222) = 0;
            BYTE4(v222) = 1;
            if ( v127 )
            {
              v128 = *((unsigned __int8 *)v198 + 8);
              if ( (_BYTE)v128 != 12
                && (unsigned __int8)v128 > 3u
                && (_BYTE)v128 != 5
                && (v128 & 0xFB) != 0xA
                && (v128 & 0xFD) != 4 )
              {
                if ( (unsigned __int8)(*((_BYTE *)v198 + 8) - 15) > 3u && v128 != 20
                  || !(unsigned __int8)sub_BCEBA0((__int64)v198, (__int64)&v220) )
                {
                  HIBYTE(v219) = 1;
                  v159 = "base element of getelementptr must be sized";
                  goto LABEL_441;
                }
                v126 = v198;
              }
            }
            if ( sub_B4DCA0((__int64)v126, (__int64)v187, v127) )
            {
              LOBYTE(v216) = 0;
              if ( v211 )
              {
                v213.m128i_i32[2] = v208.m128i_i32[2];
                if ( v208.m128i_i32[2] > 0x40u )
                  sub_C43780((__int64)&v213, (const void **)&v208);
                else
                  v213.m128i_i64[0] = v208.m128i_i64[0];
                v215 = v210;
                if ( v210 > 0x40 )
                  sub_C43780((__int64)&v214, (const void **)&v209);
                else
                  v214 = v209;
                LOBYTE(v216) = 1;
              }
              v129 = v198;
              v130 = *(_QWORD *)src;
              LOBYTE(v219) = 0;
              v179 = (unsigned __int8 *)v130;
              if ( (_BYTE)v216 )
              {
                v217.m128i_i32[2] = v213.m128i_i32[2];
                if ( v213.m128i_i32[2] > 0x40u )
                  sub_C43780((__int64)&v217, (const void **)&v213);
                else
                  v217.m128i_i64[0] = v213.m128i_i64[0];
                v218.m128i_i32[2] = v215;
                if ( v215 > 0x40 )
                  sub_C43780((__int64)&v218, (const void **)&v214);
                else
                  v218.m128i_i64[0] = (__int64)v214;
                LOBYTE(v219) = 1;
              }
              p_src = (unsigned __int64)v179;
              v131 = sub_AD9FD0((__int64)v129, v179, v187, v127, v189, (__int64)&v217, 0);
              if ( (_BYTE)v219 )
              {
                LOBYTE(v219) = 0;
                if ( v218.m128i_i32[2] > 0x40u && v218.m128i_i64[0] )
                  j_j___libc_free_0_0(v218.m128i_i64[0]);
                if ( v217.m128i_i32[2] > 0x40u && v217.m128i_i64[0] )
                  j_j___libc_free_0_0(v217.m128i_i64[0]);
              }
              v132 = (_BYTE)v216 == 0;
              *(_QWORD *)(a2 + 136) = v131;
              if ( !v132 )
                sub_9963D0((__int64)&v213);
              if ( !BYTE4(v222) )
                _libc_free(v220.m128i_i64[1], v179);
              if ( v211 )
                sub_9963D0((__int64)&v208);
              goto LABEL_368;
            }
            HIBYTE(v219) = 1;
            v159 = "invalid getelementptr indices";
LABEL_441:
            p_src = *(_QWORD *)(a2 + 8);
            v217.m128i_i64[0] = (__int64)v159;
            LOBYTE(v219) = 3;
            sub_11FD800(v5, p_src, (__int64)&v217, 1);
            if ( !BYTE4(v222) )
              _libc_free(v220.m128i_i64[1], p_src);
            goto LABEL_372;
          }
        }
        v171 = v207[0].m128i_i32[2];
        v207[0].m128i_i32[2] = 0;
        v213.m128i_i32[2] = v162;
        v217.m128i_i32[2] = v171;
        v204[0].m128i_i32[2] = 0;
        v217.m128i_i64[0] = v207[0].m128i_i64[0];
        v213.m128i_i64[0] = v204[0].m128i_i64[0];
        sub_AADC30((__int64)&v220, (__int64)&v213, v217.m128i_i64);
        if ( v213.m128i_i32[2] > 0x40u && v213.m128i_i64[0] )
          j_j___libc_free_0_0(v213.m128i_i64[0]);
        if ( v217.m128i_i32[2] > 0x40u && v217.m128i_i64[0] )
          j_j___libc_free_0_0(v217.m128i_i64[0]);
        goto LABEL_451;
      case 0x18Eu:
        *((_DWORD *)a1 + 60) = sub_1205200((__int64)(a1 + 22));
        if ( (unsigned __int8)sub_120AFE0((__int64)a1, 12, "expected '(' after vector splat") )
          return 1;
        if ( (unsigned __int8)sub_1224A40(a1, &src) )
          return 1;
        v10 = sub_120AFE0((__int64)a1, 13, "expected ')' at end of vector splat");
        if ( (_BYTE)v10 )
          return 1;
        v63 = src;
        *(_DWORD *)a2 = 13;
        *(_QWORD *)(a2 + 136) = v63;
        return v10;
      case 0x18Fu:
        src = "extractvalue constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x190u:
        src = "insertvalue constexprs are no longer supported";
        LOWORD(v240) = 259;
        goto LABEL_6;
      case 0x191u:
        LOBYTE(v225[0]) = 0;
        *((_DWORD *)a1 + 60) = sub_1205200((__int64)(a1 + 22));
        v223 = v225;
        v220.m128i_i32[0] = 0;
        v220.m128i_i64[1] = 0;
        v222 = 0;
        v224 = 0;
        v226 = v228;
        v227 = 0;
        LOBYTE(v228[0]) = 0;
        v230 = 1;
        v229 = 0;
        v231 = 0;
        v55 = sub_C33320();
        sub_C3B1B0((__int64)&src, 0.0);
        sub_C407B0(&v232, (__int64 *)&src, v55);
        sub_C338F0((__int64)&src);
        v234 = 0;
        v240 = v242;
        v243 = v245;
        LOBYTE(v245[0]) = 0;
        v235 = 0;
        LODWORD(src) = 0;
        v237 = 0;
        v239 = 0;
        v241 = 0;
        LOBYTE(v242[0]) = 0;
        v244 = 0;
        v247 = 1;
        v246 = 0;
        v248 = 0;
        sub_C3B1B0((__int64)&v217, 0.0);
        sub_C407B0(&v249, v217.m128i_i64, v55);
        sub_C338F0((__int64)&v217);
        v251 = 0;
        v252 = 0;
        if ( (unsigned __int8)sub_120AFE0((__int64)a1, 12, "expected '(' in block address expression") )
          goto LABEL_217;
        if ( (unsigned __int8)sub_1221570(a1, &v220, a3, 0) )
          goto LABEL_217;
        if ( (unsigned __int8)sub_120AFE0((__int64)a1, 4, "expected comma in block address expression") )
          goto LABEL_217;
        if ( (unsigned __int8)sub_1221570(a1, &src, a3, 0) )
          goto LABEL_217;
        v10 = sub_120AFE0((__int64)a1, 13, "expected ')' in block address expression");
        if ( (_BYTE)v10 )
          goto LABEL_217;
        v56 = v220.m128i_i32[0];
        v57 = a4;
        if ( (v220.m128i_i32[0] & 0xFFFFFFFD) != 1 )
          goto LABEL_216;
        if ( ((unsigned int)src & 0xFFFFFFFD) != 0 )
        {
          HIBYTE(v219) = 1;
          v145 = "expected basic block name in blockaddress";
          goto LABEL_416;
        }
        if ( v220.m128i_i32[0] == 1 )
        {
          v138 = *((_DWORD *)a1 + 304);
          if ( !v138 )
            goto LABEL_182;
          v139 = v138 - 1;
          for ( m = v139 & (37 * (_DWORD)v221); ; m = v139 & v172 )
          {
            v141 = (__int64)&a1[150][2 * m];
            if ( (_DWORD)v221 == *(_DWORD *)v141 )
              break;
            if ( *(_DWORD *)v141 == -1 )
              goto LABEL_182;
            v172 = v56 + m;
            ++v56;
          }
          v142 = *(_BYTE **)(v141 + 8);
        }
        else
        {
          v58 = sub_1213320((__int64)(a1 + 137), (__int64)&v223);
          v57 = a4;
          if ( a1 + 138 != (_QWORD **)v58 )
            goto LABEL_182;
          v149 = sub_BA8B30((__int64)a1[43], (__int64)v223, v224);
          v57 = a4;
          v142 = (_BYTE *)v149;
        }
        if ( v142 )
        {
          if ( *v142 )
          {
LABEL_216:
            v217.m128i_i64[0] = (__int64)"expected function name in blockaddress";
            v219 = 259;
            sub_11FD800(v5, v220.m128i_u64[1], (__int64)&v217, 1);
LABEL_217:
            v10 = 1;
            goto LABEL_184;
          }
          v195 = v142;
          LOBYTE(v143) = sub_B2FC80((__int64)v142);
          v10 = v143;
          if ( (_BYTE)v143 )
          {
            v217.m128i_i64[0] = (__int64)"cannot take blockaddress inside a declaration";
            v219 = 259;
            sub_11FD800(v5, v220.m128i_u64[1], (__int64)&v217, 1);
            goto LABEL_184;
          }
          v144 = a1[166];
          if ( v144 && (_BYTE *)v144[1] == v195 )
          {
            if ( (_DWORD)src )
              v156 = sub_121DBC0(v144, (__int64)&v240, v237);
            else
              v156 = sub_121E0D0((__int64)v144, v238, v237);
            v157 = (__int64)v195;
            if ( !v156 )
              goto LABEL_475;
          }
          else
          {
            if ( !(_DWORD)src )
            {
              HIBYTE(v219) = 1;
              v145 = "cannot take address of numeric label after the function is defined";
LABEL_416:
              v217.m128i_i64[0] = (__int64)v145;
              v10 = 1;
              LOBYTE(v219) = 3;
              sub_11FD800(v5, v237, (__int64)&v217, 1);
LABEL_184:
              if ( v251 )
                j_j___libc_free_0_0(v251);
              v62 = sub_C33340();
              if ( v249 == v62 )
              {
                if ( v250 )
                {
                  v73 = &v250[3 * (_QWORD)*(v250 - 1)];
                  while ( v250 != v73 )
                  {
                    v73 -= 3;
                    if ( v62 == *v73 )
                      sub_969EE0((__int64)v73);
                    else
                      sub_C338F0((__int64)v73);
                  }
                  j_j_j___libc_free_0_0(v73 - 1);
                }
              }
              else
              {
                sub_C338F0((__int64)&v249);
              }
              if ( v247 > 0x40 && v246 )
                j_j___libc_free_0_0(v246);
              if ( v243 != v245 )
                j_j___libc_free_0(v243, v245[0] + 1LL);
              if ( v240 != v242 )
                j_j___libc_free_0(v240, v242[0] + 1LL);
              if ( v234 )
                j_j___libc_free_0_0(v234);
              if ( v62 == v232 )
              {
                if ( v233 )
                {
                  v72 = &v233[3 * (_QWORD)*(v233 - 1)];
                  while ( v233 != v72 )
                  {
                    v72 -= 3;
                    if ( v62 == *v72 )
                      sub_969EE0((__int64)v72);
                    else
                      sub_C338F0((__int64)v72);
                  }
                  j_j_j___libc_free_0_0(v72 - 1);
                }
              }
              else
              {
                sub_C338F0((__int64)&v232);
              }
              if ( v230 > 0x40 && v229 )
                j_j___libc_free_0_0(v229);
              if ( v226 != v228 )
                j_j___libc_free_0(v226, v228[0] + 1LL);
              if ( v223 != v225 )
                j_j___libc_free_0(v223, v225[0] + 1LL);
              return v10;
            }
            v156 = (_BYTE *)sub_1209B90(*((_QWORD *)v195 + 14), v240, v241);
            v157 = (__int64)v195;
            if ( !v156 || *v156 != 23 )
            {
LABEL_475:
              HIBYTE(v219) = 1;
              v145 = "referenced value is not a basic block";
              goto LABEL_416;
            }
          }
          v158 = sub_ACC1C0(v157, (__int64)v156);
          *(_DWORD *)a2 = 12;
          *(_QWORD *)(a2 + 136) = v158;
          goto LABEL_184;
        }
LABEL_182:
        v174 = v57;
        v59 = sub_12200E0(a1 + 160, (__int64)&v220);
        v60 = sub_12210B0(v59, (__int64)&src);
        if ( *v60 )
        {
LABEL_183:
          v61 = *v60;
          *(_DWORD *)a2 = 12;
          *(_QWORD *)(a2 + 136) = v61;
          goto LABEL_184;
        }
        if ( v174 )
        {
          if ( *(_BYTE *)(v174 + 8) != 14 )
          {
            sub_1207630(v207[0].m128i_i64, v174);
            sub_95D570(&v208, "type of blockaddress must be a pointer and not '", (__int64)v207);
            sub_94F930(&v213, (__int64)&v208, "'");
            v148 = *(_QWORD *)(a2 + 8);
            v217.m128i_i64[0] = (__int64)&v213;
            v219 = 260;
            sub_11FD800(v5, v148, (__int64)&v217, 1);
            sub_2240A30(&v213);
            sub_2240A30(&v208);
            v10 = 1;
            sub_2240A30(v207);
            goto LABEL_184;
          }
          v151 = *(_DWORD *)(v174 + 8) >> 8;
        }
        else
        {
          if ( !a3 )
            BUG();
          v151 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 8) + 8LL) + 8LL) >> 8;
        }
        v196 = v60;
        v152 = sub_BCB2B0(*a1);
        v213.m128i_i32[0] = v151;
        v153 = (_QWORD *)v152;
        v219 = 257;
        v213.m128i_i8[4] = 1;
        v154 = sub_BD2C40(88, unk_3F0FAE8);
        v60 = v196;
        v155 = v154;
        if ( v154 )
        {
          sub_B30000((__int64)v154, (__int64)a1[43], v153, 0, 7, 0, (__int64)&v217, 0, 0, v213.m128i_i64[0], 0);
          v60 = v196;
        }
        *v60 = v155;
        goto LABEL_183;
      case 0x192u:
        LOBYTE(v245[0]) = 0;
        *((_DWORD *)a1 + 60) = sub_1205200((__int64)(a1 + 22));
        v240 = v242;
        v243 = v245;
        LODWORD(src) = 0;
        v237 = 0;
        v239 = 0;
        v241 = 0;
        LOBYTE(v242[0]) = 0;
        v244 = 0;
        v247 = 1;
        v246 = 0;
        v248 = 0;
        v49 = sub_C33320();
        sub_C3B1B0((__int64)&v220, 0.0);
        sub_C407B0(&v249, v220.m128i_i64, v49);
        sub_C338F0((__int64)&v220);
        v251 = 0;
        v252 = 0;
        v10 = sub_1221570(a1, &src, a3, 0);
        if ( (_BYTE)v10 )
          goto LABEL_162;
        v50 = (int)src;
        if ( ((unsigned int)src & 0xFFFFFFFD) == 1 )
        {
          if ( (_DWORD)src == 1 )
          {
            v83 = *((_DWORD *)a1 + 304);
            v84 = a1[150];
            if ( !v83 )
              goto LABEL_296;
            v85 = v83 - 1;
            v86 = (v83 - 1) & (37 * v238);
            v87 = (unsigned int *)&v84[2 * v86];
            v88 = *v87;
            if ( v238 != (_DWORD)v88 )
            {
              while ( (_DWORD)v88 != -1 )
              {
                v86 = v85 & (v50 + v86);
                v87 = (unsigned int *)&v84[2 * v86];
                v88 = *v87;
                if ( v238 == (_DWORD)v88 )
                  goto LABEL_282;
                ++v50;
              }
              goto LABEL_296;
            }
LABEL_282:
            v89 = *((_QWORD *)v87 + 1);
            if ( !v89 )
            {
LABEL_296:
              v51 = a1 + 173;
LABEL_160:
              v52 = sub_1220BF0(v51, (__int64)&src);
              v53 = *v52;
              if ( !*v52 )
              {
                v111 = sub_BCB2B0(*a1);
                LOWORD(v223) = 257;
                v194 = (_QWORD *)v111;
                v217.m128i_i8[4] = 0;
                v112 = sub_BD2C40(88, unk_3F0FAE8);
                v113 = v112;
                if ( v112 )
                  sub_B30000((__int64)v112, (__int64)a1[43], v194, 0, 7, 0, (__int64)&v220, 0, 0, v217.m128i_i64[0], 0);
                *v52 = (__int64)v113;
                v53 = (__int64)v113;
              }
              goto LABEL_161;
            }
          }
          else
          {
            if ( a1 + 138 != (_QWORD **)sub_1213320((__int64)(a1 + 137), (__int64)&v240) )
            {
LABEL_159:
              v51 = a1 + 167;
              goto LABEL_160;
            }
            v88 = (__int64)v240;
            v89 = sub_BA8B30((__int64)a1[43], (__int64)v240, v241);
            if ( !v89 )
            {
              if ( (_DWORD)src != 1 )
                goto LABEL_159;
              goto LABEL_296;
            }
          }
          if ( *(_BYTE *)(*(_QWORD *)(v89 + 24) + 8LL) == 13 )
          {
            v53 = sub_ACC6E0(v89, v88, (__int64)v87);
LABEL_161:
            *(_QWORD *)(a2 + 136) = v53;
            *(_DWORD *)a2 = 12;
            goto LABEL_162;
          }
          BYTE1(v223) = 1;
          v76 = "expected a function, alias to function, or ifunc in dso_local_equivalent";
        }
        else
        {
          BYTE1(v223) = 1;
          v76 = "expected global value name in dso_local_equivalent";
        }
        v220.m128i_i64[0] = (__int64)v76;
        v10 = 1;
        LOBYTE(v223) = 3;
        sub_11FD800(v5, v237, (__int64)&v220, 1);
LABEL_162:
        if ( v251 )
          j_j___libc_free_0_0(v251);
        v54 = sub_C33340();
        if ( v249 == v54 )
        {
          if ( v250 )
          {
            v71 = &v250[3 * (_QWORD)*(v250 - 1)];
            while ( v250 != v71 )
            {
              v71 -= 3;
              if ( v54 == *v71 )
                sub_969EE0((__int64)v71);
              else
                sub_C338F0((__int64)v71);
            }
            j_j_j___libc_free_0_0(v71 - 1);
          }
        }
        else
        {
          sub_C338F0((__int64)&v249);
        }
        if ( v247 > 0x40 && v246 )
          j_j___libc_free_0_0(v246);
        if ( v243 != v245 )
          j_j___libc_free_0(v243, v245[0] + 1LL);
        if ( v240 != v242 )
          j_j___libc_free_0(v240, v242[0] + 1LL);
        return v10;
      case 0x193u:
        *((_DWORD *)a1 + 60) = sub_1205200((__int64)(a1 + 22));
        v10 = sub_1221570(a1, a2, a3, 0);
        if ( (_BYTE)v10 )
          return 1;
        if ( (*(_DWORD *)a2 & 0xFFFFFFFD) == 1 )
        {
          *(_BYTE *)(a2 + 152) = 1;
          return v10;
        }
        BYTE1(v240) = 1;
        v28 = "expected global value name in no_cfi";
        goto LABEL_68;
      case 0x194u:
        *((_DWORD *)a1 + 60) = sub_1205200((__int64)(a1 + 22));
        v217.m128i_i64[0] = 0;
        v220.m128i_i64[0] = 0;
        if ( (unsigned __int8)sub_120AFE0((__int64)a1, 12, "expected '(' in constant ptrauth expression")
          || (unsigned __int8)sub_1224A40(a1, &v208)
          || (unsigned __int8)sub_120AFE0((__int64)a1, 4, "expected comma in constant ptrauth expression")
          || (unsigned __int8)sub_1224A40(a1, &v213)
          || *((_DWORD *)a1 + 60) == 4
          && (unsigned __int8)sub_1205540((__int64)a1)
          && ((unsigned __int8)sub_1224A40(a1, &v217)
           || *((_DWORD *)a1 + 60) == 4
           && (unsigned __int8)sub_1205540((__int64)a1)
           && (unsigned __int8)sub_1224A40(a1, &v220)) )
        {
          return 1;
        }
        v10 = sub_120AFE0((__int64)a1, 13, "expected ')' in constant ptrauth expression");
        if ( (_BYTE)v10 )
          return 1;
        if ( *(_BYTE *)(*(_QWORD *)(v208.m128i_i64[0] + 8) + 8LL) == 14 )
        {
          v114 = v213.m128i_i64[0];
          if ( *(_BYTE *)v213.m128i_i64[0] == 17 && *(_DWORD *)(v213.m128i_i64[0] + 32) == 32 )
          {
            v136 = v217.m128i_i64[0];
            if ( v217.m128i_i64[0] )
            {
              if ( *(_BYTE *)v217.m128i_i64[0] != 17 || *(_DWORD *)(v217.m128i_i64[0] + 32) != 64 )
              {
                BYTE1(v240) = 1;
                v28 = "constant ptrauth integer discriminator must be i64 constant";
                goto LABEL_68;
              }
            }
            else
            {
              v160 = sub_BCB2E0(*a1);
              v136 = sub_ACD640(v160, 0, 0);
            }
            v161 = v220.m128i_i64[0];
            if ( v220.m128i_i64[0] )
            {
              if ( *(_BYTE *)(*(_QWORD *)(v220.m128i_i64[0] + 8) + 8LL) != 14 )
              {
                BYTE1(v240) = 1;
                v28 = "constant ptrauth address discriminator must be a pointer";
                goto LABEL_68;
              }
            }
            else
            {
              v197 = v136;
              v166 = (__int64 **)sub_BCE3C0(*a1, 0);
              v167 = sub_AC9EC0(v166);
              v136 = v197;
              v220.m128i_i64[0] = v167;
              v161 = v167;
            }
            v168 = sub_AD0290(v208.m128i_i64[0], v114, v136, v161);
            *(_DWORD *)a2 = 12;
            *(_QWORD *)(a2 + 136) = v168;
            return v10;
          }
          BYTE1(v240) = 1;
          v28 = "constant ptrauth key must be i32 constant";
        }
        else
        {
          BYTE1(v240) = 1;
          v28 = "constant ptrauth base pointer must be a pointer";
        }
LABEL_68:
        src = v28;
        v7 = *(_QWORD *)(a2 + 8);
        LOBYTE(v240) = 3;
        goto LABEL_6;
      default:
        break;
    }
  }
LABEL_5:
  src = "expected value token";
  LOWORD(v240) = 259;
LABEL_6:
  sub_11FD800(v5, v7, (__int64)&src, 1);
  return 1;
}
