// Function: sub_2872B00
// Address: 0x2872b00
//
void __fastcall sub_2872B00(__int64 *a1)
{
  __int64 v1; // r12
  char v2; // al
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rbx
  _QWORD *v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rcx
  _QWORD *v9; // rsi
  __int64 v10; // r8
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rsi
  _QWORD **v15; // r14
  unsigned int v16; // r15d
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r14
  __m128i v23; // xmm1
  __int64 v24; // r15
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rsi
  int v28; // eax
  __int64 v29; // rcx
  const __m128i *v30; // r10
  __int64 *v31; // r9
  __int64 v32; // r10
  char v33; // al
  __int64 v34; // r8
  __int64 v35; // r9
  int v36; // edx
  __int64 v37; // rcx
  _QWORD *v38; // r10
  __int64 v39; // rdx
  unsigned int v40; // eax
  __int64 v41; // rsi
  __int64 v42; // rsi
  unsigned __int64 v43; // rbx
  unsigned __int64 v44; // rax
  unsigned int v45; // edx
  _QWORD *v46; // rsi
  __int64 *v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rbx
  __int64 v52; // rsi
  __int64 *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // r15
  __int64 *v56; // rbx
  __int64 *v57; // r12
  __int64 v58; // rsi
  int v59; // edi
  char v60; // al
  __int64 *v61; // rcx
  __int64 v62; // rdx
  int v63; // esi
  __int64 *v64; // rax
  __int64 v65; // r8
  bool v66; // al
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  char v70; // al
  __int64 v71; // rdx
  __int64 v72; // r9
  __int64 v73; // r8
  __int64 *v74; // rax
  unsigned __int64 v75; // rsi
  char v76; // al
  unsigned __int64 v77; // r9
  __int64 v78; // rcx
  __int64 v79; // rdi
  __int64 v80; // r15
  __int64 v81; // r12
  __int64 v82; // r13
  unsigned __int64 v83; // rax
  __int64 v84; // rdi
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 v89; // rax
  char v90; // al
  __int64 v91; // rax
  __int64 v92; // rax
  int v93; // esi
  unsigned int v94; // r11d
  __int64 v95; // rdi
  int v96; // edi
  __int64 *v97; // rsi
  __int64 v98; // [rsp+10h] [rbp-240h]
  __int64 v99; // [rsp+10h] [rbp-240h]
  __int64 *v100; // [rsp+18h] [rbp-238h]
  char v101; // [rsp+27h] [rbp-229h]
  __m128i *v102; // [rsp+28h] [rbp-228h]
  __int64 v103; // [rsp+28h] [rbp-228h]
  _BYTE *v104; // [rsp+28h] [rbp-228h]
  unsigned int v105; // [rsp+28h] [rbp-228h]
  __int64 v106; // [rsp+28h] [rbp-228h]
  unsigned __int64 v107; // [rsp+30h] [rbp-220h]
  __int64 v108; // [rsp+38h] [rbp-218h]
  const __m128i *v109; // [rsp+38h] [rbp-218h]
  __int64 v110; // [rsp+38h] [rbp-218h]
  _QWORD *v111; // [rsp+38h] [rbp-218h]
  _QWORD *v112; // [rsp+38h] [rbp-218h]
  _QWORD *v113; // [rsp+38h] [rbp-218h]
  __int64 v114; // [rsp+38h] [rbp-218h]
  _QWORD *v115; // [rsp+38h] [rbp-218h]
  int v116; // [rsp+38h] [rbp-218h]
  _QWORD *v117; // [rsp+38h] [rbp-218h]
  __int64 v118; // [rsp+40h] [rbp-210h]
  __int64 v119; // [rsp+48h] [rbp-208h]
  _QWORD *v120; // [rsp+68h] [rbp-1E8h] BYREF
  __m128i v121; // [rsp+70h] [rbp-1E0h]
  __int64 v122; // [rsp+80h] [rbp-1D0h] BYREF
  __m128i v123; // [rsp+88h] [rbp-1C8h] BYREF
  __int64 v124; // [rsp+A0h] [rbp-1B0h] BYREF
  __int64 v125; // [rsp+A8h] [rbp-1A8h]
  __int64 v126; // [rsp+B0h] [rbp-1A0h]
  __int64 v127; // [rsp+B8h] [rbp-198h]
  __int64 v128; // [rsp+C0h] [rbp-190h] BYREF
  __int64 v129; // [rsp+C8h] [rbp-188h]
  __int64 v130; // [rsp+D0h] [rbp-180h]
  __int64 v131; // [rsp+D8h] [rbp-178h]
  char v132[8]; // [rsp+E0h] [rbp-170h] BYREF
  unsigned __int64 v133; // [rsp+E8h] [rbp-168h]
  char v134; // [rsp+FCh] [rbp-154h]
  char v135[16]; // [rsp+100h] [rbp-150h] BYREF
  __int128 v136; // [rsp+110h] [rbp-140h] BYREF
  __int64 v137; // [rsp+120h] [rbp-130h]
  _OWORD *v138; // [rsp+128h] [rbp-128h]
  __int128 v139; // [rsp+130h] [rbp-120h]
  _OWORD v140[2]; // [rsp+140h] [rbp-110h] BYREF
  __int64 v141; // [rsp+168h] [rbp-E8h]
  __int64 v142; // [rsp+170h] [rbp-E0h]
  char v143; // [rsp+178h] [rbp-D8h]
  __int64 v144; // [rsp+180h] [rbp-D0h] BYREF
  char *v145; // [rsp+188h] [rbp-C8h]
  __int64 v146; // [rsp+190h] [rbp-C0h]
  int v147; // [rsp+198h] [rbp-B8h]
  char v148; // [rsp+19Ch] [rbp-B4h]
  char v149; // [rsp+1A0h] [rbp-B0h] BYREF

  v1 = (__int64)a1;
  v2 = sub_DFA300(a1[6]);
  v148 = 1;
  v101 = v2;
  v145 = &v149;
  v3 = *a1;
  v144 = 0;
  v4 = *(_QWORD *)(v3 + 208);
  v146 = 16;
  v147 = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v118 = v3 + 200;
  if ( v4 != v3 + 200 )
  {
    while ( 1 )
    {
      if ( !v4 )
        BUG();
      v5 = *(_QWORD *)(v4 - 8);
      *(_QWORD *)&v136 = *(_QWORD *)(v4 + 40);
      if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
      {
        v6 = *(_QWORD **)(v5 - 8);
        v7 = (__int64)&v6[4 * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)];
      }
      else
      {
        v7 = v5;
        v6 = (_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
      }
      v9 = sub_284FF00(v6, v7, (__int64 *)&v136);
      if ( *(_BYTE *)(v1 + 36884) )
        break;
      if ( sub_C8CA60(v1 + 36856, (__int64)v9) )
      {
LABEL_10:
        v4 = *(_QWORD *)(v4 + 8);
        if ( v118 == v4 )
          goto LABEL_11;
      }
      else
      {
        v10 = *(_QWORD *)(v4 + 40);
LABEL_16:
        v108 = 0xFFFFFFFFLL;
        v15 = 0;
        v16 = 0;
        if ( (unsigned __int8)sub_2851CA0(*(_QWORD *)(v1 + 48), (char *)v5, v10, v8, v10) )
        {
          v16 = 2;
          v15 = (_QWORD **)sub_284F800(*(_QWORD *)(v1 + 48), v5, *(_QWORD *)(v4 + 40));
          v108 = v45;
        }
        v120 = (_QWORD *)sub_22AD250(*(_QWORD *)v1, v4 - 32);
        if ( !v120 )
          goto LABEL_10;
        sub_C8CD80((__int64)v132, (__int64)v135, v4 + 48, v17, v18, v19);
        if ( *(_BYTE *)v5 == 82 )
        {
          if ( v101 && *MEMORY[0xFFFFFFFFFFFFFFA0] == 82 && MEMORY[0xFFFFFFFFFFFFFFA0] == v5 )
            goto LABEL_38;
          if ( (*(_WORD *)(v5 + 2) & 0x3Fu) - 32 <= 1 )
          {
            v65 = *(_QWORD *)(v5 - 32);
            if ( *(_QWORD *)(v4 + 40) != v65 )
            {
LABEL_60:
              v103 = v65;
              v100 = sub_DD8400(*(_QWORD *)(v1 + 8), v65);
              v66 = sub_DADE90(*(_QWORD *)(v1 + 8), (__int64)v100, *(_QWORD *)(v1 + 56));
              v69 = v103;
              if ( !v66 )
                goto LABEL_61;
              v76 = sub_F80610(v1 + 80, (__int64)v100, v67, v68, v103, (__int64)v100);
              v69 = v103;
              if ( !v76 )
                goto LABEL_61;
              v77 = (unsigned __int64)v100;
              if ( *(_BYTE *)(*(_QWORD *)(v103 + 8) + 8LL) != 14
                || (v99 = v103,
                    v106 = sub_D97190(*(_QWORD *)(v1 + 8), (__int64)v100),
                    v91 = sub_D97190(*(_QWORD *)(v1 + 8), (__int64)v120),
                    v77 = (unsigned __int64)v100,
                    v69 = v99,
                    v106 == v91) )
              {
                v75 = sub_1055B50(v77, (__int64)v132, *(_QWORD *)(v1 + 8), 1);
                if ( !v75 )
                  goto LABEL_38;
LABEL_69:
                v16 = 3;
                v120 = sub_DCC810(*(__int64 **)(v1 + 8), v75, (__int64)v120, 0, 0);
              }
              else
              {
LABEL_61:
                v104 = (_BYTE *)v69;
                v70 = sub_D48480(*(_QWORD *)(v1 + 56), v69, v67, v68);
                v73 = (__int64)v104;
                if ( v70 )
                {
                  if ( *v104 <= 0x1Cu
                    || (v90 = sub_B19D00(
                                *(_QWORD *)(v1 + 16),
                                (__int64)v104,
                                **(_QWORD **)(*(_QWORD *)(v1 + 56) + 32LL)),
                        v73 = (__int64)v104,
                        v90) )
                  {
                    if ( *(_BYTE *)(*(_QWORD *)(v73 + 8) + 8LL) != 14 )
                    {
                      v74 = sub_DA3860(*(_QWORD **)(v1 + 8), v73);
                      v75 = sub_1055B50((unsigned __int64)v74, (__int64)v132, *(_QWORD *)(v1 + 8), 1);
                      if ( !v75 )
                        goto LABEL_38;
                      goto LABEL_69;
                    }
                  }
                }
              }
              v78 = *(unsigned int *)(v1 + 1104);
              v79 = v1 + 968;
              if ( *(_DWORD *)(v1 + 1104) )
              {
                v105 = v16;
                v80 = v1;
                v81 = 0;
                v98 = v4;
                v82 = v78;
                do
                {
                  v71 = *(_QWORD *)(*(_QWORD *)(v80 + 1096) + 8 * v81);
                  if ( v71 != -1 )
                  {
                    *(_QWORD *)&v136 = -v71;
                    sub_2872920(v79, (__int64 *)&v136, -v71, v78, v73, v72);
                  }
                  ++v81;
                }
                while ( v81 != v82 );
                v1 = v80;
                v4 = v98;
                v16 = v105;
              }
              *(_QWORD *)&v136 = -1;
              sub_2872920(v79, (__int64 *)&v136, v71, v78, v73, v72);
              goto LABEL_24;
            }
            v85 = *(_QWORD *)(v5 - 64);
            if ( v85 )
            {
              if ( v65 )
              {
                v86 = *(_QWORD *)(v5 - 24);
                **(_QWORD **)(v5 - 16) = v86;
                if ( v86 )
                  *(_QWORD *)(v86 + 16) = *(_QWORD *)(v5 - 16);
              }
              *(_QWORD *)(v5 - 32) = v85;
              v87 = *(_QWORD *)(v85 + 16);
              *(_QWORD *)(v5 - 24) = v87;
              if ( v87 )
                *(_QWORD *)(v87 + 16) = v5 - 24;
              *(_QWORD *)(v5 - 16) = v85 + 16;
              *(_QWORD *)(v85 + 16) = v5 - 32;
              goto LABEL_89;
            }
            if ( v65 )
            {
              v92 = *(_QWORD *)(v5 - 24);
              **(_QWORD **)(v5 - 16) = v92;
              if ( v92 )
                *(_QWORD *)(v92 + 16) = *(_QWORD *)(v5 - 16);
              *(_QWORD *)(v5 - 32) = 0;
LABEL_89:
              if ( *(_QWORD *)(v5 - 64) )
              {
                v88 = *(_QWORD *)(v5 - 56);
                **(_QWORD **)(v5 - 48) = v88;
                if ( v88 )
                  *(_QWORD *)(v88 + 16) = *(_QWORD *)(v5 - 48);
              }
              *(_QWORD *)(v5 - 64) = v65;
              if ( v65 )
              {
                v89 = *(_QWORD *)(v65 + 16);
                *(_QWORD *)(v5 - 56) = v89;
                if ( v89 )
                  *(_QWORD *)(v89 + 16) = v5 - 56;
                *(_QWORD *)(v5 - 48) = v65 + 16;
                *(_QWORD *)(v65 + 16) = v5 - 64;
              }
              v65 = *(_QWORD *)(v5 - 32);
            }
            *(_BYTE *)(v1 + 952) = 1;
            goto LABEL_60;
          }
        }
LABEL_24:
        sub_2857EE0((__int64)&v122, v1, (__int64 *)&v120, v16, v15, v108);
        v22 = v122;
        v139 = 0;
        v23 = _mm_loadu_si128(&v123);
        v24 = *(_QWORD *)(v1 + 1320) + 2184 * v122;
        v137 = 0;
        *(_QWORD *)&v139 = 2;
        v138 = v140;
        DWORD2(v139) = 0;
        BYTE12(v139) = 1;
        v136 = 0;
        memset(v140, 0, sizeof(v140));
        v25 = *(unsigned int *)(v24 + 64);
        v26 = *(unsigned int *)(v24 + 68);
        v121 = v23;
        v27 = v25 + 1;
        v28 = v25;
        if ( v25 + 1 > v26 )
        {
          v83 = *(_QWORD *)(v24 + 56);
          v84 = v24 + 56;
          if ( v83 > (unsigned __int64)&v136 || (v25 = v83 + 80 * v25, (unsigned __int64)&v136 >= v25) )
          {
            sub_2851200(v84, v27, v25, v26, v20, v21);
            v25 = *(unsigned int *)(v24 + 64);
            v29 = *(_QWORD *)(v24 + 56);
            v30 = (const __m128i *)&v136;
            v28 = *(_DWORD *)(v24 + 64);
          }
          else
          {
            v114 = *(_QWORD *)(v24 + 56);
            sub_2851200(v84, v27, v25, v26, v20, v21);
            v29 = *(_QWORD *)(v24 + 56);
            v25 = *(unsigned int *)(v24 + 64);
            v30 = (const __m128i *)((char *)&v136 + v29 - v114);
            v28 = *(_DWORD *)(v24 + 64);
          }
        }
        else
        {
          v29 = *(_QWORD *)(v24 + 56);
          v30 = (const __m128i *)&v136;
        }
        v31 = (__int64 *)(v29 + 80 * v25);
        if ( v31 )
        {
          v102 = (__m128i *)(v29 + 80 * v25);
          *v31 = v30->m128i_i64[0];
          v109 = v30;
          v31[1] = v30->m128i_i64[1];
          sub_C8CF70((__int64)(v31 + 2), v31 + 6, 2, (__int64)v30[3].m128i_i64, (__int64)v30[1].m128i_i64);
          v31 = (__int64 *)v102;
          v102[4] = _mm_loadu_si128(v109 + 4);
          v28 = *(_DWORD *)(v24 + 64);
        }
        *(_DWORD *)(v24 + 64) = v28 + 1;
        if ( !BYTE12(v139) )
          _libc_free((unsigned __int64)v138);
        v32 = *(_QWORD *)(v24 + 56) + 80LL * *(unsigned int *)(v24 + 64) - 80;
        *(_QWORD *)v32 = v5;
        *(_QWORD *)(v32 + 8) = *(_QWORD *)(v4 + 40);
        if ( (char *)(v32 + 16) != v132 )
        {
          v110 = v32;
          sub_C8CE00(v32 + 16, v32 + 48, (__int64)v132, v29, v20, (__int64)v31);
          v32 = v110;
        }
        v111 = (_QWORD *)v32;
        *(_QWORD *)(v32 + 64) = v121.m128i_i64[0];
        *(_BYTE *)(v32 + 72) = v121.m128i_i8[8];
        v33 = sub_2855480((_QWORD *)v32, *(_QWORD *)(v1 + 56));
        v36 = v131;
        *(_BYTE *)(v24 + 744) &= v33;
        v37 = v129;
        v38 = v111;
        if ( v36 )
        {
          v39 = (unsigned int)(v36 - 1);
          v40 = v39 & (((0xBF58476D1CE4E5B9LL * v22) >> 31) ^ (484763065 * v22));
          v41 = *(_QWORD *)(v129 + 8LL * v40);
          if ( v22 == v41 )
            goto LABEL_34;
          v59 = 1;
          while ( v41 != -1 )
          {
            v34 = (unsigned int)(v59 + 1);
            v40 = v39 & (v59 + v40);
            v41 = *(_QWORD *)(v129 + 8LL * v40);
            if ( v22 == v41 )
              goto LABEL_34;
            ++v59;
          }
        }
        v60 = sub_2855480(v111, *(_QWORD *)(v1 + 56));
        v38 = v111;
        if ( !v60 )
        {
          v61 = *(__int64 **)(v1 + 8);
          v62 = *(_QWORD *)(v1 + 56);
          *(_QWORD *)&v140[0] = 0x400000000LL;
          v136 = 0u;
          LOBYTE(v137) = 0;
          LOBYTE(v138) = 0;
          *(_QWORD *)&v139 = 0;
          *((_QWORD *)&v139 + 1) = (char *)v140 + 8;
          v141 = 0;
          v142 = 0;
          v143 = 0;
          sub_285B840((__int64)&v136, v120, v62, v61);
          sub_285BFD0(v1 + 1176, (__int64)&v136, (__int64)&v144, (__int64)&v124, v24, 0);
          v63 = v131;
          v38 = v111;
          if ( (_DWORD)v131 )
          {
            v34 = (unsigned int)(v131 - 1);
            v39 = ((unsigned int)((0xBF58476D1CE4E5B9LL * v22) >> 31) ^ (484763065 * (_DWORD)v22)) & ((_DWORD)v131 - 1);
            v64 = (__int64 *)(v129 + 8 * v39);
            v37 = *v64;
            if ( v22 == *v64 )
            {
LABEL_57:
              if ( *((_OWORD **)&v139 + 1) != (_OWORD *)((char *)v140 + 8) )
              {
                v113 = v38;
                _libc_free(*((unsigned __int64 *)&v139 + 1));
                v38 = v113;
              }
              goto LABEL_34;
            }
            v116 = 1;
            v35 = 0;
            while ( v37 != -1 )
            {
              if ( v37 == -2 && !v35 )
                v35 = (__int64)v64;
              v39 = (unsigned int)v34 & (v116 + (_DWORD)v39);
              v64 = (__int64 *)(v129 + 8LL * (unsigned int)v39);
              v37 = *v64;
              if ( v22 == *v64 )
                goto LABEL_57;
              ++v116;
            }
            v63 = v131;
            if ( v35 )
              v64 = (__int64 *)v35;
            ++v128;
            v39 = (unsigned int)(v130 + 1);
            if ( 4 * (int)v39 < (unsigned int)(3 * v131) )
            {
              v37 = (unsigned int)(v131 - HIDWORD(v130) - v39);
              if ( (unsigned int)v37 <= (unsigned int)v131 >> 3 )
              {
                v107 = ((0xBF58476D1CE4E5B9LL * v22) >> 31) ^ (0xBF58476D1CE4E5B9LL * v22);
                v117 = v38;
                sub_A32210((__int64)&v128, v131);
                if ( !(_DWORD)v131 )
                {
LABEL_149:
                  LODWORD(v130) = v130 + 1;
                  BUG();
                }
                v34 = (unsigned int)(v131 - 1);
                v35 = v129;
                v93 = 1;
                v38 = v117;
                v94 = v34 & v107;
                v39 = (unsigned int)(v130 + 1);
                v37 = 0;
                v64 = (__int64 *)(v129 + 8LL * ((unsigned int)v34 & (unsigned int)v107));
                v95 = *v64;
                if ( v22 != *v64 )
                {
                  while ( v95 != -1 )
                  {
                    if ( v95 == -2 && !v37 )
                      v37 = (__int64)v64;
                    v94 = v34 & (v93 + v94);
                    v64 = (__int64 *)(v129 + 8LL * v94);
                    v95 = *v64;
                    if ( v22 == *v64 )
                      goto LABEL_106;
                    ++v93;
                  }
                  if ( v37 )
                    v64 = (__int64 *)v37;
                }
              }
              goto LABEL_106;
            }
          }
          else
          {
            ++v128;
          }
          v115 = v38;
          sub_A32210((__int64)&v128, 2 * v63);
          if ( !(_DWORD)v131 )
            goto LABEL_149;
          v35 = (unsigned int)(v131 - 1);
          v38 = v115;
          v39 = (unsigned int)(v130 + 1);
          v37 = (unsigned int)v35 & ((unsigned int)((0xBF58476D1CE4E5B9LL * v22) >> 31) ^ (484763065 * (_DWORD)v22));
          v64 = (__int64 *)(v129 + 8 * v37);
          v34 = *v64;
          if ( v22 != *v64 )
          {
            v96 = 1;
            v97 = 0;
            while ( v34 != -1 )
            {
              if ( v34 == -2 && !v97 )
                v97 = v64;
              v37 = (unsigned int)v35 & (v96 + (_DWORD)v37);
              v64 = (__int64 *)(v129 + 8LL * (unsigned int)v37);
              v34 = *v64;
              if ( v22 == *v64 )
                goto LABEL_106;
              ++v96;
            }
            if ( v97 )
              v64 = v97;
          }
LABEL_106:
          LODWORD(v130) = v39;
          if ( *v64 != -1 )
            --HIDWORD(v130);
          *v64 = v22;
          goto LABEL_57;
        }
LABEL_34:
        v42 = *(_QWORD *)(v24 + 752);
        if ( !v42
          || (v112 = v38,
              v43 = sub_D97050(*(_QWORD *)(v1 + 8), v42),
              v44 = sub_D97050(*(_QWORD *)(v1 + 8), *(_QWORD *)(v112[1] + 8LL)),
              v38 = v112,
              v43 < v44) )
        {
          *(_QWORD *)(v24 + 752) = *(_QWORD *)(v38[1] + 8LL);
        }
        if ( !*(_DWORD *)(v24 + 768) )
        {
          v46 = v120;
          if ( !(unsigned __int8)sub_F80610(v1 + 80, (__int64)v120, v39, v37, v34, v35) )
            *(_BYTE *)(v24 + 745) = 1;
          v47 = *(__int64 **)(v1 + 8);
          v48 = *(_QWORD *)(v1 + 56);
          *(_QWORD *)&v140[0] = 0x400000000LL;
          v136 = 0u;
          LOBYTE(v137) = 0;
          LOBYTE(v138) = 0;
          *(_QWORD *)&v139 = 0;
          *((_QWORD *)&v139 + 1) = (char *)v140 + 8;
          v141 = 0;
          v142 = 0;
          v143 = 0;
          sub_285B840((__int64)&v136, v46, v48, v47);
          sub_2862B30(v1, v24, v22, (unsigned __int64)&v136, v49, v50);
          if ( *((_OWORD **)&v139 + 1) != (_OWORD *)((char *)v140 + 8) )
            _libc_free(*((unsigned __int64 *)&v139 + 1));
          v51 = *(_QWORD *)(v24 + 760) + 112LL * *(unsigned int *)(v24 + 768) - 112;
          v52 = *(_QWORD *)(v51 + 88);
          if ( v52 )
            sub_285AF50(v1 + 36280, v52, v22);
          v53 = *(__int64 **)(v51 + 40);
          v54 = *(unsigned int *)(v51 + 48);
          if ( v53 != &v53[v54] )
          {
            v119 = v1;
            v55 = v1 + 36280;
            v56 = &v53[v54];
            v57 = v53;
            do
            {
              v58 = *v57++;
              sub_285AF50(v55, v58, v22);
            }
            while ( v56 != v57 );
            v1 = v119;
          }
        }
LABEL_38:
        if ( v134 )
          goto LABEL_10;
        _libc_free(v133);
        v4 = *(_QWORD *)(v4 + 8);
        if ( v118 == v4 )
        {
LABEL_11:
          v13 = v129;
          v14 = 8LL * (unsigned int)v131;
          goto LABEL_12;
        }
      }
    }
    v11 = *(_QWORD **)(v1 + 36864);
    v12 = &v11[*(unsigned int *)(v1 + 36876)];
    if ( v11 != v12 )
    {
      while ( (_QWORD *)*v11 != v9 )
      {
        if ( v12 == ++v11 )
          goto LABEL_16;
      }
      goto LABEL_10;
    }
    goto LABEL_16;
  }
  v13 = 0;
  v14 = 0;
LABEL_12:
  sub_C7D6A0(v13, v14, 8);
  sub_C7D6A0(v125, 8LL * (unsigned int)v127, 8);
  if ( !v148 )
    _libc_free((unsigned __int64)v145);
}
