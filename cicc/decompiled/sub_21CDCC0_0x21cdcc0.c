// Function: sub_21CDCC0
// Address: 0x21cdcc0
//
void __fastcall sub_21CDCC0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 *a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  unsigned int v9; // r15d
  __int64 v12; // r12
  const __m128i *v13; // rax
  __int64 v14; // rsi
  __m128i v15; // xmm1
  __int64 v16; // rbx
  __int64 v17; // rdx
  bool v18; // cc
  _QWORD *v19; // rax
  unsigned int v20; // ebx
  char *v21; // rdx
  char v22; // al
  const void **v23; // rdx
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // r9
  __int64 v27; // r10
  int v28; // edx
  int v29; // r8d
  __int64 v30; // rdx
  __m128i v31; // xmm4
  const __m128i *v32; // rax
  const __m128i *v33; // r9
  __int64 v34; // rdx
  const __m128i *v35; // rax
  unsigned __int64 v36; // r11
  __int64 *v37; // rcx
  __int64 v38; // rdx
  int v39; // esi
  __m128i *v40; // rdx
  __int64 v41; // rdi
  unsigned __int32 v42; // r11d
  unsigned __int8 v43; // si
  __int64 v44; // r9
  int v45; // r8d
  __int64 v46; // r9
  __int64 v47; // r10
  __int64 v48; // rbx
  unsigned int v49; // r13d
  __int64 v50; // r14
  unsigned __int64 v51; // r15
  __int64 v52; // rax
  char *v53; // rdx
  char v54; // al
  const void **v55; // rdx
  __int64 v56; // rbx
  unsigned __int8 v57; // al
  __int64 v58; // rdx
  __int64 v59; // r9
  unsigned __int8 *v60; // rdx
  __int64 v61; // r8
  __int64 v62; // rax
  int v63; // r15d
  unsigned int v64; // r12d
  __int64 v65; // r13
  __int64 v66; // rbx
  unsigned __int8 *v67; // rax
  unsigned __int8 *v68; // rax
  __int64 v69; // r15
  __int64 v70; // rdx
  __int64 v71; // r8
  __int64 v72; // rax
  const __m128i *v73; // r9
  __int64 v74; // rdx
  _OWORD *v75; // rcx
  __int64 v76; // r14
  const __m128i *v77; // r12
  __int64 v78; // r13
  unsigned int v79; // r15d
  __int64 i; // rbx
  __int64 v81; // r9
  __int64 *v82; // rcx
  __int64 v83; // r9
  __int64 v84; // rax
  int v85; // r9d
  __int64 v86; // r12
  int v87; // r8d
  __int64 v88; // rax
  __int64 *v89; // r15
  unsigned int v90; // edx
  unsigned int v91; // r14d
  __int64 v92; // r13
  unsigned int v93; // ebx
  __int64 v94; // r12
  __int64 *v95; // rax
  _BYTE *v96; // rdx
  __int64 v97; // rdx
  __int64 *v98; // r8
  __int64 v99; // r9
  __int64 v100; // rax
  __int64 **v101; // rax
  __int64 v102; // rax
  __int64 *v103; // rax
  _BYTE *v104; // rdi
  unsigned __int8 *v105; // rdi
  __int64 v106; // rax
  const __m128i *v107; // rbx
  unsigned __int64 v108; // rcx
  const __m128i *v109; // r15
  unsigned __int64 v110; // rdx
  __m128 *v111; // rax
  __int32 v112; // esi
  __int64 v113; // rax
  int v114; // edx
  __int64 v115; // rbx
  int v116; // r8d
  int v117; // r9d
  __int64 v118; // r14
  __int64 v119; // rdx
  __int64 v120; // r15
  __int64 v121; // rdx
  __int64 *v122; // rdx
  __int64 v123; // rax
  __int64 *v124; // rax
  __int16 v125; // si
  __int64 v126; // rax
  __int64 v127; // rax
  unsigned __int8 *v128; // rax
  __int64 v129; // rax
  __int64 *v130; // rax
  int v131; // r8d
  int v132; // r9d
  __int64 *v133; // rdx
  __int64 *v134; // r15
  __int64 *v135; // r14
  __int64 v136; // rdx
  __int64 **v137; // rdx
  __int64 v138; // rax
  __int64 *v139; // rax
  __int64 v140; // rax
  const void **v141; // r8
  __int64 v142; // rax
  __int64 v143; // rdx
  const void **v144; // rdx
  int v145; // edx
  __int64 v146; // [rsp-18h] [rbp-1F8h]
  unsigned __int8 v147; // [rsp-10h] [rbp-1F0h]
  __int128 v148; // [rsp-10h] [rbp-1F0h]
  __int128 v149; // [rsp-10h] [rbp-1F0h]
  __int128 v150; // [rsp-10h] [rbp-1F0h]
  __int128 v151; // [rsp-10h] [rbp-1F0h]
  int v152; // [rsp-10h] [rbp-1F0h]
  __int64 v153; // [rsp-8h] [rbp-1E8h]
  __int64 v154; // [rsp-8h] [rbp-1E8h]
  int v155; // [rsp+8h] [rbp-1D8h]
  __int64 v156; // [rsp+10h] [rbp-1D0h]
  __int64 v157; // [rsp+10h] [rbp-1D0h]
  __int64 v158; // [rsp+18h] [rbp-1C8h]
  __int64 v159; // [rsp+18h] [rbp-1C8h]
  __int64 v160; // [rsp+18h] [rbp-1C8h]
  __int64 v161; // [rsp+18h] [rbp-1C8h]
  __int64 *v162; // [rsp+20h] [rbp-1C0h]
  __int64 v163; // [rsp+20h] [rbp-1C0h]
  const __m128i *v164; // [rsp+20h] [rbp-1C0h]
  unsigned int v165; // [rsp+28h] [rbp-1B8h]
  const __m128i *v166; // [rsp+30h] [rbp-1B0h]
  unsigned __int16 v167; // [rsp+38h] [rbp-1A8h]
  __int64 v168; // [rsp+38h] [rbp-1A8h]
  __int64 v169; // [rsp+38h] [rbp-1A8h]
  char v170; // [rsp+40h] [rbp-1A0h]
  __int64 v171; // [rsp+40h] [rbp-1A0h]
  __int64 v172; // [rsp+40h] [rbp-1A0h]
  __int64 v173; // [rsp+48h] [rbp-198h]
  __m128i v174; // [rsp+50h] [rbp-190h] BYREF
  __int64 v175; // [rsp+60h] [rbp-180h]
  __int64 v176; // [rsp+68h] [rbp-178h]
  __int64 v177; // [rsp+70h] [rbp-170h] BYREF
  int v178; // [rsp+78h] [rbp-168h]
  __int64 v179; // [rsp+80h] [rbp-160h] BYREF
  const void **v180; // [rsp+88h] [rbp-158h]
  __m128i v181; // [rsp+90h] [rbp-150h] BYREF
  _BYTE v182[32]; // [rsp+A0h] [rbp-140h] BYREF
  unsigned __int8 *v183; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v184; // [rsp+C8h] [rbp-118h]
  _BYTE v185[80]; // [rsp+D0h] [rbp-110h] BYREF
  __m128i v186; // [rsp+120h] [rbp-C0h] BYREF
  _OWORD v187[3]; // [rsp+130h] [rbp-B0h] BYREF
  char v188; // [rsp+160h] [rbp-80h]
  __int64 v189; // [rsp+168h] [rbp-78h]

  v12 = a1;
  v13 = *(const __m128i **)(a1 + 32);
  v14 = *(_QWORD *)(a1 + 72);
  v15 = _mm_loadu_si128(v13);
  v16 = v13[2].m128i_i64[1];
  v177 = v14;
  v174 = v15;
  if ( v14 )
    sub_1623A60((__int64)&v177, v14, 2);
  v17 = *(_QWORD *)(v16 + 88);
  v18 = *(_DWORD *)(v17 + 32) <= 0x40u;
  v178 = *(_DWORD *)(a1 + 64);
  v19 = *(_QWORD **)(v17 + 24);
  if ( !v18 )
    v19 = (_QWORD *)*v19;
  v20 = (unsigned int)v19;
  if ( (unsigned int)v19 <= 0x1001 )
  {
    if ( (unsigned int)v19 > 0xFCA )
    {
      switch ( (int)v19 )
      {
        case 4043:
        case 4057:
          sub_2172D80(a1, a2, a3, a7, *(double *)v15.m128i_i64, a9);
          goto LABEL_10;
        case 4060:
        case 4061:
        case 4062:
        case 4069:
        case 4070:
        case 4071:
          v21 = *(char **)(a1 + 40);
          v22 = *v21;
          v23 = (const void **)*((_QWORD *)v21 + 1);
          LOBYTE(v179) = v22;
          v180 = v23;
          if ( v22 )
          {
            if ( (unsigned __int8)(v22 - 14) <= 0x5Fu )
            {
              v165 = (unsigned __int16)word_435D740[(unsigned __int8)(v22 - 14)];
              switch ( v22 )
              {
                case 24:
                case 25:
                case 26:
                case 27:
                case 28:
                case 29:
                case 30:
                case 31:
                case 32:
                case 62:
                case 63:
                case 64:
                case 65:
                case 66:
                case 67:
                  v181.m128i_i8[0] = 3;
                  v181.m128i_i64[1] = 0;
                  break;
                case 33:
                case 34:
                case 35:
                case 36:
                case 37:
                case 38:
                case 39:
                case 40:
                case 68:
                case 69:
                case 70:
                case 71:
                case 72:
                case 73:
                  v181.m128i_i8[0] = 4;
                  v181.m128i_i64[1] = 0;
                  break;
                case 41:
                case 42:
                case 43:
                case 44:
                case 45:
                case 46:
                case 47:
                case 48:
                case 74:
                case 75:
                case 76:
                case 77:
                case 78:
                case 79:
                  v181.m128i_i8[0] = 5;
                  v181.m128i_i64[1] = 0;
                  break;
                case 49:
                case 50:
                case 51:
                case 52:
                case 53:
                case 54:
                case 80:
                case 81:
                case 82:
                case 83:
                case 84:
                case 85:
                  v181.m128i_i8[0] = 6;
                  v181.m128i_i64[1] = 0;
                  break;
                case 55:
                  v181.m128i_i8[0] = 7;
                  v181.m128i_i64[1] = 0;
                  break;
                case 86:
                case 87:
                case 88:
                case 98:
                case 99:
                case 100:
                  v181.m128i_i8[0] = 8;
                  v181.m128i_i64[1] = 0;
                  break;
                case 89:
                case 90:
                case 91:
                case 92:
                case 93:
                case 101:
                case 102:
                case 103:
                case 104:
                case 105:
                  v181.m128i_i8[0] = 9;
                  v181.m128i_i64[1] = 0;
                  break;
                case 94:
                case 95:
                case 96:
                case 97:
                case 106:
                case 107:
                case 108:
                case 109:
                  v181.m128i_i8[0] = 10;
                  v181.m128i_i64[1] = 0;
                  break;
                default:
                  v181.m128i_i8[0] = 2;
                  v181.m128i_i64[1] = 0;
                  break;
              }
              goto LABEL_89;
            }
          }
          else if ( sub_1F58D20((__int64)&v179) )
          {
            v165 = sub_1F58D30((__int64)&v179);
            v181.m128i_i8[0] = sub_1F596B0((__int64)&v179);
            v181.m128i_i64[1] = v24;
            if ( !v181.m128i_i8[0] )
            {
              v25 = sub_1F58D40((__int64)&v181);
              goto LABEL_17;
            }
LABEL_89:
            v25 = sub_1F3E310(&v181);
LABEL_17:
            v170 = 0;
            if ( v25 <= 0xF )
            {
              v181.m128i_i8[0] = 4;
              v181.m128i_i64[1] = 0;
              v170 = 1;
            }
            if ( v165 == 2 )
            {
              if ( v20 > 0xFDE )
              {
                if ( v20 - 4069 > 2 )
                  goto LABEL_10;
                v167 = 663;
              }
              else
              {
                v167 = 661;
              }
              v27 = sub_1D25E70(
                      (__int64)a2,
                      v181.m128i_u32[0],
                      v181.m128i_i64[1],
                      v181.m128i_u32[0],
                      v181.m128i_i64[1],
                      v26,
                      1,
                      0);
              v29 = v145;
            }
            else
            {
              if ( v165 != 4 )
                goto LABEL_10;
              if ( v20 <= 0xFDE )
              {
                v167 = 662;
              }
              else
              {
                v167 = 664;
                if ( v20 - 4069 > 2 )
                  goto LABEL_10;
              }
              a7 = _mm_load_si128(&v181);
              v188 = 1;
              v189 = 0;
              v186 = a7;
              v187[0] = a7;
              v187[1] = a7;
              v187[2] = a7;
              v27 = sub_1D25C30((__int64)a2, (unsigned __int8 *)&v186, 5);
              v29 = v28;
            }
            v30 = *(unsigned int *)(a1 + 56);
            v31 = _mm_load_si128(&v174);
            v186.m128i_i64[0] = (__int64)v187;
            v186.m128i_i64[1] = 0x800000001LL;
            v32 = *(const __m128i **)(a1 + 32);
            v30 *= 40;
            v187[0] = v31;
            v33 = (const __m128i *)((char *)v32 + v30);
            v34 = v30 - 80;
            v35 = v32 + 5;
            v36 = 0xCCCCCCCCCCCCCCCDLL * (v34 >> 3);
            if ( (unsigned __int64)v34 > 0x118 )
            {
              v155 = v29;
              v157 = v27;
              v164 = v35;
              v166 = v33;
              v174.m128i_i64[0] = 0xCCCCCCCCCCCCCCCDLL * (v34 >> 3);
              sub_16CD150((__int64)&v186, v187, v36 + 1, 16, v29, (int)v33);
              v39 = v186.m128i_i32[2];
              v37 = (__int64 *)v186.m128i_i64[0];
              LODWORD(v36) = v174.m128i_i32[0];
              v33 = v166;
              v35 = v164;
              v27 = v157;
              v29 = v155;
              v38 = 2LL * v186.m128i_u32[2];
            }
            else
            {
              v37 = (__int64 *)v187;
              v38 = 2;
              v39 = 1;
            }
            v40 = (__m128i *)&v37[v38];
            if ( v35 != v33 )
            {
              do
              {
                if ( v40 )
                  *v40 = _mm_loadu_si128(v35);
                v35 = (const __m128i *)((char *)v35 + 40);
                ++v40;
              }
              while ( v33 != v35 );
              v37 = (__int64 *)v186.m128i_i64[0];
              v39 = v186.m128i_i32[2];
            }
            v41 = *(_QWORD *)(a1 + 96);
            v42 = v39 + v36;
            v43 = *(_BYTE *)(v12 + 88);
            v186.m128i_i32[2] = v42;
            v44 = *(_QWORD *)(v12 + 104);
            v174.m128i_i64[0] = (__int64)&v177;
            v168 = sub_1D24DC0(a2, v167, (__int64)&v177, v27, v29, v44, v37, v42, v43, v41);
            v183 = v185;
            v184 = 0x400000000LL;
            v47 = v158;
            v48 = 0;
            v162 = a2;
            v156 = a3;
            v49 = v9;
            do
            {
              v50 = v168;
              v51 = (unsigned int)v48;
              if ( byte_4FD3A00 )
              {
                v52 = *(_QWORD *)(v168 + 40) + 16 * v48;
                switch ( *(_BYTE *)v52 )
                {
                  case 2:
                    v125 = 616;
                    goto LABEL_91;
                  case 4:
                    v125 = 614;
                    goto LABEL_91;
                  case 5:
                    v125 = 618;
                    goto LABEL_91;
                  case 6:
                    v125 = 620;
                    goto LABEL_91;
                  case 9:
                    v125 = 479;
                    goto LABEL_91;
                  case 0xA:
                    v125 = 481;
LABEL_91:
                    *((_QWORD *)&v149 + 1) = (unsigned int)v48;
                    LOBYTE(v49) = *(_BYTE *)v52;
                    *(_QWORD *)&v149 = v168;
                    v159 = v47;
                    v126 = sub_1D2CC80(v162, v125, v174.m128i_i64[0], v49, *(_QWORD *)(v52 + 8), v46, v149);
                    v47 = v159;
                    v50 = v126;
                    v51 = 0;
                    break;
                  default:
                    goto LABEL_101;
                }
              }
              if ( v170 )
              {
                if ( (_BYTE)v179 )
                {
                  switch ( (char)v179 )
                  {
                    case 14:
                    case 15:
                    case 16:
                    case 17:
                    case 18:
                    case 19:
                    case 20:
                    case 21:
                    case 22:
                    case 23:
                    case 56:
                    case 57:
                    case 58:
                    case 59:
                    case 60:
                    case 61:
                      LOBYTE(v140) = 2;
                      break;
                    case 24:
                    case 25:
                    case 26:
                    case 27:
                    case 28:
                    case 29:
                    case 30:
                    case 31:
                    case 32:
                    case 62:
                    case 63:
                    case 64:
                    case 65:
                    case 66:
                    case 67:
                      LOBYTE(v140) = 3;
                      break;
                    case 33:
                    case 34:
                    case 35:
                    case 36:
                    case 37:
                    case 38:
                    case 39:
                    case 40:
                    case 68:
                    case 69:
                    case 70:
                    case 71:
                    case 72:
                    case 73:
                      LOBYTE(v140) = 4;
                      break;
                    case 41:
                    case 42:
                    case 43:
                    case 44:
                    case 45:
                    case 46:
                    case 47:
                    case 48:
                    case 74:
                    case 75:
                    case 76:
                    case 77:
                    case 78:
                    case 79:
                      LOBYTE(v140) = 5;
                      break;
                    case 49:
                    case 50:
                    case 51:
                    case 52:
                    case 53:
                    case 54:
                    case 80:
                    case 81:
                    case 82:
                    case 83:
                    case 84:
                    case 85:
                      LOBYTE(v140) = 6;
                      break;
                    case 55:
                      LOBYTE(v140) = 7;
                      break;
                    case 86:
                    case 87:
                    case 88:
                    case 98:
                    case 99:
                    case 100:
                      LOBYTE(v140) = 8;
                      break;
                    case 89:
                    case 90:
                    case 91:
                    case 92:
                    case 93:
                    case 101:
                    case 102:
                    case 103:
                    case 104:
                    case 105:
                      LOBYTE(v140) = 9;
                      break;
                    case 94:
                    case 95:
                    case 96:
                    case 97:
                    case 106:
                    case 107:
                    case 108:
                    case 109:
                      LOBYTE(v140) = 10;
                      break;
                    default:
                      BUG();
                  }
                  v141 = 0;
                }
                else
                {
                  LOBYTE(v140) = sub_1F596B0((__int64)&v179);
                  v47 = v140;
                  v141 = v144;
                }
                *((_QWORD *)&v151 + 1) = v51;
                LOBYTE(v47) = v140;
                *(_QWORD *)&v151 = v50;
                v160 = v47;
                v142 = sub_1D309E0(
                         v162,
                         145,
                         v174.m128i_i64[0],
                         (unsigned int)v47,
                         v141,
                         0,
                         *(double *)a7.m128i_i64,
                         *(double *)v15.m128i_i64,
                         *(double *)a9.m128i_i64,
                         v151);
                v45 = v152;
                v47 = v160;
                v175 = v142;
                v50 = v142;
                v46 = v154;
                v176 = v143;
                v51 = (unsigned int)v143 | v51 & 0xFFFFFFFF00000000LL;
              }
              v127 = (unsigned int)v184;
              if ( (unsigned int)v184 >= HIDWORD(v184) )
              {
                v161 = v47;
                sub_16CD150((__int64)&v183, v185, 0, 16, v45, v46);
                v127 = (unsigned int)v184;
                v47 = v161;
              }
              v128 = &v183[16 * v127];
              ++v48;
              *(_QWORD *)v128 = v50;
              *((_QWORD *)v128 + 1) = v51;
              v129 = (unsigned int)(v184 + 1);
              LODWORD(v184) = v184 + 1;
            }
            while ( v165 != v48 );
            *((_QWORD *)&v150 + 1) = v129;
            *(_QWORD *)&v150 = v183;
            v130 = sub_1D359D0(
                     v162,
                     104,
                     v174.m128i_i64[0],
                     v179,
                     v180,
                     0,
                     *(double *)a7.m128i_i64,
                     *(double *)v15.m128i_i64,
                     a9,
                     v150);
            v134 = v133;
            v135 = v130;
            v136 = *(unsigned int *)(v156 + 8);
            if ( (unsigned int)v136 >= *(_DWORD *)(v156 + 12) )
            {
              sub_16CD150(v156, (const void *)(v156 + 16), 0, 16, v131, v132);
              v136 = *(unsigned int *)(v156 + 8);
            }
            v137 = (__int64 **)(*(_QWORD *)v156 + 16 * v136);
            *v137 = v135;
            v137[1] = v134;
            v138 = (unsigned int)(*(_DWORD *)(v156 + 8) + 1);
            *(_DWORD *)(v156 + 8) = v138;
            if ( *(_DWORD *)(v156 + 12) <= (unsigned int)v138 )
            {
              sub_16CD150(v156, (const void *)(v156 + 16), 0, 16, v131, v132);
              v138 = *(unsigned int *)(v156 + 8);
            }
            v139 = (__int64 *)(*(_QWORD *)v156 + 16 * v138);
            v139[1] = v165;
            *v139 = v168;
            ++*(_DWORD *)(v156 + 8);
LABEL_101:
            if ( v183 != v185 )
              _libc_free((unsigned __int64)v183);
LABEL_86:
            v105 = (unsigned __int8 *)v186.m128i_i64[0];
            if ( (_OWORD *)v186.m128i_i64[0] != v187 )
              goto LABEL_68;
            goto LABEL_10;
          }
          v106 = *(unsigned int *)(a1 + 56);
          v107 = *(const __m128i **)(a1 + 32);
          v186.m128i_i64[0] = (__int64)v187;
          v108 = 40 * v106;
          v186.m128i_i64[1] = 0x400000000LL;
          v109 = (const __m128i *)((char *)v107 + 40 * v106);
          v110 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v106) >> 3);
          v111 = (__m128 *)v187;
          v112 = 0;
          if ( v108 > 0xA0 )
          {
            v174.m128i_i64[0] = v110;
            sub_16CD150((__int64)&v186, v187, v110, 16, a5, (int)a6);
            v112 = v186.m128i_i32[2];
            LODWORD(v110) = v174.m128i_i32[0];
            v111 = (__m128 *)(v186.m128i_i64[0] + 16LL * v186.m128i_u32[2]);
          }
          if ( v107 != v109 )
          {
            do
            {
              if ( v111 )
              {
                a9 = _mm_loadu_si128(v107);
                *v111 = (__m128)a9;
              }
              v107 = (const __m128i *)((char *)v107 + 40);
              ++v111;
            }
            while ( v109 != v107 );
            v112 = v186.m128i_i32[2];
          }
          v186.m128i_i32[2] = v112 + v110;
          v113 = sub_1D252B0((__int64)a2, 4, 0, 1, 0);
          v115 = sub_1D24DC0(
                   a2,
                   0x2Cu,
                   (__int64)&v177,
                   v113,
                   v114,
                   *(_QWORD *)(a1 + 104),
                   (__int64 *)v186.m128i_i64[0],
                   v186.m128i_u32[2],
                   3u,
                   0);
          v118 = sub_1D309E0(
                   a2,
                   145,
                   (__int64)&v177,
                   3,
                   0,
                   0,
                   *(double *)a7.m128i_i64,
                   *(double *)v15.m128i_i64,
                   *(double *)a9.m128i_i64,
                   (unsigned __int64)v115);
          v120 = v119;
          v121 = *(unsigned int *)(a3 + 8);
          if ( (unsigned int)v121 >= *(_DWORD *)(a3 + 12) )
          {
            sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v116, v117);
            v121 = *(unsigned int *)(a3 + 8);
          }
          v122 = (__int64 *)(*(_QWORD *)a3 + 16 * v121);
          *v122 = v118;
          v122[1] = v120;
          v123 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
          *(_DWORD *)(a3 + 8) = v123;
          if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v123 )
          {
            sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v116, v117);
            v123 = *(unsigned int *)(a3 + 8);
          }
          v124 = (__int64 *)(*(_QWORD *)a3 + 16 * v123);
          *v124 = v115;
          v124[1] = 1;
          ++*(_DWORD *)(a3 + 8);
          goto LABEL_86;
        case 4096:
        case 4097:
          goto LABEL_34;
        default:
          break;
      }
    }
LABEL_9:
    sub_2179030((__int64 *)a1, (__int64 **)a2, a3, a7, v15, a9, a4, a5, a6);
    goto LABEL_10;
  }
  if ( (_DWORD)v19 != 4175 )
    goto LABEL_9;
LABEL_34:
  v53 = *(char **)(a1 + 40);
  v54 = *v53;
  v55 = (const void **)*((_QWORD *)v53 + 1);
  LOBYTE(v179) = v54;
  v180 = v55;
  if ( v54 )
  {
    if ( (unsigned __int8)(v54 - 14) > 0x5Fu )
      goto LABEL_10;
    v56 = (unsigned __int16)word_435D740[(unsigned __int8)(v54 - 14)];
    switch ( v54 )
    {
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v57 = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v57 = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v57 = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v57 = 6;
        break;
      case 55:
        v57 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v57 = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v57 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v57 = 10;
        break;
      default:
        v57 = 2;
        break;
    }
    v59 = 0;
  }
  else
  {
    if ( !sub_1F58D20((__int64)&v179) )
      goto LABEL_10;
    v56 = (unsigned int)sub_1F58D30((__int64)&v179);
    v57 = sub_1F596B0((__int64)&v179);
    v59 = v58;
  }
  v60 = v185;
  v61 = v57;
  v183 = v185;
  v184 = 0x500000000LL;
  if ( (_DWORD)v56 )
  {
    v174.m128i_i64[0] = (__int64)a2;
    v62 = 0;
    v63 = 0;
    v173 = a3;
    v64 = v56;
    v65 = v61;
    v66 = v59;
    while ( 1 )
    {
      ++v63;
      v67 = &v60[16 * v62];
      *(_QWORD *)v67 = v65;
      *((_QWORD *)v67 + 1) = v66;
      v62 = (unsigned int)(v184 + 1);
      LODWORD(v184) = v184 + 1;
      if ( v63 == v64 )
        break;
      if ( HIDWORD(v184) <= (unsigned int)v62 )
      {
        sub_16CD150((__int64)&v183, v185, 0, 16, v61, v59);
        v62 = (unsigned int)v184;
      }
      v60 = v183;
    }
    v56 = v64;
    a3 = v173;
    v12 = a1;
    a2 = (__int64 *)v174.m128i_i64[0];
    if ( HIDWORD(v184) <= (unsigned int)v62 )
    {
      v174.m128i_i64[0] = 1;
      sub_16CD150((__int64)&v183, v185, 0, 16, 1, v59);
      v62 = (unsigned int)v184;
    }
  }
  else
  {
    v62 = 0;
  }
  v68 = &v183[16 * v62];
  *(_QWORD *)v68 = 1;
  *((_QWORD *)v68 + 1) = 0;
  LODWORD(v184) = v184 + 1;
  v69 = sub_1D25C30((__int64)a2, v183, (unsigned int)v184);
  v71 = v70;
  v186.m128i_i64[0] = (__int64)v187;
  v186.m128i_i64[1] = 0x800000000LL;
  v72 = *(unsigned int *)(v12 + 56);
  if ( (_DWORD)v72 )
  {
    v73 = *(const __m128i **)(v12 + 32);
    v74 = 0;
    v174.m128i_i64[0] = (__int64)a2;
    v75 = v187;
    v76 = v12;
    v171 = a3;
    v77 = v73;
    v78 = 40 * v72;
    v169 = v69;
    v79 = v56;
    for ( i = 40; ; i += 40 )
    {
      a7 = _mm_loadu_si128(v77);
      v75[v74] = a7;
      v74 = (unsigned int)++v186.m128i_i32[2];
      if ( v78 == i )
        break;
      v77 = (const __m128i *)(i + *(_QWORD *)(v76 + 32));
      if ( v186.m128i_i32[3] <= (unsigned int)v74 )
      {
        v163 = v71;
        sub_16CD150((__int64)&v186, v187, 0, 16, v71, (int)v73);
        v74 = v186.m128i_u32[2];
        v71 = v163;
      }
      v75 = (_OWORD *)v186.m128i_i64[0];
    }
    v56 = v79;
    v12 = v76;
    a3 = v171;
    v81 = (unsigned int)v74;
    a2 = (__int64 *)v174.m128i_i64[0];
    v69 = v169;
    v82 = (__int64 *)v186.m128i_i64[0];
  }
  else
  {
    v82 = (__int64 *)v187;
    v81 = 0;
  }
  v153 = *(_QWORD *)(v12 + 96);
  v147 = *(_BYTE *)(v12 + 88);
  v146 = v81;
  v83 = *(_QWORD *)(v12 + 104);
  v174.m128i_i64[0] = (__int64)&v177;
  v84 = sub_1D24DC0(a2, 0x2Cu, (__int64)&v177, v69, v71, v83, v82, v146, v147, v153);
  v181.m128i_i64[0] = (__int64)v182;
  v86 = v84;
  v181.m128i_i64[1] = 0x200000000LL;
  if ( (_DWORD)v56 )
  {
    v87 = 0;
    v172 = a3;
    v88 = 0;
    v89 = a2;
    v90 = 2;
    v91 = v56;
    v92 = v86;
    v93 = 0;
    while ( 1 )
    {
      v94 = v93;
      if ( (unsigned int)v88 >= v90 )
      {
        sub_16CD150((__int64)&v181, v182, 0, 16, v87, v85);
        v88 = v181.m128i_u32[2];
      }
      v95 = (__int64 *)(v181.m128i_i64[0] + 16 * v88);
      ++v93;
      *v95 = v92;
      v95[1] = v94;
      v88 = (unsigned int)++v181.m128i_i32[2];
      if ( v93 == v91 )
        break;
      v90 = v181.m128i_u32[3];
    }
    v86 = v92;
    v96 = (_BYTE *)v181.m128i_i64[0];
    a3 = v172;
    v56 = v91;
    a2 = v89;
  }
  else
  {
    v88 = 0;
    v96 = v182;
  }
  *((_QWORD *)&v148 + 1) = v88;
  *(_QWORD *)&v148 = v96;
  v98 = sub_1D359D0(
          a2,
          104,
          v174.m128i_i64[0],
          (unsigned int)v179,
          v180,
          0,
          *(double *)a7.m128i_i64,
          *(double *)v15.m128i_i64,
          a9,
          v148);
  v99 = v97;
  v100 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v100 >= *(_DWORD *)(a3 + 12) )
  {
    v174.m128i_i64[0] = (__int64)v98;
    v174.m128i_i64[1] = v97;
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, (int)v98, v97);
    v100 = *(unsigned int *)(a3 + 8);
    v99 = v174.m128i_i64[1];
    v98 = (__int64 *)v174.m128i_i64[0];
  }
  v101 = (__int64 **)(*(_QWORD *)a3 + 16 * v100);
  *v101 = v98;
  v101[1] = (__int64 *)v99;
  v102 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v102;
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v102 )
  {
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, (int)v98, v99);
    v102 = *(unsigned int *)(a3 + 8);
  }
  v103 = (__int64 *)(*(_QWORD *)a3 + 16 * v102);
  *v103 = v86;
  v103[1] = v56;
  v104 = (_BYTE *)v181.m128i_i64[0];
  ++*(_DWORD *)(a3 + 8);
  if ( v104 != v182 )
    _libc_free((unsigned __int64)v104);
  if ( (_OWORD *)v186.m128i_i64[0] != v187 )
    _libc_free(v186.m128i_u64[0]);
  v105 = v183;
  if ( v183 != v185 )
LABEL_68:
    _libc_free((unsigned __int64)v105);
LABEL_10:
  if ( v177 )
    sub_161E7C0((__int64)&v177, v177);
}
