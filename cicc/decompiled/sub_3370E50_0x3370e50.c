// Function: sub_3370E50
// Address: 0x3370e50
//
__int64 __fastcall sub_3370E50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, int *a7)
{
  unsigned int v7; // r14d
  __int64 v8; // r13
  __int64 v10; // r10
  __int128 *v12; // r12
  __int64 v13; // rbx
  _BYTE *v14; // rax
  _BYTE *v15; // rdx
  _BYTE *i; // rdx
  __int64 v17; // rax
  unsigned int v18; // r14d
  __int64 v19; // r15
  unsigned __int16 v20; // cx
  __int64 v21; // r12
  __int64 v22; // rbx
  __int64 (__fastcall *v23)(__int64, __int64, __int64, __int64, unsigned __int64); // rax
  __int64 v24; // rax
  const __m128i *v25; // rax
  const __m128i *j; // rbx
  __int128 *v27; // r10
  __int64 v28; // r11
  unsigned int *v29; // rax
  __int64 v30; // r14
  int v31; // eax
  int v32; // edx
  int v33; // r12d
  __int64 v34; // rdx
  int v35; // r9d
  __int64 v36; // r12
  unsigned int v37; // edx
  __int64 v38; // r14
  __m128i *v39; // rax
  int v40; // eax
  unsigned __int16 v41; // dx
  __int64 v42; // rax
  __int64 v43; // rbx
  unsigned __int16 v44; // cx
  __int64 v45; // rdx
  unsigned int v46; // eax
  unsigned __int64 v47; // rax
  unsigned int v48; // eax
  int v49; // edx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rbx
  __int128 v53; // rax
  int v54; // r9d
  int v55; // esi
  int v56; // ecx
  __int64 v57; // r11
  __int128 *v58; // r10
  __int64 v59; // rax
  __int32 v60; // edx
  __int32 v61; // ebx
  __int64 v62; // rdx
  const __m128i *v63; // rax
  __int64 v64; // rax
  int v65; // edx
  int v66; // edi
  __int64 v67; // rdx
  unsigned __int64 v68; // rax
  unsigned __int64 v69; // r14
  int v70; // eax
  int v71; // edx
  int v72; // r9d
  unsigned int v73; // r14d
  __int64 v74; // rax
  int v75; // eax
  int v76; // edx
  int v77; // r12d
  __int64 v78; // rdx
  int v79; // r9d
  __int64 v80; // rax
  unsigned int v81; // edx
  int v82; // eax
  __int64 v83; // rax
  __int32 v84; // edx
  __int32 v85; // ebx
  __int64 v86; // rdx
  const __m128i *v87; // rax
  unsigned int v88; // esi
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rdi
  __int64 (__fastcall *v92)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v93; // rax
  unsigned __int64 v94; // rbx
  __int64 v95; // rax
  __int64 (__fastcall *v96)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v97; // rax
  unsigned __int64 v98; // rbx
  __int64 (__fastcall *v99)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v100; // rax
  unsigned __int64 v101; // rcx
  unsigned __int64 v102; // rdx
  unsigned __int64 v103; // rdx
  unsigned __int64 v104; // rdx
  __int128 v105; // [rsp-20h] [rbp-310h]
  __int128 v106; // [rsp-10h] [rbp-300h]
  __int128 v107; // [rsp-10h] [rbp-300h]
  __int128 v108; // [rsp-10h] [rbp-300h]
  __int64 v109; // [rsp-8h] [rbp-2F8h]
  __int64 v110; // [rsp+8h] [rbp-2E8h]
  __int64 v111; // [rsp+10h] [rbp-2E0h]
  __int64 v112; // [rsp+20h] [rbp-2D0h]
  __int16 v113; // [rsp+32h] [rbp-2BEh]
  __int64 v114; // [rsp+38h] [rbp-2B8h]
  __int64 v115; // [rsp+40h] [rbp-2B0h]
  __int64 v116; // [rsp+48h] [rbp-2A8h]
  __int64 v117; // [rsp+50h] [rbp-2A0h]
  __m128i v118; // [rsp+60h] [rbp-290h]
  __int128 *v119; // [rsp+70h] [rbp-280h]
  __int64 v120; // [rsp+78h] [rbp-278h]
  __int64 v122; // [rsp+88h] [rbp-268h]
  __int64 v123; // [rsp+98h] [rbp-258h]
  unsigned int v124; // [rsp+B0h] [rbp-240h]
  unsigned int v125; // [rsp+B4h] [rbp-23Ch]
  __int128 *v127; // [rsp+C0h] [rbp-230h]
  unsigned int v128; // [rsp+C8h] [rbp-228h]
  __int64 v129; // [rsp+C8h] [rbp-228h]
  int v130; // [rsp+D0h] [rbp-220h]
  __int128 *v131; // [rsp+D0h] [rbp-220h]
  int v132; // [rsp+D8h] [rbp-218h]
  int v133; // [rsp+D8h] [rbp-218h]
  __int64 v134; // [rsp+D8h] [rbp-218h]
  __int64 v135; // [rsp+E0h] [rbp-210h]
  __int64 v136; // [rsp+E0h] [rbp-210h]
  __int64 v137; // [rsp+E0h] [rbp-210h]
  __int64 v138; // [rsp+E0h] [rbp-210h]
  __int128 *v139; // [rsp+E0h] [rbp-210h]
  __int64 v140; // [rsp+E0h] [rbp-210h]
  __int64 v141; // [rsp+E0h] [rbp-210h]
  __int64 v142; // [rsp+E8h] [rbp-208h]
  int v143; // [rsp+E8h] [rbp-208h]
  __int128 *v144; // [rsp+E8h] [rbp-208h]
  __int128 *v145; // [rsp+E8h] [rbp-208h]
  __int128 *v146; // [rsp+E8h] [rbp-208h]
  int v147; // [rsp+E8h] [rbp-208h]
  __int64 v148; // [rsp+E8h] [rbp-208h]
  __int128 *v149; // [rsp+E8h] [rbp-208h]
  __int128 *v150; // [rsp+E8h] [rbp-208h]
  __int64 v151; // [rsp+F0h] [rbp-200h]
  unsigned int v152; // [rsp+F8h] [rbp-1F8h]
  unsigned __int16 v153; // [rsp+FEh] [rbp-1F2h]
  unsigned int v155; // [rsp+128h] [rbp-1C8h]
  __int128 *v156; // [rsp+130h] [rbp-1C0h]
  unsigned __int64 v157; // [rsp+138h] [rbp-1B8h]
  unsigned __int16 v158; // [rsp+17Ah] [rbp-176h] BYREF
  unsigned int v159; // [rsp+17Ch] [rbp-174h] BYREF
  __int64 v160; // [rsp+180h] [rbp-170h] BYREF
  __int64 v161; // [rsp+188h] [rbp-168h]
  __int64 v162; // [rsp+190h] [rbp-160h] BYREF
  unsigned __int64 v163; // [rsp+198h] [rbp-158h]
  __int64 v164; // [rsp+1A0h] [rbp-150h] BYREF
  unsigned __int64 v165; // [rsp+1A8h] [rbp-148h]
  __int64 v166; // [rsp+1B0h] [rbp-140h] BYREF
  __int64 v167; // [rsp+1B8h] [rbp-138h]
  __int64 v168; // [rsp+1C0h] [rbp-130h]
  __int64 v169; // [rsp+1C8h] [rbp-128h]
  __int64 v170; // [rsp+1D0h] [rbp-120h]
  int v171; // [rsp+1D8h] [rbp-118h]
  _BYTE *v172; // [rsp+1E0h] [rbp-110h] BYREF
  __int64 v173; // [rsp+1E8h] [rbp-108h]
  _BYTE v174[64]; // [rsp+1F0h] [rbp-100h] BYREF
  const __m128i *v175; // [rsp+230h] [rbp-C0h] BYREF
  __int64 v176; // [rsp+238h] [rbp-B8h]
  _BYTE v177[176]; // [rsp+240h] [rbp-B0h] BYREF

  v7 = *(_DWORD *)(a1 + 8);
  if ( !v7 )
    return 0;
  v10 = a1;
  v12 = (__int128 *)a5;
  v13 = v7;
  v114 = *(_QWORD *)(a2 + 16);
  v14 = v174;
  v15 = v174;
  v172 = v174;
  v173 = 0x400000000LL;
  if ( v7 > 4 )
  {
    sub_C8D5F0((__int64)&v172, v174, v7, 0x10u, a5, a6);
    v15 = v172;
    v10 = a1;
    v14 = &v172[16 * (unsigned int)v173];
  }
  for ( i = &v15[16 * v7]; i != v14; v14 += 16 )
  {
    if ( v14 )
    {
      *(_QWORD *)v14 = 0;
      *((_DWORD *)v14 + 2) = 0;
    }
  }
  LODWORD(v173) = v7;
  v175 = (const __m128i *)v177;
  v176 = 0x800000000LL;
  v17 = *(unsigned int *)(v10 + 8);
  if ( (_DWORD)v17 )
  {
    v119 = v12;
    v18 = 0;
    v19 = v10;
    v115 = 2 * v17;
    v122 = 0;
    while ( 1 )
    {
      v120 = 8 * v122;
      v118 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)v19 + 8 * v122));
      v124 = *(_DWORD *)(*(_QWORD *)(v19 + 144) + 2 * v122);
      v20 = *(_WORD *)(*(_QWORD *)(v19 + 80) + v122);
      v153 = v20;
      if ( !*(_BYTE *)(v19 + 180) )
        goto LABEL_16;
      v21 = *(_QWORD *)(a2 + 64);
      v22 = *(_QWORD *)v114;
      v23 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, unsigned __int64))(*(_QWORD *)v114 + 744LL);
      if ( v23 == sub_2FE9BB0 )
      {
        LOWORD(v160) = *(_WORD *)(*(_QWORD *)(v19 + 80) + v122);
        v161 = 0;
        if ( v20 )
        {
          v153 = *(_WORD *)(v114 + 2LL * v20 + 2852);
          goto LABEL_16;
        }
        if ( sub_30070B0((__int64)&v160) )
        {
          LOWORD(v166) = 0;
          v167 = 0;
          LOWORD(v162) = 0;
          sub_2FE8D10(v114, v21, (unsigned int)v160, 0, &v166, (unsigned int *)&v164, (unsigned __int16 *)&v162);
          v153 = v162;
        }
        else
        {
          if ( !sub_3007070((__int64)&v160) )
            goto LABEL_124;
          v92 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(v22 + 592);
          if ( v92 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v166, v114, v21, v160, v161);
            v93 = v112;
            LOWORD(v93) = v167;
            v94 = v168;
            v112 = v93;
          }
          else
          {
            v112 = v92(v114, v21, v160, 0);
            v94 = v102;
          }
          v163 = v94;
          v95 = (unsigned __int16)v112;
          v162 = v112;
          if ( (_WORD)v112 )
            goto LABEL_98;
          if ( sub_30070B0((__int64)&v162) )
          {
            LOWORD(v166) = 0;
            LOWORD(v159) = 0;
            v167 = 0;
            sub_2FE8D10(v114, v21, (unsigned int)v162, v94, &v166, (unsigned int *)&v164, (unsigned __int16 *)&v159);
            a5 = v109;
            v153 = v159;
          }
          else
          {
            if ( !sub_3007070((__int64)&v162) )
              goto LABEL_124;
            v96 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v114 + 592LL);
            if ( v96 == sub_2D56A50 )
            {
              sub_2FE6CC0((__int64)&v166, v114, v21, v162, v163);
              v97 = v111;
              LOWORD(v97) = v167;
              v98 = v168;
              v111 = v97;
            }
            else
            {
              v111 = v96(v114, v21, v162, v94);
              v98 = v103;
            }
            v165 = v98;
            v95 = (unsigned __int16)v111;
            v164 = v111;
            if ( (_WORD)v111 )
            {
LABEL_98:
              v153 = *(_WORD *)(v114 + 2 * v95 + 2852);
              goto LABEL_16;
            }
            if ( sub_30070B0((__int64)&v164) )
            {
              LOWORD(v166) = 0;
              v158 = 0;
              v167 = 0;
              sub_2FE8D10(v114, v21, (unsigned int)v164, v98, &v166, &v159, &v158);
              v153 = v158;
            }
            else
            {
              if ( !sub_3007070((__int64)&v164) )
LABEL_124:
                BUG();
              v99 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v114 + 592LL);
              if ( v99 == sub_2D56A50 )
              {
                sub_2FE6CC0((__int64)&v166, v114, v21, v164, v165);
                v100 = v110;
                LOWORD(v100) = v167;
                v101 = v168;
                v110 = v100;
              }
              else
              {
                v110 = v99(v114, v21, v164, v98);
                v101 = v104;
              }
              v153 = sub_2FE98B0(v114, v21, (unsigned int)v110, v101);
            }
          }
        }
      }
      else
      {
        v153 = v23(v114, *(_QWORD *)(a2 + 64), *(unsigned int *)(v19 + 176), v20, 0);
      }
LABEL_16:
      v24 = (unsigned int)v176;
      if ( v124 != (unsigned __int64)(unsigned int)v176 )
      {
        if ( v124 >= (unsigned __int64)(unsigned int)v176 )
        {
          if ( v124 > (unsigned __int64)HIDWORD(v176) )
          {
            sub_C8D5F0((__int64)&v175, v177, v124, 0x10u, a5, a6);
            v24 = (unsigned int)v176;
          }
          v25 = &v175[v24];
          for ( j = &v175[v124]; j != v25; ++v25 )
          {
            if ( v25 )
            {
              v25->m128i_i64[0] = 0;
              v25->m128i_i32[2] = 0;
            }
          }
        }
        LODWORD(v176) = v124;
      }
      if ( v124 )
      {
        v155 = v18;
        v27 = v119;
        v28 = v142;
        v151 = 0;
        v125 = v18 + v124;
        while ( 1 )
        {
          v29 = (unsigned int *)(*(_QWORD *)(v19 + 112) + 4LL * v155);
          v135 = *(_QWORD *)v27;
          v143 = *((_DWORD *)v27 + 2);
          if ( a6 )
          {
            LOWORD(v28) = v153;
            v127 = v27;
            v30 = *(_QWORD *)a6;
            v132 = *(_DWORD *)(a6 + 8);
            v152 = *v29;
            v128 = v28;
            v31 = sub_33E5B50(a2, v28, 0, 1, 0, *v29, 262, 0);
            v130 = v32;
            v33 = v31;
            v166 = v135;
            LODWORD(v167) = v143;
            v168 = sub_33F0B60(a2, v152, v128, 0);
            v171 = v132;
            v169 = v34;
            *((_QWORD *)&v106 + 1) = 2 - ((v30 == 0) - 1LL);
            *(_QWORD *)&v106 = &v166;
            v170 = v30;
            v36 = sub_3411630(a2, 50, a4, v33, v130, v35, v106);
            v38 = v37;
            v157 = v37 | v157 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)a6 = v36;
            *(_DWORD *)(a6 + 8) = 2;
            v27 = v127;
          }
          else
          {
            v73 = *v29;
            v74 = v123;
            LOWORD(v74) = v153;
            v129 = v28;
            v131 = v27;
            v123 = v74;
            v75 = sub_33E5110(a2, (unsigned int)v74, 0, 1, 0);
            v133 = v76;
            v77 = v75;
            v166 = v135;
            LODWORD(v167) = v143;
            v168 = sub_33F0B60(a2, v73, (unsigned int)v123, 0);
            v169 = v78;
            *((_QWORD *)&v108 + 1) = 2;
            *(_QWORD *)&v108 = &v166;
            v80 = sub_3411630(a2, 50, a4, v77, v133, v79, v108);
            v27 = v131;
            v36 = v80;
            v38 = v81;
            v28 = v129;
            v157 = v81 | v157 & 0xFFFFFFFF00000000LL;
          }
          v39 = (__m128i *)&v175[v151];
          *(_QWORD *)v27 = v36;
          *((_DWORD *)v27 + 2) = 1;
          v39->m128i_i64[0] = v36;
          v39->m128i_i32[2] = v38;
          v40 = *(_DWORD *)(*(_QWORD *)(v19 + 112) + 4LL * v155);
          if ( v40 >= 0 )
            goto LABEL_58;
          v41 = v153 - 17;
          if ( (unsigned __int16)(v153 - 2) > 7u && v41 > 0x6Cu && (unsigned __int16)(v153 - 176) > 0x1Fu )
            goto LABEL_58;
          v42 = v40 & 0x7FFFFFFF;
          if ( (unsigned int)v42 >= *(_DWORD *)(a3 + 1096) )
            goto LABEL_58;
          v43 = *(_QWORD *)(a3 + 1088) + 40 * v42;
          if ( *(char *)(v43 + 3) >= 0 )
            goto LABEL_58;
          v44 = v153;
          if ( v41 <= 0xD3u )
            v44 = word_4456580[v153 - 1];
          if ( v44 <= 1u || (unsigned __int16)(v44 - 504) <= 7u )
            BUG();
          v45 = *(_QWORD *)&byte_444C4A0[16 * v44 - 16];
          v46 = *(_DWORD *)(v43 + 16);
          if ( v46 > 0x40 )
          {
            v134 = v28;
            v139 = v27;
            v147 = *(_QWORD *)&byte_444C4A0[16 * v44 - 16];
            LODWORD(v47) = sub_C44500(v43 + 8);
            LODWORD(v45) = v147;
            v28 = v134;
            v27 = v139;
            if ( v147 == (_DWORD)v47 )
              goto LABEL_68;
          }
          else
          {
            if ( !v46 )
            {
              if ( !(_DWORD)v45 )
              {
LABEL_68:
                HIWORD(v82) = v113;
                LOWORD(v82) = v153;
                v148 = v28;
                v156 = v27;
                v83 = sub_3400BD0(a2, 0, a4, v82, 0, 0, 0);
                v27 = v156;
                v28 = v148;
                v85 = v84;
                v86 = v83;
                v87 = v175;
                v175[v151].m128i_i64[0] = v86;
                v87[v151].m128i_i32[2] = v85;
                goto LABEL_58;
              }
              goto LABEL_44;
            }
            v47 = ~(*(_QWORD *)(v43 + 8) << (64 - (unsigned __int8)v46));
            if ( !v47 )
            {
              if ( (_DWORD)v45 == 64 )
                goto LABEL_68;
              LODWORD(v47) = 64;
LABEL_71:
              v88 = v45 - v47;
              if ( (_DWORD)v45 - (_DWORD)v47 == 1 )
              {
                LOWORD(v89) = 2;
              }
              else
              {
                switch ( v88 )
                {
                  case 2u:
                    LOWORD(v89) = 3;
                    break;
                  case 4u:
                    LOWORD(v89) = 4;
                    break;
                  case 8u:
                    LOWORD(v89) = 5;
                    break;
                  case 0x10u:
                    LOWORD(v89) = 6;
                    break;
                  case 0x20u:
                    LOWORD(v89) = 7;
                    break;
                  case 0x40u:
                    LOWORD(v89) = 8;
                    break;
                  case 0x80u:
                    LOWORD(v89) = 9;
                    break;
                  default:
                    v140 = v28;
                    v149 = v27;
                    v89 = sub_3007020(*(_QWORD **)(a2 + 64), v88);
                    v27 = v149;
                    v28 = v140;
                    v116 = v89;
LABEL_82:
                    v91 = v116;
                    v141 = v28;
                    v150 = v27;
                    LOWORD(v91) = v89;
                    v116 = v91;
                    *(_QWORD *)&v53 = sub_33F7D60(a2, (unsigned int)v91, v90);
                    v55 = 4;
                    v56 = v153;
                    v58 = v150;
                    v57 = v141;
                    goto LABEL_57;
                }
              }
              v90 = 0;
              goto LABEL_82;
            }
            _BitScanReverse64(&v47, v47);
            LODWORD(v47) = v47 ^ 0x3F;
            if ( (_DWORD)v45 == (_DWORD)v47 )
              goto LABEL_68;
          }
          if ( (_DWORD)v47 )
            goto LABEL_71;
LABEL_44:
          v48 = *(_DWORD *)v43 & 0x7FFFFFFF;
          if ( v48 > 1 )
          {
            v49 = v45 - v48;
            if ( v49 )
            {
              switch ( v49 )
              {
                case 1:
                  LOWORD(v50) = 3;
                  break;
                case 3:
                  LOWORD(v50) = 4;
                  break;
                case 7:
                  LOWORD(v50) = 5;
                  break;
                case 15:
                  LOWORD(v50) = 6;
                  break;
                case 31:
                  LOWORD(v50) = 7;
                  break;
                case 63:
                  LOWORD(v50) = 8;
                  break;
                case 127:
                  LOWORD(v50) = 9;
                  break;
                default:
                  v136 = v28;
                  v144 = v27;
                  v50 = sub_3007020(*(_QWORD **)(a2 + 64), v49 + 1);
                  v27 = v144;
                  v28 = v136;
                  v117 = v50;
                  goto LABEL_56;
              }
            }
            else
            {
              LOWORD(v50) = 2;
            }
            v51 = 0;
LABEL_56:
            v52 = v117;
            v137 = v28;
            v145 = v27;
            LOWORD(v52) = v50;
            v117 = v52;
            *(_QWORD *)&v53 = sub_33F7D60(a2, (unsigned int)v52, v51);
            v55 = 3;
            v56 = v153;
            v57 = v137;
            v58 = v145;
LABEL_57:
            v138 = v57;
            v157 = v38 | v157 & 0xFFFFFFFF00000000LL;
            *((_QWORD *)&v105 + 1) = v157;
            *(_QWORD *)&v105 = v36;
            v146 = v58;
            v59 = sub_3406EB0(a2, v55, a4, v56, 0, v54, v105, v53);
            v27 = v146;
            v28 = v138;
            v61 = v60;
            v62 = v59;
            v63 = v175;
            v175[v151].m128i_i64[0] = v62;
            v63[v151].m128i_i32[2] = v61;
          }
LABEL_58:
          ++v155;
          ++v151;
          if ( v155 == v125 )
          {
            v142 = v28;
            goto LABEL_60;
          }
        }
      }
      v125 = v18;
LABEL_60:
      v64 = sub_336BB10(
              a2,
              a4,
              v175,
              v124,
              v153,
              a7,
              v118.m128i_u64[0],
              v118.m128i_u64[1],
              *v119,
              *(_QWORD *)(v19 + 176),
              v166,
              0);
      v18 = v125;
      v66 = v65;
      v67 = v64;
      v68 = (unsigned __int64)v172;
      v122 += 2;
      *(_QWORD *)&v172[v120] = v67;
      *(_DWORD *)(v68 + v120 + 8) = v66;
      LODWORD(v176) = 0;
      if ( v115 == v122 )
      {
        v13 = (unsigned int)v173;
        v10 = v19;
        break;
      }
    }
  }
  v69 = (unsigned __int64)v172;
  v70 = sub_33E5830(a2, *(_QWORD *)v10);
  *((_QWORD *)&v107 + 1) = v13;
  *(_QWORD *)&v107 = v69;
  v8 = sub_3411630(a2, 55, a4, v70, v71, v72, v107);
  if ( v175 != (const __m128i *)v177 )
    _libc_free((unsigned __int64)v175);
  if ( v172 != v174 )
    _libc_free((unsigned __int64)v172);
  return v8;
}
