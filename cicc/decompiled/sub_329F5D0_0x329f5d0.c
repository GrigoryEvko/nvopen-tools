// Function: sub_329F5D0
// Address: 0x329f5d0
//
__int64 __fastcall sub_329F5D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  __int64 v10; // rbx
  __int64 v11; // rdx
  char v12; // r14
  _BYTE *v13; // rsi
  __int16 *v14; // rdx
  unsigned __int16 v15; // ax
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 result; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rax
  __int32 v23; // edx
  __int64 v24; // rsi
  __int64 v25; // rax
  __int32 v26; // edx
  __int64 v27; // rax
  int v28; // ecx
  int v29; // edx
  __int64 v30; // rdx
  unsigned int *v31; // rcx
  unsigned int *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // r10
  __int64 v37; // r11
  unsigned __int16 *v38; // rbx
  __int64 v39; // r14
  __int64 v40; // r13
  int v41; // r12d
  __int64 v42; // rsi
  char v43; // dl
  char v44; // dl
  char v45; // dl
  bool v46; // zf
  char v47; // dl
  char v48; // dl
  __m128i v49; // xmm0
  __int32 v50; // eax
  __m128i v51; // xmm0
  int v52; // esi
  __int64 v53; // rax
  __m128i v54; // xmm1
  unsigned int v55; // r10d
  const __m128i *v56; // rax
  __int32 v57; // r8d
  __int64 v58; // r9
  __m128i v59; // xmm3
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int128 *v62; // rbx
  int v63; // r9d
  int v64; // edx
  __int64 v65; // rax
  _BYTE *v66; // rcx
  char v67; // al
  int v68; // r9d
  __int64 v69; // r10
  char v70; // cl
  __int64 v71; // rdi
  char v72; // al
  int v73; // esi
  __int64 v74; // rax
  int v75; // edx
  int v76; // esi
  __int64 v77; // rdx
  __int64 v78; // rsi
  __int64 v79; // rdx
  __int64 v80; // r13
  __int64 v81; // rbx
  __int64 v82; // rdx
  __int64 v83; // rdi
  __int64 v84; // rdx
  int v85; // esi
  __int64 v86; // rax
  int v87; // edx
  int v88; // esi
  __int64 v89; // rdx
  int v90; // edx
  __int64 v91; // rax
  __int64 *v92; // rax
  __int64 v93; // r13
  __int128 v94; // rcx
  __int64 v95; // r14
  __int128 v96; // rax
  int v97; // r9d
  __int128 v98; // rax
  int v99; // r9d
  __int64 v100; // rax
  int v101; // edx
  int v102; // esi
  __int64 v103; // rdx
  __int64 *v104; // rax
  __int64 v105; // r13
  __int128 v106; // rcx
  __int64 v107; // r14
  __int128 v108; // rax
  int v109; // r9d
  __int128 v110; // rax
  int v111; // r9d
  __int64 v112; // rax
  int v113; // edx
  int v114; // esi
  __int64 v115; // rdx
  __int128 v116; // [rsp-20h] [rbp-260h]
  __int128 v117; // [rsp-20h] [rbp-260h]
  __int128 v118; // [rsp-10h] [rbp-250h]
  __m128i v119; // [rsp+0h] [rbp-240h]
  unsigned __int128 v120; // [rsp+10h] [rbp-230h]
  unsigned int v121; // [rsp+10h] [rbp-230h]
  char v122; // [rsp+10h] [rbp-230h]
  __m128i v123; // [rsp+10h] [rbp-230h]
  unsigned int v124; // [rsp+30h] [rbp-210h]
  unsigned __int32 v125; // [rsp+34h] [rbp-20Ch]
  __int64 v126; // [rsp+38h] [rbp-208h]
  __int64 v127; // [rsp+50h] [rbp-1F0h]
  bool v128; // [rsp+58h] [rbp-1E8h]
  int v129; // [rsp+58h] [rbp-1E8h]
  int v130; // [rsp+58h] [rbp-1E8h]
  bool v131; // [rsp+68h] [rbp-1D8h]
  char v132; // [rsp+70h] [rbp-1D0h]
  __int64 v133; // [rsp+70h] [rbp-1D0h]
  __int64 v134; // [rsp+70h] [rbp-1D0h]
  unsigned int v135; // [rsp+70h] [rbp-1D0h]
  __int64 v136; // [rsp+78h] [rbp-1C8h]
  char v137; // [rsp+80h] [rbp-1C0h]
  __int128 v138; // [rsp+80h] [rbp-1C0h]
  __int128 v139; // [rsp+80h] [rbp-1C0h]
  unsigned int v140; // [rsp+80h] [rbp-1C0h]
  unsigned int v141; // [rsp+80h] [rbp-1C0h]
  unsigned int v142; // [rsp+80h] [rbp-1C0h]
  __m128i v143; // [rsp+E0h] [rbp-160h]
  __int64 v144; // [rsp+120h] [rbp-120h] BYREF
  __int64 v145; // [rsp+128h] [rbp-118h]
  __m128i v146; // [rsp+130h] [rbp-110h] BYREF
  __int64 v147; // [rsp+140h] [rbp-100h] BYREF
  __int64 v148; // [rsp+148h] [rbp-F8h]
  __m128i v149; // [rsp+150h] [rbp-F0h] BYREF
  __m128i v150; // [rsp+160h] [rbp-E0h] BYREF
  __int64 v151; // [rsp+170h] [rbp-D0h] BYREF
  __int32 v152; // [rsp+178h] [rbp-C8h]
  __int64 v153; // [rsp+180h] [rbp-C0h] BYREF
  __int32 v154; // [rsp+188h] [rbp-B8h]
  __m128i v155; // [rsp+190h] [rbp-B0h] BYREF
  unsigned __int128 v156; // [rsp+1A0h] [rbp-A0h] BYREF
  _DWORD v157[4]; // [rsp+1B0h] [rbp-90h] BYREF
  __int64 (__fastcall *v158)(_QWORD *, _DWORD *, int); // [rsp+1C0h] [rbp-80h]
  __int64 (__fastcall *v159)(unsigned int *, __int64, __int64); // [rsp+1C8h] [rbp-78h]
  __m128i *v160; // [rsp+1D0h] [rbp-70h] BYREF
  __int64 *v161; // [rsp+1D8h] [rbp-68h]
  __int64 v162; // [rsp+1E0h] [rbp-60h]
  __int64 v163; // [rsp+1E8h] [rbp-58h]
  __int64 *v164; // [rsp+1F0h] [rbp-50h]
  unsigned __int128 *v165; // [rsp+1F8h] [rbp-48h]
  __m128i *v166; // [rsp+200h] [rbp-40h]

  v10 = 16LL * (unsigned int)a3;
  v146.m128i_i64[1] = a3;
  v11 = *(_QWORD *)(a2 + 48);
  v12 = *(_BYTE *)(a1 + 33);
  v146.m128i_i64[0] = a2;
  v13 = *(_BYTE **)(a1 + 8);
  v14 = (__int16 *)(v10 + v11);
  v144 = a4;
  v15 = *v14;
  v16 = *((_QWORD *)v14 + 1);
  v145 = a5;
  LOWORD(v147) = v15;
  v148 = v16;
  if ( v12 )
  {
    if ( v15 == 1 )
    {
      v17 = 1;
      v137 = v13[7107] == 0;
      v132 = v13[7108] == 0;
      v131 = v13[7109] == 0;
LABEL_4:
      v12 = v13[500 * v17 + 6610] == 0;
      goto LABEL_16;
    }
    if ( v15 )
    {
      v137 = 0;
      v17 = v15;
      if ( *(_QWORD *)&v13[8 * v15 + 112] )
      {
        v66 = &v13[500 * v15];
        v137 = v66[6607] == 0;
        if ( *(_QWORD *)&v13[8 * v15 + 112] )
        {
          v132 = v66[6608] == 0;
          v131 = v66[6609] == 0;
          goto LABEL_4;
        }
      }
      v132 = 0;
      v131 = 0;
      v12 = 0;
LABEL_16:
      if ( (unsigned __int16)(v15 - 2) > 7u )
        goto LABEL_6;
      goto LABEL_17;
    }
    v131 = 0;
    v12 = 0;
    v132 = 0;
    v137 = 0;
LABEL_5:
    if ( !sub_30070A0((__int64)&v147) )
      goto LABEL_6;
    goto LABEL_17;
  }
  if ( v15 == 1 )
  {
    v43 = v13[7107];
    v137 = 1;
    if ( v43 )
      v137 = v43 == 4;
    v44 = v13[7108];
    v132 = 1;
    if ( v44 )
    {
      v46 = v44 == 4;
      v45 = v13[7109];
      v132 = v46;
      if ( v45 )
        goto LABEL_56;
    }
    else
    {
      v45 = v13[7109];
      if ( v45 )
        goto LABEL_56;
    }
    v131 = 1;
    v30 = 1;
    goto LABEL_40;
  }
  if ( v15 )
  {
    v137 = 0;
    if ( !*(_QWORD *)&v13[8 * v15 + 112] )
      goto LABEL_36;
    v47 = v13[500 * v15 + 6607];
    if ( v47 )
    {
      v137 = v47 == 4;
      if ( !*(_QWORD *)&v13[8 * v15 + 112] )
        goto LABEL_36;
    }
    else
    {
      v137 = 1;
    }
    v48 = v13[500 * v15 + 6608];
    if ( v48 )
    {
      v132 = v48 == 4;
      if ( !*(_QWORD *)&v13[8 * v15 + 112] )
        goto LABEL_37;
    }
    else
    {
      v132 = 1;
    }
    v45 = v13[500 * v15 + 6609];
    if ( !v45 )
    {
      v131 = 1;
      goto LABEL_38;
    }
LABEL_56:
    v46 = v45 == 4;
    v30 = 1;
    v131 = v46;
    if ( v15 == 1 )
      goto LABEL_40;
    goto LABEL_38;
  }
  v137 = 0;
LABEL_36:
  v132 = 0;
LABEL_37:
  v131 = 0;
LABEL_38:
  if ( !v15 )
    goto LABEL_5;
  v30 = v15;
  if ( !*(_QWORD *)&v13[8 * v15 + 112] )
    goto LABEL_16;
LABEL_40:
  v12 = (v13[500 * v30 + 6610] & 0xFB) == 0;
  if ( (unsigned __int16)(v15 - 2) > 7u )
    goto LABEL_6;
LABEL_17:
  sub_2FE6CC0((__int64)&v160, (__int64)v13, *(_QWORD *)(*(_QWORD *)a1 + 64LL), v147, v148);
  if ( (_BYTE)v160 == 1 && (_WORD)v147 )
  {
    v19 = *(_QWORD *)(a1 + 8) + 500LL * (unsigned __int16)v147;
    v137 |= *(_BYTE *)(v19 + 6607) == 4;
    v132 |= *(_BYTE *)(v19 + 6608) == 4;
  }
LABEL_6:
  if ( *(_BYTE *)(a1 + 33) && !v137 && !v132 && !v12 && !v131 )
    return 0;
  if ( *(_DWORD *)(a2 + 24) != 216
    || *(_DWORD *)(v144 + 24) != 216
    || (v31 = *(unsigned int **)(v144 + 40),
        v32 = *(unsigned int **)(a2 + 40),
        v33 = *(_QWORD *)(*(_QWORD *)v31 + 48LL) + 16LL * v31[2],
        v34 = *(_QWORD *)(*(_QWORD *)v32 + 48LL) + 16LL * v32[2],
        *(_WORD *)v34 != *(_WORD *)v33)
    || *(_QWORD *)(v34 + 8) != *(_QWORD *)(v33 + 8) && !*(_WORD *)v34
    || (v36 = sub_329F5D0(a1, *(_QWORD *)v32, *((_QWORD *)v32 + 1), *(_QWORD *)v31, *((_QWORD *)v31 + 1), a6, 0),
        v37 = v35,
        !v36) )
  {
    v20 = *(_QWORD *)a1;
    v149.m128i_i64[0] = 0;
    v149.m128i_i32[2] = 0;
    v150.m128i_i64[0] = 0;
    v150.m128i_i32[2] = 0;
    sub_325FAC0(v20, v146.m128i_i64[0], v146.m128i_i32[2], (__int64)&v149, (__int64)&v150);
    v21 = *(_QWORD *)a1;
    v151 = 0;
    v152 = 0;
    v153 = 0;
    v154 = 0;
    sub_325FAC0(v21, v144, v145, (__int64)&v151, (__int64)&v153);
    if ( v149.m128i_i64[0] )
    {
      v22 = sub_3285210(*(_QWORD *)a1, v149.m128i_i64[0], v144, v145, (__int64)&v153, a6);
      if ( v22 )
      {
        v151 = v22;
        v152 = v23;
      }
    }
    v24 = v151;
    if ( !v151 )
      return 0;
    v25 = sub_3285210(*(_QWORD *)a1, v151, v146.m128i_i64[0], v146.m128i_u32[2], (__int64)&v150, a6);
    if ( v25 )
    {
      v149.m128i_i64[0] = v25;
      v149.m128i_i32[2] = v26;
    }
    v27 = v151;
    if ( !v151 )
      return 0;
    if ( !v149.m128i_i64[0] )
      return 0;
    v28 = *(_DWORD *)(v149.m128i_i64[0] + 24);
    v29 = *(_DWORD *)(v151 + 24);
    if ( v29 == v28 )
      return 0;
    if ( v29 == 190 )
    {
      v49 = _mm_load_si128(&v146);
      v146.m128i_i64[0] = v144;
      v146.m128i_i32[2] = v145;
      v144 = v49.m128i_i64[0];
      LODWORD(v145) = v49.m128i_i32[2];
      v143 = _mm_loadu_si128(&v149);
      v149.m128i_i64[0] = v151;
      v50 = v152;
      v152 = v143.m128i_i32[2];
      v149.m128i_i32[2] = v50;
      v27 = v143.m128i_i64[0];
      v151 = v143.m128i_i64[0];
      v51 = _mm_loadu_si128(&v150);
      v150.m128i_i64[0] = v153;
      v150.m128i_i32[2] = v154;
      v153 = v51.m128i_i64[0];
      v154 = v51.m128i_i32[2];
    }
    else if ( v28 != 190 )
    {
      return 0;
    }
    if ( *(_DWORD *)(v27 + 24) != 192 )
      return 0;
    v52 = sub_32844A0((unsigned __int16 *)&v147, v24);
    v53 = *(_QWORD *)(v149.m128i_i64[0] + 40);
    v54 = _mm_loadu_si128((const __m128i *)v53);
    v55 = *(_DWORD *)(v53 + 8);
    v127 = *(_QWORD *)v53;
    v155 = _mm_loadu_si128((const __m128i *)(v53 + 40));
    v56 = *(const __m128i **)(v151 + 40);
    v57 = v56->m128i_i32[2];
    v58 = v56->m128i_i64[0];
    v59 = _mm_loadu_si128(v56);
    v60 = v56[2].m128i_i64[1];
    v61 = v56[3].m128i_i64[0];
    v164 = &v147;
    v160 = &v150;
    v161 = &v153;
    v62 = (__int128 *)&v155;
    v126 = v58;
    v125 = v57;
    v156 = __PAIR128__(v61, v60);
    v120 = __PAIR128__(v61, v60);
    v162 = a1;
    v163 = a6;
    v165 = &v156;
    v166 = &v155;
    v128 = v127 == v58 && v55 == v57;
    if ( !v128 && !v12 && !v131 )
    {
      v135 = v55;
      if ( (_WORD)v147
        && *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL * (unsigned __int16)v147 + 112)
        && (unsigned __int8)sub_3286E00(&v146)
        && (unsigned __int8)sub_3286E00(&v144) )
      {
        v157[0] = v52;
        v159 = sub_3261E90;
        v158 = sub_325DB00;
        if ( (unsigned __int8)sub_33CACD0(
                                v155.m128i_i32[0],
                                v155.m128i_i32[2],
                                v120,
                                DWORD2(v120),
                                (unsigned int)v157,
                                0,
                                0) )
        {
          sub_A17130((__int64)v157);
          v64 = 1;
          v65 = *(_QWORD *)(v127 + 56);
          if ( v65 )
          {
            do
            {
              if ( v135 == *(_DWORD *)(v65 + 8) )
              {
                if ( !v64 )
                  goto LABEL_125;
                v65 = *(_QWORD *)(v65 + 32);
                if ( !v65 )
                  goto LABEL_134;
                if ( v135 == *(_DWORD *)(v65 + 8) )
                  goto LABEL_125;
                v64 = 0;
              }
              v65 = *(_QWORD *)(v65 + 32);
            }
            while ( v65 );
            if ( v64 == 1 )
              goto LABEL_125;
LABEL_134:
            if ( *(_DWORD *)(v127 + 24) == 187 )
            {
              v92 = *(__int64 **)(v127 + 40);
              if ( v126 == *v92 && v125 == *((_DWORD *)v92 + 2) )
              {
                v93 = v92[5];
                *(_QWORD *)&v94 = v126;
                *((_QWORD *)&v94 + 1) = v125;
                v95 = *((unsigned int *)v92 + 12);
                goto LABEL_139;
              }
              if ( v126 == v92[5] && v125 == *((_DWORD *)v92 + 12) )
              {
                v93 = *v92;
                *(_QWORD *)&v94 = v126;
                *((_QWORD *)&v94 + 1) = v125;
                v95 = *((unsigned int *)v92 + 2);
LABEL_139:
                *(_QWORD *)&v96 = sub_3406EB0(*(_QWORD *)a1, 193, a6, v147, v148, v63, v94, *(_OWORD *)&v155);
                *((_QWORD *)&v116 + 1) = v95;
                *(_QWORD *)&v116 = v93;
                v138 = v96;
                *(_QWORD *)&v98 = sub_3406EB0(*(_QWORD *)a1, 190, a6, v147, v148, v97, v116, *(_OWORD *)&v155);
                v100 = sub_3406EB0(*(_QWORD *)a1, 187, a6, v147, v148, v99, v138, v98);
                v102 = v101;
                v103 = v100;
                LODWORD(v100) = v102;
                v78 = v103;
                v79 = (unsigned int)v100;
                return sub_325D810((__int64)&v160, v78, v79);
              }
            }
          }
LABEL_125:
          v90 = 1;
          v91 = *(_QWORD *)(v126 + 56);
          if ( !v91 )
            return 0;
          do
          {
            if ( v125 == *(_DWORD *)(v91 + 8) )
            {
              if ( !v90 )
                return 0;
              v91 = *(_QWORD *)(v91 + 32);
              if ( !v91 )
                goto LABEL_144;
              if ( v125 == *(_DWORD *)(v91 + 8) )
                return 0;
              v90 = 0;
            }
            v91 = *(_QWORD *)(v91 + 32);
          }
          while ( v91 );
          if ( v90 == 1 )
            return 0;
LABEL_144:
          if ( *(_DWORD *)(v126 + 24) != 187 )
            return 0;
          v104 = *(__int64 **)(v126 + 40);
          if ( v127 == *v104 && v135 == *((_DWORD *)v104 + 2) )
          {
            v105 = v104[5];
            *(_QWORD *)&v106 = v127;
            *((_QWORD *)&v106 + 1) = v135;
            v107 = *((unsigned int *)v104 + 12);
            goto LABEL_149;
          }
          if ( v127 == v104[5] && v135 == *((_DWORD *)v104 + 12) )
          {
            v105 = *v104;
            *(_QWORD *)&v106 = v127;
            *((_QWORD *)&v106 + 1) = v135;
            v107 = *((unsigned int *)v104 + 2);
LABEL_149:
            *(_QWORD *)&v108 = sub_3406EB0(*(_QWORD *)a1, 193, a6, v147, v148, v63, v106, *(_OWORD *)&v155);
            *((_QWORD *)&v117 + 1) = v107;
            *(_QWORD *)&v117 = v105;
            v139 = v108;
            *(_QWORD *)&v110 = sub_3406EB0(*(_QWORD *)a1, 192, a6, v147, v148, v109, v117, v156);
            v112 = sub_3406EB0(*(_QWORD *)a1, 187, a6, v147, v148, v111, v139, v110);
            v114 = v113;
            v115 = v112;
            LODWORD(v112) = v114;
            v78 = v115;
            v79 = (unsigned int)v112;
            return sub_325D810((__int64)&v160, v78, v79);
          }
        }
        else
        {
          sub_A17130((__int64)v157);
        }
      }
      return 0;
    }
    v157[0] = v52;
    v159 = sub_3261E90;
    v121 = v55;
    v158 = sub_325DB00;
    v67 = sub_33CACD0(v155.m128i_i32[0], v155.m128i_i32[2], v156, DWORD2(v156), (unsigned int)v157, 0, 0);
    v69 = v121;
    v70 = v67;
    if ( v158 )
    {
      v122 = v67;
      v124 = v69;
      v158(v157, v157, 3);
      v69 = v124;
      v70 = v122;
    }
    if ( v70 )
    {
      v71 = *(_QWORD *)a1;
      v72 = *(_BYTE *)(a1 + 33);
      if ( v128 )
      {
        if ( v137 || v132 )
        {
          if ( v72 && !v137 )
            goto LABEL_97;
          goto LABEL_142;
        }
        if ( !v12 && !v131 )
        {
          if ( v72 )
          {
LABEL_97:
            v62 = (__int128 *)&v156;
            v73 = 194;
LABEL_98:
            v74 = sub_3406EB0(v71, v73, a6, v147, v148, v68, *(_OWORD *)&v54, *v62);
            v76 = v75;
            v77 = v74;
            LODWORD(v74) = v76;
            v78 = v77;
            v79 = (unsigned int)v74;
            return sub_325D810((__int64)&v160, v78, v79);
          }
LABEL_142:
          v73 = 193;
          goto LABEL_98;
        }
      }
      if ( !v72 || v131 )
      {
        v85 = 195;
      }
      else
      {
        v62 = (__int128 *)&v156;
        v85 = 196;
      }
      v86 = sub_340F900(v71, v85, a6, v147, v148, v68, *(_OWORD *)&v54, *(_OWORD *)&v59, *v62);
      v88 = v87;
      v89 = v86;
      LODWORD(v86) = v88;
      v78 = v89;
      v79 = (unsigned int)v86;
      return sub_325D810((__int64)&v160, v78, v79);
    }
    if ( !((unsigned __int8)v132 | (unsigned __int8)v137) && !v12 && !v131 || v150.m128i_i64[0] || v153 )
      return 0;
    v80 = v155.m128i_i64[0];
    v81 = v156;
    v123 = _mm_loadu_si128(&v155);
    v119 = _mm_loadu_si128((const __m128i *)&v156);
    if ( (unsigned int)(*(_DWORD *)(v155.m128i_i64[0] + 24) - 213) <= 3
      && (unsigned int)(*(_DWORD *)(v156 + 24) - 213) <= 3 )
    {
      v82 = *(_QWORD *)(v155.m128i_i64[0] + 40);
      v80 = *(_QWORD *)v82;
      v83 = *(unsigned int *)(v82 + 8);
      v84 = *(_QWORD *)(v156 + 40);
      v81 = *(_QWORD *)v84;
      v123.m128i_i64[1] = v83 | v123.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v119.m128i_i64[1] = *(unsigned int *)(v84 + 8) | v119.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    }
    if ( v128 && (unsigned __int8)v132 | (unsigned __int8)v137 )
    {
      if ( a7 )
      {
        if ( *(_DWORD *)(v81 + 24) == 186 )
        {
          if ( *(_DWORD *)(v80 + 24) == 186 )
            return 0;
        }
        else
        {
          v129 = v69;
          result = sub_32780C0(
                     (__int64 *)a1,
                     v54.m128i_i64[0],
                     v54.m128i_i64[1],
                     v155.m128i_i64[0],
                     v155.m128i_i64[1],
                     v137,
                     v156,
                     SDWORD2(v156),
                     v80,
                     v123.m128i_i64[1],
                     v81,
                     v119.m128i_i64[1],
                     193,
                     194,
                     a6);
          LODWORD(v69) = v129;
          if ( result )
            return result;
          if ( *(_DWORD *)(v80 + 24) == 186 )
          {
            if ( *(_DWORD *)(v81 + 24) == 186 )
              return 0;
            goto LABEL_157;
          }
        }
      }
      else
      {
        v130 = v69;
        result = sub_32780C0(
                   (__int64 *)a1,
                   v54.m128i_i64[0],
                   v54.m128i_i64[1],
                   v155.m128i_i64[0],
                   v155.m128i_i64[1],
                   v137,
                   v156,
                   SDWORD2(v156),
                   v80,
                   v123.m128i_i64[1],
                   v81,
                   v119.m128i_i64[1],
                   193,
                   194,
                   a6);
        LODWORD(v69) = v130;
        if ( result )
          return result;
      }
      v142 = v69;
      result = sub_32780C0(
                 (__int64 *)a1,
                 v59.m128i_i64[0],
                 v59.m128i_i64[1],
                 v156,
                 *((__int64 *)&v156 + 1),
                 v132,
                 v155.m128i_i8[0],
                 v155.m128i_i32[2],
                 v81,
                 v119.m128i_i64[1],
                 v80,
                 v123.m128i_i64[1],
                 194,
                 193,
                 a6);
      v69 = v142;
      if ( result )
        return result;
    }
    if ( !a7 )
    {
      v141 = v69;
      result = sub_329ECD0(
                 (__int64 *)a1,
                 v54.m128i_i64[0],
                 v54.m128i_i64[1],
                 v59.m128i_i64[0],
                 v59.m128i_i64[1],
                 v131,
                 *(_OWORD *)&v155,
                 v156,
                 v80,
                 v123.m128i_i64[1],
                 v81,
                 v119.m128i_i64[1],
                 195,
                 196,
                 a6);
      v69 = v141;
      if ( result )
        return result;
LABEL_117:
      result = sub_329ECD0(
                 (__int64 *)a1,
                 v127,
                 v69 | v54.m128i_i64[1] & 0xFFFFFFFF00000000LL,
                 v126,
                 v125 | v59.m128i_i64[1] & 0xFFFFFFFF00000000LL,
                 v12,
                 v156,
                 *(_OWORD *)&v155,
                 v81,
                 v119.m128i_i64[1],
                 v80,
                 v123.m128i_i64[1],
                 196,
                 195,
                 a6);
      if ( result )
        return result;
      return 0;
    }
    if ( *(_DWORD *)(v81 + 24) == 186 )
    {
LABEL_116:
      if ( *(_DWORD *)(v80 + 24) == 186 )
        return 0;
      goto LABEL_117;
    }
LABEL_157:
    v140 = v69;
    result = sub_329ECD0(
               (__int64 *)a1,
               v54.m128i_i64[0],
               v54.m128i_i64[1],
               v59.m128i_i64[0],
               v59.m128i_i64[1],
               v131,
               *(_OWORD *)&v155,
               v156,
               v80,
               v123.m128i_i64[1],
               v81,
               v119.m128i_i64[1],
               195,
               196,
               a6);
    v69 = v140;
    if ( result )
      return result;
    goto LABEL_116;
  }
  v38 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + v10);
  v39 = *(_QWORD *)a1;
  v40 = *((_QWORD *)v38 + 1);
  v41 = *v38;
  v42 = *(_QWORD *)(v146.m128i_i64[0] + 80);
  v160 = (__m128i *)v42;
  if ( v42 )
  {
    v133 = v36;
    v136 = v35;
    sub_B96E90((__int64)&v160, v42, 1);
    v36 = v133;
    v37 = v136;
  }
  *((_QWORD *)&v118 + 1) = v37;
  *(_QWORD *)&v118 = v36;
  LODWORD(v161) = *(_DWORD *)(v146.m128i_i64[0] + 72);
  result = sub_33FAF80(v39, 216, (unsigned int)&v160, v41, v40, (unsigned int)&v160, v118);
  if ( v160 )
  {
    v134 = result;
    sub_B91220((__int64)&v160, (__int64)v160);
    return v134;
  }
  return result;
}
