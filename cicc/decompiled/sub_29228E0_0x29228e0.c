// Function: sub_29228E0
// Address: 0x29228e0
//
void __fastcall sub_29228E0(
        __int64 a1,
        char a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __m128i *v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rax
  bool v20; // zf
  __int64 v21; // rax
  int v22; // esi
  __int64 v23; // rdx
  __int64 v24; // rdx
  __m128i *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __m128i *v28; // rsi
  __m128i *v29; // r12
  __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // r15
  __int64 v34; // rax
  _QWORD *v35; // rax
  int v36; // esi
  __int64 v37; // rdx
  __int64 v38; // rdx
  __m128i *v39; // rax
  __m128i *v40; // rsi
  __int64 v41; // r12
  __int64 v42; // r15
  __int64 v43; // r14
  __int64 v44; // r9
  __int64 v45; // rbx
  __int64 v46; // r13
  __int64 v47; // r9
  __int64 *v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rax
  __m128i *v51; // rbx
  __int64 v52; // rax
  __m128i *v53; // r13
  __int64 *v54; // r14
  __int64 v55; // rbx
  __int64 v56; // r12
  __int64 v57; // r13
  __int64 v58; // rsi
  __int64 v59; // r9
  __int64 *v60; // rdi
  __int64 v61; // rax
  __int64 v62; // rax
  _QWORD *v63; // r12
  __int64 v64; // rax
  __m128i *v65; // r13
  unsigned __int64 v66; // rdi
  __int64 v67; // rax
  _QWORD *v68; // rcx
  __int64 v69; // rdi
  __int64 v70; // rax
  __int64 v71; // rsi
  __int64 v72; // rax
  unsigned int v73; // r13d
  __int64 v74; // rdi
  __int64 v75; // r13
  __int64 v76; // rax
  __int64 v77; // r8
  __int64 v78; // r9
  int v79; // eax
  unsigned int v80; // esi
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // r13
  __int64 v84; // r8
  __int64 v85; // r9
  int v86; // eax
  unsigned int v87; // esi
  __int64 v88; // rax
  __int64 v89; // rdx
  unsigned int v90; // r13d
  __int64 *v91; // rdi
  __int64 v92; // rax
  __int64 v93; // rdx
  unsigned int v94; // r13d
  __int64 *v95; // rdi
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rbx
  _QWORD *v100; // rdx
  __int64 v101; // rax
  unsigned int v102; // [rsp+4h] [rbp-37Ch]
  int v103; // [rsp+10h] [rbp-370h]
  __int64 v104; // [rsp+18h] [rbp-368h]
  __int64 v105; // [rsp+18h] [rbp-368h]
  unsigned int v106; // [rsp+18h] [rbp-368h]
  __int64 v109; // [rsp+30h] [rbp-350h]
  __int64 v110; // [rsp+30h] [rbp-350h]
  unsigned int v111; // [rsp+30h] [rbp-350h]
  __int64 v113; // [rsp+38h] [rbp-348h]
  __int64 v114; // [rsp+40h] [rbp-340h]
  __int64 v115; // [rsp+40h] [rbp-340h]
  __int64 v116; // [rsp+40h] [rbp-340h]
  __int64 v117; // [rsp+48h] [rbp-338h]
  __int64 v118; // [rsp+48h] [rbp-338h]
  __int64 v119; // [rsp+48h] [rbp-338h]
  __int64 v120; // [rsp+48h] [rbp-338h]
  __int64 v121; // [rsp+48h] [rbp-338h]
  __int64 v122; // [rsp+48h] [rbp-338h]
  char v123; // [rsp+50h] [rbp-330h]
  char v124; // [rsp+50h] [rbp-330h]
  _QWORD *v125; // [rsp+50h] [rbp-330h]
  _QWORD *v126; // [rsp+50h] [rbp-330h]
  unsigned int v127; // [rsp+50h] [rbp-330h]
  unsigned int v128; // [rsp+50h] [rbp-330h]
  __int64 v130; // [rsp+58h] [rbp-328h]
  int v131; // [rsp+60h] [rbp-320h]
  int v132; // [rsp+60h] [rbp-320h]
  __int64 v135; // [rsp+70h] [rbp-310h]
  __m128i *v136; // [rsp+70h] [rbp-310h]
  __int64 v137; // [rsp+70h] [rbp-310h]
  unsigned __int64 v138; // [rsp+80h] [rbp-300h] BYREF
  __int64 v139; // [rsp+88h] [rbp-2F8h]
  __int64 v140; // [rsp+90h] [rbp-2F0h] BYREF
  __int64 v141; // [rsp+98h] [rbp-2E8h]
  _QWORD *v142; // [rsp+A0h] [rbp-2E0h] BYREF
  __int64 v143; // [rsp+A8h] [rbp-2D8h]
  __m128i v144; // [rsp+B0h] [rbp-2D0h] BYREF
  __int64 v145; // [rsp+C0h] [rbp-2C0h]
  __m128i v146; // [rsp+D0h] [rbp-2B0h] BYREF
  __int64 v147; // [rsp+E0h] [rbp-2A0h]
  __m128i v148; // [rsp+F0h] [rbp-290h] BYREF
  __int64 v149; // [rsp+100h] [rbp-280h]
  __int64 v150; // [rsp+110h] [rbp-270h] BYREF
  __int64 v151; // [rsp+118h] [rbp-268h]
  __int64 v152; // [rsp+120h] [rbp-260h]
  unsigned int v153; // [rsp+128h] [rbp-258h]
  __m128i v154; // [rsp+130h] [rbp-250h] BYREF
  __m128i v155; // [rsp+140h] [rbp-240h] BYREF
  __int64 v156; // [rsp+150h] [rbp-230h]
  __m128i v157; // [rsp+160h] [rbp-220h] BYREF
  _BYTE v158[48]; // [rsp+170h] [rbp-210h] BYREF
  __m128i v159; // [rsp+1A0h] [rbp-1E0h] BYREF
  __m128i v160; // [rsp+1B0h] [rbp-1D0h] BYREF
  __int64 v161; // [rsp+1C0h] [rbp-1C0h]

  if ( (*(_BYTE *)(a5 + 7) & 0x20) == 0 )
    return;
  v114 = sub_B91C10(a5, 38);
  if ( v114 )
  {
    v109 = sub_AE94B0(v114);
    v114 = v8;
    if ( (*(_BYTE *)(a5 + 7) & 0x20) == 0 )
      goto LABEL_124;
  }
  else
  {
    if ( (*(_BYTE *)(a5 + 7) & 0x20) == 0 )
      return;
    v109 = 0;
  }
  v9 = sub_B91C10(a5, 38);
  if ( v9 )
  {
    v10 = *(_QWORD *)(v9 + 8);
    v11 = (__m128i *)(v10 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v10 & 4) == 0 )
      v11 = 0;
    sub_B967C0(&v157, v11);
    if ( v109 != v114 || v157.m128i_i32[2] )
      goto LABEL_9;
    v66 = v157.m128i_i64[0];
    if ( (_BYTE *)v157.m128i_i64[0] != v158 )
    {
LABEL_90:
      _libc_free(v66);
      return;
    }
    return;
  }
LABEL_124:
  v157.m128i_i64[0] = (__int64)v158;
  v157.m128i_i64[1] = 0x600000000LL;
  if ( v109 != v114 )
  {
LABEL_9:
    v150 = 0;
    v151 = 0;
    v152 = 0;
    v153 = 0;
    if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    {
      v12 = sub_B91C10(a1, 38);
      if ( v12 )
      {
        v13 = sub_AE94B0(v12);
        v135 = v14;
        if ( v14 != v13 )
        {
          v15 = v13;
          while ( 1 )
          {
            v16 = *(_QWORD *)(v15 + 24);
            v17 = *(_QWORD *)(*(_QWORD *)(v16 + 32 * (2LL - (*(_DWORD *)(v16 + 4) & 0x7FFFFFF))) + 24LL);
            sub_AF47B0((__int64)&v146, *(unsigned __int64 **)(v17 + 16), *(unsigned __int64 **)(v17 + 24));
            v18 = sub_B10D40(v16 + 48);
            v19 = *(_QWORD *)(*(_QWORD *)(v16 + 32 * (1LL - (*(_DWORD *)(v16 + 4) & 0x7FFFFFF))) + 24LL);
            v160.m128i_i8[8] = 0;
            v161 = v18;
            v159.m128i_i64[0] = v19;
            v20 = (unsigned __int8)sub_2921690((__int64)&v150, (__int64)&v159, &v148) == 0;
            v21 = v148.m128i_i64[0];
            if ( v20 )
              break;
LABEL_19:
            v24 = v147;
            v25 = (__m128i *)(v21 + 40);
            *v25 = _mm_loadu_si128(&v146);
            v25[1].m128i_i64[0] = v24;
            v15 = *(_QWORD *)(v15 + 8);
            if ( v15 == v135 )
              goto LABEL_20;
          }
          v22 = v153;
          v154.m128i_i64[0] = v148.m128i_i64[0];
          ++v150;
          if ( 4 * ((int)v152 + 1) >= 3 * v153 )
          {
            v22 = 2 * v153;
          }
          else if ( v153 - HIDWORD(v152) - ((_DWORD)v152 + 1) > v153 >> 3 )
          {
            LODWORD(v152) = v152 + 1;
            if ( *(_QWORD *)v148.m128i_i64[0] )
            {
LABEL_17:
              --HIDWORD(v152);
LABEL_18:
              *(__m128i *)v21 = _mm_loadu_si128(&v159);
              *(__m128i *)(v21 + 16) = _mm_loadu_si128(&v160);
              v23 = v161;
              *(_QWORD *)(v21 + 56) = 0;
              *(_QWORD *)(v21 + 32) = v23;
              *(_OWORD *)(v21 + 40) = 0;
              goto LABEL_19;
            }
LABEL_121:
            if ( !*(_BYTE *)(v21 + 24) && !*(_QWORD *)(v21 + 32) )
              goto LABEL_18;
            goto LABEL_17;
          }
          sub_2922710((__int64)&v150, v22);
          sub_2921690((__int64)&v150, (__int64)&v159, &v154);
          v21 = v154.m128i_i64[0];
          LODWORD(v152) = v152 + 1;
          if ( *(_QWORD *)v154.m128i_i64[0] )
            goto LABEL_17;
          goto LABEL_121;
        }
      }
LABEL_20:
      if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
      {
        v26 = sub_B91C10(a1, 38);
        if ( v26 )
        {
          v27 = *(_QWORD *)(v26 + 8);
          v28 = (__m128i *)(v27 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v27 & 4) == 0 )
            v28 = 0;
          sub_B967C0(&v159, v28);
          v29 = (__m128i *)v159.m128i_i64[0];
          v136 = (__m128i *)(v159.m128i_i64[0] + 8LL * v159.m128i_u32[2]);
          if ( (__m128i *)v159.m128i_i64[0] != v136 )
          {
            while ( 1 )
            {
              v30 = v29->m128i_i64[0];
              v31 = sub_B11F60(v29->m128i_i64[0] + 80);
              sub_AF47B0((__int64)&v144, *(unsigned __int64 **)(v31 + 16), *(unsigned __int64 **)(v31 + 24));
              v32 = *(_QWORD *)(v30 + 24);
              v148.m128i_i64[0] = v32;
              if ( v32 )
                sub_B96E90((__int64)&v148, v32, 1);
              v33 = sub_B10D40((__int64)&v148);
              v34 = sub_B12000(v30 + 72);
              v155.m128i_i8[8] = 0;
              v154.m128i_i64[0] = v34;
              v156 = v33;
              if ( v148.m128i_i64[0] )
                sub_B91220((__int64)&v148, v148.m128i_i64[0]);
              v20 = (unsigned __int8)sub_2921690((__int64)&v150, (__int64)&v154, &v142) == 0;
              v35 = v142;
              if ( !v20 )
                goto LABEL_35;
              v36 = v153;
              v148.m128i_i64[0] = (__int64)v142;
              ++v150;
              if ( 4 * ((int)v152 + 1) >= 3 * v153 )
              {
                v36 = 2 * v153;
              }
              else if ( v153 - HIDWORD(v152) - ((_DWORD)v152 + 1) > v153 >> 3 )
              {
                LODWORD(v152) = v152 + 1;
                if ( !*v142 )
                  goto LABEL_116;
                goto LABEL_33;
              }
              sub_2922710((__int64)&v150, v36);
              sub_2921690((__int64)&v150, (__int64)&v154, &v148);
              v35 = (_QWORD *)v148.m128i_i64[0];
              LODWORD(v152) = v152 + 1;
              if ( !*(_QWORD *)v148.m128i_i64[0] )
              {
LABEL_116:
                if ( !*((_BYTE *)v35 + 24) && !v35[4] )
                  goto LABEL_34;
              }
LABEL_33:
              --HIDWORD(v152);
LABEL_34:
              *(__m128i *)v35 = _mm_loadu_si128(&v154);
              *((__m128i *)v35 + 1) = _mm_loadu_si128(&v155);
              v37 = v156;
              v35[7] = 0;
              v35[4] = v37;
              *(_OWORD *)(v35 + 5) = 0;
LABEL_35:
              v38 = v145;
              v39 = (__m128i *)(v35 + 5);
              v29 = (__m128i *)((char *)v29 + 8);
              *v39 = _mm_loadu_si128(&v144);
              v39[1].m128i_i64[0] = v38;
              if ( v136 == v29 )
              {
                v136 = (__m128i *)v159.m128i_i64[0];
                break;
              }
            }
          }
          if ( v136 != &v160 )
            _libc_free((unsigned __int64)v136);
        }
      }
    }
    v103 = sub_BD5C60(a6);
    v40 = (__m128i *)sub_B43CA0(a5);
    sub_AE0470((__int64)&v159, v40->m128i_i64, 0, 0);
    v130 = 0;
    if ( v109 != v114 )
    {
      v41 = v104;
      v42 = v109;
      while ( 1 )
      {
        v43 = *(_QWORD *)(v42 + 24);
        v123 = 0;
        v44 = v43 + 48;
        v45 = *(_QWORD *)(*(_QWORD *)(v43 + 32 * (2LL - (*(_DWORD *)(v43 + 4) & 0x7FFFFFF))) + 24LL);
        if ( a2 )
        {
          v67 = sub_B10D40(v43 + 48);
          v105 = v67;
          v40 = (__m128i *)v67;
          v68 = *(_QWORD **)(*(_QWORD *)(v43 + 32 * (1LL - (*(_DWORD *)(v43 + 4) & 0x7FFFFFF))) + 24LL);
          v125 = v68;
          v119 = v151;
          if ( !v153 )
            goto LABEL_44;
          v148.m128i_i64[0] = v67;
          v40 = (__m128i *)&v140;
          v154.m128i_i64[0] = 0;
          v155.m128i_i8[8] = 0;
          v156 = 0;
          LODWORD(v140) = 0;
          v142 = v68;
          v131 = 1;
          v102 = v153 - 1;
          v111 = v102 & sub_F11290((__int64 *)&v142, &v140, v148.m128i_i64);
          while ( 1 )
          {
            v69 = v119 + ((unsigned __int64)v111 << 6);
            if ( v125 == *(_QWORD **)v69 && !*(_BYTE *)(v69 + 24) && v105 == *(_QWORD *)(v69 + 32) )
              break;
            v40 = &v154;
            if ( sub_F34140(v69, (__int64)&v154) )
              goto LABEL_44;
            v111 = v102 & (v131 + v111);
            ++v131;
          }
          v123 = 0;
          if ( v69 == v151 + ((unsigned __int64)v153 << 6) )
            goto LABEL_44;
          v148 = _mm_loadu_si128((const __m128i *)(v69 + 40));
          v149 = *(_QWORD *)(v69 + 56);
          sub_AF47B0((__int64)&v154, *(unsigned __int64 **)(v45 + 16), *(unsigned __int64 **)(v45 + 24));
          v40 = (__m128i *)a3;
          v83 = v154.m128i_i64[1];
          v86 = sub_29136F0(
                  *(_QWORD *)(*(_QWORD *)(v43 + 32 * (1LL - (*(_DWORD *)(v43 + 4) & 0x7FFFFFF))) + 24LL),
                  a3,
                  a4,
                  &v138,
                  v84,
                  v85,
                  v148.m128i_u64[0],
                  v148.m128i_i64[1],
                  v149,
                  v154.m128i_i64[0],
                  v154.m128i_u64[1],
                  v155.m128i_i8[0]);
          if ( v86 == 2 )
            goto LABEL_44;
          v44 = v43 + 48;
          if ( !v86 )
          {
            v87 = v139;
            if ( v155.m128i_i8[0] )
            {
              if ( v154.m128i_i64[0] == v138 && v83 == v139 )
                goto LABEL_46;
              v87 = v139 - v83;
              v139 -= v83;
            }
            v88 = sub_B0E470(v45, v87, v138);
            v44 = v43 + 48;
            v141 = v89;
            v140 = v88;
            if ( (_BYTE)v89 )
            {
              v45 = v140;
            }
            else
            {
              v94 = v138;
              v128 = v139;
              v95 = (__int64 *)(*(_QWORD *)(v45 + 8) & 0xFFFFFFFFFFFFFFF8LL);
              if ( (*(_QWORD *)(v45 + 8) & 4) != 0 )
                v95 = (__int64 *)*v95;
              v96 = sub_B0D000(v95, 0, 0, 0, 1);
              v97 = sub_B0E470(v96, v128, v94);
              v44 = v43 + 48;
              v99 = v98;
              v100 = (_QWORD *)v97;
              v101 = v99;
              v142 = v100;
              v45 = (__int64)v100;
              v143 = v101;
              v123 = a2;
            }
          }
        }
LABEL_46:
        if ( !v130 )
        {
          v121 = v44;
          v130 = sub_AF40E0(v103, 1u);
          sub_B99FD0(a6, 0x26u, v130);
          v44 = v121;
        }
        v46 = a8;
        if ( !a8 )
        {
          v120 = v44;
          v70 = sub_B58EB0(v43, 0);
          v44 = v120;
          v46 = v70;
        }
        v47 = sub_B10CD0(v44);
        v48 = (__int64 *)(*(_QWORD *)(v45 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v45 + 8) & 4) != 0 )
          v48 = (__int64 *)*v48;
        v117 = v47;
        v49 = sub_B0D000(v48, 0, 0, 0, 1);
        v50 = sub_ADE690(
                &v159,
                a6,
                v46,
                *(_QWORD *)(*(_QWORD *)(v43 + 32 * (1LL - (*(_DWORD *)(v43 + 4) & 0x7FFFFFF))) + 24LL),
                v45,
                a7,
                v49,
                v117);
        v51 = (__m128i *)sub_291D890(v50);
        if ( a8
          && ((v52 = *(_DWORD *)(v43 + 4) & 0x7FFFFFF, **(_BYTE **)(*(_QWORD *)(v43 - 32 * v52) + 24LL) == 4)
           || !sub_AF4590(*(_QWORD *)(*(_QWORD *)(v43 + 32 * (2 - v52)) + 24LL)))
          || v123 )
        {
          sub_F507F0((__int64)v51);
        }
        LOWORD(v41) = 0;
        v53 = v51 + 3;
        sub_B444E0(v51, v43 + 24, v41);
        v40 = *(__m128i **)(v43 + 48);
        v154.m128i_i64[0] = (__int64)v40;
        if ( v40 )
        {
          sub_B96E90((__int64)&v154, (__int64)v40, 1);
          if ( v53 == &v154 )
          {
            v40 = (__m128i *)v154.m128i_i64[0];
            if ( v154.m128i_i64[0] )
              sub_B91220((__int64)&v154, v154.m128i_i64[0]);
            goto LABEL_44;
          }
          v40 = (__m128i *)v51[3].m128i_i64[0];
          if ( !v40 )
            goto LABEL_60;
        }
        else
        {
          if ( v53 == &v154 )
            goto LABEL_44;
          v40 = (__m128i *)v51[3].m128i_i64[0];
          if ( !v40 )
            goto LABEL_44;
        }
        sub_B91220((__int64)v51[3].m128i_i64, (__int64)v40);
LABEL_60:
        v40 = (__m128i *)v154.m128i_i64[0];
        v51[3].m128i_i64[0] = v154.m128i_i64[0];
        if ( v40 )
        {
          sub_B976B0((__int64)&v154, (unsigned __int8 *)v40, (__int64)v51[3].m128i_i64);
          v42 = *(_QWORD *)(v42 + 8);
          if ( v42 == v114 )
            break;
        }
        else
        {
LABEL_44:
          v42 = *(_QWORD *)(v42 + 8);
          if ( v42 == v114 )
            break;
        }
      }
    }
    v54 = (__int64 *)v157.m128i_i64[0];
    v110 = v157.m128i_i64[0] + 8LL * v157.m128i_u32[2];
    if ( v157.m128i_i64[0] == v110 )
    {
LABEL_89:
      sub_AE9130((__int64)&v159, (__int64)v40);
      sub_C7D6A0(v151, (unsigned __int64)v153 << 6, 8);
      v66 = v157.m128i_i64[0];
      if ( (_BYTE *)v157.m128i_i64[0] == v158 )
        return;
      goto LABEL_90;
    }
    while ( 1 )
    {
      v55 = *v54;
      v113 = *v54 + 80;
      v124 = 0;
      v56 = sub_B11F60(v113);
      v137 = v55 + 72;
      if ( a2 )
      {
        v71 = *(_QWORD *)(v55 + 24);
        v154.m128i_i64[0] = v71;
        if ( v71 )
          sub_B96E90((__int64)&v154, v71, 1);
        v122 = sub_B10D40((__int64)&v154);
        v72 = sub_B12000(v137);
        v40 = (__m128i *)v154.m128i_i64[0];
        v126 = (_QWORD *)v72;
        if ( v154.m128i_i64[0] )
          sub_B91220((__int64)&v154, v154.m128i_i64[0]);
        v116 = v151;
        if ( !v153 )
          goto LABEL_67;
        v40 = (__m128i *)&v140;
        v154.m128i_i64[0] = 0;
        v73 = v153 - 1;
        v155.m128i_i8[8] = 0;
        v148.m128i_i64[0] = v122;
        v156 = 0;
        LODWORD(v140) = 0;
        v142 = v126;
        v132 = 1;
        v106 = v73 & sub_F11290((__int64 *)&v142, &v140, v148.m128i_i64);
        while ( 1 )
        {
          v74 = v116 + ((unsigned __int64)v106 << 6);
          if ( v126 == *(_QWORD **)v74 && !*(_BYTE *)(v74 + 24) && v122 == *(_QWORD *)(v74 + 32) )
            break;
          v40 = &v154;
          if ( sub_F34140(v74, (__int64)&v154) )
            goto LABEL_67;
          v106 = v73 & (v132 + v106);
          ++v132;
        }
        v124 = 0;
        if ( v74 == v151 + ((unsigned __int64)v153 << 6) )
          goto LABEL_67;
        v148 = _mm_loadu_si128((const __m128i *)(v74 + 40));
        v149 = *(_QWORD *)(v74 + 56);
        sub_AF47B0((__int64)&v154, *(unsigned __int64 **)(v56 + 16), *(unsigned __int64 **)(v56 + 24));
        v75 = v154.m128i_i64[1];
        v76 = sub_B12000(v137);
        v40 = (__m128i *)a3;
        v79 = sub_29136F0(
                v76,
                a3,
                a4,
                &v138,
                v77,
                v78,
                v148.m128i_u64[0],
                v148.m128i_i64[1],
                v149,
                v154.m128i_i64[0],
                v154.m128i_u64[1],
                v155.m128i_i8[0]);
        if ( v79 == 2 )
          goto LABEL_67;
        if ( !v79 )
        {
          v80 = v139;
          if ( v155.m128i_i8[0] )
          {
            if ( v138 == v154.m128i_i64[0] && v139 == v75 )
              goto LABEL_69;
            v80 = v139 - v75;
            v139 -= v75;
          }
          v81 = sub_B0E470(v56, v80, v138);
          v141 = v82;
          v140 = v81;
          if ( (_BYTE)v82 )
          {
            v56 = v140;
          }
          else
          {
            v90 = v138;
            v127 = v139;
            v91 = (__int64 *)(*(_QWORD *)(v56 + 8) & 0xFFFFFFFFFFFFFFF8LL);
            if ( (*(_QWORD *)(v56 + 8) & 4) != 0 )
              v91 = (__int64 *)*v91;
            v92 = sub_B0D000(v91, 0, 0, 0, 1);
            v142 = (_QWORD *)sub_B0E470(v92, v127, v90);
            v56 = (__int64)v142;
            v143 = v93;
            v124 = a2;
          }
        }
      }
LABEL_69:
      if ( !v130 )
      {
        v130 = sub_AF40E0(v103, 1u);
        sub_B99FD0(a6, 0x26u, v130);
      }
      v57 = a8;
      if ( !a8 )
        v57 = sub_B12A50(v55, 0);
      v58 = *(_QWORD *)(v55 + 24);
      v154.m128i_i64[0] = v58;
      if ( v58 )
        sub_B96E90((__int64)&v154, v58, 1);
      v59 = sub_B10CD0((__int64)&v154);
      v60 = (__int64 *)(*(_QWORD *)(v56 + 8) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(v56 + 8) & 4) != 0 )
        v60 = (__int64 *)*v60;
      v115 = v59;
      v118 = sub_B0D000(v60, 0, 0, 0, 1);
      v61 = sub_B12000(v137);
      v62 = sub_ADE690(&v159, a6, v57, v61, v56, a7, v118, v115);
      v63 = (_QWORD *)sub_291D880(v62);
      if ( v154.m128i_i64[0] )
        sub_B91220((__int64)&v154, v154.m128i_i64[0]);
      if ( a8 && (**(_BYTE **)(v55 + 40) == 4 || (v64 = sub_B11F60(v113), !sub_AF4590(v64))) || v124 )
        sub_B13710((__int64)v63);
      v65 = (__m128i *)(v63 + 3);
      sub_B14390(v63, (__int64 *)v55);
      v40 = *(__m128i **)(v55 + 24);
      v154.m128i_i64[0] = (__int64)v40;
      if ( v40 )
      {
        sub_B96E90((__int64)&v154, (__int64)v40, 1);
        if ( v65 == &v154 )
        {
          v40 = (__m128i *)v154.m128i_i64[0];
          if ( v154.m128i_i64[0] )
            sub_B91220((__int64)&v154, v154.m128i_i64[0]);
          goto LABEL_67;
        }
        v40 = (__m128i *)v63[3];
        if ( !v40 )
          goto LABEL_87;
      }
      else
      {
        if ( v65 == &v154 )
          goto LABEL_67;
        v40 = (__m128i *)v63[3];
        if ( !v40 )
          goto LABEL_67;
      }
      sub_B91220((__int64)(v63 + 3), (__int64)v40);
LABEL_87:
      v40 = (__m128i *)v154.m128i_i64[0];
      v63[3] = v154.m128i_i64[0];
      if ( v40 )
      {
        ++v54;
        sub_B976B0((__int64)&v154, (unsigned __int8 *)v40, (__int64)(v63 + 3));
        if ( (__int64 *)v110 == v54 )
          goto LABEL_89;
      }
      else
      {
LABEL_67:
        if ( (__int64 *)v110 == ++v54 )
          goto LABEL_89;
      }
    }
  }
}
