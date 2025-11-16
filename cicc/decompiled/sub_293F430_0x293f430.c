// Function: sub_293F430
// Address: 0x293f430
//
__int64 __fastcall sub_293F430(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int64 v2; // r13
  __int64 v4; // rbx
  unsigned __int8 v5; // al
  __m128i v6; // xmm1
  __m128i v7; // xmm2
  unsigned __int8 v8; // r9
  __int64 v9; // r14
  unsigned int v10; // r15d
  __m128i v12; // xmm4
  unsigned int v13; // edi
  int v14; // edx
  __int64 v15; // r9
  unsigned int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // r8
  __int64 v25; // rdx
  _QWORD *v26; // rax
  unsigned __int64 v27; // rcx
  _BYTE *v28; // rdx
  __int64 v29; // rcx
  _BYTE *v30; // rax
  _BYTE *i; // rsi
  __int64 v32; // rdx
  char v33; // al
  __int64 v34; // r9
  unsigned int v35; // r13d
  __int64 v36; // r8
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  _BYTE *v39; // rbx
  unsigned __int64 v40; // r12
  unsigned __int64 v41; // rdi
  __int64 v42; // rbx
  __m128i *v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // r8
  unsigned __int64 v49; // r15
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // r8
  unsigned __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // r15
  unsigned __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // r8
  unsigned __int64 v59; // rdx
  __int64 v60; // rbx
  __int64 v61; // r8
  _BYTE *v62; // rdx
  _QWORD *v63; // rax
  _QWORD *j; // rdx
  _QWORD *v65; // rax
  _QWORD *v66; // rdx
  _QWORD *k; // rdx
  __int32 v68; // eax
  unsigned int v69; // r13d
  unsigned int m; // r15d
  __int64 v71; // rbx
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rax
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // rdx
  __int64 v80; // rax
  __m128i v81; // rax
  char v82; // al
  _QWORD *v83; // rcx
  unsigned __int64 v84; // rsi
  __int64 v85; // rax
  __int64 v86; // rdx
  __m128i v87; // xmm7
  __int64 v88; // [rsp+20h] [rbp-8A0h]
  __int64 v89; // [rsp+20h] [rbp-8A0h]
  unsigned __int8 v90; // [rsp+40h] [rbp-880h]
  __int64 v91; // [rsp+40h] [rbp-880h]
  unsigned __int64 v92; // [rsp+48h] [rbp-878h]
  unsigned __int8 v93; // [rsp+50h] [rbp-870h]
  __int64 v94; // [rsp+58h] [rbp-868h]
  __int64 v95; // [rsp+60h] [rbp-860h]
  __int64 v96; // [rsp+68h] [rbp-858h]
  __int64 v97; // [rsp+68h] [rbp-858h]
  unsigned __int8 v98; // [rsp+70h] [rbp-850h]
  int v99; // [rsp+70h] [rbp-850h]
  char v100; // [rsp+70h] [rbp-850h]
  unsigned __int8 v101; // [rsp+70h] [rbp-850h]
  __int64 v102; // [rsp+70h] [rbp-850h]
  unsigned __int8 v103; // [rsp+70h] [rbp-850h]
  unsigned __int8 v104; // [rsp+70h] [rbp-850h]
  char v105; // [rsp+70h] [rbp-850h]
  unsigned __int32 v106; // [rsp+70h] [rbp-850h]
  __int64 v107; // [rsp+70h] [rbp-850h]
  __int64 v108; // [rsp+78h] [rbp-848h]
  unsigned __int8 v109; // [rsp+78h] [rbp-848h]
  unsigned __int64 v110; // [rsp+78h] [rbp-848h]
  unsigned __int8 v111; // [rsp+78h] [rbp-848h]
  unsigned __int8 v112; // [rsp+83h] [rbp-83Dh]
  unsigned int v113; // [rsp+84h] [rbp-83Ch]
  unsigned __int8 v114; // [rsp+88h] [rbp-838h]
  unsigned __int8 v115; // [rsp+88h] [rbp-838h]
  __int64 v116; // [rsp+88h] [rbp-838h]
  int v117; // [rsp+88h] [rbp-838h]
  __m128i v118; // [rsp+90h] [rbp-830h] BYREF
  __m128i v119; // [rsp+A0h] [rbp-820h] BYREF
  __int64 v120; // [rsp+B0h] [rbp-810h]
  __m128i v121; // [rsp+C0h] [rbp-800h] BYREF
  __m128i v122; // [rsp+D0h] [rbp-7F0h]
  __int64 v123; // [rsp+E0h] [rbp-7E0h]
  _BYTE *v124; // [rsp+F0h] [rbp-7D0h] BYREF
  __int64 v125; // [rsp+F8h] [rbp-7C8h]
  _BYTE v126[32]; // [rsp+100h] [rbp-7C0h] BYREF
  __m128i v127; // [rsp+120h] [rbp-7A0h] BYREF
  __m128i v128; // [rsp+130h] [rbp-790h] BYREF
  __int64 v129; // [rsp+140h] [rbp-780h]
  _QWORD v130[4]; // [rsp+150h] [rbp-770h] BYREF
  __int16 v131; // [rsp+170h] [rbp-750h]
  __m128i v132; // [rsp+180h] [rbp-740h] BYREF
  __m128i v133; // [rsp+190h] [rbp-730h]
  __int64 v134; // [rsp+1A0h] [rbp-720h]
  void *v135; // [rsp+1B0h] [rbp-710h] BYREF
  __int64 v136; // [rsp+1B8h] [rbp-708h]
  _BYTE v137[48]; // [rsp+1C0h] [rbp-700h] BYREF
  _QWORD *v138; // [rsp+1F0h] [rbp-6D0h] BYREF
  __int64 v139; // [rsp+1F8h] [rbp-6C8h]
  _QWORD v140[8]; // [rsp+200h] [rbp-6C0h] BYREF
  _BYTE *v141; // [rsp+240h] [rbp-680h] BYREF
  __int64 v142; // [rsp+248h] [rbp-678h]
  _BYTE v143[64]; // [rsp+250h] [rbp-670h] BYREF
  __m128i v144; // [rsp+290h] [rbp-630h] BYREF
  _QWORD v145[2]; // [rsp+2A0h] [rbp-620h] BYREF
  char v146; // [rsp+2B0h] [rbp-610h]
  __m128i v147; // [rsp+2E0h] [rbp-5E0h] BYREF
  __int64 v148[2]; // [rsp+2F0h] [rbp-5D0h] BYREF
  __m128i v149; // [rsp+300h] [rbp-5C0h] BYREF
  __m128i v150; // [rsp+310h] [rbp-5B0h] BYREF
  __int8 v151; // [rsp+320h] [rbp-5A0h]
  __int64 v152; // [rsp+328h] [rbp-598h]
  char *v153; // [rsp+330h] [rbp-590h] BYREF
  char v154; // [rsp+340h] [rbp-580h] BYREF
  void *v155; // [rsp+360h] [rbp-560h]
  _BYTE *v156; // [rsp+380h] [rbp-540h] BYREF
  __int64 v157; // [rsp+388h] [rbp-538h]
  _BYTE v158[1328]; // [rsp+390h] [rbp-530h] BYREF

  v2 = (unsigned __int64)a2;
  if ( *(_DWORD *)(a1 + 1152) && !sub_293A020(a1, a2) )
    return 0;
  v4 = *((_QWORD *)a2 + 1);
  if ( *(_BYTE *)(v4 + 8) != 15 )
  {
    v123 = 0;
    v121 = 0;
    v122 = 0;
    goto LABEL_12;
  }
  v5 = sub_2939FC0(*((_QWORD *)a2 + 1));
  v121 = 0;
  v114 = v5;
  v123 = 0;
  v122 = 0;
  if ( !v5 )
  {
LABEL_12:
    sub_2939E80((__int64)&v118, a1, v4);
    v12 = _mm_loadu_si128(&v119);
    v8 = 0;
    v121 = _mm_loadu_si128(&v118);
    v123 = v120;
    v122 = v12;
    goto LABEL_5;
  }
  sub_2939E80((__int64)&v118, a1, **(_QWORD **)(v4 + 16));
  v6 = _mm_loadu_si128(&v118);
  v7 = _mm_loadu_si128(&v119);
  v8 = v114;
  v123 = v120;
  v121 = v6;
  v122 = v7;
LABEL_5:
  if ( !(_BYTE)v123 )
    return 0;
  v9 = *((_QWORD *)a2 - 4);
  if ( !v9 )
    return 0;
  if ( *(_BYTE *)v9 )
    return 0;
  if ( *((_QWORD *)a2 + 10) != *(_QWORD *)(v9 + 24) )
    return 0;
  v13 = *(_DWORD *)(v9 + 36);
  v115 = v8;
  v113 = v13;
  if ( !v13 )
    return 0;
  v10 = sub_9B74D0(v13, *(_QWORD *)(a1 + 1120));
  if ( !(_BYTE)v10 )
    return 0;
  v14 = *a2;
  v15 = v115;
  if ( v14 == 40 )
  {
    v16 = sub_B491D0((__int64)a2);
    v15 = v115;
    v116 = 32LL * v16;
  }
  else
  {
    v116 = 0;
    if ( v14 != 85 )
    {
      v116 = 64;
      if ( v14 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_54;
  v98 = v15;
  v17 = sub_BD2BC0((__int64)a2);
  v15 = v98;
  v108 = v18 + v17;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v108 >> 4) )
LABEL_153:
      BUG();
LABEL_54:
    v22 = 0;
    goto LABEL_28;
  }
  v19 = sub_BD2BC0((__int64)a2);
  v15 = v98;
  if ( !(unsigned int)((v108 - v19) >> 4) )
    goto LABEL_54;
  v109 = v98;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_153;
  v99 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v20 = sub_BD2BC0((__int64)a2);
  v15 = v109;
  v22 = 32LL * (unsigned int)(*(_DWORD *)(v20 + v21 - 4) - v99);
LABEL_28:
  v23 = *((_DWORD *)a2 + 1);
  v24 = 0x800000000LL;
  v139 = 0x800000000LL;
  v25 = 32LL * (v23 & 0x7FFFFFF) - 32 - v116 - v22;
  v26 = v140;
  v96 = v25 >> 5;
  v27 = (unsigned int)(v25 >> 5);
  v117 = v25 >> 5;
  v138 = v140;
  v110 = v27;
  if ( !v117 )
  {
    v157 = 0x800000000LL;
    v156 = v158;
    v135 = v137;
    HIDWORD(v136) = 12;
LABEL_41:
    LODWORD(v136) = v96;
    goto LABEL_42;
  }
  v28 = &v140[v27];
  if ( v27 > 8 )
  {
    v94 = v27;
    v103 = v15;
    sub_C8D5F0((__int64)&v138, v140, v27, 8u, 0x800000000LL, v15);
    v24 = 0x800000000LL;
    v15 = v103;
    v26 = &v138[(unsigned int)v139];
    v28 = &v138[v94];
    if ( &v138[v94] == v26 )
    {
      v157 = 0x800000000LL;
      LODWORD(v139) = v96;
      v156 = v158;
LABEL_146:
      v104 = v15;
      sub_293A5B0((__int64)&v156, v110, (__int64)v28, v29, 0x800000000LL, v15);
      v30 = v156;
      v15 = v104;
      v28 = &v156[160 * (unsigned int)v157];
      goto LABEL_34;
    }
  }
  do
  {
    if ( v26 )
      *v26 = 0;
    ++v26;
  }
  while ( v26 != (_QWORD *)v28 );
  v29 = 0x800000000LL;
  v157 = 0x800000000LL;
  LODWORD(v139) = v96;
  v30 = v158;
  v28 = v158;
  v156 = v158;
  if ( v110 > 8 )
    goto LABEL_146;
LABEL_34:
  for ( i = &v30[160 * v110]; i != v28; v28 += 160 )
  {
    if ( v28 )
    {
      memset(v28, 0, 0xA0u);
      *((_DWORD *)v28 + 23) = 8;
      *((_QWORD *)v28 + 10) = v28 + 96;
    }
  }
  LODWORD(v157) = v96;
  v135 = v137;
  v136 = 0xC00000000LL;
  if ( (__int64)v110 <= 12 )
  {
    if ( 4 * v110 )
    {
      v100 = v15;
      memset(v137, 255, 4 * v110);
      LOBYTE(v15) = v100;
    }
    goto LABEL_41;
  }
  v105 = v15;
  sub_C8D5F0((__int64)&v135, v137, v110, 4u, v24, v15);
  memset(v135, 255, 4 * v110);
  LOBYTE(v15) = v105;
  LODWORD(v136) = v96;
LABEL_42:
  v32 = *(_QWORD *)(a1 + 1120);
  v124 = v126;
  v101 = v15;
  v125 = 0x300000000LL;
  v33 = sub_9B76D0(v13, 0xFFFFFFFF, v32);
  v34 = v101;
  if ( v33 )
  {
    v57 = (unsigned int)v125;
    v58 = v122.m128i_i64[0];
    v59 = (unsigned int)v125 + 1LL;
    if ( v59 > HIDWORD(v125) )
    {
      v93 = v101;
      v107 = v122.m128i_i64[0];
      sub_C8D5F0((__int64)&v124, v126, v59, 8u, v122.m128i_i64[0], v34);
      v57 = (unsigned int)v125;
      v34 = v93;
      v58 = v107;
    }
    *(_QWORD *)&v124[8 * v57] = v58;
    LODWORD(v125) = v125 + 1;
  }
  if ( (_BYTE)v34 && *(_DWORD *)(v4 + 12) > 1u )
  {
    v92 = v2;
    v35 = 1;
    while ( 1 )
    {
      sub_2939E80((__int64)&v147, a1, *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL * v35));
      if ( !v149.m128i_i8[0] || v147.m128i_i32[2] != v121.m128i_i32[2] )
        break;
      if ( sub_9B7850(v13, v35, *(_QWORD *)(a1 + 1120)) )
      {
        v37 = (unsigned int)v125;
        v34 = v148[0];
        v38 = (unsigned int)v125 + 1LL;
        if ( v38 > HIDWORD(v125) )
        {
          v91 = v148[0];
          sub_C8D5F0((__int64)&v124, v126, v38, 8u, v36, v148[0]);
          v37 = (unsigned int)v125;
          v34 = v91;
        }
        *(_QWORD *)&v124[8 * v37] = v34;
        LODWORD(v125) = v125 + 1;
      }
      if ( ++v35 >= *(_DWORD *)(v4 + 12) )
      {
        v10 = (unsigned __int8)v10;
        v2 = v92;
        goto LABEL_70;
      }
    }
LABEL_55:
    v10 = 0;
    goto LABEL_56;
  }
LABEL_70:
  if ( (_DWORD)v96 )
  {
    v42 = 0;
    v90 = v10;
    do
    {
      v49 = *(_QWORD *)(v2 + 32 * (v42 - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)));
      v50 = *(_QWORD *)(v49 + 8);
      if ( *(_BYTE *)(v50 + 8) == 17 )
      {
        sub_2939E80((__int64)&v144, a1, v50);
        if ( !v146 || v144.m128i_i32[2] != v121.m128i_i32[2] )
          goto LABEL_55;
        sub_293CE40(&v147, (_QWORD *)a1, v2, v49, &v144);
        v43 = (__m128i *)&v156[160 * v42];
        v43->m128i_i64[0] = v147.m128i_i64[0];
        v43 += 5;
        v43[-5].m128i_i64[1] = v147.m128i_i64[1];
        v43[-4].m128i_i16[0] = v148[0];
        v43[-4].m128i_i64[1] = v148[1];
        v43[-3] = _mm_loadu_si128(&v149);
        v43[-2] = _mm_loadu_si128(&v150);
        v43[-1].m128i_i8[0] = v151;
        v43[-1].m128i_i64[1] = v152;
        sub_293A290((__int64)v43, &v153, v44, v45, v46, v47);
        if ( v153 != &v154 )
          _libc_free((unsigned __int64)v153);
        if ( sub_9B76D0(v113, v42, *(_QWORD *)(a1 + 1120)) )
        {
          *((_DWORD *)v135 + v42) = v125;
          v54 = (unsigned int)v125;
          v55 = v145[0];
          v56 = (unsigned int)v125 + 1LL;
          if ( v56 > HIDWORD(v125) )
          {
            sub_C8D5F0((__int64)&v124, v126, v56, 8u, v48, v34);
            v54 = (unsigned int)v125;
          }
          *(_QWORD *)&v124[8 * v54] = v55;
          LODWORD(v125) = v125 + 1;
        }
      }
      else
      {
        v138[v42] = v49;
        if ( sub_9B76D0(v113, v42, *(_QWORD *)(a1 + 1120)) )
        {
          v51 = (unsigned int)v125;
          v52 = *(_QWORD *)(v49 + 8);
          v53 = (unsigned int)v125 + 1LL;
          if ( v53 > HIDWORD(v125) )
          {
            v89 = *(_QWORD *)(v49 + 8);
            sub_C8D5F0((__int64)&v124, v126, v53, 8u, v52, v34);
            v51 = (unsigned int)v125;
            v52 = v89;
          }
          *(_QWORD *)&v124[8 * v51] = v52;
          LODWORD(v125) = v125 + 1;
        }
      }
      ++v42;
    }
    while ( v42 != v110 );
    v10 = v90;
  }
  v60 = v121.m128i_u32[3];
  v61 = v121.m128i_u32[3];
  v141 = v143;
  v142 = 0x800000000LL;
  if ( v121.m128i_i32[3] )
  {
    v62 = v143;
    v63 = v143;
    if ( v121.m128i_u32[3] > 8uLL )
    {
      v106 = v121.m128i_u32[3];
      sub_C8D5F0((__int64)&v141, v143, v121.m128i_u32[3], 8u, v121.m128i_u32[3], v34);
      v62 = v141;
      v61 = v106;
      v63 = &v141[8 * (unsigned int)v142];
    }
    for ( j = &v62[8 * v60]; j != v63; ++v63 )
    {
      if ( v63 )
        *v63 = 0;
    }
    LODWORD(v142) = v61;
  }
  v65 = v145;
  v144.m128i_i64[1] = 0x800000000LL;
  v66 = v145;
  v144.m128i_i64[0] = (__int64)v145;
  if ( v110 )
  {
    if ( v110 > 8 )
    {
      sub_C8D5F0((__int64)&v144, v145, v110, 8u, v61, v34);
      v66 = (_QWORD *)v144.m128i_i64[0];
      v65 = (_QWORD *)(v144.m128i_i64[0] + 8LL * v144.m128i_u32[2]);
    }
    for ( k = &v66[v110]; k != v65; ++v65 )
    {
      if ( v65 )
        *v65 = 0;
    }
    v144.m128i_i32[2] = v96;
  }
  v102 = sub_B6E160(*(__int64 **)(v9 + 40), v113, (__int64)v124, (unsigned int)v125);
  sub_23D0AB0((__int64)&v147, v2, 0, 0, 0);
  v68 = v121.m128i_i32[3];
  if ( v121.m128i_i32[3] )
  {
    v112 = v10;
    v97 = v2;
    v69 = 0;
    while ( v68 - 1 != v69 || !v122.m128i_i64[1] )
    {
      v144.m128i_i32[2] = 0;
      if ( v117 )
      {
        v111 = 0;
        goto LABEL_111;
      }
LABEL_124:
      LODWORD(v130[0]) = v69;
      v131 = 265;
      v81.m128i_i64[0] = (__int64)sub_BD5D20(v97);
      v127 = v81;
      v128.m128i_i64[0] = (__int64)".i";
      v82 = v131;
      LOWORD(v129) = 773;
      if ( (_BYTE)v131 )
      {
        if ( (_BYTE)v131 == 1 )
        {
          v132 = _mm_loadu_si128(&v127);
          v87 = _mm_loadu_si128(&v128);
          v134 = v129;
          v133 = v87;
        }
        else
        {
          if ( HIBYTE(v131) == 1 )
          {
            v95 = v130[1];
            v83 = (_QWORD *)v130[0];
          }
          else
          {
            v83 = v130;
            v82 = 2;
          }
          v133.m128i_i64[0] = (__int64)v83;
          v132.m128i_i64[0] = (__int64)&v127;
          v133.m128i_i64[1] = v95;
          LOBYTE(v134) = 2;
          BYTE1(v134) = v82;
        }
      }
      else
      {
        LOWORD(v134) = 256;
      }
      v84 = 0;
      if ( v102 )
        v84 = *(_QWORD *)(v102 + 24);
      v85 = sub_921880((unsigned int **)&v147, v84, v102, v144.m128i_i32[0], v144.m128i_i32[2], (__int64)&v132, 0);
      v86 = v69++;
      *(_QWORD *)&v141[8 * v86] = v85;
      v68 = v121.m128i_i32[3];
      if ( v121.m128i_i32[3] <= v69 )
      {
        v10 = v112;
        v2 = v97;
        goto LABEL_133;
      }
    }
    v144.m128i_i32[2] = 0;
    *(_QWORD *)v124 = v122.m128i_i64[1];
    v111 = v112;
    if ( v117 )
    {
LABEL_111:
      for ( m = 0; m != v117; ++m )
      {
        if ( sub_9B75A0(v113, m, *(_QWORD *)(a1 + 1120)) )
        {
          v71 = v138[m];
          v72 = v144.m128i_u32[2];
          v73 = v144.m128i_u32[2] + 1LL;
          if ( v73 > v144.m128i_u32[3] )
          {
            sub_C8D5F0((__int64)&v144, v145, v73, 8u, v74, v75);
            v72 = v144.m128i_u32[2];
          }
          *(_QWORD *)(v144.m128i_i64[0] + 8 * v72) = v71;
          ++v144.m128i_i32[2];
        }
        else
        {
          v76 = sub_293BC00((__int64)&v156[160 * m], v69);
          v79 = v144.m128i_u32[2];
          if ( (unsigned __int64)v144.m128i_u32[2] + 1 > v144.m128i_u32[3] )
          {
            v88 = v76;
            sub_C8D5F0((__int64)&v144, v145, v144.m128i_u32[2] + 1LL, 8u, v77, v78);
            v79 = v144.m128i_u32[2];
            v76 = v88;
          }
          *(_QWORD *)(v144.m128i_i64[0] + 8 * v79) = v76;
          ++v144.m128i_i32[2];
          if ( v111 && *((int *)v135 + m) >= 0 )
          {
            v80 = sub_293BC00((__int64)&v156[160 * m], v69);
            *(_QWORD *)&v124[8 * *((int *)v135 + m)] = *(_QWORD *)(v80 + 8);
          }
        }
      }
      if ( !v111 )
        goto LABEL_124;
    }
    v102 = sub_B6E160(*(__int64 **)(v9 + 40), v113, (__int64)v124, (unsigned int)v125);
    goto LABEL_124;
  }
LABEL_133:
  sub_293CAB0(a1, v2, (__int64)&v141, (__int64)&v121);
  nullsub_61();
  v155 = &unk_49DA100;
  nullsub_63();
  if ( (__int64 *)v147.m128i_i64[0] != v148 )
    _libc_free(v147.m128i_u64[0]);
  if ( (_QWORD *)v144.m128i_i64[0] != v145 )
    _libc_free(v144.m128i_u64[0]);
  if ( v141 != v143 )
    _libc_free((unsigned __int64)v141);
LABEL_56:
  if ( v124 != v126 )
    _libc_free((unsigned __int64)v124);
  if ( v135 != v137 )
    _libc_free((unsigned __int64)v135);
  v39 = v156;
  v40 = (unsigned __int64)&v156[160 * (unsigned int)v157];
  if ( v156 != (_BYTE *)v40 )
  {
    do
    {
      v40 -= 160LL;
      v41 = *(_QWORD *)(v40 + 80);
      if ( v41 != v40 + 96 )
        _libc_free(v41);
    }
    while ( v39 != (_BYTE *)v40 );
    v40 = (unsigned __int64)v156;
  }
  if ( (_BYTE *)v40 != v158 )
    _libc_free(v40);
  if ( v138 != v140 )
    _libc_free((unsigned __int64)v138);
  return v10;
}
