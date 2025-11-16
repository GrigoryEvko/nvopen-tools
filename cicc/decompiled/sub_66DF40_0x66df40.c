// Function: sub_66DF40
// Address: 0x66df40
//
__int64 __fastcall sub_66DF40(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        _DWORD *a6,
        __int64 a7)
{
  __int64 v8; // r14
  bool v10; // bl
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rbx
  __int64 v20; // rax
  char v21; // bl
  unsigned __int16 v22; // cx
  __int64 v23; // r13
  __int64 v24; // rdx
  int v25; // eax
  char i; // al
  unsigned int v27; // r14d
  __int16 v28; // r12
  unsigned __int64 v29; // rsi
  __int64 v30; // rdi
  _DWORD *v31; // rcx
  unsigned __int64 v32; // xmm0_8
  __m128i v33; // xmm2
  __m128i v34; // xmm3
  __int64 v35; // rdx
  const __m128i *v36; // r14
  __int8 v37; // al
  _BOOL4 v38; // r14d
  __int64 v39; // r12
  const __m128i *v40; // rdx
  __m128i *v41; // rax
  const __m128i *v42; // roff
  __m128i v43; // xmm4
  const __m128i *v44; // rdx
  __m128i *v45; // rax
  const __m128i *v46; // rdi
  __int64 v47; // rax
  __int64 v48; // r14
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 m; // rax
  __int64 v52; // r12
  __int64 v53; // r13
  __int64 v54; // r15
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  unsigned __int8 v58; // al
  __int64 v60; // rdi
  __int64 v61; // rax
  __m128i v62; // xmm5
  __m128i v63; // xmm6
  __m128i v64; // xmm7
  __int64 v65; // rdx
  __int64 v66; // rcx
  const __m128i *v67; // rdi
  const __m128i *v68; // r12
  __int8 v69; // al
  __int64 v70; // rdi
  const __m128i *v71; // rdx
  __m128i *v72; // rax
  const __m128i *v73; // roff
  const __m128i *v74; // rdi
  unsigned __int16 *v75; // rdi
  __int64 k; // rax
  __int64 v77; // rax
  char v78; // dl
  char v79; // al
  __m128i *v80; // r12
  __m128i *v81; // r13
  __int64 v82; // rbx
  unsigned __int8 v83; // al
  __int64 v84; // rax
  const __m128i *v85; // rdx
  __m128i *v86; // rax
  const __m128i *v87; // roff
  __int16 v88; // ax
  __int64 v89; // rax
  __int64 v90; // r12
  __int64 v91; // r14
  const __m128i *v92; // r12
  __m128i *v93; // rax
  __m128i *v94; // r12
  _QWORD *v95; // rdi
  __m128i **v96; // rdx
  __int8 v97; // al
  __int64 v98; // rsi
  __int64 v99; // rdi
  __int64 j; // rax
  __int64 v101; // rax
  __int64 v103; // [rsp+10h] [rbp-170h]
  __int64 v104; // [rsp+18h] [rbp-168h]
  unsigned int v105; // [rsp+20h] [rbp-160h]
  int v106; // [rsp+24h] [rbp-15Ch]
  const __m128i *v107; // [rsp+28h] [rbp-158h]
  __int64 v108; // [rsp+30h] [rbp-150h]
  bool v109; // [rsp+3Bh] [rbp-145h]
  unsigned int v110; // [rsp+3Ch] [rbp-144h]
  __int64 v112; // [rsp+48h] [rbp-138h]
  int v113; // [rsp+50h] [rbp-130h]
  unsigned int v114; // [rsp+54h] [rbp-12Ch]
  unsigned __int8 v116; // [rsp+92h] [rbp-EEh]
  _BOOL4 v117; // [rsp+94h] [rbp-ECh]
  unsigned int v118; // [rsp+98h] [rbp-E8h]
  int v119; // [rsp+9Ch] [rbp-E4h]
  __int16 v120; // [rsp+A2h] [rbp-DEh]
  int v121; // [rsp+A4h] [rbp-DCh]
  __int16 v122; // [rsp+A8h] [rbp-D8h]
  __int16 v123; // [rsp+AAh] [rbp-D6h]
  unsigned int v124; // [rsp+ACh] [rbp-D4h]
  __int64 v125; // [rsp+B0h] [rbp-D0h]
  _QWORD *v126; // [rsp+B8h] [rbp-C8h]
  int v127; // [rsp+C0h] [rbp-C0h]
  bool v128; // [rsp+C5h] [rbp-BBh]
  __int16 v129; // [rsp+C6h] [rbp-BAh]
  __int64 v130; // [rsp+C8h] [rbp-B8h]
  unsigned int v131; // [rsp+DCh] [rbp-A4h] BYREF
  __m128i *v132; // [rsp+E0h] [rbp-A0h] BYREF
  __m128i *v133; // [rsp+E8h] [rbp-98h] BYREF
  const __m128i *v134; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v135; // [rsp+F8h] [rbp-88h] BYREF
  __int64 v136; // [rsp+100h] [rbp-80h] BYREF
  __m128i *v137; // [rsp+108h] [rbp-78h] BYREF
  __m128i v138; // [rsp+110h] [rbp-70h] BYREF
  __m128i v139; // [rsp+120h] [rbp-60h]
  __m128i v140; // [rsp+130h] [rbp-50h]
  __m128i v141; // [rsp+140h] [rbp-40h]

  v8 = a1;
  v112 = *(_QWORD *)a1;
  v10 = (*(_BYTE *)(a1 + 161) & 0x10) != 0;
  v109 = v10;
  v117 = v10;
  v132 = (__m128i *)sub_724DC0(a1, a2, a3, a4, a5, a6);
  v133 = (__m128i *)sub_724DC0(a1, a2, v11, v12, v13, v14);
  v134 = (const __m128i *)sub_724DC0(a1, a2, v15, v16, v17, v18);
  v108 = *(_QWORD *)(*(_QWORD *)(a1 + 176) + 8LL);
  if ( !a5 )
  {
    LODWORD(v19) = 0;
LABEL_79:
    v103 = *(_QWORD *)(v112 + 96);
    if ( v108 )
    {
      if ( (unsigned int)sub_8DBE70(v108) )
      {
        v116 = byte_4CFDE80;
      }
      else
      {
        v84 = v108;
        if ( *(_BYTE *)(v108 + 140) == 12 )
        {
          do
            v84 = *(_QWORD *)(v84 + 160);
          while ( *(_BYTE *)(v84 + 140) == 12 );
        }
        else
        {
          v84 = v108;
        }
        v116 = *(_BYTE *)(v84 + 160);
      }
    }
    else
    {
      v116 = 13;
    }
    goto LABEL_5;
  }
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 176) + 8LL) || v10 )
  {
    LODWORD(v19) = 0;
    if ( (*(_BYTE *)(a1 + 162) & 0x18) == 0x10 && *(char *)(a5 + 177) < 0 )
    {
      if ( v117 && sub_5F2660() )
        LODWORD(v19) = v117;
      else
        v19 = (a3 >> 9) & 1;
    }
    goto LABEL_79;
  }
  v116 = 13;
  LODWORD(v19) = 0;
  v103 = *(_QWORD *)(v112 + 96);
LABEL_5:
  v135 = *(_QWORD *)&dword_4F063F8;
  sub_854980(v112, 0);
  if ( word_4D04898
    && !(dword_4F077BC | dword_4D0488C)
    && unk_4F04C50
    && (*(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 193LL) & 2) != 0 )
  {
    if ( !(_DWORD)qword_4F077B4 || qword_4F077A0 <= 0x765Bu || (v99 = 5, !(unsigned int)sub_729F80(dword_4F063F8)) )
      v99 = 8;
    sub_684AA0(v99, 2407, &v135);
  }
  v114 = -1;
  v20 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(_BYTE *)(v20 + 4) == 7 )
    v114 = *(_DWORD *)v20;
  v105 = 0;
  if ( (_DWORD)v19 )
  {
    v105 = dword_4F06650[0];
    sub_7BDB60(1);
  }
  if ( word_4F06418[0] == 73 )
    v136 = *(_QWORD *)&dword_4F063F8;
  else
    v136 = unk_4F077C8;
  sub_7BE280(73, 130, 0, 0);
  if ( v117 )
    *(_QWORD *)(v8 + 168) = sub_8600D0(16, 0xFFFFFFFFLL, v8, 0);
  **(_BYTE **)(v8 + 176) |= 1u;
  if ( dword_4F077C4 == 2 || (v21 = dword_4F077C0, dword_4F077C0) )
  {
    v104 = 0;
    v21 = *(_QWORD *)(v112 + 72) != 0;
  }
  else
  {
    v104 = sub_7259C0(2);
    v88 = *(_WORD *)(v104 + 160);
    *(_QWORD *)(v104 + 168) = v8;
    *(_WORD *)(v104 + 160) = v88 & 0xF700 | 5;
    sub_8D6090(v104);
  }
  v22 = word_4F06418[0];
  if ( word_4F06418[0] == 74 && dword_4F077C4 == 2 )
  {
    v107 = 0;
    v106 = 1;
    v113 = 0;
    goto LABEL_63;
  }
  if ( !v108 && v109 )
  {
    v108 = sub_72BA30(5);
    v22 = word_4F06418[0];
  }
  v23 = v8;
  v130 = 0;
  v107 = 0;
  v106 = 1;
  v24 = qword_4F061C8;
  v25 = 5;
  v113 = 0;
  ++*(_BYTE *)(qword_4F061C8 + 82LL);
  if ( v116 != 13 )
    v25 = v116;
  v128 = !v109 && v114 != -1;
  v110 = v25;
  for ( i = *(_BYTE *)(v24 + 75); ; *(_BYTE *)(qword_4F061C8 + 75LL) = i )
  {
    v27 = unk_4F07370;
    v28 = unk_4F07374;
    LODWORD(v29) = unk_4F07378;
    v30 = unk_4F0737C;
    *(_BYTE *)(v24 + 75) = i + 1;
    ++*(_BYTE *)(v24 + 64);
    v124 = v27;
    v123 = v28;
    v127 = v29;
    v129 = v30;
    if ( v22 == 1 )
    {
      v125 = sub_869D30();
      v62 = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
      v63 = _mm_loadu_si128(&xmmword_4D04A20);
      v64 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
      v138 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
      v139 = v62;
      v118 = dword_4F063F8;
      v140 = v63;
      v122 = unk_4F063FC;
      v141 = v64;
      v29 = (unsigned int)qword_4F063F0;
      v119 = qword_4F063F0;
      v120 = WORD2(qword_4F063F0);
      sub_7B8B50(v30, (unsigned int)qword_4F063F0, v65, v66);
      v31 = dword_4F07508;
      v126 = 0;
      *(_QWORD *)dword_4F07508 = v138.m128i_i64[1];
      if ( unk_4D043CC )
      {
        v30 = 19;
        v126 = (_QWORD *)sub_5CC190(19);
      }
    }
    else
    {
      sub_7BE280(1, 40, 0, 0);
      v30 = (unsigned __int16)v30;
      v122 = v28;
      v29 = (unsigned int)v29;
      v118 = v27;
      v32 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v120 = v30;
      v33 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v34 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v119 = v29;
      v139 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v138.m128i_i64[0] = v32;
      v139.m128i_i8[1] |= 0x20u;
      v126 = 0;
      v138.m128i_i64[1] = *(_QWORD *)dword_4F07508;
      v125 = 0;
      v140 = v33;
      v141 = v34;
    }
    v35 = word_4F06418[0];
    if ( (_DWORD)qword_4F077B4 && word_4F06418[0] == 142 )
    {
      v30 = 12;
      v77 = sub_5CC190(12);
      if ( v77 )
      {
        while ( 1 )
        {
          v78 = *(_BYTE *)(v77 + 8);
          if ( v78 != 6 && v78 != 82 )
            break;
          v77 = *(_QWORD *)v77;
          if ( !v77 )
            goto LABEL_142;
        }
        v29 = (unsigned __int64)&dword_4F063F8;
        v30 = 3569;
        sub_684B30(3569, &dword_4F063F8);
        v35 = word_4F06418[0];
      }
      else
      {
LABEL_142:
        v35 = word_4F06418[0];
      }
    }
    v36 = v134;
    --*(_BYTE *)(qword_4F061C8 + 64LL);
    v36[9].m128i_i64[0] = 0;
    if ( (_WORD)v35 == 56 )
    {
      sub_7B8B50(v30, v29, v35, v31);
      v124 = dword_4F063F8;
      v123 = unk_4F063FC;
      sub_6C9ED0(v108, 1, v134);
      sub_73A770(v134);
      v68 = v134;
      v134[10].m128i_i8[9] &= ~2u;
      v127 = unk_4F061D8;
      v129 = unk_4F061DC;
      v69 = v68[10].m128i_i8[13];
      if ( !v69 )
        goto LABEL_93;
      if ( v69 == 12 )
      {
        v121 = 1;
        v38 = 0;
        goto LABEL_39;
      }
      if ( v116 == 13 && !v109 )
      {
        v121 = unk_4D042DC;
        if ( unk_4D042DC )
        {
          v121 = 0;
          v38 = 0;
          goto LABEL_39;
        }
        v38 = sub_621140((__int64)v68, (__int64)v68, byte_4CFDE80);
        if ( v38 )
        {
          v38 = 0;
          goto LABEL_39;
        }
        if ( !dword_4D04964 || byte_4F07472[0] != 8 )
        {
          for ( j = v134[8].m128i_i64[0]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          if ( *(_QWORD *)(j + 128) <= unk_4F06B20 )
          {
            LODWORD(v137) = 0;
            v101 = sub_72BA30(5);
            sub_712540(v134, v101, 1, 1, &v137, dword_4F07508);
            if ( dword_4D04964 )
              sub_684B30(66, dword_4F07508);
            goto LABEL_94;
          }
        }
LABEL_38:
        v38 = 1;
        sub_6851C0(66, dword_4F07508);
        v121 = 0;
        goto LABEL_39;
      }
      v91 = sub_72BA30(v110);
      v121 = sub_621140((__int64)v68, (__int64)v68, v110);
      if ( v121 )
      {
        v98 = v91;
        v38 = 0;
        LODWORD(v137) = 0;
        sub_712540(v68, v98, 1, 1, &v137, dword_4F07508);
        v121 = 0;
      }
      else
      {
        v38 = 1;
        sub_685360(1749, dword_4F07508);
      }
    }
    else
    {
      if ( !v130 )
      {
        v67 = v36;
        v38 = 0;
        sub_72BAF0(v67, 0, v110);
        v121 = 0;
        goto LABEL_39;
      }
      v37 = v36[10].m128i_i8[13];
      if ( !v37 )
      {
LABEL_93:
        v38 = 1;
LABEL_94:
        v121 = 0;
        goto LABEL_39;
      }
      if ( v37 == 12 )
      {
        v74 = v36;
        v38 = 0;
        sub_740980(v74);
        v121 = 1;
        goto LABEL_39;
      }
      if ( v116 != 13 || v109 )
      {
        sub_72BA30(v110);
        v121 = sub_621210((__int64)v36, v110);
        if ( v121 )
        {
          v38 = 1;
          sub_685360(1749, dword_4F07508);
          v121 = 0;
        }
        else
        {
          v75 = (unsigned __int16 *)&v36[11];
          v38 = 0;
          sub_621300(v75);
        }
      }
      else
      {
        v121 = sub_621210((__int64)v36, byte_4CFDE80);
        if ( v121 )
          goto LABEL_38;
        for ( k = v134[8].m128i_i64[0]; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        if ( sub_621210((__int64)v134, *(_BYTE *)(k + 160)) )
        {
          v92 = v134;
          v92[8].m128i_i64[0] = sub_72BA30((unsigned __int8)byte_4CFDE80);
        }
        v38 = 0;
        sub_621300((unsigned __int16 *)&v134[11]);
      }
    }
LABEL_39:
    if ( dword_4F077BC )
    {
      if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) == 6 )
      {
        sub_7CFB70(&v138, 64);
        if ( v139.m128i_i64[1] )
        {
          if ( *(_BYTE *)(v139.m128i_i64[1] + 80) == 16 && (*(_BYTE *)(v139.m128i_i64[1] + 96) & 4) == 0 )
            sub_881DB0(v139.m128i_i64[1]);
        }
      }
    }
    if ( v128 )
    {
      v39 = sub_886170(&v138, a5, v114);
LABEL_43:
      *a6 = 1;
      if ( v38 )
        goto LABEL_90;
      goto LABEL_44;
    }
    v39 = sub_647630(2u, (__int64)&v138, (unsigned int)dword_4F04C5C, 0);
    if ( !v117 )
      goto LABEL_43;
    if ( v38 )
    {
LABEL_90:
      sub_72C970(v134);
      v106 = 0;
      goto LABEL_47;
    }
LABEL_44:
    if ( !v121 )
    {
      if ( v113 )
      {
        if ( (int)sub_621060((__int64)v134, (__int64)v132) <= 0 )
        {
          if ( (int)sub_621060((__int64)v134, (__int64)v133) < 0 )
          {
            v85 = v134;
            v86 = v133;
            v87 = v134;
            *v133 = _mm_loadu_si128(v134);
            v86[1] = _mm_loadu_si128(v87 + 1);
            v86[2] = _mm_loadu_si128(v85 + 2);
            v86[3] = _mm_loadu_si128(v85 + 3);
            v86[4] = _mm_loadu_si128(v85 + 4);
            v86[5] = _mm_loadu_si128(v85 + 5);
            v86[6] = _mm_loadu_si128(v85 + 6);
            v86[7] = _mm_loadu_si128(v85 + 7);
            v86[8] = _mm_loadu_si128(v85 + 8);
            v86[9] = _mm_loadu_si128(v85 + 9);
            v86[10] = _mm_loadu_si128(v85 + 10);
            v86[11] = _mm_loadu_si128(v85 + 11);
            v86[12] = _mm_loadu_si128(v85 + 12);
          }
        }
        else
        {
          v71 = v134;
          v72 = v132;
          v73 = v134;
          *v132 = _mm_loadu_si128(v134);
          v72[1] = _mm_loadu_si128(v73 + 1);
          v72[2] = _mm_loadu_si128(v71 + 2);
          v72[3] = _mm_loadu_si128(v71 + 3);
          v72[4] = _mm_loadu_si128(v71 + 4);
          v72[5] = _mm_loadu_si128(v71 + 5);
          v72[6] = _mm_loadu_si128(v71 + 6);
          v72[7] = _mm_loadu_si128(v71 + 7);
          v72[8] = _mm_loadu_si128(v71 + 8);
          v72[9] = _mm_loadu_si128(v71 + 9);
          v72[10] = _mm_loadu_si128(v71 + 10);
          v72[11] = _mm_loadu_si128(v71 + 11);
          v72[12] = _mm_loadu_si128(v71 + 12);
        }
      }
      else
      {
        v40 = v134;
        v41 = v132;
        v42 = v134;
        *v132 = _mm_loadu_si128(v134);
        v41[1] = _mm_loadu_si128(v42 + 1);
        v41[2] = _mm_loadu_si128(v40 + 2);
        v41[3] = _mm_loadu_si128(v40 + 3);
        v41[4] = _mm_loadu_si128(v40 + 4);
        v41[5] = _mm_loadu_si128(v40 + 5);
        v41[6] = _mm_loadu_si128(v40 + 6);
        v41[7] = _mm_loadu_si128(v40 + 7);
        v41[8] = _mm_loadu_si128(v40 + 8);
        v41[9] = _mm_loadu_si128(v40 + 9);
        v41[10] = _mm_loadu_si128(v40 + 10);
        v41[11] = _mm_loadu_si128(v40 + 11);
        v43 = _mm_loadu_si128(v40 + 12);
        v44 = v134;
        v41[12] = v43;
        v45 = v133;
        *v133 = _mm_loadu_si128(v44);
        v113 = 1;
        v45[1] = _mm_loadu_si128(v44 + 1);
        v45[2] = _mm_loadu_si128(v44 + 2);
        v45[3] = _mm_loadu_si128(v44 + 3);
        v45[4] = _mm_loadu_si128(v44 + 4);
        v45[5] = _mm_loadu_si128(v44 + 5);
        v45[6] = _mm_loadu_si128(v44 + 6);
        v45[7] = _mm_loadu_si128(v44 + 7);
        v45[8] = _mm_loadu_si128(v44 + 8);
        v45[9] = _mm_loadu_si128(v44 + 9);
        v45[10] = _mm_loadu_si128(v44 + 10);
        v45[11] = _mm_loadu_si128(v44 + 11);
        v45[12] = _mm_loadu_si128(v44 + 12);
      }
    }
LABEL_47:
    sub_7296C0(&v131);
    v46 = v134;
    if ( v134[10].m128i_i8[13] == 12 || (v21 & 1) != 0 )
    {
      sub_70FDD0(v134, v134, v134[8].m128i_i64[0], 0);
      v46 = v134;
    }
    v47 = sub_740630(v46);
    *(_BYTE *)(v47 + 170) |= 0x10u;
    v48 = v47;
    if ( v128 )
    {
      *(_QWORD *)(v47 + 40) = *(_QWORD *)(*(_QWORD *)(a5 + 168) + 152LL);
    }
    else
    {
      v60 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
      if ( (unsigned __int8)(*(_BYTE *)(v60 + 4) - 1) <= 1u )
      {
        sub_732EF0(v60);
        v60 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
      }
      v61 = *(_QWORD *)(v60 + 184);
      *(_QWORD *)(v48 + 40) = v61;
      if ( v61 && (*(_BYTE *)(v61 - 8) & 1) == 0 )
        *(_QWORD *)(v48 + 40) = 0;
    }
    sub_729730(v131);
    sub_877D80(v48, v39);
    *(_BYTE *)(v48 + 88) = *(_BYTE *)(v23 + 88) & 0x70 | *(_BYTE *)(v48 + 88) & 0x8F;
    *(_QWORD *)(v39 + 88) = v48;
    if ( dword_4F077C0 )
      goto LABEL_54;
    if ( dword_4F077C4 != 2 )
    {
      *(_QWORD *)(v48 + 128) = v104;
LABEL_54:
      sub_8756F0(3, v39, &v138.m128i_u64[1], v125);
      if ( v130 )
        goto LABEL_55;
      if ( v117 )
        goto LABEL_98;
      goto LABEL_118;
    }
    if ( v117 )
    {
      sub_8756F0(3, v39, &v138.m128i_u64[1], v125);
      if ( v130 )
        goto LABEL_55;
LABEL_98:
      v107 = (const __m128i *)v48;
      *(_QWORD *)(*(_QWORD *)(v23 + 168) + 96LL) = v48;
      goto LABEL_56;
    }
    if ( a5 )
    {
      sub_877E20(v39, v48, a5);
      *(_BYTE *)(v48 + 88) = *(_BYTE *)(v23 + 88) & 3 | *(_BYTE *)(v48 + 88) & 0xFC;
    }
    else if ( (*(_BYTE *)(v112 + 81) & 0x10) == 0 && *(_QWORD *)(v112 + 64) )
    {
      sub_877E90(v39, v48);
    }
    sub_8756F0(3, v39, &v138.m128i_u64[1], v125);
    if ( v130 )
    {
LABEL_55:
      *(_QWORD *)(v130 + 120) = v48;
      goto LABEL_56;
    }
LABEL_118:
    *(_QWORD *)(v23 + 168) = v48;
    v107 = (const __m128i *)v48;
LABEL_56:
    if ( v126 )
      sub_5CEC90(v126, v48, 2);
    v49 = *(_QWORD *)(v48 + 72);
    if ( v49 )
    {
      *(_DWORD *)v49 = v118;
      *(_WORD *)(v49 + 12) = v120;
      *(_WORD *)(v49 + 4) = v122;
      *(_WORD *)(v49 + 36) = v123;
      *(_DWORD *)(v49 + 8) = v119;
      *(_WORD *)(v49 + 44) = v129;
      *(_DWORD *)(v49 + 32) = v124;
      *(_DWORD *)(v49 + 40) = v127;
    }
    v137 = *(__m128i **)&dword_4F063F8;
    if ( !(unsigned int)sub_7BE800(67) )
    {
      v8 = v23;
      v50 = qword_4F061C8;
      --*(_BYTE *)(qword_4F061C8 + 75LL);
      goto LABEL_62;
    }
    if ( word_4F06418[0] == 74 )
      break;
    v130 = v48;
    v24 = qword_4F061C8;
    v22 = word_4F06418[0];
    i = *(_BYTE *)(qword_4F061C8 + 75LL) - 1;
  }
  v8 = v23;
  if ( dword_4F077C4 == 1 )
    goto LABEL_120;
  if ( dword_4F077C4 == 2 )
  {
    if ( unk_4F07778 > 201102 || dword_4F07774 )
      goto LABEL_120;
LABEL_109:
    if ( v116 != 13 || v109 )
      goto LABEL_120;
    v70 = 4;
    if ( dword_4D04964 )
      v70 = unk_4F07471;
    sub_684AA0(v70, 228, &v137);
    v50 = qword_4F061C8;
    --*(_BYTE *)(qword_4F061C8 + 75LL);
  }
  else
  {
    if ( unk_4F07778 <= 199900 )
      goto LABEL_109;
LABEL_120:
    v50 = qword_4F061C8;
    --*(_BYTE *)(qword_4F061C8 + 75LL);
  }
LABEL_62:
  --*(_BYTE *)(v50 + 82);
LABEL_63:
  v138.m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)(a7 + 40) = *(_QWORD *)&dword_4F063F8;
  if ( v105 )
  {
    for ( m = *(_QWORD *)(v112 + 64); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
      ;
    v52 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)m + 96LL) + 80LL);
    v53 = *(_QWORD *)(v103 + 32);
    v54 = sub_888280(v112, v53, v105, dword_4F06650[0]);
    sub_7BDC00();
    sub_7AE700(unk_4F061C0 + 24LL, *(unsigned int *)(v54 + 24), dword_4F06650[0], 1, v53);
    if ( v117 )
      sub_7AE210(v53);
    sub_879080(v53, 0, *(_QWORD *)(v52 + 32));
  }
  sub_7BE280(74, 67, 3196, &v136);
  if ( v117 )
    sub_863FC0(74, 67, v55, v56, v57);
  if ( HIDWORD(qword_4F077B4) )
  {
    if ( word_4F06418[0] == 142 )
    {
      v93 = (__m128i *)sub_5CC970(3);
      v94 = v93;
      if ( v93 )
      {
        v137 = v93;
        v138.m128i_i64[0] = unk_4D04178;
        *(_QWORD *)(a7 + 40) = unk_4F061D8;
        v95 = (_QWORD *)(a2 + 208);
        if ( *(_QWORD *)(a2 + 208) )
          v95 = sub_5CB9F0((_QWORD **)v95);
        v96 = &v137;
        do
        {
          v97 = v94->m128i_i8[8];
          if ( v97 == 51 || v97 == 57 )
          {
            v96 = (__m128i **)v94;
            v94 = (__m128i *)v94->m128i_i64[0];
          }
          else
          {
            *v95 = v94;
            v94->m128i_i8[10] = 5;
            v94 = (__m128i *)(*v96)->m128i_i64[0];
            *v96 = v94;
            v95 = (_QWORD *)*v95;
          }
        }
        while ( v94 );
        sub_66A990(v137, v8, a2, 1, 0, 0);
      }
    }
  }
  sub_869D70(v8, 6);
  v58 = v116;
  if ( v116 != 13 )
  {
LABEL_74:
    *(_BYTE *)(v8 + 160) = v58;
    goto LABEL_75;
  }
  v79 = *(_BYTE *)(v8 + 161);
  v80 = v132;
  v81 = v133;
  if ( (v79 & 0x10) != 0 || !unk_4D042D8 && ((v79 & 0x20) == 0 || qword_4F077A8 <= 0x9C3Fu) && !unk_4F072F0 )
  {
LABEL_149:
    if ( v113 )
      goto LABEL_175;
    goto LABEL_150;
  }
  if ( !v113 || sub_621140((__int64)v133, (__int64)v132, byte_4F068B0[0]) )
  {
    *(_BYTE *)(v8 + 160) = byte_4F068B0[0];
    goto LABEL_149;
  }
  if ( sub_621140((__int64)v81, (__int64)v80, 1u) )
  {
    *(_BYTE *)(v8 + 160) = 1;
    goto LABEL_150;
  }
  if ( sub_621140((__int64)v81, (__int64)v80, 2u) )
  {
    *(_BYTE *)(v8 + 160) = 2;
    goto LABEL_150;
  }
  if ( sub_621140((__int64)v81, (__int64)v80, 3u) )
  {
    *(_BYTE *)(v8 + 160) = 3;
    goto LABEL_150;
  }
  if ( unk_4F06B30 < unk_4F06B20 && sub_621140((__int64)v81, (__int64)v80, 4u) )
  {
    *(_BYTE *)(v8 + 160) = 4;
    goto LABEL_150;
  }
LABEL_175:
  if ( unk_4D042DC && *(_BYTE *)(v8 + 160) == 5 )
  {
    if ( sub_621140((__int64)v81, (__int64)v80, 5u) )
    {
      *(_BYTE *)(v8 + 160) = 5;
    }
    else if ( sub_621140((__int64)v81, (__int64)v80, 6u) )
    {
      *(_BYTE *)(v8 + 160) = 6;
    }
    else if ( sub_621140((__int64)v81, (__int64)v80, 7u) )
    {
      *(_BYTE *)(v8 + 160) = 7;
    }
    else if ( sub_621140((__int64)v81, (__int64)v80, 8u) )
    {
      *(_BYTE *)(v8 + 160) = 8;
    }
    else
    {
      if ( dword_4D04964 && !unk_4D04298 )
        goto LABEL_239;
      if ( sub_621140((__int64)v81, (__int64)v80, 9u) )
      {
        *(_BYTE *)(v8 + 160) = 9;
        goto LABEL_150;
      }
      if ( (!dword_4D04964 || unk_4D04298) && sub_621140((__int64)v81, (__int64)v80, 0xAu) )
      {
        *(_BYTE *)(v8 + 160) = 10;
      }
      else
      {
LABEL_239:
        *(_BYTE *)(v8 + 160) = byte_4CFDE80;
        if ( v106 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
          sub_684AA0(dword_4D04964 == 0 ? 5 : 8, 1420, v8 + 64);
      }
    }
  }
LABEL_150:
  if ( !HIDWORD(qword_4F077B4) || v109 )
    goto LABEL_75;
  v82 = *(unsigned __int8 *)(v8 + 160);
  if ( dword_4F077C0 )
  {
    v89 = sub_7259C0(2);
    *(_BYTE *)(v89 + 161) &= ~8u;
    v90 = v89;
    *(_BYTE *)(v89 + 160) = v82;
    *(_QWORD *)(v89 + 168) = v8;
    sub_8D6090(v89);
    sub_6671B0(v107, v90);
    if ( v107 )
      goto LABEL_154;
LABEL_157:
    if ( dword_4F077C0 )
    {
      v58 = byte_4B6DF80[v82];
      goto LABEL_74;
    }
    goto LABEL_215;
  }
  if ( !v107 )
  {
LABEL_215:
    **(_BYTE **)(v8 + 176) |= 4u;
    goto LABEL_75;
  }
LABEL_154:
  if ( v113 )
  {
    v83 = byte_4B6DF80[(unsigned __int8)v82];
    if ( v83 != (_BYTE)v82 && sub_621140((__int64)v133, (__int64)v132, v83) )
      goto LABEL_157;
  }
LABEL_75:
  sub_8D6090(v8);
  *(_BYTE *)(v8 + 141) &= ~0x20u;
  if ( dword_4F077C4 == 2 )
    sub_6671B0(v107, v8);
  sub_880400(v112);
  sub_6030F0((unsigned int *)&v138);
  sub_724E30(&v132);
  sub_724E30(&v133);
  return sub_724E30(&v134);
}
