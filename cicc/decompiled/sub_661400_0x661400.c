// Function: sub_661400
// Address: 0x661400
//
__int64 __fastcall sub_661400(unsigned int *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 *a5)
{
  char v5; // al
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int16 v10; // ax
  _QWORD *v11; // rcx
  __m128i v12; // xmm5
  __m128i v13; // xmm6
  __m128i v14; // xmm7
  __int64 v15; // rdx
  int v16; // ebx
  __int64 v17; // r9
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // rbx
  __int64 v24; // r13
  char v25; // r13
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // r8
  int v29; // eax
  __int64 v30; // rsi
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // r14
  __int64 v34; // rdx
  __int64 v35; // rcx
  __m128i v36; // xmm5
  __m128i v37; // xmm6
  __m128i v38; // xmm7
  __int64 v39; // rcx
  __int64 v40; // rsi
  int v41; // eax
  __int64 v42; // rax
  unsigned int v43; // r15d
  char v44; // dl
  __int64 v45; // rsi
  __int64 v46; // rdi
  __int64 v47; // rdx
  __int64 v48; // r8
  _QWORD *v49; // rcx
  __int64 result; // rax
  _QWORD **v51; // rbx
  __int64 v52; // rax
  __m128i v53; // xmm1
  __m128i v54; // xmm2
  __m128i v55; // xmm3
  __m128i v56; // xmm5
  __m128i v57; // xmm6
  __m128i v58; // xmm7
  __int16 v59; // ax
  __m128i v60; // xmm5
  __m128i v61; // xmm6
  __m128i v62; // xmm7
  __int64 v63; // rdx
  __int64 v64; // rdx
  __int64 v65; // rdx
  __int64 v66; // rcx
  __m128i v67; // xmm1
  __m128i v68; // xmm2
  __m128i v69; // xmm3
  unsigned __int64 v70; // xmm0_8
  __m128i v71; // xmm2
  __m128i v72; // xmm3
  __int64 v73; // rax
  _BYTE *v74; // rdi
  __int64 v75; // r13
  __int64 v76; // rdx
  unsigned __int16 v77; // ax
  __m128i v78; // xmm1
  __m128i v79; // xmm2
  __m128i v80; // xmm3
  __int64 v81; // rax
  __int64 v82; // r13
  __int64 v83; // rdi
  char v84; // al
  __int64 v85; // rdx
  __int64 v86; // rdx
  unsigned __int64 v87; // xmm0_8
  __m128i v88; // xmm2
  __m128i v89; // xmm3
  __int16 v90; // [rsp+8h] [rbp-F8h]
  char v91; // [rsp+8h] [rbp-F8h]
  __int16 v92; // [rsp+10h] [rbp-F0h]
  int v93; // [rsp+10h] [rbp-F0h]
  unsigned int v94; // [rsp+10h] [rbp-F0h]
  __int16 v95; // [rsp+10h] [rbp-F0h]
  __int64 v96; // [rsp+10h] [rbp-F0h]
  __int16 v97; // [rsp+10h] [rbp-F0h]
  unsigned int v98; // [rsp+18h] [rbp-E8h]
  unsigned int v99; // [rsp+20h] [rbp-E0h]
  bool v100; // [rsp+27h] [rbp-D9h]
  unsigned int v101; // [rsp+28h] [rbp-D8h]
  __int16 v102; // [rsp+2Ch] [rbp-D4h]
  __int16 v103; // [rsp+2Eh] [rbp-D2h]
  int v105; // [rsp+38h] [rbp-C8h]
  unsigned int v106; // [rsp+3Ch] [rbp-C4h]
  unsigned int *v107; // [rsp+40h] [rbp-C0h]
  unsigned int v108; // [rsp+48h] [rbp-B8h]
  __int64 v109; // [rsp+50h] [rbp-B0h]
  unsigned int v110; // [rsp+58h] [rbp-A8h]
  int v111; // [rsp+5Ch] [rbp-A4h]
  unsigned int v112; // [rsp+64h] [rbp-9Ch] BYREF
  __m128i *v113; // [rsp+68h] [rbp-98h] BYREF
  __int64 v114; // [rsp+70h] [rbp-90h] BYREF
  __int64 v115; // [rsp+78h] [rbp-88h] BYREF
  __int64 v116; // [rsp+80h] [rbp-80h] BYREF
  __int64 v117; // [rsp+88h] [rbp-78h] BYREF
  __m128i v118; // [rsp+90h] [rbp-70h] BYREF
  __m128i v119; // [rsp+A0h] [rbp-60h]
  __m128i v120; // [rsp+B0h] [rbp-50h]
  __m128i v121; // [rsp+C0h] [rbp-40h]

  v107 = a1;
  v106 = a2;
  v108 = a3;
  v115 = *(_QWORD *)&dword_4F063F8;
  v112 = 0;
  v113 = 0;
  v116 = unk_4F077C8;
  v5 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 14);
  *a5 = 0;
  v100 = (v5 & 0x10) != 0;
  if ( (_DWORD)a2 )
  {
    v111 = 0;
    v114 = *a4;
  }
  else
  {
    if ( word_4F06418[0] == 154 )
    {
      sub_7B8B50(a1, a2, a3, a4);
      v108 = 1;
    }
    v114 = *(_QWORD *)&dword_4F063F8;
    if ( unk_4D04324 )
    {
      a2 = 877;
      a1 = &dword_4F063F8;
      sub_684AB0(&dword_4F063F8, 877);
    }
    sub_7B8B50(a1, a2, a3, a4);
    v111 = 1;
    if ( unk_4D043D8 )
    {
      if ( word_4F06418[0] == 25 )
      {
        if ( dword_4D043F8 )
        {
          a2 = 0;
          if ( (unsigned __int16)sub_7BE840(0, 0) == 25 )
            v116 = *(_QWORD *)&dword_4F063F8;
        }
      }
      v111 = 1;
      v113 = (__m128i *)sub_5CC190(15);
    }
  }
  sub_7296C0(&v118);
  v6 = sub_869D30();
  v7 = v118.m128i_u32[0];
  v109 = v6;
  sub_729730(v118.m128i_u32[0]);
  v10 = word_4F06418[0];
  if ( word_4F06418[0] == 1 )
  {
    v56 = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
    v57 = _mm_loadu_si128(&xmmword_4D04A20);
    v58 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
    v118 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
    v119 = v56;
    v59 = WORD2(qword_4F063F0);
    v120 = v57;
    v121 = v58;
    v105 = qword_4F063F0;
    v103 = v59;
    sub_7B8B50(v7, a2, v8, v9);
    v10 = word_4F06418[0];
    if ( word_4F06418[0] == 146 )
    {
      sub_7B8B50(v7, a2, v15, v11);
      v111 = unk_4D043D4;
      if ( unk_4D043D4 )
      {
        if ( dword_4F077C4 != 2 || unk_4F07778 <= 201702 )
        {
          if ( (_DWORD)qword_4F077B4 )
          {
            if ( !dword_4CFDE64 )
            {
              v7 = dword_4F063F8;
              if ( !(unsigned int)sub_729F80(dword_4F063F8) )
              {
                a2 = (__int64)&dword_4F063F8;
                v7 = 2922;
                sub_684B30(2922, &dword_4F063F8);
                dword_4CFDE64 = 1;
              }
            }
          }
        }
        v11 = (_QWORD *)dword_4D043D0;
        v99 = dword_4D043D0;
        v77 = word_4F06418[0];
        if ( dword_4D043D0 )
        {
          v99 = 0;
          if ( word_4F06418[0] == 154 )
          {
            if ( dword_4F077C4 != 2 || unk_4F07778 <= 202001 )
            {
              if ( (_DWORD)qword_4F077B4 )
              {
                if ( !dword_4CFDE60 )
                {
                  v7 = dword_4F063F8;
                  if ( !(unsigned int)sub_729F80(dword_4F063F8) )
                  {
                    a2 = (__int64)&dword_4F063F8;
                    v7 = 2945;
                    sub_684B30(2945, &dword_4F063F8);
                    dword_4CFDE60 = 1;
                  }
                }
              }
            }
            sub_7B8B50(v7, a2, v76, v11);
            v77 = word_4F06418[0];
            v99 = 1;
          }
        }
        if ( v77 == 1 )
        {
          v110 = 1;
          v111 = 0;
          goto LABEL_11;
        }
        a2 = (__int64)&dword_4F063F8;
        v7 = 40;
        sub_6851C0(40, &dword_4F063F8);
        v112 = 1;
        v110 = 1;
        v78 = _mm_loadu_si128(&xmmword_4F06660[1]);
        v111 = 0;
        v79 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v80 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v118.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
        v119 = v78;
        v120 = v79;
        v119.m128i_i8[1] = v78.m128i_i8[1] | 0x20;
        v118.m128i_i64[1] = *(_QWORD *)dword_4F07508;
        v10 = word_4F06418[0];
        v121 = v80;
      }
      else
      {
        v7 = 283;
        a2 = (__int64)&v118.m128i_i64[1];
        sub_6851C0(283, &v118.m128i_u64[1]);
        v112 = 1;
        v87 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
        v88 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v89 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v119 = _mm_loadu_si128(&xmmword_4F06660[1]);
        v118.m128i_i64[0] = v87;
        v119.m128i_i8[1] |= 0x20u;
        v120 = v88;
        v118.m128i_i64[1] = *(_QWORD *)dword_4F07508;
        v121 = v89;
LABEL_261:
        v10 = word_4F06418[0];
        while ( v10 == 1 )
        {
          sub_7B8B50(283, &v118.m128i_u64[1], v15, v11);
          v10 = word_4F06418[0];
          if ( word_4F06418[0] == 146 )
          {
            sub_7B8B50(283, &v118.m128i_u64[1], v15, v11);
            goto LABEL_261;
          }
        }
        v99 = 0;
        v110 = 0;
      }
    }
    else
    {
      v99 = 0;
      v110 = 0;
      v111 = 0;
    }
  }
  else
  {
    v105 = unk_4F077C8;
    v103 = unk_4F077CC;
    if ( dword_4F077C4 != 2 )
    {
LABEL_5:
      v11 = &unk_4F077C8;
      v99 = 0;
      v110 = 0;
      v12 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v13 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v14 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v15 = unk_4F077C8;
      v118.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v119 = v12;
      v118.m128i_i64[1] = unk_4F077C8;
      v120 = v13;
      v121 = v14;
      goto LABEL_6;
    }
    a2 = 0;
    v7 = 0;
    if ( !(unsigned int)sub_7C0F00(0, 0) )
    {
      v10 = word_4F06418[0];
      goto LABEL_5;
    }
    v67 = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
    v68 = _mm_loadu_si128(&xmmword_4D04A20);
    v69 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
    v118 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
    v119 = v67;
    v120 = v68;
    v121 = v69;
    if ( (v67.m128i_i8[1] & 0x20) == 0 )
    {
      if ( (v119.m128i_i8[0] & 0x58) != 0 )
      {
        v7 = 502;
        a2 = (__int64)dword_4F07508;
      }
      else
      {
        if ( (v119.m128i_i8[0] & 1) == 0 )
          sub_721090(0);
        v7 = 283;
        a2 = (__int64)dword_4F07508;
      }
      sub_6851C0(v7, dword_4F07508);
      v112 = 1;
      v70 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v71 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v72 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v119 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v119.m128i_i8[1] |= 0x20u;
      v118.m128i_i64[0] = v70;
      v118.m128i_i64[1] = *(_QWORD *)dword_4F07508;
      v120 = v71;
      v121 = v72;
    }
    sub_7B8B50(v7, a2, v65, v66);
    v10 = word_4F06418[0];
    v99 = 0;
    v110 = 0;
    v111 = 0;
  }
LABEL_6:
  if ( v10 != 142 )
    goto LABEL_7;
  if ( !dword_4D043E0 || dword_4F077BC && qword_4F077A8 <= 0x9D07u )
  {
    v11 = (_QWORD *)v110;
    v15 = ((unsigned __int8)v111 ^ 1) & 1;
    if ( !v110 )
    {
LABEL_142:
      v52 = qword_4F061C8;
      v11 = (_QWORD *)*(unsigned __int8 *)(qword_4F061C8 + 83LL);
      *(_BYTE *)(qword_4F061C8 + 83LL) = (_BYTE)v11 + 1;
      a2 = *(unsigned __int8 *)(v52 + 81);
      v7 = (unsigned int)(a2 + 1);
      *(_BYTE *)(v52 + 81) = a2 + 1;
      if ( !(_BYTE)v15 && !v113 )
      {
        v7 = 1;
        sub_7BE280(1, 40, 0, 0);
        v52 = qword_4F061C8;
        a2 = (unsigned int)*(unsigned __int8 *)(qword_4F061C8 + 81LL) - 1;
        v11 = (_QWORD *)((unsigned int)*(unsigned __int8 *)(qword_4F061C8 + 83LL) - 1);
      }
      *(_BYTE *)(v52 + 81) = a2;
      *(_BYTE *)(v52 + 83) = (_BYTE)v11;
      v113 = 0;
      v53 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v54 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v55 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v118.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v119 = v53;
      v119.m128i_i8[1] = v53.m128i_i8[1] | 0x20;
      v120 = v54;
      v118.m128i_i64[1] = *(_QWORD *)dword_4F07508;
      v121 = v55;
      v101 = dword_4F063F8;
      v102 = unk_4F063FC;
      goto LABEL_49;
    }
LABEL_11:
    v16 = v108 & (v106 ^ 1);
    if ( v16 )
    {
      v7 = 2768;
      sub_6851C0(2768, &v115);
      v110 = v108 & (v106 ^ 1);
      v16 = 0;
    }
LABEL_13:
    v101 = dword_4F063F8;
    v102 = unk_4F063FC;
    goto LABEL_14;
  }
  v7 = 16;
  v51 = sub_5CB9F0((_QWORD **)&v113);
  *v51 = (_QWORD *)sub_5CC970(16);
  v10 = word_4F06418[0];
LABEL_7:
  if ( v10 != 73 )
  {
    v15 = ((unsigned __int8)v111 ^ 1) & 1;
    if ( v10 != 56 || !(_BYTE)v15 )
    {
      v11 = (_QWORD *)v110;
      if ( v110 )
        goto LABEL_11;
      goto LABEL_142;
    }
    v111 = 0;
    v7 = v106;
    if ( !v106 )
    {
      v16 = 1;
      if ( v108 )
      {
        v7 = 2353;
        sub_6851C0(2353, &v115);
        v16 = v108;
      }
      goto LABEL_13;
    }
  }
  v101 = dword_4F063F8;
  v102 = unk_4F063FC;
  if ( v110 )
  {
    v16 = 0;
LABEL_14:
    a2 = (__int64)v113;
    if ( v113 )
    {
      a2 = (__int64)&v113[3].m128i_i64[1];
      v7 = 1098;
      sub_6851C0(1098, &v113[3].m128i_u64[1]);
      v113 = 0;
    }
    else if ( (_DWORD)v116 )
    {
      a2 = (__int64)&v116;
      v7 = 1098;
      sub_6851C0(1098, &v116);
    }
    v15 = (__int64)&dword_4F04C34;
    LODWORD(v17) = v16 + 74;
    if ( dword_4F04C5C != dword_4F04C34 )
    {
      if ( v16 )
      {
        v11 = qword_4F04C68;
        v18 = *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 4);
        if ( v18 == 2 || v18 == 17 )
        {
          LODWORD(v17) = 75;
          if ( v111 )
            goto LABEL_21;
LABEL_52:
          if ( !v112 )
          {
            v94 = v17;
            if ( v16 )
            {
              a2 = 0;
              v7 = (__int64)&v118;
              v27 = sub_7CFB70(&v118, 0);
              v17 = v94;
              v22 = v27;
              if ( !v27 )
              {
                *(_WORD *)v107 = v94;
                goto LABEL_68;
              }
            }
            else
            {
              v7 = (__int64)&v118;
              v73 = sub_7D22B0(&v118);
              v17 = v94;
              v22 = v73;
              if ( !v73 )
              {
                *(_WORD *)v107 = v94;
                goto LABEL_84;
              }
            }
            v28 = 0;
            if ( *(_BYTE *)(v22 + 80) == 23 )
              v28 = *(_BYTE *)(*(_QWORD *)(v22 + 88) + 124LL) & 1;
            if ( v119.m128i_i64[1] && (*(_BYTE *)(v119.m128i_i64[1] + 82) & 4) != 0 )
            {
              a2 = 0;
              v7 = (__int64)&v118;
              v91 = v28;
              v97 = v17;
              sub_87DC80(&v118, 0, 0, 1, v28, v17);
              LOBYTE(v28) = v91;
              LOWORD(v17) = v97;
            }
            if ( (v119.m128i_i8[1] & 0x20) == 0 )
            {
              v15 = (__int64)&dword_4D044B8;
              v29 = *(unsigned __int8 *)(v22 + 80);
              if ( dword_4D044B8 )
              {
                v15 = (unsigned int)(v29 - 4);
                if ( (unsigned __int8)(v29 - 4) <= 2u )
                {
LABEL_65:
                  if ( (v119.m128i_i8[2] & 1) == 0 )
                  {
LABEL_205:
                    *(_WORD *)v107 = v17;
                    if ( !v16 )
                      goto LABEL_84;
                    goto LABEL_67;
                  }
LABEL_66:
                  v7 = 101;
                  v95 = v17;
                  a2 = (__int64)&v118.m128i_i64[1];
                  sub_6851A0(101, &v118.m128i_u64[1], *(_QWORD *)(v118.m128i_i64[0] + 8));
                  *(_WORD *)v107 = v95;
                  if ( !v16 )
                    goto LABEL_84;
                  goto LABEL_67;
                }
                if ( (_BYTE)v29 == 3 )
                {
                  if ( !*(_BYTE *)(v22 + 104) )
                    goto LABEL_66;
                  goto LABEL_65;
                }
              }
              if ( (((unsigned __int8)v16 ^ (unsigned __int8)v28) & 1) != 0 || (_BYTE)v29 != 23 )
                goto LABEL_66;
            }
            *(_WORD *)v107 = v17;
            if ( !v16 )
            {
LABEL_28:
              v23 = *(_QWORD *)(v22 + 88);
              v24 = 1;
              v93 = 0;
              if ( !v23 )
                goto LABEL_29;
              goto LABEL_93;
            }
            goto LABEL_68;
          }
          goto LABEL_82;
        }
        a2 = (__int64)&v114;
        v7 = 726;
        sub_6851C0(726, &v114);
        v112 = 1;
        LOWORD(v17) = 75;
        v60 = _mm_loadu_si128(&xmmword_4F06660[1]);
        v61 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v62 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v118.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
        v119 = v60;
        v120 = v61;
        v119.m128i_i8[1] = v60.m128i_i8[1] | 0x20;
        v118.m128i_i64[1] = *(_QWORD *)dword_4F07508;
        v121 = v62;
        if ( v111 )
        {
          v15 = 75;
          v22 = 0;
          *(_WORD *)v107 = 75;
          goto LABEL_68;
        }
LABEL_82:
        *(_WORD *)v107 = v17;
        if ( v16 )
        {
LABEL_67:
          v22 = 0;
          goto LABEL_68;
        }
        if ( !v111 )
          goto LABEL_84;
LABEL_173:
        sub_854B40();
        if ( word_4F06418[0] == 73 )
        {
          sub_7BDC20(0);
          *(_WORD *)v107 = 74;
        }
        result = v109;
        if ( v109 && !*(_BYTE *)(v109 + 16) )
          return sub_869FD0(v109, (unsigned int)dword_4F04C64);
        return result;
      }
LABEL_80:
      a2 = (__int64)&v114;
      v7 = 724;
      sub_6851C0(724, &v114);
      v112 = 1;
      v36 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v37 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v38 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v118.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v119 = v36;
      v120 = v37;
      v119.m128i_i8[1] = v36.m128i_i8[1] | 0x20;
      v118.m128i_i64[1] = *(_QWORD *)dword_4F07508;
      v121 = v38;
      if ( v111 )
      {
        *(_WORD *)v107 = 74;
        goto LABEL_173;
      }
      v111 = 1;
      v16 = 0;
      LOWORD(v17) = 74;
      goto LABEL_82;
    }
    goto LABEL_51;
  }
LABEL_49:
  if ( dword_4F04C5C != dword_4F04C34 )
  {
    v110 = 0;
    goto LABEL_80;
  }
  v110 = 0;
  LODWORD(v17) = 74;
  v16 = 0;
LABEL_51:
  if ( !v111 )
    goto LABEL_52;
LABEL_21:
  if ( v112 )
    goto LABEL_205;
  v19 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v20 = *(_QWORD *)(v19 + 24);
  v21 = v19 + 32;
  if ( !v20 )
    v20 = v21;
  v22 = *(_QWORD *)(v20 + 120);
  if ( !v22 )
  {
    v90 = v17;
    v96 = v20;
    v81 = sub_87F730(&dword_4F063F8);
    LOWORD(v17) = v90;
    v22 = v81;
    *(_QWORD *)(v96 + 120) = v81;
  }
  a2 = (__int64)&v118;
  v7 = v22;
  v92 = v17;
  sub_878710(v22, &v118);
  v118.m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
  *(_WORD *)v107 = v92;
  if ( !v16 )
  {
    if ( v22 )
      goto LABEL_28;
LABEL_84:
    v22 = (__int64)qword_4D049A8;
    v93 = dword_4F04C64;
    if ( v118.m128i_i64[0] != *qword_4D049A8 || dword_4F04C64 || (v119.m128i_i8[1] & 0x20) != 0 )
    {
      if ( qword_4D049A0
        && v118.m128i_i64[0] == *qword_4D049A0
        && (v85 = qword_4F04C68[0] + 776LL * dword_4F04C64, (unsigned __int8)(*(_BYTE *)(v85 + 4) - 3) <= 1u)
        && *(_QWORD *)(v85 + 224) == qword_4D049A8[11]
        && (v119.m128i_i8[1] & 0x20) == 0 )
      {
        v22 = (__int64)qword_4D049A0;
        v24 = 3;
        sub_886FD0(&v118);
        v93 = 0;
      }
      else
      {
        v22 = (__int64)qword_4D049B8;
        if ( v118.m128i_i64[0] != *qword_4D049B8 || dword_4F04C64 || (v119.m128i_i8[1] & 0x20) != 0 )
        {
          if ( qword_4D049B0
            && v118.m128i_i64[0] == *(_QWORD *)qword_4D049B0
            && (v86 = qword_4F04C68[0] + 776LL * dword_4F04C64, *(_BYTE *)(v86 + 4) == 4)
            && *(_QWORD *)(v86 + 224) == qword_4D049B8[11]
            && (v119.m128i_i8[1] & 0x20) == 0 )
          {
            v22 = qword_4D049B0;
            v24 = 3;
            sub_887050(&v118);
            v93 = 0;
          }
          else
          {
            v22 = (__int64)qword_4D04998;
            if ( v118.m128i_i64[0] != *qword_4D04998 || dword_4F04C64 || (v119.m128i_i8[1] & 0x20) != 0 )
            {
              v24 = 1;
              v93 = 0;
              v22 = sub_885AD0(23, &v118, (unsigned int)dword_4F04C64, 1);
            }
            else
            {
              v24 = 3;
              sub_886F70(&v118);
            }
          }
        }
        else
        {
          v24 = 3;
          sub_886510(&v118);
          v93 = 1;
        }
      }
    }
    else
    {
      v24 = 3;
      sub_8870D0(&v118);
    }
    v23 = *(_QWORD *)(v22 + 88);
    if ( !v23 )
    {
LABEL_29:
      if ( v113 )
        sub_5CF700(v113->m128i_i64);
      v23 = sub_726DA0(0);
      sub_877D80(v23, v22);
      v25 = v108 & 1;
      if ( v111 )
      {
        *(_QWORD *)(v23 + 8) = 0;
        sub_877E90(v22, v23);
        *(_BYTE *)(v23 + 88) = *(_BYTE *)(v23 + 88) & 0x8F | 0x20;
        *(_BYTE *)(v23 + 124) = (2 * v25) | *(_BYTE *)(v23 + 124) & 0xFD;
        *(_QWORD *)(v22 + 88) = v23;
      }
      else
      {
        sub_877E90(v22, v23);
        *(_BYTE *)(v23 + 88) = *(_BYTE *)(v23 + 88) & 0x8F | 0x20;
        *(_BYTE *)(v23 + 124) = *(_BYTE *)(v23 + 124) & 0xFD | (2 * v25);
        *(_QWORD *)(v22 + 88) = v23;
        if ( (*(_BYTE *)(v22 + 81) & 0x10) != 0 )
          goto LABEL_34;
        v74 = *(_BYTE **)(v22 + 64);
        if ( !v74 )
          goto LABEL_34;
        if ( (v74[124] & 1) != 0 )
          v74 = (_BYTE *)sub_735B70(v74);
        if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v74 + 96LL) + 200LL) & 2) == 0 )
          goto LABEL_34;
      }
      *(_BYTE *)(*(_QWORD *)(v22 + 96) + 200LL) |= 2u;
      if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
      {
        v84 = *(_BYTE *)(v23 + 88);
        *(_BYTE *)(v23 + 124) |= 4u;
        *(_BYTE *)(v23 + 88) = v84 & 0x8F | 0x10;
      }
LABEL_34:
      sub_7331A0(v23);
      if ( !v110 )
        sub_854980(v22, 0);
      if ( !unk_4D042AC || (v98 = 0, qword_4D049B8 != (_QWORD *)v22) )
      {
        sub_8602E0(3, v23);
        v98 = 1;
        *(_QWORD *)(*(_QWORD *)(v23 + 128) + 32LL) = v23;
      }
      v24 = 3;
      if ( v111 | v108 )
        sub_650620(v23, v108, 1, v26);
      goto LABEL_104;
    }
LABEL_93:
    if ( !v110 )
    {
      sub_854980(v22, 0);
      v23 = *(_QWORD *)(v22 + 88);
    }
    if ( v108 != ((*(_BYTE *)(v23 + 124) & 2) != 0) )
    {
      if ( v108 )
      {
        sub_6853B0(unk_4F07471, 2354, &v115, v22);
        if ( dword_4F077BC )
        {
          *(_BYTE *)(v23 + 124) |= 2u;
          sub_650620(v23, 1, 0, v39);
        }
      }
      else
      {
        sub_6853B0(4, 2355, &v115, v22);
      }
    }
    if ( !unk_4D042AC || (v98 = 0, qword_4D049B8 != (_QWORD *)v22) )
    {
      v40 = v23;
      if ( (*(_BYTE *)(v23 + 124) & 1) != 0 )
        v40 = sub_735B70(v23);
      sub_8602E0(4, v40);
      v98 = 1;
      v41 = v93 << 27;
      BYTE1(v41) = 32;
      *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8) = *(_DWORD *)(qword_4F04C68[0]
                                                                            + 776LL * dword_4F04C64
                                                                            + 8)
                                                                & 0xF7FFDFFF
                                                                | v41 & 0x8002000;
    }
LABEL_104:
    sub_8756F0(v24, v22, &v118.m128i_u64[1], v109);
    if ( v109 && *(_BYTE *)(v109 + 16) == 53 )
    {
      v75 = *(_QWORD *)(v109 + 24);
      *(_QWORD *)(v75 + 48) = sub_5CF190(v113);
    }
    sub_5CEC90(v113, v23, 28);
    if ( word_4F06418[0] == 73 )
    {
      v117 = *(_QWORD *)&dword_4F063F8;
    }
    else
    {
      v117 = unk_4F077C8;
      if ( word_4F06418[0] == 1 && (v110 & 1) != 0 )
      {
        sub_661400(v107, 1, v99, &v114, a5);
        v45 = 28;
        v46 = v23;
        sub_869D70(v23, 28);
        goto LABEL_124;
      }
    }
    if ( (unsigned int)sub_7BE280(73, 130, 0, 0) )
    {
      v42 = qword_4F061C8;
      v43 = dword_4F066AC;
      v44 = *(_BYTE *)(qword_4F061C8 + 82LL);
      *(_BYTE *)(qword_4F061C8 + 82LL) = v44 + 1;
      if ( word_4F06418[0] != 9 && word_4F06418[0] != 74 )
      {
        do
          sub_660E20(1, 0, 0, 0, 0, 0, 0);
        while ( word_4F06418[0] != 9 && word_4F06418[0] != 74 );
        v42 = qword_4F061C8;
        v44 = *(_BYTE *)(qword_4F061C8 + 82LL) - 1;
      }
      *(_BYTE *)(v42 + 82) = v44;
      if ( (v111 & 1) != 0 && v100 && dword_4F066AC == v43 )
        sub_6851C0(3105, &v114);
      sub_854430();
      if ( v106 )
        *a5 = v22;
    }
    else
    {
      sub_854B40();
    }
    v45 = 28;
    v46 = v23;
    sub_869D70(v23, 28);
    if ( !v110 )
    {
      v46 = 74;
      if ( (unsigned int)sub_7BE5B0(74, 67, 3196, &v117) )
        sub_854AB0();
      else
        sub_854B40();
      v45 = v98;
      if ( !v98 )
        goto LABEL_131;
      goto LABEL_125;
    }
LABEL_124:
    v48 = v98;
    if ( !v98 )
    {
LABEL_128:
      if ( ((v106 ^ 1) & v110) != 0 && *a5 )
        sub_854980(*a5, 0);
      goto LABEL_131;
    }
LABEL_125:
    v49 = qword_4F04C68;
    if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 704) )
    {
      v46 = 1;
      sub_5D09C0(1);
    }
    sub_863FD0(v46, v45, v47, v49, v48);
    goto LABEL_128;
  }
LABEL_68:
  sub_7B8B50(v7, a2, v15, v11);
  ++*(_BYTE *)(qword_4F061C8 + 83LL);
  if ( dword_4F077C4 == 2 )
  {
    if ( word_4F06418[0] == 1 && (unk_4D04A11 & 2) != 0 || (unsigned int)sub_7C0F00(0, 0) )
      goto LABEL_70;
LABEL_171:
    sub_854B40();
    v23 = 0;
    sub_6851D0(40);
    goto LABEL_155;
  }
  if ( word_4F06418[0] != 1 )
    goto LABEL_171;
LABEL_70:
  v30 = 5;
  v31 = 0;
  v32 = sub_7BF130(0, 5, &v112);
  v33 = v32;
  if ( v112 )
  {
LABEL_152:
    v23 = 0;
    goto LABEL_153;
  }
  if ( v32 )
  {
    if ( (*(_DWORD *)(unk_4D04A18 + 80LL) & 0x400FF) == 0x40018 )
    {
      v31 = 266;
      sub_6854E0(266, unk_4D04A18);
      goto LABEL_75;
    }
    if ( *(_BYTE *)(v32 + 80) == 23 )
    {
      if ( v22 )
      {
        v23 = *(_QWORD *)(v22 + 88);
        if ( v23 )
        {
          v82 = *(_QWORD *)(v22 + 88);
          if ( (*(_BYTE *)(v23 + 124) & 1) != 0 )
            v82 = sub_735B70(*(_QWORD *)(v22 + 88));
          v83 = *(_QWORD *)(v33 + 88);
          if ( (*(_BYTE *)(v83 + 124) & 1) != 0 )
            v83 = sub_735B70(v83);
          if ( v82 == v83 )
            sub_8756F0(1, v22, &v118.m128i_u64[1], v109);
          else
            sub_685920(&v118.m128i_u64[1], v22, 8);
LABEL_244:
          v30 = v33;
          sub_8767A0(4, v33, &dword_4F063F8, 1);
          v31 = v112;
          if ( !v112 )
            goto LABEL_78;
LABEL_153:
          sub_854B40();
          goto LABEL_154;
        }
        v23 = sub_726DA0(1);
      }
      else
      {
        v23 = sub_726DA0(1);
        v22 = sub_885AD0(23, &v118, (unsigned int)dword_4F04C5C, 1);
      }
      *(_QWORD *)(v23 + 128) = *(_QWORD *)(v33 + 88);
      sub_877D80(v23, v22);
      sub_877E90(v22, v23);
      *(_BYTE *)(v23 + 88) = *(_BYTE *)(v23 + 88) & 0x8F | 0x20;
      *(_QWORD *)(v22 + 88) = v23;
      sub_7331A0(v23);
      sub_8756F0(3, v22, &v118.m128i_u64[1], v109);
      goto LABEL_244;
    }
  }
  v31 = 725;
  sub_6851C0(725, dword_4F07508);
LABEL_75:
  v30 = v112;
  if ( v112 || !v22 )
    goto LABEL_152;
  v23 = 0;
LABEL_78:
  v30 = 0;
  v31 = v22;
  sub_854980(v22, 0);
LABEL_154:
  sub_7B8B50(v31, v30, v34, v35);
LABEL_155:
  --*(_BYTE *)(qword_4F061C8 + 83LL);
  if ( (unsigned int)sub_7BE5B0(75, 65, 0, 0) )
  {
    sub_854AB0();
    result = v109;
    if ( v109 )
      goto LABEL_132;
    goto LABEL_157;
  }
  sub_854B40();
LABEL_131:
  result = v109;
  if ( v109 )
  {
LABEL_132:
    result = *(unsigned __int8 *)(result + 16);
    if ( (_BYTE)result )
    {
      if ( !v23 )
        return result;
      if ( (_BYTE)result == 28 )
      {
        result = *(_QWORD *)(v23 + 72);
        if ( result )
          goto LABEL_161;
        result = sub_7274B0(1);
        *(_QWORD *)(v23 + 72) = result;
      }
      else
      {
        if ( (_BYTE)result != 53 )
          return result;
        result = sub_7274B0(1);
        *(_QWORD *)(*(_QWORD *)(v109 + 24) + 8LL) = result;
      }
      if ( !result )
        return result;
LABEL_161:
      *(_QWORD *)(result + 16) = v114;
      *(_DWORD *)(result + 24) = v105;
      *(_WORD *)(result + 28) = v103;
      v63 = v118.m128i_i64[1];
      *(_DWORD *)(result + 32) = v101;
      *(_QWORD *)result = v63;
      *(_DWORD *)(result + 8) = v105;
      v64 = qword_4F063F0;
      *(_WORD *)(result + 12) = v103;
      *(_WORD *)(result + 36) = v102;
      *(_QWORD *)(result + 40) = v64;
      return result;
    }
    result = sub_869FD0(v109, (unsigned int)dword_4F04C64);
    if ( !v23 )
      return result;
    goto LABEL_158;
  }
LABEL_157:
  if ( !v23 )
    return result;
LABEL_158:
  result = *(_QWORD *)(v23 + 72);
  if ( !result )
  {
    result = sub_7274B0(1);
    *(_QWORD *)(v23 + 72) = result;
  }
  if ( !*(_DWORD *)(result + 16) )
    goto LABEL_161;
  return result;
}
