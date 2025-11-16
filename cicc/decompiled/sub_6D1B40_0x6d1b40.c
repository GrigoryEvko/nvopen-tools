// Function: sub_6D1B40
// Address: 0x6d1b40
//
__int64 __fastcall sub_6D1B40(__int64 a1, __m128i *a2, __int16 a3, __int64 a4, __int64 a5)
{
  __int16 v5; // r14
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rax
  int v10; // ecx
  __int64 v11; // rcx
  bool v12; // r14
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // r15d
  __int64 v17; // r13
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rcx
  unsigned int v25; // r14d
  __int64 v26; // rax
  __int64 v27; // rdx
  bool v28; // zf
  __int64 v29; // rdx
  __int64 v30; // rcx
  char v31; // r15
  int v32; // r14d
  __int64 v33; // rdx
  __int64 v34; // rax
  _QWORD *v35; // rcx
  __m128i *v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 result; // rax
  __int64 v41; // rax
  int v42; // r13d
  int v43; // ebx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rcx
  unsigned int v55; // ebx
  unsigned int v56; // r8d
  unsigned int v57; // ecx
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  char v61; // al
  __int64 v62; // rbx
  __int64 v63; // rax
  __int64 *v64; // r8
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // rdi
  unsigned int v68; // r15d
  __int64 v69; // rax
  __int64 v70; // r13
  __int64 v71; // rdi
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rax
  __int64 v77; // rbx
  _QWORD *v78; // rax
  _QWORD *v79; // r14
  __int64 v80; // rax
  __int64 v81; // rdi
  __int64 v82; // rcx
  unsigned __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // rdx
  bool v86; // [rsp+7h] [rbp-289h]
  unsigned int v87; // [rsp+10h] [rbp-280h]
  int v88; // [rsp+14h] [rbp-27Ch]
  unsigned int v89; // [rsp+18h] [rbp-278h]
  int v90; // [rsp+20h] [rbp-270h]
  unsigned __int16 v91; // [rsp+24h] [rbp-26Ch]
  int v92; // [rsp+24h] [rbp-26Ch]
  int v93; // [rsp+24h] [rbp-26Ch]
  char v95; // [rsp+28h] [rbp-268h]
  unsigned int v96; // [rsp+38h] [rbp-258h]
  _BOOL4 v97; // [rsp+38h] [rbp-258h]
  int v98; // [rsp+48h] [rbp-248h] BYREF
  int v99; // [rsp+4Ch] [rbp-244h] BYREF
  __int64 v100; // [rsp+50h] [rbp-240h] BYREF
  __int64 *v101; // [rsp+58h] [rbp-238h] BYREF
  __int64 v102; // [rsp+60h] [rbp-230h] BYREF
  __int64 v103; // [rsp+68h] [rbp-228h] BYREF
  __int64 v104; // [rsp+70h] [rbp-220h] BYREF
  __int64 v105; // [rsp+78h] [rbp-218h] BYREF
  __m128i v106[33]; // [rsp+80h] [rbp-210h] BYREF

  v5 = a3;
  v6 = a1;
  v7 = a3 & 8;
  v103 = *(_QWORD *)&dword_4F063F8;
  if ( !(_DWORD)v7 )
    sub_7B8B50(a1, a2, v7, a4);
  v8 = HIDWORD(qword_4F077B4);
  if ( HIDWORD(qword_4F077B4) && word_4F06418[0] == 73 )
  {
    v106[0].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
    v32 = qword_4F077B4;
    if ( (_DWORD)qword_4F077B4 )
    {
      v32 = 0;
    }
    else if ( qword_4F077A8 <= 0xC3B3u )
    {
      if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
      {
        v32 = 1;
        if ( (unsigned int)sub_6E5430(a1, HIDWORD(qword_4F077B4), v7, a4, a5) )
        {
          a1 = 28;
          sub_6851C0(0x1Cu, v106);
        }
      }
      else
      {
        a1 = 28;
        v32 = sub_6E91E0(28, v106) != 0;
      }
    }
    v33 = qword_4D03C50;
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) == 5 )
      goto LABEL_153;
    v34 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v35 = (_QWORD *)*(unsigned __int8 *)(v34 + 4);
    v36 = (__m128i *)((unsigned __int8)((_BYTE)v35 - 15) & 0xFD);
    if ( ((((_BYTE)v35 - 15) & 0xFD) == 0 || (_BYTE)v35 == 2) && (v35 = &qword_4F04C50, qword_4F04C50) )
    {
      if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x40) == 0 )
      {
        if ( (*(_BYTE *)(v34 + 12) & 1) != 0 )
        {
          if ( !v32 )
            goto LABEL_61;
LABEL_63:
          sub_7BDC10(a1, v36, v33);
          if ( word_4F06418[0] == 74 )
            sub_7B8B50(a1, v36, v37, v38);
          v102 = sub_72C930(a1);
LABEL_66:
          sub_6E6260(v6);
LABEL_67:
          *(_DWORD *)(v6 + 68) = v103;
          *(_WORD *)(v6 + 72) = WORD2(v103);
          *(_QWORD *)dword_4F07508 = *(_QWORD *)(v6 + 68);
          v39 = *(_QWORD *)&dword_4F063F8;
          *(_QWORD *)(v6 + 76) = *(_QWORD *)&dword_4F063F8;
          *(_QWORD *)&dword_4F061D8 = v39;
          sub_6E3280(v6, &v103);
          sub_7BE280(28, 18, 0, 0);
          return sub_6E26D0(2, v6);
        }
LABEL_153:
        sub_6E1DD0(&v105);
        v68 = dword_4F04C3C;
        if ( (*(_DWORD *)(v105 + 16) & 0x200200) == 0 )
          dword_4F04C3C = 1;
        v69 = sub_86FD00(0, 0, 0, 1, *(_BYTE *)(v105 + 20) >> 7, &v102);
        v70 = v69;
        if ( *(_BYTE *)(v69 + 40) == 11 )
        {
          v84 = *(_QWORD *)(*(_QWORD *)(v69 + 80) + 8LL);
          if ( v84 )
            *(_BYTE *)(v84 + 29) |= 0x10u;
        }
        v71 = v105;
        sub_6E1DF0(v105);
        dword_4F04C3C = v68;
        *(_BYTE *)(qword_4D03C50 + 21LL) |= 0x40u;
        if ( v32 )
          goto LABEL_66;
        v104 = 0;
        if ( !v102 )
          v102 = sub_72CBE0(v71, 0, v72, v73, v74, v75);
        v76 = *(_QWORD *)(v70 + 72);
        if ( v76 )
        {
          do
          {
            v77 = v76;
            v76 = *(_QWORD *)(v76 + 16);
          }
          while ( v76 );
          if ( *(_BYTE *)(v77 + 40) == 25 )
          {
            if ( !*(_QWORD *)(v77 + 48) )
              v104 = *(_QWORD *)(v77 + 72);
            if ( (unsigned int)sub_8D2600(v102) )
            {
              v71 = *(_QWORD *)(v77 + 48);
              if ( v71 )
                sub_7304E0(v71);
            }
            else
            {
              v71 = v102;
              if ( (unsigned int)sub_8DD010(v102) )
              {
                if ( (unsigned int)sub_6E5430(v71, 0, v72, v73, v74) )
                  sub_6851C0(0x57Du, v106);
                goto LABEL_66;
              }
            }
          }
        }
        v105 = sub_724DC0(v71, 0, v72, v73, v74, v75);
        v78 = (_QWORD *)sub_726700(17);
        v78[7] = v70;
        v79 = v78;
        *v78 = v102;
        if ( v104 )
        {
          v80 = sub_6ECAE0(v102, 0, 0, 1, 4, (unsigned int)v106, (__int64)&v104);
          *(_QWORD *)(v104 + 56) = v79;
          v79 = (_QWORD *)v80;
        }
        if ( (unsigned int)sub_719770(v79, v105, 0, 0) )
        {
          v81 = v105;
          *(_BYTE *)(v105 + 171) |= 2u;
          *(_QWORD *)(v81 + 144) = v79;
          sub_6E6A50(v81, v6);
        }
        else
        {
          sub_6E70E0(v79, v6);
        }
        sub_724E30(&v105);
        if ( qword_4F04C50 )
          *(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 204LL) |= 0x80u;
        if ( dword_4D04320 )
          sub_684B30(0x648u, v106);
        goto LABEL_67;
      }
      if ( v32 )
        goto LABEL_63;
    }
    else
    {
      if ( v32 )
        goto LABEL_63;
      if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x40) == 0 )
      {
LABEL_61:
        if ( (unsigned int)sub_6E5430(a1, v36, qword_4D03C50, v35, a5) )
        {
          v36 = v106;
          a1 = 1166;
          sub_6851C0(0x48Eu, v106);
        }
        goto LABEL_63;
      }
    }
    if ( (unsigned int)sub_6E5430(a1, v36, qword_4D03C50, v35, a5) )
    {
      v36 = v106;
      a1 = 1234;
      sub_6851C0(0x4D2u, v106);
    }
    goto LABEL_63;
  }
  v100 = 0;
  v101 = 0;
  v96 = 1;
  if ( dword_4F04C64 != -1
    && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) != 0
    && unk_4D04404
    && word_4F06418[0] != 76 )
  {
    v96 = sub_869470(&v100);
  }
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  v9 = qword_4D03C50;
  ++*(_QWORD *)(qword_4D03C50 + 40LL);
  if ( *(_BYTE *)(v9 + 16) && (unsigned int)sub_679C10(0x25u) )
  {
    if ( (*(_BYTE *)(qword_4D03C50 + 20LL) & 8) != 0 )
    {
      if ( unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0 )
      {
        v31 = dword_4F04C44 != -1;
        v95 = v31;
      }
      else
      {
        v95 = 1;
        v31 = 1;
      }
    }
    else
    {
      v95 = 0;
      v31 = 0;
    }
    sub_867030(v100);
    if ( (v5 & 0x2000) != 0 )
    {
      v5 &= ~0x2000u;
      sub_6851C0(0xBDEu, &v103);
    }
    v52 = qword_4D03C50;
    *(_BYTE *)(qword_4D03C50 + 20LL) &= ~8u;
    v104 = *(_QWORD *)&dword_4F063F8;
    v102 = sub_65CDF0(*(_BYTE *)(v52 + 16) <= 3u, 0, &v98, &v99, (__int64)&v101);
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
    if ( !dword_4D0477C || word_4F06418[0] != 73 )
    {
      v55 = dword_4D0478C;
      v56 = dword_4D0478C;
      v57 = dword_4D0478C != 0;
      if ( v101 )
      {
        v97 = dword_4D0478C != 0;
        sub_644730(v101);
        v57 = v97;
        v56 = dword_4D0478C;
      }
      v93 = sub_68E4D0(&v102, (__int64)&v104, v98, v57, v56, v31);
      if ( v99
        && dword_4F077C4 == 2
        && (!dword_4F077BC || qword_4F077A8 > 0x76BFu)
        && (unsigned int)sub_6E5430(&v102, &v104, v58, v59, v60) )
      {
        sub_6851C0(0xFFu, &v104);
      }
      *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
      sub_6C9F90(v102, 0, 0x11u, (__m128i *)a1, v106, 0);
      if ( v95 && (unsigned int)sub_8D23B0(v102) )
        sub_697260(v102, a1, (__int64)&v104);
      v105 = *(_QWORD *)(a1 + 76);
      if ( v55 && !dword_4D0478C && (unsigned int)sub_8D3410(v102) )
      {
        v67 = v102;
        if ( !(unsigned int)sub_68BAB0(v102, (_BYTE *)v6, &v104, v65, v66) )
          v102 = sub_72C930(v67);
      }
      else
      {
        v61 = *(_BYTE *)(a1 + 16);
        if ( v61 == 1 )
        {
          v62 = *(_QWORD *)(a1 + 144);
        }
        else
        {
          v62 = 0;
          if ( v61 == 2 )
          {
            v62 = *(_QWORD *)(a1 + 288);
            if ( !v62 && *(_BYTE *)(a1 + 317) == 12 && *(_BYTE *)(a1 + 320) == 1 )
              v62 = sub_72E9A0(a1 + 144);
          }
        }
        sub_6BF2D0(v102, (__m128i *)a1, v106, 1u, v5, v93, &v104, &v103, &v105);
        sub_6E41D0(a1, v62, 1, &v103, &v104, v102);
      }
      goto LABEL_119;
    }
    if ( v99 )
    {
      if ( dword_4F077C4 != 2 )
      {
        v64 = v101;
        if ( !v101 )
        {
LABEL_134:
          sub_68D9C0((__int64)&v102, (__int64)&v103, &v104, 0, 0, a1, v5);
          v105 = *(_QWORD *)&dword_4F061D8;
LABEL_119:
          *(_DWORD *)(v6 + 68) = v103;
          *(_WORD *)(v6 + 72) = WORD2(v103);
          *(_QWORD *)dword_4F07508 = *(_QWORD *)(v6 + 68);
          v63 = v105;
          *(_QWORD *)(v6 + 76) = v105;
          *(_QWORD *)&dword_4F061D8 = v63;
          return sub_6E3280(v6, &v103);
        }
LABEL_132:
        if ( unk_4F07778 > 201111 )
        {
          memset(v106, 0, 0x1D8u);
          v106[9].m128i_i64[1] = (__int64)v106;
          v106[1].m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
          if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
            v106[11].m128i_i8[2] |= 1u;
          v106[12].m128i_i64[0] = (__int64)&v101;
          sub_5CF030(&v102, v64, (__int64)v106);
          goto LABEL_134;
        }
LABEL_133:
        sub_644730(v64);
        goto LABEL_134;
      }
      if ( dword_4F077BC )
      {
        v64 = v101;
        if ( !v101 )
          goto LABEL_134;
        goto LABEL_133;
      }
      if ( (unsigned int)sub_6E5430(28, 18, v53, v54, dword_4F077BC) )
        sub_6851C0(0xFFu, &dword_4F063F8);
    }
    v64 = v101;
    if ( !v101 )
      goto LABEL_134;
    if ( dword_4F077C4 == 2 )
      goto LABEL_133;
    goto LABEL_132;
  }
  v10 = v5 & 0x9306;
  if ( !unk_4D04404 )
  {
    v11 = v10 | 0x30u;
    if ( v96 )
      goto LABEL_10;
LABEL_22:
    v16 = 0;
    v12 = 1;
    goto LABEL_27;
  }
  if ( word_4F06418[0] != 76 )
  {
    v11 = v10 | 0x430u;
    if ( v96 )
    {
LABEL_10:
      if ( qword_4D0495C )
        v11 = (unsigned int)v11 | v5 & 0x40;
      sub_69ED20(a1, a2, 0, v11);
      if ( !unk_4D04404 )
        goto LABEL_16;
      v12 = 0;
      if ( word_4F06418[0] == 28 )
        goto LABEL_15;
LABEL_14:
      v8 = 0;
      if ( (unsigned __int16)sub_7BE840(0, 0) != 76 )
        goto LABEL_15;
LABEL_30:
      v17 = v100;
      v18 = (__int64)v106;
      v90 = v12;
      sub_6E1BE0(v106);
      v91 = word_4F06418[0];
      if ( word_4F06418[0] != 76 )
      {
        v86 = v96 != 0;
        v89 = dword_4F06650[0];
        v105 = *(_QWORD *)&dword_4F063F8;
        if ( word_4F06418[0] > 0x43u )
        {
          if ( (unsigned __int16)(word_4F06418[0] - 147) <= 1u )
          {
LABEL_34:
            v88 = 0;
            v87 = word_4F06418[0];
            goto LABEL_35;
          }
        }
        else if ( word_4F06418[0] > 0x20u )
        {
          v19 = 0x7FF9EFFCFLL;
          if ( _bittest64(&v19, (unsigned int)word_4F06418[0] - 33) )
            goto LABEL_34;
        }
LABEL_70:
        v18 = 2857;
        v8 = (__int64)&v105;
        sub_6851C0(0xB29u, &v105);
        v87 = 67;
        v91 = 67;
        v88 = 1;
LABEL_35:
        sub_7B8B50(v18, v8, v19, v20);
        v104 = *(_QWORD *)&dword_4F063F8;
        if ( (unsigned int)sub_8670F0() )
        {
          v18 = (__int64)&v104;
          sub_867590(&v104);
          v90 = 1;
          v96 = 0;
        }
        else if ( v96 )
        {
          v90 = 0;
          v96 = 1;
        }
        sub_7B8B50(v18, v8, v21, v22);
        v25 = 0;
        if ( v86 )
        {
          while ( 1 )
          {
            sub_6E31E0(v6, 0, 0, v106);
            v8 = 1;
            v26 = sub_867630(v17, 1);
            if ( v26 )
            {
              v27 = v106[0].m128i_i64[1];
              v28 = *(_BYTE *)(v106[0].m128i_i64[1] + 8) == 0;
              *(_QWORD *)(v106[0].m128i_i64[1] + 16) = v26;
              if ( v28 )
                *(_QWORD *)(*(_QWORD *)(v27 + 24) + 136LL) = v26;
              v8 = (__int64)&dword_4F04C64;
              v25 = 1;
              if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 0x10) == 0 )
              {
                v8 = v89;
                *(_DWORD *)(v26 + 20) = v89;
              }
            }
            v18 = v17;
            if ( !(unsigned int)sub_866C00(v17) )
              break;
            sub_69ED20(v6, 0, 17, 0);
            sub_7B8B50(v6, 0, v29, v30);
            if ( word_4F06418[0] == 76 )
              sub_867610();
          }
        }
        if ( word_4F06418[0] == 28 )
        {
          if ( !v96 )
          {
            v42 = 1;
LABEL_122:
            sub_6D02D0((__m128i *)v6, a2, &v103, &v105, &qword_4F063F0, v106[0].m128i_i64[0], v87, v42, v96, v25);
            if ( !(v88 | v90) )
            {
              sub_6851C0(0xB2Du, &v104);
              sub_6E6260(v6);
            }
            v105 = qword_4F063F0;
            sub_7BE280(28, 18, 0, 0);
            --*(_BYTE *)(qword_4F061C8 + 36LL);
            --*(_QWORD *)(qword_4D03C50 + 40LL);
            *(_QWORD *)dword_4F07508 = v103;
            *(_QWORD *)(v6 + 68) = v103;
            result = v105;
            *(_QWORD *)&dword_4F061D8 = v105;
            *(_QWORD *)(v6 + 76) = v105;
            return result;
          }
          v92 = sub_869470(&v102);
          sub_867590(&v104);
          v42 = v96;
        }
        else
        {
          if ( v96 )
          {
            v8 = (__int64)&dword_4F063F8;
            v23 = *(_QWORD *)&dword_4F063F8;
            v105 = *(_QWORD *)&dword_4F063F8;
          }
          if ( word_4F06418[0] == v91 )
          {
            sub_7B8B50(v18, v8, v23, v24);
          }
          else
          {
            sub_6851C0(0xB28u, &dword_4F063F8);
            v88 = 1;
          }
          v42 = 0;
          v92 = sub_869470(&v102);
          sub_867590(&v104);
          if ( !v96 )
          {
            v42 = 0;
            if ( v102 )
              sub_866870(v102);
          }
        }
LABEL_81:
        if ( v92 )
        {
          v43 = v90;
          do
          {
            sub_69ED20(v6, 0, 17, 0);
            if ( (unsigned int)sub_8670F0() )
            {
              v43 = v96;
              if ( !v96 )
              {
                v43 = 1;
                sub_6851C0(0xB2Au, &v104);
                v88 = 1;
              }
            }
            sub_6E31E0(v6, 0, 0, v106);
            v44 = sub_867630(v102, 1);
            if ( v44 )
            {
              v45 = v106[0].m128i_i64[1];
              v25 = 1;
              v28 = *(_BYTE *)(v106[0].m128i_i64[1] + 8) == 0;
              *(_QWORD *)(v106[0].m128i_i64[1] + 16) = v44;
              if ( v28 )
                *(_QWORD *)(*(_QWORD *)(v45 + 24) + 136LL) = v44;
            }
          }
          while ( (unsigned int)sub_866C00(v102) );
          v90 = v43;
        }
        else
        {
          v90 = 1;
        }
        goto LABEL_122;
      }
      if ( v17 )
      {
        v86 = v96 != 0;
        v89 = dword_4F06650[0];
        v105 = *(_QWORD *)&dword_4F063F8;
        goto LABEL_70;
      }
      v104 = *(_QWORD *)&dword_4F063F8;
      sub_7B8B50(v106, v8, v19, v20);
      v82 = (__int64)&dword_4F063F8;
      v105 = *(_QWORD *)&dword_4F063F8;
      if ( word_4F06418[0] > 0x43u )
      {
        v83 = (unsigned int)word_4F06418[0] - 147;
        if ( (unsigned __int16)(word_4F06418[0] - 147) <= 1u )
        {
LABEL_190:
          v87 = word_4F06418[0];
          v88 = 0;
LABEL_191:
          sub_7B8B50(v18, v8, v83, v82);
          v25 = 0;
          v42 = 1;
          v92 = sub_869470(&v102);
          sub_867590(&v104);
          v96 = 1;
          goto LABEL_81;
        }
      }
      else if ( word_4F06418[0] > 0x20u )
      {
        v82 = 0x7FF9EFFCFLL;
        v83 = (unsigned int)word_4F06418[0] - 33;
        if ( _bittest64(&v82, v83) )
          goto LABEL_190;
      }
      v18 = 2857;
      v8 = (__int64)&v105;
      sub_6851C0(0xB29u, &v105);
      v87 = 67;
      v88 = 1;
      goto LABEL_191;
    }
    goto LABEL_22;
  }
  v16 = 1;
  v12 = v96 == 0;
LABEL_27:
  sub_6E6260(a1);
  if ( !unk_4D04404 )
    goto LABEL_16;
  if ( word_4F06418[0] != 28 )
  {
    if ( v16 )
      goto LABEL_30;
    goto LABEL_14;
  }
LABEL_15:
  sub_867030(v100);
LABEL_16:
  v105 = qword_4F063F0;
  sub_7BE280(28, 18, 0, 0);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  --*(_QWORD *)(qword_4D03C50 + 40LL);
  v13 = *(_BYTE *)(a1 + 16);
  if ( v13 == 2 )
  {
    v48 = *(_QWORD *)(a1 + 288);
    *(_BYTE *)(a1 + 20) |= 2u;
    if ( v48 )
    {
      *(_BYTE *)(v48 + 26) |= 0x20u;
      v48 = *(_QWORD *)(a1 + 288);
    }
    v49 = v103;
    if ( (*(_BYTE *)(a1 + 18) & 8) != 0 )
    {
      *(_DWORD *)(a1 + 18) |= 0x20020u;
      *(_QWORD *)dword_4F07508 = v49;
      *(_QWORD *)(a1 + 68) = v49;
      v85 = v105;
      *(_QWORD *)(a1 + 76) = v105;
      *(_QWORD *)&dword_4F061D8 = v85;
    }
    else
    {
      *(_QWORD *)(a1 + 68) = v103;
      *(_QWORD *)dword_4F07508 = v49;
      v50 = v105;
      *(_QWORD *)(a1 + 76) = v105;
      *(_QWORD *)&dword_4F061D8 = v50;
      *(_BYTE *)(a1 + 20) |= 2u;
    }
    if ( v48 )
      *(_BYTE *)(v48 + 26) |= 0x20u;
  }
  else
  {
    *(_BYTE *)(a1 + 20) |= 2u;
    if ( v13 == 1 )
    {
      *(_BYTE *)(*(_QWORD *)(a1 + 144) + 26LL) |= 0x20u;
      v46 = v103;
      if ( (*(_BYTE *)(a1 + 18) & 8) != 0 )
      {
        *(_DWORD *)(a1 + 18) |= 0x20020u;
        *(_QWORD *)dword_4F07508 = v46;
        *(_QWORD *)(a1 + 68) = v46;
        v47 = v105;
        *(_QWORD *)(a1 + 76) = v105;
        *(_QWORD *)&dword_4F061D8 = v47;
      }
      else
      {
        *(_QWORD *)(a1 + 68) = v103;
        *(_QWORD *)dword_4F07508 = v46;
        v51 = v105;
        *(_QWORD *)(a1 + 76) = v105;
        *(_QWORD *)&dword_4F061D8 = v51;
        *(_BYTE *)(a1 + 20) |= 2u;
      }
      *(_BYTE *)(*(_QWORD *)(a1 + 144) + 26LL) |= 0x20u;
    }
    else
    {
      if ( (*(_BYTE *)(a1 + 18) & 8) == 0 )
      {
        v41 = v103;
        *(_QWORD *)(a1 + 68) = v103;
        *(_QWORD *)dword_4F07508 = v41;
        result = v105;
        *(_QWORD *)(a1 + 76) = v105;
        *(_QWORD *)&dword_4F061D8 = result;
        *(_BYTE *)(a1 + 20) |= 2u;
        return result;
      }
      v14 = v103;
      *(_DWORD *)(a1 + 18) |= 0x20020u;
      *(_QWORD *)dword_4F07508 = v14;
      *(_QWORD *)(a1 + 68) = v14;
      v15 = v105;
      *(_QWORD *)(a1 + 76) = v105;
      *(_QWORD *)&dword_4F061D8 = v15;
    }
  }
  result = *(unsigned __int8 *)(a1 + 18);
  if ( (result & 8) != 0 )
  {
    result = (unsigned int)result | 0x20;
    *(_BYTE *)(a1 + 18) = result;
  }
  return result;
}
