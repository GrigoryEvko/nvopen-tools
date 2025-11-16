// Function: sub_6333F0
// Address: 0x6333f0
//
__int16 __fastcall sub_6333F0(__int64 *a1, __int64 a2, __m128i *a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // r15
  bool v8; // zf
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // r8
  __int64 v14; // rsi
  __int64 v15; // r14
  __int64 v16; // rdi
  __int64 v17; // rdx
  unsigned int *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  _QWORD *v21; // rax
  __int64 v22; // rdi
  char v23; // al
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rcx
  __m128i v28; // xmm1
  __m128i v29; // xmm2
  __m128i v30; // xmm3
  __int64 v31; // rax
  __int64 v32; // r8
  int v33; // ecx
  __int64 *v34; // rdi
  __int64 v35; // rsi
  __int8 v36; // r11
  int v37; // edi
  __int64 v38; // rax
  __int64 v39; // rsi
  int v40; // r9d
  __int64 v41; // rax
  __int64 v42; // rsi
  __int8 v43; // al
  __int64 v44; // rax
  __int64 v45; // rdi
  __int8 v46; // al
  __int64 v47; // r8
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // r8
  char v51; // al
  __int64 v52; // rdi
  __int8 v53; // al
  char v54; // al
  __int64 v55; // rax
  __int64 v56; // rax
  _QWORD *v57; // rdx
  __int64 v58; // r9
  bool v59; // r11
  _QWORD *v60; // rax
  __int64 v61; // rsi
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rsi
  __int64 v67; // r9
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rcx
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rsi
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rsi
  __int64 v78; // rax
  __int64 v79; // rcx
  __int64 v80; // rsi
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rsi
  __int64 i; // rax
  unsigned int *v86; // rbx
  __int64 v87; // r14
  __int64 v88; // rax
  __int64 v89; // r13
  unsigned __int8 v90; // al
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // r14
  __int64 v96; // rax
  __int64 v97; // rsi
  __int64 v98; // rax
  __int64 v100; // [rsp+8h] [rbp-108h]
  __int64 v101; // [rsp+10h] [rbp-100h]
  __int64 v102; // [rsp+10h] [rbp-100h]
  __int64 v103; // [rsp+18h] [rbp-F8h]
  __int64 v105; // [rsp+30h] [rbp-E0h]
  __int64 v106; // [rsp+40h] [rbp-D0h]
  int v107; // [rsp+40h] [rbp-D0h]
  __int64 v108; // [rsp+40h] [rbp-D0h]
  __int64 v109; // [rsp+40h] [rbp-D0h]
  __int64 v110; // [rsp+40h] [rbp-D0h]
  __int64 v111; // [rsp+40h] [rbp-D0h]
  __int64 v112; // [rsp+40h] [rbp-D0h]
  __int64 v113; // [rsp+40h] [rbp-D0h]
  int v114; // [rsp+48h] [rbp-C8h]
  __int64 v115; // [rsp+48h] [rbp-C8h]
  int v116; // [rsp+48h] [rbp-C8h]
  __int64 v117; // [rsp+48h] [rbp-C8h]
  int v118; // [rsp+48h] [rbp-C8h]
  int v119; // [rsp+48h] [rbp-C8h]
  int v120; // [rsp+48h] [rbp-C8h]
  __int64 v121; // [rsp+48h] [rbp-C8h]
  int v122; // [rsp+48h] [rbp-C8h]
  int v123; // [rsp+48h] [rbp-C8h]
  __int64 v124; // [rsp+50h] [rbp-C0h]
  __int64 v125; // [rsp+58h] [rbp-B8h]
  __int64 v126; // [rsp+58h] [rbp-B8h]
  char v127; // [rsp+60h] [rbp-B0h]
  bool v128; // [rsp+66h] [rbp-AAh]
  char v129; // [rsp+67h] [rbp-A9h]
  __int64 v130; // [rsp+68h] [rbp-A8h]
  char v131; // [rsp+70h] [rbp-A0h]
  __int64 v133; // [rsp+80h] [rbp-90h] BYREF
  __int64 v134; // [rsp+88h] [rbp-88h] BYREF
  __int64 v135; // [rsp+90h] [rbp-80h] BYREF
  __int64 v136; // [rsp+98h] [rbp-78h] BYREF
  _OWORD v137[7]; // [rsp+A0h] [rbp-70h] BYREF

  v5 = a2;
  v8 = *(_BYTE *)(a2 + 140) == 12;
  v105 = *a1;
  v133 = *a1;
  if ( v8 )
  {
    do
      v5 = *(_QWORD *)(v5 + 160);
    while ( *(_BYTE *)(v5 + 140) == 12 );
  }
  v129 = *(_BYTE *)(v105 + 8);
  if ( !v129 && (dword_4F077C4 == 2 || unk_4F07778 > 199900 || dword_4F077C0) )
  {
    if ( (unsigned int)sub_696450(v105, v5) )
    {
      LOWORD(v18) = sub_631120(a1, v5, a3, (__int64)a5);
      return (__int16)v18;
    }
    v129 = *(_BYTE *)(v133 + 8);
  }
  v9 = *(_QWORD *)(v5 + 160);
  v10 = a3[1].m128i_i64[1];
  v134 = v9;
  v103 = v10;
  if ( (*(_BYTE *)(v5 + 177) & 8) == 0 )
    a3[1].m128i_i64[1] = v5;
  v11 = 0;
  v134 = sub_72FD90(v9, 7);
  if ( unk_4D04418 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(v5 + 168) + 8LL);
    if ( (a3[2].m128i_i8[8] & 0x40) != 0 )
      goto LABEL_12;
  }
  else if ( (a3[2].m128i_i8[8] & 0x40) != 0 )
  {
LABEL_12:
    *a5 = 0;
    goto LABEL_13;
  }
  v19 = sub_724D50(10);
  v20 = v133;
  *a5 = v19;
  *(_QWORD *)(v19 + 128) = v5;
  v21 = (_QWORD *)sub_6E1A20(v20);
  v22 = v133;
  *(_QWORD *)(*a5 + 64) = *v21;
  if ( *(_BYTE *)(v22 + 8) != 2 )
    *(_QWORD *)(*a5 + 112) = *(_QWORD *)sub_6E1A60(v22);
  v23 = v129 == 1;
  v24 = *a5;
  if ( (a3[2].m128i_i32[2] & 0x20008000) == 0x20000000 )
  {
    *(_BYTE *)(v24 + 169) = *(_BYTE *)(v24 + 169) & 0xBF | (v23 << 6);
LABEL_13:
    if ( v129 != 1 )
      goto LABEL_14;
    goto LABEL_44;
  }
  *(_BYTE *)(v24 + 169) = (32 * v23) | *(_BYTE *)(v24 + 169) & 0xDF;
  if ( v129 != 1 )
  {
LABEL_14:
    if ( v134 | v11 )
    {
      v12 = v133;
      v127 = 0;
      v13 = v133;
      if ( (a3[2].m128i_i8[10] & 1) != 0 )
      {
        if ( (a3[2].m128i_i16[4] & 0x220) == 0 )
        {
          v82 = sub_6E1A20(v133);
          sub_6851C0(2360, v82);
          v12 = v133;
        }
        a3[2].m128i_i8[9] |= 2u;
        v13 = v12;
        v127 = 0;
      }
    }
    else
    {
      v127 = dword_4F077C0;
      if ( dword_4F077C0 )
      {
        v93 = sub_6E1A20(v133);
        sub_684B30(1162, v93);
        v17 = v134;
        v133 = 0;
        if ( v134 && (a3[2].m128i_i8[9] & 0x20) == 0 )
        {
          v127 = 0;
          v11 = 0;
          goto LABEL_102;
        }
        goto LABEL_35;
      }
      *(_QWORD *)&v137[0] = 0;
      sub_631120(&v133, v5, a3, (__int64)v137);
      v13 = v133;
    }
    goto LABEL_19;
  }
LABEL_44:
  a4 = v133 + 40;
  v105 = *(_QWORD *)(v133 + 24);
  v13 = v105;
  v133 = v105;
  if ( !v105 )
  {
    if ( dword_4F077C4 == 2 || dword_4F077C0 )
    {
      v17 = v134;
      v127 = (a3[2].m128i_i8[9] & 0x20) != 0;
      goto LABEL_32;
    }
    sub_6851C0(29, a4);
    v13 = v133;
  }
  v127 = (a3[2].m128i_i8[9] & 0x20) != 0;
LABEL_19:
  if ( !v13 )
  {
LABEL_31:
    v17 = v134;
LABEL_32:
    if ( !v17 )
      goto LABEL_100;
LABEL_33:
    if ( (a3[2].m128i_i8[9] & 0x20) != 0 )
      goto LABEL_34;
LABEL_102:
    sub_632A80(*a5, v5, v17, v11, a3, a4, 0);
    if ( v129 == 1 )
      goto LABEL_103;
LABEL_35:
    *a1 = v133;
    goto LABEL_36;
  }
  while ( *(_BYTE *)(v13 + 8) != 2 )
  {
    if ( !v11 )
    {
      v25 = v13;
      while ( v134 )
      {
        if ( dword_4F077C4 == 2
          && (a3[2].m128i_i8[8] & 0x60) == 0x20
          && (!unk_4D03C50 || (*(_BYTE *)(unk_4D03C50 + 21LL) & 0x10) == 0) )
        {
          *(_BYTE *)(v25 + 9) |= 8u;
        }
        sub_636E20(&v133, &v134, a3, *a5, a4);
        v25 = v133;
        if ( !v133 )
        {
          v17 = v134;
          v11 = 0;
          if ( v134 )
            goto LABEL_33;
          goto LABEL_34;
        }
        if ( *(_BYTE *)(v133 + 8) == 2 )
        {
          v11 = 0;
          v13 = v133;
          goto LABEL_52;
        }
      }
      goto LABEL_34;
    }
    v14 = *(_QWORD *)(v11 + 40);
    v15 = *a5;
    if ( (a3[2].m128i_i8[9] & 0x20) != 0 )
      v14 = *(_QWORD *)&dword_4D03B80;
    sub_634B10(&v133, v14, 0, a3, a4, v137);
    if ( (a3[2].m128i_i8[8] & 0x40) == 0 )
    {
      v16 = *(_QWORD *)&v137[0];
      if ( *(_QWORD *)&v137[0] )
      {
        *(_BYTE *)(*(_QWORD *)&v137[0] + 171LL) |= 0x80u;
        sub_72A690(v16, v15, v11, 0);
      }
    }
    if ( (a3[2].m128i_i8[9] & 0x20) == 0 )
      v11 = *(_QWORD *)(v11 + 8);
    v13 = v133;
LABEL_30:
    if ( !v13 )
      goto LABEL_31;
  }
LABEL_52:
  if ( v129 != 1 && (a3[2].m128i_i8[9] & 0x40) == 0 )
    goto LABEL_31;
  a3[2].m128i_i8[9] &= ~0x40u;
  v135 = v13;
  v124 = *a5;
  v131 = a3[2].m128i_i8[12] & 1;
  v125 = v134;
  if ( (*(_BYTE *)(v5 + 177) & 8) != 0 )
  {
    v26 = a3[1].m128i_i64[1];
    v27 = v26;
    goto LABEL_56;
  }
  v27 = 0;
  v26 = v5;
  if ( dword_4F077C4 != 2 )
  {
LABEL_56:
    if ( *(_QWORD *)(v13 + 24) )
      goto LABEL_57;
LABEL_80:
    v39 = sub_6E1A20(v13);
    sub_6851C0(1045, v39);
    v33 = 0;
    goto LABEL_62;
  }
  if ( *(_BYTE *)(*(_QWORD *)(v5 + 168) + 113LL) == 2 )
  {
    do
      v26 = *(_QWORD *)(*(_QWORD *)(v26 + 40) + 32LL);
    while ( *(_BYTE *)(*(_QWORD *)(v26 + 168) + 113LL) == 2 );
  }
  v27 = v5;
  if ( !*(_QWORD *)(v13 + 24) )
    goto LABEL_80;
LABEL_57:
  v106 = v27;
  v28 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v29 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v30 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v137[0] = _mm_loadu_si128(xmmword_4F06660);
  v137[1] = v28;
  v137[2] = v29;
  v137[3] = v30;
  *((_QWORD *)&v137[0] + 1) = *(_QWORD *)sub_6E1A20(v13);
  *(_QWORD *)&v137[0] = *(_QWORD *)(v135 + 24);
  v31 = sub_7D2AC0(v137, v26, 0);
  v32 = v31;
  if ( !v31 )
  {
    if ( (a3[2].m128i_i8[8] & 0x20) == 0 )
    {
      v108 = *(_QWORD *)v5;
      v115 = *(_QWORD *)(*(_QWORD *)&v137[0] + 8LL);
      v56 = sub_6E1A20(v135);
      sub_686A10(136, v56, v115, v108);
    }
    goto LABEL_61;
  }
  if ( *(_BYTE *)(v31 + 80) != 8 )
  {
    if ( (a3[2].m128i_i8[8] & 0x20) == 0 )
    {
      v117 = *(_QWORD *)(*(_QWORD *)&v137[0] + 8LL);
      v66 = sub_6E1A20(v135);
      sub_6851A0(1577, v66, v117);
    }
LABEL_61:
    a3[2].m128i_i8[9] |= 2u;
    v33 = 0;
    goto LABEL_62;
  }
  v57 = *(_QWORD **)(v31 + 96);
  v134 = *(_QWORD *)(v31 + 88);
  v58 = *(_QWORD *)(*(_QWORD *)(v134 + 40) + 32LL);
  v59 = v58 != v5;
  if ( v106 && v58 != v106 )
  {
    if ( v58 )
    {
      if ( dword_4F07588 )
      {
        v78 = *(_QWORD *)(v106 + 32);
        if ( *(_QWORD *)(v58 + 32) == v78 )
        {
          if ( v78 )
          {
            if ( !v57 || v58 == v5 )
              goto LABEL_210;
LABEL_208:
            v69 = *(_QWORD *)(v58 + 32);
            if ( *(_QWORD *)(v5 + 32) == v69 && v69 )
            {
LABEL_210:
              v33 = 1;
              goto LABEL_62;
            }
            goto LABEL_176;
          }
        }
      }
    }
    if ( v57 )
    {
      v60 = v57;
      while ( 1 )
      {
        v62 = v60[8];
        if ( v106 == v62 )
          break;
        if ( v62 )
        {
          if ( dword_4F07588 )
          {
            v61 = *(_QWORD *)(v62 + 32);
            if ( *(_QWORD *)(v106 + 32) == v61 )
            {
              if ( v61 )
                break;
            }
          }
        }
        v60 = (_QWORD *)v60[12];
        if ( !v60 )
          goto LABEL_260;
      }
    }
    else
    {
LABEL_260:
      if ( (a3[2].m128i_i8[8] & 0x20) == 0 )
      {
        v128 = v58 != v5;
        v100 = *(_QWORD *)(*(_QWORD *)(v134 + 40) + 32LL);
        v102 = v32;
        v111 = *(_QWORD *)v5;
        v121 = *(_QWORD *)(*(_QWORD *)&v137[0] + 8LL);
        v92 = sub_6E1A20(v135);
        sub_686A10(136, v92, v121, v111);
        v59 = v128;
        v58 = v100;
        v32 = v102;
      }
      a3[2].m128i_i8[9] |= 2u;
      v57 = *(_QWORD **)(v32 + 96);
    }
  }
  if ( !v57 || !v59 )
    goto LABEL_210;
  if ( v58 && dword_4F07588 )
    goto LABEL_208;
LABEL_176:
  if ( dword_4F077C4 != 2 && (!dword_4F077C0 || (_DWORD)qword_4F077B4 || qword_4F077A8 > 0x9E97u) || unk_4D04790 )
  {
LABEL_184:
    while ( 1 )
    {
      v64 = v57[8];
      if ( v64 == v5 )
        break;
      if ( v64 )
      {
        if ( dword_4F07588 )
        {
          v63 = *(_QWORD *)(v64 + 32);
          if ( *(_QWORD *)(v5 + 32) == v63 )
          {
            if ( v63 )
              break;
          }
        }
      }
      v57 = (_QWORD *)v57[12];
      if ( !v57 )
      {
        if ( (a3[2].m128i_i8[8] & 0x20) == 0 )
        {
          v95 = *(_QWORD *)(*(_QWORD *)&v137[0] + 8LL);
          v130 = *(_QWORD *)v5;
          v96 = sub_6E1A20(v135);
          sub_686A10(136, v96, v95, v130);
        }
        v43 = a3[2].m128i_i8[9] | 2;
        a3[2].m128i_i8[9] = v43;
        v17 = v134;
        goto LABEL_99;
      }
    }
    v35 = v57[11];
    v33 = 0;
    v13 = 0;
    v134 = v35;
    if ( dword_4F077C4 != 2 )
      goto LABEL_70;
    goto LABEL_144;
  }
  if ( dword_4F077BC )
  {
    if ( !(_DWORD)qword_4F077B4 )
    {
      if ( qword_4F077A8 > 0x138E3u )
        goto LABEL_184;
      goto LABEL_252;
    }
    goto LABEL_296;
  }
  if ( (_DWORD)qword_4F077B4 )
  {
LABEL_296:
    if ( qword_4F077A0 )
      goto LABEL_184;
  }
LABEL_252:
  v119 = unk_4D04790;
  v80 = sub_6E1A20(v135);
  sub_6851C0(2358, v80);
  v33 = v119;
LABEL_62:
  v34 = (__int64 *)v135;
  if ( (*(_BYTE *)(v135 + 10) & 2) != 0 )
    a3[2].m128i_i8[12] |= 1u;
  v13 = *v34;
  if ( *v34 && *(_BYTE *)(v13 + 8) == 3 )
  {
    v116 = v33;
    v65 = sub_6BBB10(v34);
    v33 = v116;
    v13 = v65;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( !v33 )
      goto LABEL_81;
LABEL_144:
    v54 = *(_BYTE *)(v5 + 140);
    if ( v54 == 11 || (*(_BYTE *)(v5 + 177) & 0x20) != 0 )
      goto LABEL_69;
    v8 = v54 == 12;
    v55 = v5;
    if ( v8 )
    {
      do
        v55 = *(_QWORD *)(v55 + 160);
      while ( *(_BYTE *)(v55 + 140) == 12 );
    }
    if ( *(char *)(*(_QWORD *)(*(_QWORD *)v55 + 96LL) + 178LL) < 0 || unk_4D04790 )
      goto LABEL_69;
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 > 0x138E3u )
        {
          v17 = v134;
          v35 = v134;
          if ( !v134 )
          {
            if ( v33 )
            {
              v135 = v13;
              if ( !v125 && !v11 )
                goto LABEL_136;
LABEL_115:
              v37 = unk_4D04790;
LABEL_116:
              if ( v37 && (a3[2].m128i_i16[4] & 0x240) == 0 && v125 )
              {
                if ( *(_BYTE *)(v5 + 140) == 11 )
                  goto LABEL_120;
                v79 = v11;
                v11 = 0;
                sub_632A80(v124, v5, v125, v79, a3, a4, v17);
              }
              goto LABEL_135;
            }
LABEL_113:
            if ( !v11 && v125 == v17 )
            {
              v11 = 0;
LABEL_135:
              if ( *(_BYTE *)(v5 + 140) == 11 )
              {
LABEL_120:
                if ( (a3[2].m128i_i8[8] & 0x40) != 0 )
                  goto LABEL_121;
LABEL_137:
                v126 = sub_724D50(13);
                v49 = sub_72CBE0();
                v50 = v126;
                *(_QWORD *)(v126 + 128) = v49;
                v51 = *(_BYTE *)(v126 + 176);
                *(_BYTE *)(v126 + 176) = v51 | 1;
                if ( (a3[2].m128i_i8[12] & 1) != 0 )
                  *(_BYTE *)(v126 + 176) = v51 | 5;
                v52 = v133;
                *(_QWORD *)(v126 + 184) = v134;
                *(_QWORD *)(v50 + 64) = *(_QWORD *)sub_6E1A20(v52);
                sub_72A690(v126, v124, 0, 0);
                if ( *(_BYTE *)(v5 + 140) != 11 )
                {
                  *(_BYTE *)(v124 + 170) |= 0x40u;
                  v47 = v135;
                  if ( v135 )
                  {
LABEL_122:
                    if ( *(_BYTE *)(v47 + 8) != 2 )
                    {
                      sub_636E20(&v135, &v134, a3, v124, a4);
LABEL_132:
                      v13 = v135;
LABEL_133:
                      a3[2].m128i_i8[12] = v131 | a3[2].m128i_i8[12] & 0xFE;
                      v133 = v13;
                      goto LABEL_30;
                    }
                    v48 = v134;
                    if ( dword_4F077C4 == 2 && (*(_BYTE *)(v134 + 144) & 0x10) == 0 )
                    {
                      if ( *(_BYTE *)(v5 + 140) == 12 )
                      {
                        v70 = v5;
                        do
                          v70 = *(_QWORD *)(v70 + 160);
                        while ( *(_BYTE *)(v70 + 140) == 12 );
                        v71 = *(_QWORD *)(*(_QWORD *)v70 + 96LL);
                        v72 = v5;
                        if ( !*(_QWORD *)(v71 + 24) )
                        {
                          if ( *(_QWORD *)(v134 + 112) )
                          {
LABEL_126:
                            sub_6368A0(&v135, *(_QWORD *)(v48 + 120), a3, &v136);
                            if ( *(_BYTE *)(v5 + 140) == 11 )
                            {
                              v134 = 0;
                            }
                            else if ( (a3[2].m128i_i8[9] & 0x20) == 0 )
                            {
                              v134 = sub_72FD90(*(_QWORD *)(v134 + 112), 7);
                            }
                            if ( (a3[2].m128i_i8[8] & 0x40) == 0 && v136 )
                              sub_72A690(v136, v124, 0, 0);
                            goto LABEL_132;
                          }
LABEL_212:
                          if ( (unsigned int)sub_8D2430(*(_QWORD *)(v48 + 120)) )
                            sub_62FA90(v135, v134, (__int64)a3);
                          v48 = v134;
                          goto LABEL_126;
                        }
                        do
                          v72 = *(_QWORD *)(v72 + 160);
                        while ( *(_BYTE *)(v72 + 140) == 12 );
                        v73 = *(_QWORD *)(*(_QWORD *)v72 + 96LL);
                      }
                      else
                      {
                        v73 = *(_QWORD *)(*(_QWORD *)v5 + 96LL);
                        if ( !*(_QWORD *)(v73 + 24) )
                          goto LABEL_124;
                      }
                      if ( (*(_BYTE *)(v73 + 177) & 2) == 0 && !unk_4D04790 )
                      {
                        v74 = sub_6E1A20(v47);
                        sub_6851C0(3280, v74);
                        v48 = v134;
                      }
                    }
LABEL_124:
                    if ( *(_QWORD *)(v48 + 112) && *(_BYTE *)(v5 + 140) != 11 )
                      goto LABEL_126;
                    goto LABEL_212;
                  }
LABEL_141:
                  if ( (a3[2].m128i_i8[8] & 0x20) != 0 )
                  {
                    v53 = a3[2].m128i_i8[12];
                    a3[2].m128i_i8[9] |= 2u;
                    a3[2].m128i_i8[12] = v131 | v53 & 0xFE;
                    v133 = 0;
                    goto LABEL_31;
                  }
                  v75 = sub_6E1A20(v133);
                  sub_6851C0(2914, v75);
                  v13 = v135;
                  a3[2].m128i_i8[9] |= 2u;
                  goto LABEL_133;
                }
LABEL_121:
                v47 = v135;
                if ( v135 )
                  goto LABEL_122;
                goto LABEL_141;
              }
LABEL_136:
              a3[2].m128i_i8[9] |= 0x10u;
              if ( (a3[2].m128i_i8[8] & 0x40) != 0 )
                goto LABEL_121;
              goto LABEL_137;
            }
            goto LABEL_115;
          }
LABEL_154:
          v36 = a3[2].m128i_i8[8];
          if ( (v36 & 0x40) != 0 )
            goto LABEL_111;
          v37 = unk_4D04790;
          goto LABEL_73;
        }
LABEL_304:
        v35 = v134;
        if ( v125 == v134 && v13 && (v33 & 1) != 0 && *(_BYTE *)(v13 + 8) != 2 )
        {
          if ( !unk_4D048F8 )
            goto LABEL_133;
          goto LABEL_70;
        }
        goto LABEL_303;
      }
    }
    else if ( !(_DWORD)qword_4F077B4 )
    {
LABEL_303:
      v113 = v13;
      v123 = v33;
      v98 = sub_6E1A20(v135);
      sub_6851C0(1563, v98);
      v35 = v134;
      v33 = v123;
      v13 = v113;
LABEL_70:
      if ( !v35 || (v36 = a3[2].m128i_i8[8], (v36 & 0x40) != 0) )
      {
        v17 = v134;
        goto LABEL_111;
      }
      v37 = unk_4D04790;
      if ( unk_4D04790 )
        goto LABEL_195;
LABEL_73:
      if ( dword_4F077BC )
      {
        if ( (_DWORD)qword_4F077B4 )
        {
          v17 = v134;
          v38 = qword_4F077A0;
          goto LABEL_159;
        }
        if ( qword_4F077A8 )
        {
LABEL_195:
          v67 = v135;
          if ( *(_BYTE *)(v5 + 140) == 11 )
          {
            v83 = v105;
            if ( v105 != v135 )
            {
              while ( *(_BYTE *)(v83 + 8) != 2 )
              {
                v83 = *(_QWORD *)v83;
                if ( v83 == v135 )
                  goto LABEL_204;
              }
              if ( (v36 & 0x20) == 0 )
              {
                v110 = v13;
                v120 = v33;
                v84 = sub_6E1A20(v135);
                sub_6851C0(2906, v84);
                v33 = v120;
                v13 = v110;
              }
              goto LABEL_203;
            }
          }
          else
          {
            v68 = v105;
            *(_QWORD *)(v135 + 56) = v35;
            if ( v105 != v67 )
            {
              while ( *(_BYTE *)(v68 + 8) != 2 || v35 != *(_QWORD *)(v68 + 56) )
              {
                v68 = *(_QWORD *)v68;
                if ( v68 == v67 )
                  goto LABEL_229;
              }
              if ( (a3[2].m128i_i8[8] & 0x20) == 0 )
              {
                v112 = v13;
                v122 = v33;
                v97 = sub_6E1A20(v67);
                sub_6851C0(2906, v97);
                v13 = v112;
                v33 = v122;
              }
              goto LABEL_203;
            }
LABEL_229:
            if ( !v125 )
            {
LABEL_234:
              if ( (a3[2].m128i_i8[8] & 0x20) == 0 )
              {
                v109 = v13;
                v118 = v33;
                v77 = sub_6E1A20(v67);
                sub_6851C0(2904, v77);
                v33 = v118;
                v13 = v109;
              }
LABEL_203:
              a3[2].m128i_i8[9] |= 2u;
              goto LABEL_204;
            }
            if ( v35 != v125 )
            {
              v76 = *(_QWORD *)(v125 + 112);
              if ( v76 )
              {
                while ( v35 != v76 )
                {
                  v76 = *(_QWORD *)(v76 + 112);
                  if ( !v76 )
                    goto LABEL_234;
                }
                if ( !v33 )
                {
                  v17 = v134;
                  goto LABEL_116;
                }
                goto LABEL_205;
              }
              goto LABEL_234;
            }
          }
LABEL_204:
          if ( !v33 )
          {
LABEL_206:
            v17 = v134;
            goto LABEL_113;
          }
LABEL_205:
          v135 = v13;
          goto LABEL_206;
        }
        v17 = v134;
        if ( !v33 )
          goto LABEL_113;
LABEL_216:
        v135 = v13;
        goto LABEL_113;
      }
      v17 = v134;
LABEL_111:
      if ( (_DWORD)qword_4F077B4 )
      {
        v38 = qword_4F077A0;
LABEL_159:
        if ( !v38 )
        {
          if ( !v33 )
            goto LABEL_113;
          v40 = v33;
LABEL_193:
          v135 = v13;
LABEL_95:
          if ( !v40 )
            goto LABEL_98;
          goto LABEL_113;
        }
        goto LABEL_160;
      }
      if ( !v33 )
        goto LABEL_113;
      goto LABEL_216;
    }
    if ( qword_4F077A0 )
    {
      v17 = v134;
      v35 = v134;
      if ( !v134 )
      {
LABEL_160:
        v40 = 1;
        goto LABEL_84;
      }
      goto LABEL_154;
    }
    goto LABEL_304;
  }
  if ( v33 )
  {
LABEL_69:
    v35 = v134;
    goto LABEL_70;
  }
LABEL_81:
  if ( (_DWORD)qword_4F077B4 && qword_4F077A0 )
  {
    v17 = v134;
    v33 = 1;
    v40 = 0;
LABEL_84:
    if ( *(_BYTE *)(v5 + 140) != 11 && v13 && *(_BYTE *)(v13 + 8) != 2 )
    {
      v41 = v125;
      if ( v125 )
      {
        if ( v125 == v17 )
          goto LABEL_94;
        while ( 1 )
        {
          v41 = *(_QWORD *)(v41 + 112);
          if ( !v41 )
            break;
          if ( v17 == v41 )
            goto LABEL_94;
        }
      }
      if ( (a3[2].m128i_i8[8] & 0x20) == 0 )
      {
        v101 = v13;
        v107 = v40;
        v114 = v33;
        v42 = sub_6E1A20(v135);
        sub_684B30(2904, v42);
        v17 = v134;
        v33 = v114;
        v40 = v107;
        v13 = v101;
      }
    }
LABEL_94:
    if ( !v33 )
      goto LABEL_95;
    goto LABEL_193;
  }
  v17 = v134;
LABEL_98:
  v43 = a3[2].m128i_i8[9];
LABEL_99:
  a3[2].m128i_i8[9] = v43 | 2;
  a3[2].m128i_i8[12] = v131 | a3[2].m128i_i8[12] & 0xFE;
  v133 = 0;
  if ( v17 )
    goto LABEL_33;
LABEL_100:
  if ( v11 && (a3[2].m128i_i8[9] & 0x20) == 0 )
    goto LABEL_102;
LABEL_34:
  if ( v129 != 1 )
    goto LABEL_35;
LABEL_103:
  v44 = *(_QWORD *)*a1;
  if ( v44 && *(_BYTE *)(v44 + 8) == 3 )
    v44 = sub_6BBB10(*a1);
  v45 = v133;
  *a1 = v44;
  if ( v45 )
  {
    if ( (a3[2].m128i_i8[8] & 0x20) != 0 )
    {
      v46 = (2 * (dword_4F077C0 == 0)) | a3[2].m128i_i8[9] & 0xFD;
      a3[2].m128i_i8[9] = v46;
      goto LABEL_109;
    }
    if ( dword_4F077C0 )
    {
      v81 = sub_6E1A20(v45);
      sub_684B30(1162, v81);
      goto LABEL_256;
    }
    v94 = sub_6E1A20(v45);
    sub_6851C0(146, v94);
    v46 = a3[2].m128i_i8[9] | 2;
    a3[2].m128i_i8[9] = v46;
  }
  else
  {
LABEL_256:
    v46 = a3[2].m128i_i8[9];
  }
LABEL_109:
  a3[2].m128i_i8[9] = (32 * (v127 & 1)) | v46 & 0xDF;
LABEL_36:
  a3[1].m128i_i64[1] = v103;
  v18 = &dword_4D048B8;
  if ( dword_4D048B8 )
  {
    LOWORD(v18) = a3[2].m128i_i16[4] & 0x8004;
    if ( (_WORD)v18 == 0x8000 )
    {
      for ( i = v5; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v18 = *(unsigned int **)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 24LL);
      if ( v18 )
      {
        v18 = (unsigned int *)*((_QWORD *)v18 + 11);
        if ( (*((_BYTE *)v18 + 194) & 8) == 0 || (*((_BYTE *)v18 + 206) & 0x10) != 0 )
        {
          v18 = (unsigned int *)sub_630000(v5, a3, a4);
          v86 = v18;
          if ( v18 )
          {
            if ( (a3[2].m128i_i8[8] & 0x40) == 0 )
            {
              v87 = *a5;
              v88 = sub_725A70((*(_BYTE *)(*a5 + 192) & 1) == 0 ? 2 : 6);
              *(_QWORD *)(v88 + 56) = v87;
              v89 = v88;
              if ( (*(_BYTE *)(v87 + 170) & 0x40) != 0 )
                *(_BYTE *)(v88 + 50) |= 0x80u;
              *(_QWORD *)(v88 + 16) = v86;
              v90 = a3[2].m128i_u8[10];
              if ( (v90 & 0x20) == 0 )
              {
                *((_BYTE *)v86 + 193) |= 0x40u;
                v90 = a3[2].m128i_u8[10];
              }
              sub_734250(v89, ((v90 >> 4) ^ 1) & 1);
              v91 = sub_724D50(9);
              *a5 = v91;
              *(_QWORD *)(v91 + 176) = v89;
              v18 = (unsigned int *)*a5;
              *(_QWORD *)(*a5 + 128) = v5;
              a3[2].m128i_i8[9] |= 4u;
            }
          }
        }
      }
    }
  }
  return (__int16)v18;
}
