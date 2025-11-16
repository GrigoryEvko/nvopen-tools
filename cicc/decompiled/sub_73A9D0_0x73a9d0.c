// Function: sub_73A9D0
// Address: 0x73a9d0
//
_QWORD *__fastcall sub_73A9D0(const __m128i *a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r14d
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  int v8; // r10d
  int v9; // r13d
  _QWORD *v11; // r15
  __int64 kk; // r13
  __int64 v13; // rax
  _QWORD *v14; // r15
  __int64 k; // r13
  __int64 v16; // rax
  _QWORD *v17; // r13
  __int64 v18; // rax
  _QWORD *v19; // r13
  __int64 v20; // rax
  _QWORD *v21; // r13
  __int64 v22; // rax
  __int8 v23; // al
  __int64 v24; // r14
  int v25; // eax
  int v26; // r10d
  _QWORD *v27; // r15
  __int64 i2; // r13
  __int64 v29; // rax
  _QWORD *v30; // r15
  __int64 i1; // r13
  __int64 v32; // rax
  int v33; // r9d
  int v34; // r10d
  __int64 v35; // rax
  unsigned __int8 *v36; // rdi
  int v37; // r9d
  __int64 v38; // rax
  _QWORD *v39; // rcx
  __int64 v40; // r13
  _QWORD *v41; // rcx
  __int64 v42; // rax
  _QWORD *v43; // r15
  __int64 n; // r13
  __int64 v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  char *v49; // rdx
  char v50; // dl
  _QWORD *v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rsi
  _QWORD *v56; // r13
  __int64 v57; // rax
  __int64 v58; // rax
  _QWORD *v59; // rax
  __int64 v60; // r15
  __int64 v61; // rax
  _QWORD *v62; // rax
  __int64 v63; // r15
  __int64 v64; // rdi
  __int64 v65; // r15
  __int64 j; // r13
  __int64 v67; // rax
  _QWORD *v68; // r15
  char v69; // al
  __int64 *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  _QWORD *v75; // rax
  _QWORD *v76; // r15
  __int64 m; // r13
  __int64 v78; // rax
  _QWORD *v79; // rax
  _QWORD *v80; // rax
  __int64 v81; // rax
  char v82; // dl
  _QWORD *v83; // rax
  _QWORD *ii; // rdx
  __int64 v85; // rdi
  __int64 v86; // rdi
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // r13
  __int64 v91; // rax
  _QWORD *v92; // rax
  __int64 v93; // rax
  int v94; // eax
  _BOOL4 v95; // eax
  _DWORD *v96; // r12
  _QWORD *v97; // rax
  __int64 v98; // [rsp+8h] [rbp-68h]
  _QWORD *v99; // [rsp+8h] [rbp-68h]
  __int64 v100; // [rsp+10h] [rbp-60h]
  __int64 v101; // [rsp+10h] [rbp-60h]
  __int64 v102; // [rsp+10h] [rbp-60h]
  int v103; // [rsp+10h] [rbp-60h]
  __int64 v104; // [rsp+10h] [rbp-60h]
  __int64 v105; // [rsp+10h] [rbp-60h]
  __int64 i; // [rsp+10h] [rbp-60h]
  _QWORD *v107; // [rsp+10h] [rbp-60h]
  __int64 v108; // [rsp+18h] [rbp-58h]
  __int64 v109; // [rsp+18h] [rbp-58h]
  __int64 jj; // [rsp+18h] [rbp-58h]
  __int64 nn; // [rsp+18h] [rbp-58h]
  __int64 mm; // [rsp+18h] [rbp-58h]
  int v113; // [rsp+18h] [rbp-58h]
  int v114; // [rsp+18h] [rbp-58h]
  int v115; // [rsp+18h] [rbp-58h]
  _QWORD *v116; // [rsp+18h] [rbp-58h]
  _QWORD *v117; // [rsp+18h] [rbp-58h]
  int v118; // [rsp+18h] [rbp-58h]
  int v119; // [rsp+18h] [rbp-58h]
  __int64 v120; // [rsp+18h] [rbp-58h]
  char v121; // [rsp+18h] [rbp-58h]
  int v122; // [rsp+18h] [rbp-58h]
  int v123; // [rsp+18h] [rbp-58h]
  const __m128i *v124; // [rsp+28h] [rbp-48h] BYREF
  __m128i v125[4]; // [rsp+30h] [rbp-40h] BYREF

  v4 = a2;
  v6 = sub_730FF0(a1);
  v7 = v6;
  if ( (a2 & 0x5000) != 0 )
  {
    v6[10] = a1[5].m128i_i64[0];
    *((_BYTE *)v6 + 26) = a1[1].m128i_i8[10] & 4 | *((_BYTE *)v6 + 26) & 0xFB;
  }
  switch ( a1[1].m128i_i8[8] )
  {
    case 0:
    case 3:
    case 4:
    case 0x10:
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x18:
    case 0x20:
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      goto LABEL_5;
    case 1:
      v8 = a2 & 1;
      if ( (a2 & 1) == 0 )
        goto LABEL_112;
      v55 = a3 + 8;
      if ( (unsigned int)sub_76ECE0(v6) )
        goto LABEL_133;
      v8 = v4 & 1;
LABEL_112:
      v56 = (_QWORD *)a1[4].m128i_i64[1];
      if ( v56 )
      {
        v118 = v8;
        v57 = sub_73A9D0(a1[4].m128i_i64[1], v4, a3);
        v8 = v118;
        v98 = v57;
        for ( i = v57; ; i = v58 )
        {
          v56 = (_QWORD *)v56[2];
          if ( !v56 )
            break;
          v119 = v8;
          v58 = sub_73A9D0(v56, v4, a3);
          v8 = v119;
          if ( v98 )
            *(_QWORD *)(i + 16) = v58;
          else
            v98 = v58;
        }
      }
      else
      {
        v98 = 0;
      }
      v9 = v4 & 0x10;
      v7[9] = v98;
      if ( a1[3].m128i_i8[8] == 91 )
      {
        v123 = v8;
        sub_7304E0(v98);
        v8 = v123;
      }
      goto LABEL_5;
    case 2:
      v51 = (_QWORD *)a1[3].m128i_i64[1];
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      if ( (a2 & 0x2800) != 0
        || (*((_BYTE *)v51 + 172) & 2) != 0
        || (*(_BYTE *)(v51 - 1) & 1) == 0 && ((*(_BYTE *)(v6 - 1) & 1) != 0 || (a2 & 0x201) != 0) )
      {
        v52 = sub_73FC90(v51, 0, (unsigned int)a2 | 0x20, a3);
        v8 = a2 & 1;
        v7[7] = v52;
        v105 = v52;
        if ( (a2 & 0x12000) != 0 )
        {
          v53 = *(_QWORD *)(a1[3].m128i_i64[1] + 144);
          if ( v53 )
          {
            v54 = sub_73A9D0(v53, (unsigned int)a2, a3);
            v8 = a2 & 1;
            *(_QWORD *)(v105 + 144) = v54;
          }
        }
      }
      goto LABEL_5;
    case 5:
      v48 = sub_73F780(a1[3].m128i_i64[1], (unsigned int)a2, a3);
      v7[7] = v48;
      v49 = *(char **)(v48 + 24);
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      if ( v49 )
      {
        v50 = *v49;
        if ( !v50 || v50 == 3 )
          *(_BYTE *)(v48 + 49) |= 1u;
      }
      goto LABEL_5;
    case 6:
      v46 = (_QWORD *)a1[4].m128i_i64[0];
      if ( (*(_BYTE *)(v46 - 1) & 1) != 0 || (*(_BYTE *)(v7 - 1) & 1) == 0 )
        goto LABEL_92;
      while ( 1 )
      {
        v46 = (_QWORD *)*v46;
        if ( !v46 )
          break;
        if ( (v46[4] & 1) == 0 )
        {
          v7 = sub_7305B0();
          v8 = a2 & 1;
          v9 = a2 & 0x10;
          goto LABEL_5;
        }
      }
LABEL_92:
      v9 = a2 & 0x10;
      v47 = sub_73F780(a1[3].m128i_i64[1], (unsigned int)a2, a3);
      v7[7] = v47;
      v8 = a2 & 1;
      if ( *(_BYTE *)(a1[3].m128i_i64[1] + 48) == 8 )
      {
        v7[8] = *(_QWORD *)(v47 + 64);
      }
      else
      {
        v7[8] = sub_740A90(a1[4].m128i_i64[0], (unsigned int)a2, a3);
        v8 = a2 & 1;
      }
LABEL_5:
      if ( !v8 )
        goto LABEL_6;
      goto LABEL_159;
    case 7:
      v62 = (_QWORD *)a1[3].m128i_i64[1];
      v64 = v62[6];
      v107 = v62;
      v99 = (_QWORD *)v7[7];
      if ( v64 )
      {
        v63 = v7[7];
        *(_QWORD *)(v63 + 48) = sub_73A9D0(v64, (unsigned int)a2, a3);
      }
      v65 = v107[3];
      if ( v65 )
      {
        v120 = sub_73A9D0(v107[3], (unsigned int)a2, a3);
        for ( j = v120; ; j = v67 )
        {
          v65 = *(_QWORD *)(v65 + 16);
          if ( !v65 )
            break;
          v67 = sub_73A9D0(v65, (unsigned int)a2, a3);
          if ( v120 )
            *(_QWORD *)(j + 16) = v67;
          else
            v120 = v67;
        }
        v99[3] = v120;
      }
      v85 = v107[5];
      if ( v85 )
        v99[5] = sub_73F780(v85, (unsigned int)a2, a3);
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      v86 = v107[4];
      if ( v86 )
      {
        v87 = sub_73F780(v86, (unsigned int)a2, a3);
        v8 = a2 & 1;
        v99[4] = v87;
      }
      goto LABEL_5;
    case 8:
      v59 = (_QWORD *)a1[3].m128i_i64[1];
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      if ( v59 )
      {
        v60 = v7[7];
        v61 = sub_73F780(v59[1], (unsigned int)a2, a3);
        v8 = a2 & 1;
        *(_QWORD *)(v60 + 8) = v61;
      }
      goto LABEL_5;
    case 9:
      v68 = (_QWORD *)a1[3].m128i_i64[1];
      if ( !v68[3] )
      {
        v90 = v6[7];
        *(_QWORD *)(v90 + 8) = sub_73F780(v68[1], (unsigned int)a2, a3);
        v91 = sub_73A9D0(v68[2], (unsigned int)a2, a3);
        v8 = a2 & 1;
        *(_QWORD *)(v90 + 16) = v91;
        v9 = a2 & 0x10;
        goto LABEL_5;
      }
      if ( (a2 & 1) == 0 )
LABEL_160:
        sub_721090();
      *(_DWORD *)(a3 + 8) = 1;
      v55 = a3 + 8;
LABEL_133:
      v9 = v4 & 0x10;
      goto LABEL_134;
    case 0xA:
      v33 = 0;
      v34 = a2 & 1;
      if ( (a2 & 1) != 0 && (*(_BYTE *)(v6 - 1) & 1) == 0 && !*(_QWORD *)(qword_4F04C50 + 88LL) )
      {
        sub_733780(0x17u, qword_4F04C50, 0, 1, 0);
        v34 = a2 & 1;
        v33 = v34;
      }
      v103 = v34;
      v114 = v33;
      if ( *(_BYTE *)qword_4F06BC0 == 4 )
      {
        v88 = sub_73A9D0(a1[3].m128i_i64[1], (unsigned int)a2, a3);
        v37 = v114;
        v8 = v103;
        v7 = (_QWORD *)v88;
      }
      else
      {
        sub_733780(0, 0, 0, *(_BYTE *)a1[4].m128i_i64[0], 0);
        v35 = sub_73A9D0(a1[3].m128i_i64[1], (unsigned int)a2, a3);
        v36 = (unsigned __int8 *)qword_4F06BC0;
        v7[8] = 0;
        v7[7] = v35;
        sub_732E60(v36, 0xDu, v7);
        sub_733F40();
        v37 = v114;
        v8 = v103;
        if ( !v7[8] )
          v7 = (_QWORD *)v7[7];
      }
      v9 = a2 & 0x10;
      if ( v37 )
      {
        v115 = v8;
        sub_733F40();
        v8 = v115;
      }
      goto LABEL_5;
    case 0xB:
      v14 = (_QWORD *)a1[3].m128i_i64[1];
      if ( !v14 )
        goto LABEL_161;
      v109 = sub_73A9D0(v14, (unsigned int)a2, a3);
      for ( k = v109; ; k = v16 )
      {
        v14 = (_QWORD *)v14[2];
        if ( !v14 )
          break;
        v16 = sub_73A9D0(v14, (unsigned int)a2, a3);
        if ( v109 )
          *(_QWORD *)(k + 16) = v16;
        else
          v109 = v16;
      }
      goto LABEL_162;
    case 0xC:
    case 0xE:
    case 0xF:
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      if ( a1[3].m128i_i8[8] )
        goto LABEL_5;
      if ( (*(_BYTE *)(v6 - 1) & 1) == 0 || (a1[-1].m128i_i8[8] & 1) != 0 )
      {
LABEL_145:
        v122 = v8;
        v6[8] = sub_73A9D0(a1[4].m128i_i64[0], (unsigned int)a2, a3);
        v8 = v122;
      }
      else
      {
        *((_BYTE *)v6 + 56) = 1;
        v6[8] = *(_QWORD *)a1[4].m128i_i64[0];
      }
      goto LABEL_5;
    case 0xD:
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      if ( !a1[3].m128i_i16[4] )
        goto LABEL_145;
      goto LABEL_5;
    case 0x11:
      if ( (a2 & 0x80u) == 0LL )
        goto LABEL_160;
      v121 = a1[1].m128i_i8[9] & 1;
      v71 = sub_73A930(a1->m128i_i64[0]);
      v7 = (_QWORD *)v71;
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      if ( !v121 )
      {
        v75 = sub_731370(v71, a2, 0, v72, v73, v74);
        v8 = a2 & 1;
        v7 = v75;
      }
      goto LABEL_5;
    case 0x12:
      v79 = *(_QWORD **)a3;
      if ( !*(_QWORD *)a3 )
        goto LABEL_158;
      break;
    case 0x17:
      v76 = (_QWORD *)a1[4].m128i_i64[0];
      if ( !v76 )
        goto LABEL_164;
      v108 = sub_73A9D0(v76, (unsigned int)a2, a3);
      for ( m = v108; ; m = v78 )
      {
        v76 = (_QWORD *)v76[2];
        if ( !v76 )
          break;
        v78 = sub_73A9D0(v76, (unsigned int)a2, a3);
        if ( v108 )
          *(_QWORD *)(m + 16) = v78;
        else
          v108 = v78;
      }
      goto LABEL_165;
    case 0x19:
      v43 = (_QWORD *)a1[3].m128i_i64[1];
      if ( !v43 )
        goto LABEL_161;
      v109 = sub_73A9D0(v43, (unsigned int)a2, a3);
      for ( n = v109; ; n = v45 )
      {
        v43 = (_QWORD *)v43[2];
        if ( !v43 )
          break;
        v45 = sub_73A9D0(v43, (unsigned int)a2, a3);
        if ( v109 )
          *(_QWORD *)(n + 16) = v45;
        else
          v109 = v45;
      }
      goto LABEL_162;
    case 0x1A:
      if ( a1[3].m128i_i64[1] )
      {
        v116 = (_QWORD *)a1[3].m128i_i64[1];
        v38 = sub_73A9D0(v116, (unsigned int)a2, a3);
        v39 = v116;
        v104 = v38;
        v40 = v38;
        while ( 1 )
        {
          v41 = (_QWORD *)v39[2];
          if ( !v41 )
            break;
          v117 = v41;
          v42 = sub_73A9D0(v41, (unsigned int)a2, a3);
          v39 = v117;
          if ( v40 )
            *(_QWORD *)(v104 + 16) = v42;
          else
            v40 = v42;
          v104 = v42;
        }
      }
      else
      {
        v40 = 0;
      }
      v7[7] = v40;
      v83 = (_QWORD *)a1[3].m128i_i64[1];
      for ( ii = (_QWORD *)a1[4].m128i_i64[0]; v83 != ii; v40 = *(_QWORD *)(v40 + 16) )
        v83 = (_QWORD *)v83[2];
      v7[8] = v40;
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      goto LABEL_5;
    case 0x1B:
      v17 = (_QWORD *)a1[3].m128i_i64[1];
      if ( v17 )
      {
        v100 = sub_73A9D0(a1[3].m128i_i64[1], (unsigned int)a2, a3);
        for ( jj = v100; ; jj = v18 )
        {
          v17 = (_QWORD *)v17[2];
          if ( !v17 )
            break;
          v18 = sub_73A9D0(v17, (unsigned int)a2, a3);
          if ( v100 )
            *(_QWORD *)(jj + 16) = v18;
          else
            v100 = v18;
        }
      }
      else
      {
        v100 = 0;
      }
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      v7[7] = v100;
      *((_BYTE *)v7 + 64) = a1[4].m128i_i8[0] & 1 | v7[8] & 0xFE;
      goto LABEL_5;
    case 0x1C:
    case 0x1D:
      v6[7] = sub_73A9D0(a1[3].m128i_i64[1], (unsigned int)a2, a3);
      v11 = (_QWORD *)a1[4].m128i_i64[0];
      if ( v11 )
      {
        v108 = sub_73A9D0(v11, (unsigned int)a2, a3);
        for ( kk = v108; ; kk = v13 )
        {
          v11 = (_QWORD *)v11[2];
          if ( !v11 )
            break;
          v13 = sub_73A9D0(v11, (unsigned int)a2, a3);
          if ( v108 )
            *(_QWORD *)(kk + 16) = v13;
          else
            v108 = v13;
        }
      }
      else
      {
LABEL_164:
        v108 = 0;
      }
LABEL_165:
      v80 = (_QWORD *)v108;
      goto LABEL_166;
    case 0x1E:
      v21 = (_QWORD *)a1[3].m128i_i64[1];
      if ( v21 )
      {
        v102 = sub_73A9D0(a1[3].m128i_i64[1], (unsigned int)a2, a3);
        for ( mm = v102; ; mm = v22 )
        {
          v21 = (_QWORD *)v21[2];
          if ( !v21 )
            break;
          v22 = sub_73A9D0(v21, (unsigned int)a2, a3);
          if ( v102 )
            *(_QWORD *)(mm + 16) = v22;
          else
            v102 = v22;
        }
      }
      else
      {
        v102 = 0;
      }
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      v7[7] = v102;
      *((_WORD *)v7 + 32) = a1[4].m128i_i16[0];
      *((_BYTE *)v7 + 66) = a1[4].m128i_i8[2] & 1 | *((_BYTE *)v7 + 66) & 0xFE;
      goto LABEL_5;
    case 0x21:
      v19 = (_QWORD *)a1[3].m128i_i64[1];
      if ( v19 )
      {
        v101 = sub_73A9D0(a1[3].m128i_i64[1], (unsigned int)a2, a3);
        for ( nn = v101; ; nn = v20 )
        {
          v19 = (_QWORD *)v19[2];
          if ( !v19 )
            break;
          v20 = sub_73A9D0(v19, (unsigned int)a2, a3);
          if ( v101 )
            *(_QWORD *)(nn + 16) = v20;
          else
            v101 = v20;
        }
      }
      else
      {
        v101 = 0;
      }
      v7[7] = v101;
      v80 = (_QWORD *)a1[4].m128i_i64[0];
LABEL_166:
      v7[8] = v80;
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      goto LABEL_5;
    case 0x22:
      v30 = (_QWORD *)a1[3].m128i_i64[1];
      if ( !v30 )
        goto LABEL_161;
      v109 = sub_73A9D0(v30, (unsigned int)a2, a3);
      for ( i1 = v109; ; i1 = v32 )
      {
        v30 = (_QWORD *)v30[2];
        if ( !v30 )
          break;
        v32 = sub_73A9D0(v30, (unsigned int)a2, a3);
        if ( v109 )
          *(_QWORD *)(i1 + 16) = v32;
        else
          v109 = v32;
      }
      goto LABEL_162;
    case 0x23:
      v27 = (_QWORD *)a1[3].m128i_i64[1];
      if ( v27 )
      {
        v109 = sub_73A9D0(v27, (unsigned int)a2, a3);
        for ( i2 = v109; ; i2 = v29 )
        {
          v27 = (_QWORD *)v27[2];
          if ( !v27 )
            break;
          v29 = sub_73A9D0(v27, (unsigned int)a2, a3);
          if ( v109 )
            *(_QWORD *)(i2 + 16) = v29;
          else
            v109 = v29;
        }
      }
      else
      {
LABEL_161:
        v109 = 0;
      }
LABEL_162:
      v79 = (_QWORD *)v109;
      goto LABEL_158;
    case 0x24:
      v8 = a2 & 1;
      v9 = a2 & 0x10;
      if ( (a2 & 0x40000) != 0 )
      {
        v89 = sub_73A9D0(a1[3].m128i_i64[1], (unsigned int)a2, a3);
        v8 = a2 & 1;
        v7 = (_QWORD *)v89;
      }
      else
      {
        v23 = a1[4].m128i_i8[0];
        if ( (v23 & 1) != 0 && (a2 & 0x10) != 0 || (v23 & 2) != 0 && (a2 & 0x20000) != 0 )
        {
          v24 = sub_73A9D0(a1[3].m128i_i64[1], (unsigned int)a2 | 0x40000, a3);
          v124 = (const __m128i *)sub_724DC0();
          v125[0] = 0u;
          v25 = sub_7A30C0(v24, 1, 0, v124);
          v26 = a2 & 1;
          if ( v25 )
          {
            v92 = sub_73A720(v124, 1);
            v26 = a2 & 1;
            v7 = v92;
          }
          else if ( dword_4F04C44 == -1 )
          {
            v93 = qword_4F04C68[0] + 776LL * dword_4F04C64;
            if ( (*(_BYTE *)(v93 + 6) & 6) == 0 && *(_BYTE *)(v93 + 4) != 12 )
            {
              v94 = sub_6E5430();
              v26 = a2 & 1;
              if ( v94 )
              {
                v95 = sub_6E5AC0();
                v26 = a2 & 1;
                if ( !v95 )
                {
                  v96 = sub_67D9D0(0x1Cu, (_DWORD *)(v24 + 28));
                  sub_67E370((__int64)v96, v125);
                  sub_685910((__int64)v96, (FILE *)v125);
                  v97 = sub_7305B0();
                  v26 = a2 & 1;
                  v7 = v97;
                }
              }
            }
          }
          v113 = v26;
          sub_724E30((__int64)&v124);
          v8 = v113;
        }
      }
      goto LABEL_5;
    default:
      goto LABEL_160;
  }
  while ( a1[3].m128i_i64[1] != v79[1] )
  {
    v79 = (_QWORD *)*v79;
    if ( !v79 )
      goto LABEL_158;
  }
  v79 = (_QWORD *)v79[2];
LABEL_158:
  v7[7] = v79;
  v9 = a2 & 0x10;
  if ( (a2 & 1) != 0 )
  {
LABEL_159:
    v55 = a3 + 8;
LABEL_134:
    sub_76E7E0(v7, v55);
    if ( !v9 )
      return v7;
  }
  else
  {
LABEL_6:
    if ( !v9 )
      return v7;
  }
  v69 = *((_BYTE *)v7 + 24);
  switch ( v69 )
  {
    case 20:
    case 3:
      sub_728F70(*(_QWORD *)v7[7]);
      return v7;
    case 2:
      v81 = v7[7];
      v82 = *(_BYTE *)(v81 + 173);
      if ( v82 == 6 )
      {
        if ( *(_BYTE *)(v81 + 176) <= 1u )
          sub_728F70(**(_QWORD **)(v81 + 184));
      }
      else if ( v82 == 7 && (*(_BYTE *)(v81 + 192) & 2) != 0 )
      {
        v70 = *(__int64 **)(v81 + 200);
        if ( v70 )
          goto LABEL_140;
      }
      break;
    case 7:
      v70 = *(__int64 **)(v7[7] + 16LL);
      if ( v70 )
LABEL_140:
        sub_728F70(*v70);
      break;
  }
  return v7;
}
