// Function: sub_1102530
// Address: 0x1102530
//
unsigned __int8 *__fastcall sub_1102530(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v8; // r13
  __int64 v10; // rdx
  unsigned __int8 *v11; // rbx
  int v12; // eax
  __int64 v13; // rcx
  unsigned __int8 v14; // al
  __int64 *v15; // r14
  _BYTE *v16; // rbx
  unsigned int v17; // r13d
  int v18; // eax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rax
  unsigned int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned int v29; // r14d
  __int64 *v30; // rax
  __int64 v31; // r12
  const char *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r14
  __int64 *v35; // rdi
  __int64 *v36; // rsi
  int v37; // r15d
  __int64 v38; // rax
  __int64 v39; // r15
  _QWORD *v40; // rax
  unsigned int v41; // ecx
  __int64 v42; // r15
  __int64 v43; // rsi
  __int64 *v44; // r12
  __int64 *v45; // rbx
  __int64 v46; // rdi
  __int64 v47; // r14
  __int64 v48; // r13
  unsigned int v49; // edx
  unsigned int v50; // eax
  unsigned int v51; // ecx
  __int64 v52; // rdi
  __int64 v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // r12
  __int64 v57; // rsi
  __int64 v58; // rax
  _BYTE *v59; // rax
  char v60; // al
  __int64 v61; // rdi
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // rdi
  __int64 v65; // rsi
  __int64 v66; // r14
  __int64 v67; // r12
  unsigned int v68; // eax
  int v69; // ebx
  __int64 v70; // r13
  _BYTE *v71; // rax
  char v72; // al
  __int64 v73; // rdi
  __int64 v74; // rsi
  __int64 v75; // rax
  __int64 v76; // rdi
  __int64 v77; // r12
  __int64 v78; // rsi
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rsi
  int v82; // edi
  __int64 v83; // rdi
  __int64 v84; // rsi
  __int64 v85; // rax
  __int64 v86; // rdi
  __int64 v87; // r12
  __int64 v88; // rsi
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // r13
  __int64 v93; // rbx
  __int64 v94; // r13
  __int64 v95; // rdx
  unsigned int v96; // esi
  __int64 *v97; // rax
  __int64 v98; // r10
  _BYTE *v99; // rdx
  __int64 *v100; // rax
  __int64 v101; // r10
  __int64 v102; // rsi
  _BYTE *v103; // rax
  __int64 v104; // r11
  __int64 v105; // rax
  unsigned int **v106; // rdi
  __int64 v107; // rax
  __int64 v108; // r11
  __int64 v109; // rsi
  __int64 v110; // rax
  unsigned int **v111; // rdi
  __int64 v112; // rax
  __int64 v113; // rdi
  __int64 v114; // rsi
  __int64 v115; // rax
  __int64 v116; // rdx
  __int64 v117; // rbx
  __int64 v118; // r12
  __int64 v119; // rdx
  unsigned int v120; // esi
  __int64 v121; // [rsp+8h] [rbp-F8h]
  unsigned int v122; // [rsp+8h] [rbp-F8h]
  __int64 *v123; // [rsp+10h] [rbp-F0h]
  unsigned int v124; // [rsp+10h] [rbp-F0h]
  __int64 v125; // [rsp+10h] [rbp-F0h]
  __int64 v126; // [rsp+10h] [rbp-F0h]
  __int64 v127; // [rsp+20h] [rbp-E0h]
  unsigned int v128; // [rsp+20h] [rbp-E0h]
  __int64 v129; // [rsp+20h] [rbp-E0h]
  __int64 v130; // [rsp+20h] [rbp-E0h]
  __int64 v131; // [rsp+20h] [rbp-E0h]
  __int64 v132; // [rsp+28h] [rbp-D8h]
  unsigned int v133; // [rsp+28h] [rbp-D8h]
  unsigned int v134; // [rsp+28h] [rbp-D8h]
  __int64 v135; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v136; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v137; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v138[4]; // [rsp+50h] [rbp-B0h] BYREF
  __int16 v139; // [rsp+70h] [rbp-90h]
  __int64 *v140; // [rsp+80h] [rbp-80h] BYREF
  __int64 v141; // [rsp+88h] [rbp-78h]
  _BYTE v142[16]; // [rsp+90h] [rbp-70h] BYREF
  __int16 v143; // [rsp+A0h] [rbp-60h]

  v8 = (__int64)sub_11005E0(a1, (unsigned __int8 *)a2, a3, a4, a5, a6);
  if ( v8 )
    return (unsigned __int8 *)v8;
  v10 = *(_QWORD *)(a2 + 8);
  v11 = *(unsigned __int8 **)(a2 - 32);
  v135 = v10;
  v12 = *v11;
  if ( (unsigned __int8)v12 <= 0x1Cu )
  {
LABEL_22:
    v15 = (__int64 *)a1[2].m128i_i64[0];
    goto LABEL_8;
  }
  v13 = *((_QWORD *)v11 + 2);
  if ( (unsigned int)(v12 - 42) > 0x11 )
  {
LABEL_14:
    if ( !v13 )
      goto LABEL_6;
    goto LABEL_15;
  }
  if ( !v13 )
  {
LABEL_6:
    v14 = *v11;
    v15 = (__int64 *)a1[2].m128i_i64[0];
LABEL_7:
    if ( v14 == 85 )
    {
      v23 = *((_QWORD *)v11 - 4);
      if ( v23 )
      {
        if ( !*(_BYTE *)v23 && *(_QWORD *)(v23 + 24) == *((_QWORD *)v11 + 10) && (*(_BYTE *)(v23 + 33) & 0x20) != 0 )
        {
          v24 = *(_DWORD *)(v23 + 36);
          if ( v24 == 250 )
            goto LABEL_31;
          if ( v24 > 0xFA )
          {
            if ( v24 > 0x136 )
            {
              if ( v24 == 355 )
                goto LABEL_31;
            }
            else if ( v24 > 0x133 )
            {
              goto LABEL_31;
            }
          }
          else if ( ((v24 - 170) & 0xFFFFFFFD) == 0 || v24 == 21 )
          {
LABEL_31:
            v25 = *(_QWORD *)&v11[-32 * (*((_DWORD *)v11 + 1) & 0x7FFFFFF)];
            v26 = *(_QWORD *)(v25 + 16);
            if ( v26 )
            {
              if ( !*(_QWORD *)(v26 + 8) )
              {
                v27 = v135;
                if ( v24 == 170 || *(_BYTE *)v25 == 75 && (v27 = v135, v135 == *(_QWORD *)(*(_QWORD *)(v25 - 32) + 8LL)) )
                {
                  HIDWORD(v138[0]) = 0;
                  v143 = 257;
                  v132 = sub_10FFB20((__int64)v15, v25, v27, LODWORD(v138[0]), (__int64)&v140, 0);
                  v28 = *((_QWORD *)v11 - 4);
                  if ( !v28 || *(_BYTE *)v28 || *(_QWORD *)(v28 + 24) != *((_QWORD *)v11 + 10) )
                    BUG();
                  v29 = *(_DWORD *)(v28 + 36);
                  v30 = (__int64 *)sub_B43CA0(a2);
                  v31 = sub_B6E160(v30, v29, (__int64)&v135, 1);
                  v140 = (__int64 *)v142;
                  v141 = 0x100000000LL;
                  sub_B56970((__int64)v11, (__int64)&v140);
                  v32 = sub_BD5D20((__int64)v11);
                  v139 = 261;
                  v138[1] = v33;
                  v138[0] = (__int64)v32;
                  v137 = v132;
                  if ( v31 )
                    v34 = *(_QWORD *)(v31 + 24);
                  else
                    v34 = 0;
                  v35 = &v140[7 * (unsigned int)v141];
                  if ( v140 == v35 )
                  {
                    v37 = 0;
                  }
                  else
                  {
                    v36 = v140;
                    v37 = 0;
                    do
                    {
                      v38 = v36[5] - v36[4];
                      v36 += 7;
                      v37 += v38 >> 3;
                    }
                    while ( v35 != v36 );
                  }
                  v121 = (unsigned int)v141;
                  v39 = (unsigned int)(v37 + 2);
                  v123 = v140;
                  LOBYTE(v132) = 16 * (_DWORD)v141 != 0;
                  v40 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v141) << 32) | v39);
                  v8 = (__int64)v40;
                  if ( v40 )
                  {
                    v41 = v39 & 0x7FFFFFF | ((_DWORD)v132 << 28);
                    v42 = (__int64)v40;
                    sub_B44260((__int64)v40, **(_QWORD **)(v34 + 16), 56, v41, 0, 0);
                    *(_QWORD *)(v8 + 72) = 0;
                    sub_B4A290(v8, v34, v31, &v137, 1, (__int64)v138, (__int64)v123, v121);
                  }
                  else
                  {
                    v42 = 0;
                  }
                  v43 = (__int64)v11;
                  sub_B45230(v42, (__int64)v11);
                  v44 = v140;
                  v45 = &v140[7 * (unsigned int)v141];
                  if ( v140 != v45 )
                  {
                    do
                    {
                      v46 = *(v45 - 3);
                      v45 -= 7;
                      if ( v46 )
                      {
                        v43 = v45[6] - v46;
                        j_j___libc_free_0(v46, v43);
                      }
                      if ( (__int64 *)*v45 != v45 + 2 )
                      {
                        v43 = v45[2] + 1;
                        j_j___libc_free_0(*v45, v43);
                      }
                    }
                    while ( v44 != v45 );
                    v44 = v140;
                  }
                  if ( v44 != (__int64 *)v142 )
                    _libc_free(v44, v43);
                  return (unsigned __int8 *)v8;
                }
              }
            }
          }
        }
      }
    }
LABEL_8:
    v8 = (__int64)sub_10FF8B0((char *)a2, v15);
    if ( !v8 )
    {
      v16 = *(_BYTE **)(a2 - 32);
      if ( (unsigned __int8)(*v16 - 72) <= 1u && sub_10FD370(*(char **)(a2 - 32), a1) )
      {
        v143 = 257;
        return (unsigned __int8 *)sub_B51D30(
                                    (unsigned int)(unsigned __int8)*v16 - 29,
                                    *((_QWORD *)v16 - 4),
                                    v135,
                                    (__int64)&v140,
                                    0,
                                    0);
      }
    }
    return (unsigned __int8 *)v8;
  }
  if ( !*(_QWORD *)(v13 + 8) )
  {
    v47 = sub_10FDD10(*((_QWORD *)v11 - 8), *(_BYTE *)(v10 + 8) == 1);
    v48 = sub_10FDD10(*((_QWORD *)v11 - 4), *(_BYTE *)(v135 + 8) == 1);
    v124 = sub_BCB090(*((_QWORD *)v11 + 1));
    v133 = sub_BCB090(v47);
    v49 = sub_BCB090(v48);
    v50 = v133;
    v128 = v49;
    if ( v133 < v49 )
      v50 = v49;
    v122 = v50;
    v51 = sub_BCB090(v135);
    switch ( *v11 )
    {
      case '+':
      case '-':
        if ( 2 * v51 + 1 > v124 || v51 < v122 )
          goto LABEL_12;
        v143 = 257;
        v52 = a1[2].m128i_i64[0];
        v53 = *((_QWORD *)v11 - 8);
        HIDWORD(v138[0]) = 0;
        v54 = sub_10FFB20(v52, v53, v135, LODWORD(v138[0]), (__int64)&v140, 0);
        v55 = a1[2].m128i_i64[0];
        v143 = 257;
        v56 = v54;
        v57 = *((_QWORD *)v11 - 4);
        HIDWORD(v138[0]) = 0;
        v58 = sub_10FFB20(v55, v57, v135, LODWORD(v138[0]), (__int64)&v140, 0);
        v143 = 257;
        v8 = sub_B504D0((unsigned int)*v11 - 29, v56, v58, (__int64)&v140, 0, 0);
        sub_B45230(v8, (__int64)v11);
        return (unsigned __int8 *)v8;
      case '/':
        if ( v133 + v128 > v124 || v51 < v122 )
          goto LABEL_12;
        v143 = 257;
        v83 = a1[2].m128i_i64[0];
        v84 = *((_QWORD *)v11 - 8);
        HIDWORD(v138[0]) = 0;
        v85 = sub_10FFB20(v83, v84, v135, LODWORD(v138[0]), (__int64)&v140, 0);
        v86 = a1[2].m128i_i64[0];
        v143 = 257;
        v87 = v85;
        v88 = *((_QWORD *)v11 - 4);
        HIDWORD(v138[0]) = 0;
        v89 = sub_10FFB20(v86, v88, v135, LODWORD(v138[0]), (__int64)&v140, 0);
        v143 = 257;
        v80 = v89;
        v81 = v87;
        v82 = 18;
        goto LABEL_81;
      case '2':
        if ( LOBYTE(qword_4F8B7A8[8]) || !LOBYTE(qword_4F8B888[8]) )
          goto LABEL_12;
        v134 = v51;
        v71 = sub_C94E20((__int64)qword_4F863F0);
        v72 = v71 ? *v71 : LOBYTE(qword_4F863F0[2]);
        if ( v72 || 2 * v134 > v124 || v134 < v122 )
          goto LABEL_12;
        v143 = 257;
        v73 = a1[2].m128i_i64[0];
        v74 = *((_QWORD *)v11 - 8);
        HIDWORD(v138[0]) = 0;
        v75 = sub_10FFB20(v73, v74, v135, LODWORD(v138[0]), (__int64)&v140, 0);
        v76 = a1[2].m128i_i64[0];
        v143 = 257;
        v77 = v75;
        v78 = *((_QWORD *)v11 - 4);
        HIDWORD(v138[0]) = 0;
        v79 = sub_10FFB20(v76, v78, v135, LODWORD(v138[0]), (__int64)&v140, 0);
        v143 = 257;
        v80 = v79;
        v81 = v77;
        v82 = 21;
LABEL_81:
        v8 = sub_B504D0(v82, v81, v80, (__int64)&v140, 0, 0);
        sub_B45260((unsigned __int8 *)v8, (__int64)v11, 1);
        return (unsigned __int8 *)v8;
      case '5':
        if ( LOBYTE(qword_4F8B7A8[8]) || !LOBYTE(qword_4F8B888[8]) )
          goto LABEL_12;
        v59 = sub_C94E20((__int64)qword_4F863F0);
        v60 = v59 ? *v59 : LOBYTE(qword_4F863F0[2]);
        if ( v124 == v122 || v60 )
          goto LABEL_12;
        v61 = a1[2].m128i_i64[0];
        v143 = 257;
        v62 = *((_QWORD *)v11 - 8);
        HIDWORD(v138[0]) = 0;
        if ( v133 >= v128 )
        {
          v112 = sub_10FFB20(v61, v62, v47, v138[0], (__int64)&v140, 0);
          v113 = a1[2].m128i_i64[0];
          v143 = 257;
          v114 = *((_QWORD *)v11 - 4);
          HIDWORD(v138[0]) = 0;
          v129 = v112;
          v66 = sub_10FFB20(v113, v114, v47, LODWORD(v138[0]), (__int64)&v140, 0);
        }
        else
        {
          v63 = sub_10FFB20(v61, v62, v48, v138[0], (__int64)&v140, 0);
          v64 = a1[2].m128i_i64[0];
          v143 = 257;
          v65 = *((_QWORD *)v11 - 4);
          HIDWORD(v138[0]) = 0;
          v129 = v63;
          v66 = sub_10FFB20(v64, v65, v48, LODWORD(v138[0]), (__int64)&v140, 0);
        }
        v139 = 257;
        v67 = a1[2].m128i_i64[0];
        v68 = sub_B45210((__int64)v11);
        BYTE4(v136) = 1;
        LODWORD(v136) = v68;
        v69 = v68;
        v137 = v136;
        if ( *(_BYTE *)(v67 + 108) )
        {
          v70 = sub_B35400(v67, 0x72u, v129, v66, v136, (__int64)v138, 0, 0, 0);
        }
        else
        {
          v70 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(v67 + 80) + 40LL))(
                  *(_QWORD *)(v67 + 80),
                  24,
                  v129,
                  v66,
                  v68);
          if ( !v70 )
          {
            v143 = 257;
            v115 = sub_B504D0(24, v129, v66, (__int64)&v140, 0, 0);
            v116 = *(_QWORD *)(v67 + 96);
            v70 = v115;
            if ( v116 )
              sub_B99FD0(v115, 3u, v116);
            sub_B45150(v70, v69);
            (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v67 + 88) + 16LL))(
              *(_QWORD *)(v67 + 88),
              v70,
              v138,
              *(_QWORD *)(v67 + 56),
              *(_QWORD *)(v67 + 64));
            v117 = *(_QWORD *)v67;
            v118 = *(_QWORD *)v67 + 16LL * *(unsigned int *)(v67 + 8);
            while ( v117 != v118 )
            {
              v119 = *(_QWORD *)(v117 + 8);
              v120 = *(_DWORD *)v117;
              v117 += 16;
              sub_B99FD0(v70, v120, v119);
            }
          }
        }
        v143 = 257;
        return (unsigned __int8 *)sub_B52350(v70, v135, (__int64)&v140, 0, 0);
      default:
LABEL_12:
        v11 = *(unsigned __int8 **)(a2 - 32);
        if ( *v11 <= 0x1Cu )
          goto LABEL_22;
        v13 = *((_QWORD *)v11 + 2);
        break;
    }
    goto LABEL_14;
  }
LABEL_15:
  if ( *(_QWORD *)(v13 + 8) )
    goto LABEL_6;
  v17 = sub_B45210(a2);
  if ( (unsigned __int8)sub_920620((__int64)v11) )
  {
    v18 = v11[1] >> 1;
    if ( v18 != 127 )
      v17 &= v18;
  }
  v140 = &v136;
  if ( !(unsigned __int8)sub_10FFFD0(&v140, v11) )
  {
    v15 = (__int64 *)a1[2].m128i_i64[0];
    if ( *v11 != 86 )
    {
LABEL_91:
      v11 = *(unsigned __int8 **)(a2 - 32);
      v14 = *v11;
      goto LABEL_7;
    }
    v97 = (__int64 *)sub_986520((__int64)v11);
    v98 = *v97;
    if ( *v97
      && (v99 = (_BYTE *)v97[4], *v99 == 75)
      && (v108 = *((_QWORD *)v99 - 4)) != 0
      && (v109 = v97[8]) != 0
      && v135 == *(_QWORD *)(v108 + 8) )
    {
      LODWORD(v138[0]) = v17;
      BYTE4(v138[0]) = 1;
      v126 = v108;
      v143 = 257;
      v131 = v98;
      v110 = sub_10FFB20((__int64)v15, v109, v135, v138[0], (__int64)&v140, 0);
      LODWORD(v138[0]) = v17;
      v111 = (unsigned int **)a1[2].m128i_i64[0];
      v140 = (__int64 *)"narrow.sel";
      BYTE4(v138[0]) = 1;
      v143 = 259;
      v107 = sub_B36280(v111, v131, v126, v110, v138[0], (__int64)&v140, (__int64)v11);
    }
    else
    {
      v100 = (__int64 *)sub_986520((__int64)v11);
      v101 = *v100;
      if ( !*v100 )
        goto LABEL_91;
      v102 = v100[4];
      if ( !v102 )
        goto LABEL_91;
      v103 = (_BYTE *)v100[8];
      if ( *v103 != 75 )
        goto LABEL_91;
      v104 = *((_QWORD *)v103 - 4);
      if ( !v104 || v135 != *(_QWORD *)(v104 + 8) )
        goto LABEL_91;
      LODWORD(v138[0]) = v17;
      BYTE4(v138[0]) = 1;
      v130 = v101;
      v125 = v104;
      v143 = 257;
      v105 = sub_10FFB20((__int64)v15, v102, v135, v138[0], (__int64)&v140, 0);
      v106 = (unsigned int **)a1[2].m128i_i64[0];
      v140 = (__int64 *)"narrow.sel";
      v143 = 259;
      LODWORD(v138[0]) = v17;
      BYTE4(v138[0]) = 1;
      v107 = sub_B36280(v106, v130, v105, v125, v138[0], (__int64)&v140, (__int64)v11);
    }
    return sub_F162A0((__int64)a1, a2, v107);
  }
  LODWORD(v138[0]) = v17;
  v19 = a1[2].m128i_i64[0];
  BYTE4(v138[0]) = 1;
  v143 = 257;
  v20 = sub_10FFB20(v19, v136, v135, v138[0], (__int64)&v140, 0);
  v21 = a1[2].m128i_i64[0];
  v139 = 257;
  v127 = v20;
  v22 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(v21 + 80) + 48LL))(
          *(_QWORD *)(v21 + 80),
          12,
          v20,
          v17);
  if ( !v22 )
  {
    v143 = 257;
    v90 = sub_B50340(12, v127, (__int64)&v140, 0, 0);
    v91 = *(_QWORD *)(v21 + 96);
    v22 = v90;
    if ( v91 )
      sub_B99FD0(v90, 3u, v91);
    sub_B45150(v22, v17);
    (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v21 + 88) + 16LL))(
      *(_QWORD *)(v21 + 88),
      v22,
      v138,
      *(_QWORD *)(v21 + 56),
      *(_QWORD *)(v21 + 64));
    v92 = 16LL * *(unsigned int *)(v21 + 8);
    v93 = *(_QWORD *)v21;
    v94 = v93 + v92;
    while ( v94 != v93 )
    {
      v95 = *(_QWORD *)(v93 + 8);
      v96 = *(_DWORD *)v93;
      v93 += 16;
      sub_B99FD0(v22, v96, v95);
    }
  }
  return sub_F162A0((__int64)a1, a2, v22);
}
