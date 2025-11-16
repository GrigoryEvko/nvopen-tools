// Function: sub_1AB45A0
// Address: 0x1ab45a0
//
_QWORD *__fastcall sub_1AB45A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  _QWORD *result; // rax
  _QWORD *v15; // r12
  __int64 v16; // r15
  _QWORD *v17; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // rax
  _QWORD *v25; // r15
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  char v29; // al
  char v30; // dl
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r15
  __int64 v36; // rdx
  char v37; // cl
  char v38; // cl
  unsigned __int64 v39; // r12
  char v40; // al
  __int64 v41; // r13
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rdi
  _QWORD *v45; // r14
  __int64 v46; // rax
  unsigned __int8 v47; // al
  unsigned int v48; // r15d
  unsigned __int64 v49; // r13
  int v50; // r12d
  _BYTE *v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // r15
  const char *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rsi
  __int64 v58; // r15
  __int64 v59; // rdx
  __int64 v60; // rcx
  _QWORD *v61; // r12
  double v62; // xmm4_8
  double v63; // xmm5_8
  __int64 v64; // rax
  char v65; // di
  __int64 *v66; // rax
  __int64 v67; // r13
  unsigned int v68; // r15d
  __int64 v69; // r11
  unsigned int v70; // edx
  __int64 v71; // rax
  __int64 v72; // rsi
  __int64 v73; // r9
  __int64 v74; // rsi
  __int64 v75; // rcx
  unsigned __int64 v76; // rcx
  __int64 v77; // rsi
  int v78; // edx
  __int64 v79; // rax
  unsigned __int64 v80; // rdx
  __int64 v81; // r15
  _QWORD *v82; // rax
  _QWORD *v83; // r13
  _QWORD *v84; // r12
  _QWORD *v85; // rax
  __int64 v86; // r15
  __int64 v87; // r12
  _QWORD *v88; // r15
  __int64 v89; // rax
  __int64 v90; // r12
  __int64 v91; // rdx
  __int64 v92; // rax
  __int64 v93; // r8
  __int64 v94; // rcx
  __int64 v95; // r15
  __int64 v96; // rdx
  __int64 v97; // r13
  _QWORD *v98; // rax
  __int64 v99; // r15
  _QWORD *v100; // rax
  _BYTE *v101; // rsi
  __int64 v102; // rsi
  unsigned __int64 v103; // r12
  unsigned __int64 v104; // r14
  __int64 v105; // rax
  __int64 v106; // rdx
  __int64 v107; // r12
  __int64 v108; // rax
  __int64 v109; // r12
  __int64 v110; // rsi
  __int64 v111; // rax
  unsigned __int64 v112; // rcx
  __int64 v113; // rdx
  unsigned __int64 v114; // rcx
  __int64 v115; // rdx
  __int64 v116; // rax
  unsigned __int64 v117; // rdx
  __int64 v118; // rax
  __int64 v119; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v120; // [rsp+8h] [rbp-A8h]
  __int64 v121; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v123; // [rsp+18h] [rbp-98h]
  char v124; // [rsp+26h] [rbp-8Ah]
  char v125; // [rsp+27h] [rbp-89h]
  bool v126; // [rsp+28h] [rbp-88h]
  __int64 v128; // [rsp+38h] [rbp-78h]
  __int64 v129; // [rsp+38h] [rbp-78h]
  __int64 v130; // [rsp+40h] [rbp-70h] BYREF
  __int64 v131; // [rsp+48h] [rbp-68h]
  __m128i v132; // [rsp+50h] [rbp-60h] BYREF
  __int64 v133; // [rsp+60h] [rbp-50h]
  __int64 v134; // [rsp+68h] [rbp-48h]
  __int64 v135; // [rsp+70h] [rbp-40h]

  result = sub_1AB4240(*(_QWORD *)(a1 + 16), a2);
  if ( result[2] )
    return result;
  v15 = result;
  LOWORD(v133) = 257;
  v16 = sub_157E9C0(a2);
  v17 = (_QWORD *)sub_22077B0(64);
  v18 = (__int64)v17;
  if ( v17 )
    sub_157FB60(v17, v16, (__int64)&v132, 0, 0);
  v19 = v15[2];
  if ( v19 != v18 )
  {
    if ( v19 != -8 && v19 != 0 && v19 != -16 )
      sub_1649B30(v15);
    v15[2] = v18;
    if ( v18 != 0 && v18 != -8 && v18 != -16 )
      sub_164C220((__int64)v15);
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    v90 = *(_QWORD *)(a1 + 32);
    v130 = (__int64)sub_1649960(a2);
    v131 = v91;
    LOWORD(v133) = 773;
    v132.m128i_i64[0] = (__int64)&v130;
    v132.m128i_i64[1] = v90;
    sub_164B780(v18, v132.m128i_i64);
  }
  if ( *(_WORD *)(a2 + 18) )
  {
    v86 = sub_159BBF0(*(_QWORD *)(a1 + 8), a2);
    v87 = sub_159BBF0(*(_QWORD *)a1, v18);
    v88 = sub_1AB4240(*(_QWORD *)(a1 + 16), v86);
    v89 = v88[2];
    if ( v87 != v89 )
    {
      if ( v89 != 0 && v89 != -8 && v89 != -16 )
        sub_1649B30(v88);
      v88[2] = v87;
      if ( v87 != 0 && v87 != -8 && v87 != -16 )
        sub_164C220((__int64)v88);
    }
  }
  v128 = a3;
  v125 = 0;
  v124 = 0;
  v123 = *(_QWORD *)(a2 + 40) & 0xFFFFFFFFFFFFFFF8LL;
  v126 = 0;
  if ( v123 != a3 )
  {
    do
    {
      v20 = v128 - 24;
      if ( !v128 )
        v20 = 0;
      v21 = sub_15F4880(v20);
      if ( *(_BYTE *)(v21 + 16) == 77 )
        goto LABEL_27;
      sub_1B75040(&v132, *(_QWORD *)(a1 + 16), *(_BYTE *)(a1 + 24) ^ 1u, 0, 0);
      sub_1B79630(&v132, v21);
      sub_1B75110(&v132);
      v22 = sub_157EB90(a2);
      v132 = (__m128i)(unsigned __int64)sub_1632FA0(v22);
      v133 = 0;
      v134 = 0;
      v135 = 0;
      v130 = sub_13E3350(v21, &v132, 0, 1, v23);
      if ( !v130 )
        goto LABEL_27;
      if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) )
      {
        sub_1A51850((unsigned __int64 *)&v132, *(_QWORD *)(a1 + 16), &v130);
        v24 = v133;
        if ( v133 )
        {
          if ( v133 != -16 && v133 != -8 )
          {
            v119 = v133;
            sub_1649B30(&v132);
            v24 = v119;
          }
          v130 = v24;
        }
      }
      if ( (unsigned __int8)sub_15F3040(v21) || sub_15F3330(v21) )
      {
LABEL_27:
        if ( (*(_BYTE *)(v20 + 23) & 0x20) != 0 )
        {
          v54 = *(_QWORD *)(a1 + 32);
          v55 = sub_1649960(v20);
          v132.m128i_i64[1] = v54;
          v130 = (__int64)v55;
          v131 = v56;
          LOWORD(v133) = 773;
          v132.m128i_i64[0] = (__int64)&v130;
          sub_164B780(v21, v132.m128i_i64);
        }
        v25 = sub_1AB4240(*(_QWORD *)(a1 + 16), v20);
        v26 = v25[2];
        if ( v21 != v26 )
        {
          if ( v26 != 0 && v26 != -8 && v26 != -16 )
            sub_1649B30(v25);
          v25[2] = v21;
          if ( v21 != -8 && v21 != -16 )
            sub_164C220((__int64)v25);
        }
        sub_157E9D0(v18 + 40, v21);
        v27 = *(_QWORD *)(v18 + 40);
        v28 = *(_QWORD *)(v21 + 24);
        *(_QWORD *)(v21 + 32) = v18 + 40;
        v27 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v21 + 24) = v27 | v28 & 7;
        *(_QWORD *)(v27 + 8) = v21 + 24;
        *(_QWORD *)(v18 + 40) = *(_QWORD *)(v18 + 40) & 7LL | (v21 + 24);
        v29 = *(_BYTE *)(v20 + 16);
        v30 = v29;
        if ( v29 != 78 )
        {
          if ( *(_QWORD *)(a1 + 40) )
          {
            v31 = v20 & 0xFFFFFFFFFFFFFFFBLL;
            if ( v29 == 29 )
            {
LABEL_39:
              v32 = v31 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v31 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                if ( *(char *)(v32 + 23) >= 0 )
                  goto LABEL_45;
                v120 = v31 & 0xFFFFFFFFFFFFFFF8LL;
                v33 = sub_1648A40(v32);
                v35 = v33 + v34;
                if ( *(char *)(v120 + 23) < 0 )
                  v35 -= sub_1648A40(v120);
                if ( (unsigned int)(v35 >> 4) )
                {
                  v92 = *(_QWORD *)(a1 + 40);
                  v132 = (__m128i)6uLL;
                  v133 = v21;
                  v93 = v92 + 8;
                  if ( v21 != -8 && v21 != -16 )
                  {
                    v121 = v92 + 8;
                    sub_164C220((__int64)&v132);
                    v93 = v121;
                  }
                  sub_1AB3EC0(v93, &v132);
                  if ( v133 != -8 && v133 != 0 && v133 != -16 )
                    sub_1649B30(&v132);
                  v30 = *(_BYTE *)(v20 + 16);
                }
                else
                {
                  v29 = *(_BYTE *)(v20 + 16);
LABEL_45:
                  v30 = v29;
                }
              }
            }
          }
          if ( v30 == 53 )
          {
            v36 = *(_QWORD *)(v20 - 24);
            v37 = v125;
            if ( *(_BYTE *)(v36 + 16) != 13 )
              v37 = 1;
            v125 = v37;
            v38 = v126;
            if ( *(_BYTE *)(v36 + 16) == 13 )
              v38 = 1;
            v126 = v38;
          }
          goto LABEL_52;
        }
        v94 = *(_QWORD *)(v20 - 24);
        if ( *(_BYTE *)(v94 + 16) || (*(_BYTE *)(v94 + 33) & 0x20) == 0 )
          v124 = 1;
        else
          v124 |= (unsigned int)(*(_DWORD *)(v94 + 36) - 35) > 3;
        v31 = v20 | 4;
        if ( *(_QWORD *)(a1 + 40) )
          goto LABEL_39;
      }
      else
      {
        v57 = v20;
        v58 = v130;
        v61 = sub_1AB4240(*(_QWORD *)(a1 + 16), v20);
        v64 = v61[2];
        if ( v58 != v64 )
        {
          LOBYTE(v60) = v64 != -8;
          if ( ((v64 != 0) & (unsigned __int8)v60) != 0 && v64 != -16 )
            sub_1649B30(v61);
          v61[2] = v58;
          LOBYTE(v59) = v58 != 0;
          if ( v58 != -8 && v58 != 0 && v58 != -16 )
            sub_164C220((__int64)v61);
        }
        sub_164BEC0(v21, v57, v59, v60, a5, a6, a7, a8, v62, v63, a11, a12);
      }
LABEL_52:
      v128 = *(_QWORD *)(v128 + 8);
    }
    while ( v123 != v128 );
  }
  v39 = sub_157EBA0(a2);
  v40 = *(_BYTE *)(v39 + 16);
  if ( v40 != 26 )
  {
    if ( v40 != 27 )
      goto LABEL_55;
    v65 = *(_BYTE *)(v39 + 23) & 0x40;
    if ( v65 )
      v66 = *(__int64 **)(v39 - 8);
    else
      v66 = (__int64 *)(v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF));
    v67 = *v66;
    if ( !*v66 )
      BUG();
    if ( *(_BYTE *)(v67 + 16) != 13 )
    {
      v110 = *(_QWORD *)(a1 + 16);
      v130 = *v66;
      sub_1A51850((unsigned __int64 *)&v132, v110, &v130);
      v67 = v133;
      if ( !v133 )
        goto LABEL_55;
      if ( v133 != -8 && v133 != -16 )
        sub_1649B30(&v132);
      if ( *(_BYTE *)(v67 + 16) != 13 )
      {
LABEL_55:
        v41 = sub_15F4880(v39);
        if ( (*(_BYTE *)(v39 + 23) & 0x20) != 0 )
        {
          v95 = *(_QWORD *)(a1 + 32);
          v130 = (__int64)sub_1649960(v39);
          LOWORD(v133) = 773;
          v131 = v96;
          v132.m128i_i64[0] = (__int64)&v130;
          v132.m128i_i64[1] = v95;
          sub_164B780(v41, v132.m128i_i64);
        }
        sub_157E9D0(v18 + 40, v41);
        v42 = *(_QWORD *)(v18 + 40);
        v43 = *(_QWORD *)(v41 + 24);
        *(_QWORD *)(v41 + 32) = v18 + 40;
        v42 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v41 + 24) = v42 | v43 & 7;
        *(_QWORD *)(v42 + 8) = v41 + 24;
        v44 = *(_QWORD *)(a1 + 16);
        *(_QWORD *)(v18 + 40) = *(_QWORD *)(v18 + 40) & 7LL | (v41 + 24);
        v45 = sub_1AB4240(v44, v39);
        v46 = v45[2];
        if ( v41 != v46 )
        {
          if ( v46 != -8 && v46 != 0 && v46 != -16 )
            sub_1649B30(v45);
          v45[2] = v41;
          if ( v41 != -8 && v41 != -16 )
            sub_164C220((__int64)v45);
        }
        if ( *(_QWORD *)(a1 + 40) )
        {
          v47 = *(_BYTE *)(v39 + 16);
          if ( v47 > 0x17u )
          {
            if ( v47 == 78 )
            {
              v103 = v39 | 4;
            }
            else
            {
              if ( v47 != 29 )
                goto LABEL_68;
              v103 = v39 & 0xFFFFFFFFFFFFFFFBLL;
            }
            v104 = v103 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v103 & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(char *)(v104 + 23) < 0 )
            {
              v105 = sub_1648A40(v103 & 0xFFFFFFFFFFFFFFF8LL);
              v107 = v105 + v106;
              if ( *(char *)(v104 + 23) < 0 )
                v107 -= sub_1648A40(v104);
              if ( (unsigned int)(v107 >> 4) )
              {
                v108 = *(_QWORD *)(a1 + 40);
                v132 = (__m128i)6uLL;
                v133 = v41;
                v109 = v108 + 8;
                if ( v41 != -8 && v41 != -16 )
                  sub_164C220((__int64)&v132);
                sub_1AB3EC0(v109, &v132);
                if ( v133 != 0 && v133 != -8 && v133 != -16 )
                  sub_1649B30(&v132);
              }
            }
          }
        }
LABEL_68:
        v48 = 0;
        v49 = sub_157EBA0(a2);
        v50 = sub_15F4D60(v49);
        if ( v50 )
        {
          do
          {
            while ( 1 )
            {
              v52 = sub_15F4DF0(v49, v48);
              v51 = *(_BYTE **)(a4 + 8);
              v132.m128i_i64[0] = v52;
              if ( v51 != *(_BYTE **)(a4 + 16) )
                break;
              ++v48;
              sub_136D8A0(a4, v51, &v132);
              if ( v50 == v48 )
                goto LABEL_75;
            }
            if ( v51 )
            {
              *(_QWORD *)v51 = v52;
              v51 = *(_BYTE **)(a4 + 8);
            }
            ++v48;
            *(_QWORD *)(a4 + 8) = v51 + 8;
          }
          while ( v50 != v48 );
        }
        goto LABEL_75;
      }
      v65 = *(_BYTE *)(v39 + 23) & 0x40;
    }
    v68 = *(_DWORD *)(v39 + 20) & 0xFFFFFFF;
    v129 = (v68 >> 1) - 1;
    v69 = v129 >> 2;
    if ( v129 >> 2 )
    {
      v69 *= 4;
      v70 = 8;
      v71 = 0;
      while ( 1 )
      {
        v73 = v71 + 1;
        v76 = v39 - 24LL * v68;
        if ( v65 )
          v76 = *(_QWORD *)(v39 - 8);
        v77 = *(_QWORD *)(v76 + 24LL * (v70 - 6));
        if ( v77 )
        {
          if ( v67 == v77 )
            break;
        }
        v72 = *(_QWORD *)(v76 + 24LL * (v70 - 4));
        if ( v72 && v67 == v72 )
        {
LABEL_180:
          v71 = v73;
          break;
        }
        v73 = v71 + 3;
        v74 = *(_QWORD *)(v76 + 24LL * (v70 - 2));
        if ( v74 && v67 == v74 )
        {
          v71 += 2;
          break;
        }
        v71 += 4;
        v75 = *(_QWORD *)(v76 + 24LL * v70);
        if ( v75 && v67 == v75 )
          goto LABEL_180;
        v70 += 8;
        if ( v69 == v71 )
          goto LABEL_193;
      }
LABEL_107:
      if ( v129 != v71 )
      {
        v78 = v71;
        if ( (_DWORD)v71 != -2 )
          goto LABEL_109;
      }
      goto LABEL_192;
    }
LABEL_193:
    v116 = v129 - v69;
    if ( v129 - v69 == 2 )
    {
      v111 = v69;
    }
    else
    {
      if ( v116 != 3 )
      {
        if ( v116 != 1 )
          goto LABEL_192;
        goto LABEL_196;
      }
      v111 = v69 + 1;
      if ( v65 )
        v112 = *(_QWORD *)(v39 - 8);
      else
        v112 = v39 - 24LL * v68;
      v113 = *(_QWORD *)(v112 + 24LL * (unsigned int)(2 * (v69 + 1)));
      if ( v113 && v67 == v113 )
      {
LABEL_191:
        v78 = v69;
        if ( v129 != v69 )
        {
LABEL_109:
          v79 = 24LL * (unsigned int)(2 * v78 + 3);
          goto LABEL_110;
        }
LABEL_192:
        v79 = 24;
LABEL_110:
        if ( v65 )
          v80 = *(_QWORD *)(v39 - 8);
        else
          v80 = v39 - 24LL * v68;
        v81 = *(_QWORD *)(v80 + v79);
        v132.m128i_i64[0] = v81;
        v82 = sub_1648A60(56, 1u);
        v83 = v82;
        if ( v82 )
          sub_15F8590((__int64)v82, v81, v18);
        v84 = sub_1AB4240(*(_QWORD *)(a1 + 16), v39);
        v85 = (_QWORD *)v84[2];
        if ( v85 == v83 )
          goto LABEL_149;
        if ( v85 + 1 == 0 || v85 == 0 )
          goto LABEL_146;
        goto LABEL_144;
      }
    }
    v69 = v111 + 1;
    if ( v65 )
      v114 = *(_QWORD *)(v39 - 8);
    else
      v114 = v39 - 24LL * v68;
    v115 = *(_QWORD *)(v114 + 24LL * (unsigned int)(2 * (v111 + 1)));
    if ( !v115 || v67 != v115 )
    {
LABEL_196:
      if ( v65 )
        v117 = *(_QWORD *)(v39 - 8);
      else
        v117 = v39 - 24LL * v68;
      v118 = *(_QWORD *)(v117 + 24LL * (unsigned int)(2 * v69 + 2));
      if ( !v118 || v67 != v118 )
        goto LABEL_192;
      v71 = v69;
      goto LABEL_107;
    }
    v69 = v111;
    goto LABEL_191;
  }
  if ( (*(_DWORD *)(v39 + 20) & 0xFFFFFFF) != 3 )
    goto LABEL_55;
  v97 = *(_QWORD *)(v39 - 72);
  if ( *(_BYTE *)(v97 + 16) != 13 )
  {
    v102 = *(_QWORD *)(a1 + 16);
    v130 = *(_QWORD *)(v39 - 72);
    sub_1A51850((unsigned __int64 *)&v132, v102, &v130);
    v97 = v133;
    if ( !v133 )
      goto LABEL_55;
    if ( v133 != -16 && v133 != -8 )
      sub_1649B30(&v132);
    if ( *(_BYTE *)(v97 + 16) != 13 )
      goto LABEL_55;
  }
  v98 = *(_QWORD **)(v97 + 24);
  if ( *(_DWORD *)(v97 + 32) > 0x40u )
    v98 = (_QWORD *)*v98;
  v99 = *(_QWORD *)(v39 - 24LL * (v98 == 0) - 24);
  v132.m128i_i64[0] = v99;
  v100 = sub_1648A60(56, 1u);
  v83 = v100;
  if ( v100 )
    sub_15F8590((__int64)v100, v99, v18);
  v84 = sub_1AB4240(*(_QWORD *)(a1 + 16), v39);
  v85 = (_QWORD *)v84[2];
  if ( v85 != v83 )
  {
    if ( v85 == 0 || v85 + 1 == 0 )
    {
LABEL_146:
      v84[2] = v83;
      if ( v83 + 1 != 0 && v83 != 0 && v83 != (_QWORD *)-16LL )
        sub_164C220((__int64)v84);
      goto LABEL_149;
    }
LABEL_144:
    if ( v85 != (_QWORD *)-16LL )
      sub_1649B30(v84);
    goto LABEL_146;
  }
LABEL_149:
  v101 = *(_BYTE **)(a4 + 8);
  if ( v101 == *(_BYTE **)(a4 + 16) )
  {
    sub_136D8A0(a4, v101, &v132);
  }
  else
  {
    if ( v101 )
    {
      *(_QWORD *)v101 = v132.m128i_i64[0];
      v101 = *(_BYTE **)(a4 + 8);
    }
    *(_QWORD *)(a4 + 8) = v101 + 8;
  }
LABEL_75:
  result = *(_QWORD **)(a1 + 40);
  if ( result )
  {
    *(_BYTE *)result |= v124;
    *(_BYTE *)(*(_QWORD *)(a1 + 40) + 1LL) |= v125;
    if ( v126 )
    {
      v53 = *(_QWORD *)(*(_QWORD *)(a2 + 56) + 80LL);
      if ( v53 )
        v126 = a2 != v53 - 24;
    }
    result = *(_QWORD **)(a1 + 40);
    *((_BYTE *)result + 1) |= v126;
  }
  return result;
}
