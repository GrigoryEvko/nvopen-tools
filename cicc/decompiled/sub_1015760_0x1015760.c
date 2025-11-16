// Function: sub_1015760
// Address: 0x1015760
//
__int64 __fastcall sub_1015760(unsigned __int64 a1, unsigned __int8 *a2, unsigned __int8 *a3, __m128i *a4, int a5)
{
  unsigned __int8 *v5; // r15
  __int64 v6; // r12
  unsigned __int8 *v7; // rbx
  unsigned __int8 v8; // al
  unsigned __int8 *v9; // r13
  bool v10; // al
  unsigned __int8 v11; // r14
  char v12; // al
  bool v13; // cl
  unsigned __int8 *v14; // r11
  char v15; // dl
  char v16; // r14
  __int64 v17; // rsi
  unsigned __int8 *v18; // r10
  char v19; // r9
  __int64 result; // rax
  unsigned int v21; // eax
  unsigned __int64 v22; // rax
  __int64 v23; // rcx
  int v24; // edx
  unsigned __int8 *v25; // r14
  unsigned __int8 *v26; // rcx
  __int64 v27; // rsi
  int v28; // edx
  unsigned __int8 *v29; // rax
  _BYTE *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rsi
  __int64 *v33; // r14
  __int64 *v34; // r15
  bool v35; // r14
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdi
  bool v39; // zf
  unsigned int v40; // r14d
  bool v41; // al
  bool v42; // al
  __int64 v43; // rax
  __int64 v44; // rdx
  unsigned int v45; // r14d
  bool v46; // al
  __int64 v47; // r14
  _BYTE *v48; // rax
  unsigned __int8 *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // r14
  _BYTE *v52; // rax
  unsigned __int8 *v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rax
  bool v56; // al
  bool v57; // al
  bool v58; // al
  bool v59; // al
  int v60; // eax
  int v61; // edx
  bool v62; // bl
  bool v63; // r14
  bool v64; // bl
  int v65; // eax
  bool v66; // bl
  bool v67; // al
  bool v68; // r14
  bool v69; // bl
  unsigned __int8 *v70; // rdx
  _BYTE *v71; // rax
  unsigned int v72; // r14d
  __int64 *v73; // rsi
  __int64 v74; // rax
  bool v75; // al
  __int64 v76; // rax
  __int64 v77; // r14
  _BYTE *v78; // rax
  unsigned int v79; // r14d
  unsigned int v80; // ebx
  unsigned __int8 *v81; // r14
  __int64 v82; // rax
  __int64 v83; // rax
  unsigned int v84; // r15d
  unsigned __int8 *v85; // r14
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 ***v88; // rdx
  __int64 v89; // rdi
  __int64 v90; // r8
  char v91; // al
  int v92; // eax
  int v93; // eax
  unsigned int v94; // edx
  __int64 v95; // rsi
  __int64 v96; // rax
  bool v97; // al
  unsigned int v98; // edx
  bool v99; // r14
  __int64 v100; // rax
  unsigned int v101; // edx
  unsigned int v102; // r14d
  int v103; // eax
  _BYTE *v104; // rax
  _BYTE *v105; // r8
  unsigned int v106; // eax
  unsigned int v107; // edi
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // rdi
  int v111; // eax
  unsigned __int8 *v112; // [rsp+8h] [rbp-D8h]
  unsigned __int8 *v113; // [rsp+8h] [rbp-D8h]
  unsigned __int8 *v114; // [rsp+10h] [rbp-D0h]
  unsigned __int8 *v115; // [rsp+10h] [rbp-D0h]
  unsigned __int8 *v116; // [rsp+10h] [rbp-D0h]
  unsigned __int8 *v117; // [rsp+10h] [rbp-D0h]
  unsigned __int8 *v118; // [rsp+10h] [rbp-D0h]
  __int64 ***v119; // [rsp+10h] [rbp-D0h]
  char v120; // [rsp+18h] [rbp-C8h]
  bool v121; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v122; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v123; // [rsp+18h] [rbp-C8h]
  __int64 v124; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v125; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v126; // [rsp+20h] [rbp-C0h]
  __int64 v127; // [rsp+20h] [rbp-C0h]
  unsigned __int8 *v128; // [rsp+20h] [rbp-C0h]
  int v129; // [rsp+20h] [rbp-C0h]
  bool v130; // [rsp+28h] [rbp-B8h]
  bool v131; // [rsp+28h] [rbp-B8h]
  char v132; // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v133; // [rsp+28h] [rbp-B8h]
  char v134; // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v135; // [rsp+28h] [rbp-B8h]
  char v136; // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v137; // [rsp+28h] [rbp-B8h]
  int v138; // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v139; // [rsp+30h] [rbp-B0h]
  unsigned __int8 *v140; // [rsp+30h] [rbp-B0h]
  unsigned __int8 *v141; // [rsp+30h] [rbp-B0h]
  char v142; // [rsp+30h] [rbp-B0h]
  unsigned __int8 *v143; // [rsp+38h] [rbp-A8h]
  __int64 v144; // [rsp+38h] [rbp-A8h]
  __int64 v145; // [rsp+38h] [rbp-A8h]
  bool v146; // [rsp+38h] [rbp-A8h]
  unsigned __int8 *v147; // [rsp+38h] [rbp-A8h]
  int v148; // [rsp+38h] [rbp-A8h]
  int v149; // [rsp+38h] [rbp-A8h]
  unsigned int v150; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v151; // [rsp+40h] [rbp-A0h]
  unsigned __int8 *v154; // [rsp+58h] [rbp-88h]
  bool v155; // [rsp+58h] [rbp-88h]
  bool v156; // [rsp+58h] [rbp-88h]
  __int64 v157; // [rsp+68h] [rbp-78h] BYREF
  __int64 *v158; // [rsp+70h] [rbp-70h] BYREF
  __int64 *v159; // [rsp+78h] [rbp-68h] BYREF
  __int64 **v160; // [rsp+80h] [rbp-60h] BYREF
  char v161; // [rsp+88h] [rbp-58h]
  __int64 ***v162; // [rsp+90h] [rbp-50h] BYREF
  __int64 ***v163; // [rsp+98h] [rbp-48h] BYREF
  char v164; // [rsp+A0h] [rbp-40h]

  v5 = a3;
  v6 = (unsigned int)a1;
  v7 = a2;
  v8 = *a2;
  v151 = a1;
  if ( (unsigned __int8)(*a2 - 42) <= 0x11u )
  {
    if ( (unsigned __int8)(*a3 - 42) > 0x11u )
    {
      v9 = 0;
      if ( !a5 )
      {
        v154 = a2;
        v9 = 0;
LABEL_21:
        result = sub_1005070(v6, v154, (__int64)v5, a4);
        goto LABEL_22;
      }
      if ( v8 != 42 )
        goto LABEL_29;
    }
    else
    {
      if ( !a5 )
      {
        v154 = a2;
        v9 = a3;
        result = sub_1005070(a1, a2, (__int64)a3, a4);
LABEL_22:
        if ( result )
          return result;
LABEL_23:
        if ( !v9 )
          goto LABEL_4;
        goto LABEL_24;
      }
      v9 = a3;
      if ( v8 != 42 )
      {
LABEL_29:
        v139 = 0;
        v13 = 0;
        v14 = 0;
        goto LABEL_30;
      }
    }
    v14 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
    v29 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    v17 = (unsigned int)(a1 - 32);
    v16 = 1;
    v139 = v29;
    v13 = a3 == v14;
    v15 = a3 == v14 || a3 == v29;
    if ( (unsigned int)v17 > 1 )
    {
      v133 = v14;
      v146 = v5 == v14;
      v155 = v5 == v14 || v5 == v29;
      v58 = sub_B532A0(a1);
      v15 = v155;
      v13 = v146;
      v14 = v133;
      v17 = (unsigned int)(a1 - 32);
      if ( !v58 || !a4[4].m128i_i8[0] || (v16 = (v7[1] & 2) != 0, (v7[1] & 2) == 0) )
      {
        v147 = v133;
        v156 = v13;
        v134 = v15;
        v59 = sub_B532B0(a1);
        v13 = v156;
        v14 = v147;
        if ( !v59 || !a4[4].m128i_i8[0] )
        {
LABEL_30:
          v154 = v7;
          if ( !v9 )
            goto LABEL_31;
          goto LABEL_14;
        }
        v17 = (unsigned int)(a1 - 32);
        v15 = (v7[1] >> 2) & v134;
        v16 = (v7[1] & 4) != 0;
      }
    }
    if ( v9 && *v9 == 42 )
    {
      v154 = v7;
      goto LABEL_16;
    }
    v154 = v7;
    if ( v15 )
    {
      v143 = 0;
      v19 = 0;
      v18 = 0;
      goto LABEL_66;
    }
LABEL_31:
    v130 = 0;
    v18 = 0;
    v143 = 0;
    goto LABEL_32;
  }
  if ( (unsigned __int8)(*a3 - 42) > 0x11u )
  {
    v154 = 0;
    v9 = 0;
    goto LABEL_4;
  }
  v154 = 0;
  v9 = a3;
  if ( a5 )
  {
    v139 = 0;
    v13 = 0;
    v14 = 0;
LABEL_14:
    if ( *v9 == 42 )
    {
      v15 = 0;
      v16 = 0;
      v17 = (unsigned int)(a1 - 32);
LABEL_16:
      v18 = (unsigned __int8 *)*((_QWORD *)v9 - 8);
      v143 = (unsigned __int8 *)*((_QWORD *)v9 - 4);
      if ( (unsigned int)v17 <= 1 )
        goto LABEL_17;
      v115 = v14;
      v121 = v13;
      v126 = (unsigned __int8 *)*((_QWORD *)v9 - 8);
      v132 = v15;
      v56 = sub_B532A0(a1);
      v15 = v132;
      v18 = v126;
      v13 = v121;
      v14 = v115;
      if ( !v56 )
        goto LABEL_149;
      if ( a4[4].m128i_i8[0] && (v9[1] & 2) != 0 )
      {
LABEL_17:
        v19 = 1;
      }
      else
      {
LABEL_149:
        v57 = sub_B532B0(a1);
        v15 = v132;
        v18 = v126;
        v13 = v121;
        v14 = v115;
        v19 = v57;
        if ( v57 )
        {
          v19 = a4[4].m128i_i8[0];
          if ( v19 )
            v19 = (v9[1] & 4) != 0;
        }
      }
      if ( !v15 )
      {
LABEL_69:
        if ( v7 == v18 || v7 == v143 )
        {
          if ( !v19 )
          {
            v130 = v14 != 0 && v18 != 0;
            goto LABEL_32;
          }
          v70 = v143;
          v116 = v14;
          v122 = v18;
          if ( v7 != v18 )
            v70 = v18;
          v135 = v70;
          v71 = (_BYTE *)sub_AD6530(*((_QWORD *)v7 + 1), v17);
          result = sub_1012FB0(a1, v71, v135, a4->m128i_i64, a5 - 1);
          if ( result )
            return result;
          v14 = v116;
          v18 = v122;
          v130 = v122 != 0 && v116 != 0;
        }
        else
        {
          v16 &= v19;
          v130 = v18 != 0 && v14 != 0;
        }
        if ( v16 )
          goto LABEL_47;
LABEL_32:
        v16 = 0;
        if ( (_DWORD)a1 != 40 )
          goto LABEL_47;
        v16 = a4[4].m128i_i8[0];
        if ( !v16 )
          goto LABEL_47;
        v22 = *v5;
        if ( (unsigned __int8)v22 <= 0x1Cu )
        {
          if ( (_BYTE)v22 != 5 )
            goto LABEL_218;
          v24 = *((unsigned __int16 *)v5 + 1);
          if ( (*((_WORD *)v5 + 1) & 0xFFFD) != 0xD && (v24 & 0xFFF7) != 0x11 )
            goto LABEL_218;
        }
        else
        {
          if ( (unsigned __int8)v22 > 0x36u )
            goto LABEL_218;
          v23 = 0x40540000000000LL;
          v24 = (unsigned __int8)v22 - 29;
          if ( !_bittest64(&v23, v22) )
            goto LABEL_218;
        }
        if ( v24 == 13 && (v5[1] & 4) != 0 )
        {
          v25 = v5;
          v26 = v7;
          goto LABEL_40;
        }
LABEL_218:
        v22 = *v7;
        v25 = v7;
        v26 = v5;
LABEL_40:
        if ( (unsigned __int8)v22 <= 0x1Cu )
        {
          if ( (_BYTE)v22 != 5 )
            goto LABEL_46;
          v28 = *((unsigned __int16 *)v25 + 1);
          if ( (*((_WORD *)v25 + 1) & 0xFFF7) != 0x11 && (v28 & 0xFFFD) != 0xD )
            goto LABEL_46;
        }
        else
        {
          if ( (unsigned __int8)v22 > 0x36u )
            goto LABEL_46;
          v27 = 0x40540000000000LL;
          v28 = (unsigned __int8)v22 - 29;
          if ( !_bittest64(&v27, v22) )
            goto LABEL_46;
        }
        if ( v28 == 13 && (v25[1] & 4) != 0 && *v26 == 42 )
        {
          v88 = (__int64 ***)*((_QWORD *)v26 - 8);
          if ( v88 )
          {
            v89 = *((_QWORD *)v26 - 4);
            v90 = v89 + 24;
            if ( *(_BYTE *)v89 != 17 )
            {
              v119 = (__int64 ***)*((_QWORD *)v26 - 8);
              if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v89 + 8) + 8LL) - 17 > 1 )
                goto LABEL_46;
              if ( *(_BYTE *)v89 > 0x15u )
                goto LABEL_46;
              v125 = v14;
              v128 = v18;
              v104 = sub_AD7630(v89, 0, (__int64)v88);
              v18 = v128;
              v14 = v125;
              v105 = v104;
              if ( !v104 || *v104 != 17 )
                goto LABEL_46;
              LOBYTE(v22) = *v25;
              v88 = v119;
              v90 = (__int64)(v105 + 24);
            }
            v162 = v88;
            v163 = &v160;
            v164 = 0;
            if ( (_BYTE)v22 == 42 && v88 == *((__int64 ****)v25 - 8) )
            {
              v117 = v14;
              v123 = v18;
              v127 = v90;
              v91 = sub_991580((__int64)&v163, *((_QWORD *)v25 - 4));
              v18 = v123;
              v14 = v117;
              v16 = v91;
              if ( v91 )
              {
                v113 = v117;
                v118 = v123;
                v124 = (__int64)v160;
                v92 = sub_C4C880(v127, (__int64)v160);
                v18 = v118;
                v14 = v113;
                if ( v92 < 0 )
                {
                  v107 = *(_DWORD *)(v127 + 8);
                  v108 = *(_QWORD *)v127;
                  if ( v107 > 0x40 )
                    v108 = *(_QWORD *)(v108 + 8LL * ((v107 - 1) >> 6));
                  if ( (v108 & (1LL << ((unsigned __int8)v107 - 1))) == 0 )
                    goto LABEL_47;
                }
                v93 = sub_C4C880(v124, v127);
                v18 = v118;
                v14 = v113;
                if ( v93 < 0 )
                {
                  v94 = *(_DWORD *)(v127 + 8);
                  v95 = *(_QWORD *)v127;
                  v96 = 1LL << ((unsigned __int8)v94 - 1);
                  if ( v94 > 0x40 )
                  {
                    if ( (*(_QWORD *)(v95 + 8LL * ((v94 - 1) >> 6)) & v96) != 0 )
                      goto LABEL_47;
                    v110 = v127;
                    v129 = *(_DWORD *)(v127 + 8);
                    v111 = sub_C444A0(v110);
                    v18 = v118;
                    v14 = v113;
                    v97 = v129 == v111;
                  }
                  else
                  {
                    if ( (v96 & v95) != 0 )
                      goto LABEL_47;
                    v97 = v95 == 0;
                  }
                  if ( !v97 )
                    v16 = 0;
LABEL_47:
                  if ( !v130 )
                    goto LABEL_57;
                  if ( v14 == v18 || v14 == v143 )
                  {
                    if ( !v16 )
                      goto LABEL_57;
                    if ( v14 == v18 )
                      v18 = v143;
                  }
                  else
                  {
                    if ( v139 != v18 && v139 != v143 || !v16 )
                      goto LABEL_57;
                    if ( v139 == v18 )
                    {
                      result = sub_1012FB0(v151, v14, v143, a4->m128i_i64, a5 - 1);
LABEL_56:
                      if ( result )
                        return result;
LABEL_57:
                      if ( !v154 )
                        goto LABEL_23;
                      goto LABEL_21;
                    }
                    v139 = v14;
                  }
                  result = sub_1012FB0(v151, v139, v18, a4->m128i_i64, a5 - 1);
                  goto LABEL_56;
                }
              }
            }
          }
        }
LABEL_46:
        v16 = 0;
        goto LABEL_47;
      }
LABEL_66:
      v114 = v18;
      v120 = v19;
      v112 = v14;
      v131 = v13;
      v30 = (_BYTE *)sub_AD6530(*((_QWORD *)v5 + 1), v17);
      v17 = (__int64)v139;
      if ( !v131 )
        v17 = (__int64)v112;
      result = sub_1012FB0(a1, (_BYTE *)v17, v30, a4->m128i_i64, a5 - 1);
      v14 = v112;
      v19 = v120;
      v18 = v114;
      if ( result )
        return result;
      goto LABEL_69;
    }
    goto LABEL_31;
  }
LABEL_24:
  v21 = sub_B52F50(v6);
  result = sub_1005070(v21, v9, (__int64)v7, a4);
  if ( result )
    return result;
LABEL_4:
  v10 = sub_B532A0(v6);
  v11 = *v7;
  if ( !v10 && v11 == 44 )
  {
    v44 = *((_QWORD *)v7 - 8);
    if ( *(_BYTE *)v44 == 17 )
    {
      v45 = *(_DWORD *)(v44 + 32);
      if ( v45 <= 0x40 )
        v46 = *(_QWORD *)(v44 + 24) == 0;
      else
        v46 = v45 == (unsigned int)sub_C444A0(v44 + 24);
    }
    else
    {
      v77 = *(_QWORD *)(v44 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v77 + 8) - 17 > 1 || *(_BYTE *)v44 > 0x15u )
        goto LABEL_6;
      v140 = (unsigned __int8 *)*((_QWORD *)v7 - 8);
      v78 = sub_AD7630(v44, 0, v44);
      if ( !v78 || *v78 != 17 )
      {
        if ( *(_BYTE *)(v77 + 8) == 17 )
        {
          v138 = *(_DWORD *)(v77 + 32);
          if ( v138 )
          {
            v98 = 0;
            v99 = 0;
            while ( 1 )
            {
              v150 = v98;
              v100 = sub_AD69F0(v140, v98);
              v101 = v150;
              if ( !v100 )
                break;
              if ( *(_BYTE *)v100 != 13 )
              {
                if ( *(_BYTE *)v100 != 17 )
                  break;
                v102 = *(_DWORD *)(v100 + 32);
                if ( v102 <= 0x40 )
                {
                  v99 = *(_QWORD *)(v100 + 24) == 0;
                }
                else
                {
                  v103 = sub_C444A0(v100 + 24);
                  v101 = v150;
                  v99 = v102 == v103;
                }
                if ( !v99 )
                  break;
              }
              v98 = v101 + 1;
              if ( v138 == v98 )
              {
                if ( v99 )
                  goto LABEL_119;
                goto LABEL_120;
              }
            }
          }
        }
        goto LABEL_120;
      }
      v79 = *((_DWORD *)v78 + 8);
      if ( v79 <= 0x40 )
        v46 = *((_QWORD *)v78 + 3) == 0;
      else
        v46 = v79 == (unsigned int)sub_C444A0((__int64)(v78 + 24));
    }
    if ( v46 )
    {
LABEL_119:
      if ( **((_BYTE **)v7 - 4) == 68 )
      {
        LOBYTE(v163) = 0;
        v162 = &v160;
        if ( (unsigned __int8)sub_991580((__int64)&v162, (__int64)v5) )
        {
          v72 = *((_DWORD *)v160 + 2);
          v73 = *v160;
          v74 = 1LL << ((unsigned __int8)v72 - 1);
          if ( v72 > 0x40 )
          {
            if ( (v73[(v72 - 1) >> 6] & v74) == 0 )
            {
              v75 = v72 == (unsigned int)sub_C444A0((__int64)v160);
              goto LABEL_201;
            }
          }
          else if ( (v74 & (unsigned __int64)v73) == 0 )
          {
            v75 = v73 == 0;
LABEL_201:
            if ( !v75 )
            {
              if ( (_DWORD)v6 == 40 || (_DWORD)v6 == 33 )
                goto LABEL_113;
              if ( (_DWORD)v6 == 39 || (_DWORD)v6 == 32 )
                goto LABEL_208;
            }
            if ( (_DWORD)v6 == 41 )
            {
LABEL_113:
              v43 = sub_1001990(*((_QWORD ***)v5 + 1));
              return sub_AD6400(v43);
            }
            if ( (_DWORD)v6 == 38 )
              goto LABEL_208;
          }
        }
      }
    }
LABEL_120:
    v11 = *v7;
  }
  if ( v11 != 54 )
    goto LABEL_6;
  v37 = *((_QWORD *)v7 - 8);
  if ( *(_BYTE *)v37 == 17 )
  {
    if ( *(_DWORD *)(v37 + 32) > 0x40u )
    {
      if ( (unsigned int)sub_C44630(v37 + 24) != 1 )
      {
LABEL_104:
        if ( v11 != 54 )
          goto LABEL_6;
        v37 = *((_QWORD *)v7 - 8);
        goto LABEL_106;
      }
      goto LABEL_92;
    }
    v55 = *(_QWORD *)(v37 + 24);
    if ( v55 && (v55 & (v55 - 1)) == 0 )
      goto LABEL_92;
  }
  else
  {
    v47 = *(_QWORD *)(v37 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v47 + 8) - 17 <= 1 && *(_BYTE *)v37 <= 0x15u )
    {
      v144 = *((_QWORD *)v7 - 8);
      v48 = sub_AD7630(v144, 0, v37);
      v49 = (unsigned __int8 *)v144;
      if ( !v48 || *v48 != 17 )
      {
        if ( *(_BYTE *)(v47 + 8) == 17 )
        {
          v148 = *(_DWORD *)(v47 + 32);
          if ( v148 )
          {
            v136 = 0;
            v141 = v7;
            v80 = 0;
            v81 = v49;
            do
            {
              v82 = sub_AD69F0(v81, v80);
              if ( !v82 )
              {
LABEL_254:
                v7 = v141;
                goto LABEL_103;
              }
              if ( *(_BYTE *)v82 != 13 )
              {
                if ( *(_BYTE *)v82 != 17 )
                  goto LABEL_254;
                if ( *(_DWORD *)(v82 + 32) > 0x40u )
                {
                  if ( (unsigned int)sub_C44630(v82 + 24) != 1 )
                    goto LABEL_254;
                }
                else
                {
                  v83 = *(_QWORD *)(v82 + 24);
                  if ( !v83 || (v83 & (v83 - 1)) != 0 )
                    goto LABEL_254;
                }
                v136 = 1;
              }
              ++v80;
            }
            while ( v148 != v80 );
            v7 = v141;
            if ( v136 )
              goto LABEL_92;
          }
        }
        goto LABEL_103;
      }
      if ( *((_DWORD *)v48 + 8) > 0x40u )
      {
        if ( (unsigned int)sub_C44630((__int64)(v48 + 24)) != 1 )
          goto LABEL_103;
      }
      else
      {
        v50 = *((_QWORD *)v48 + 3);
        if ( !v50 || (v50 & (v50 - 1)) != 0 )
          goto LABEL_103;
      }
LABEL_92:
      LOBYTE(v163) = 1;
      v162 = (__int64 ***)&v157;
      if ( (unsigned __int8)sub_991580((__int64)&v162, (__int64)v5) )
      {
        v38 = v157;
        if ( *(_DWORD *)(v157 + 8) > 0x40u )
        {
          v38 = v157;
          if ( (unsigned int)sub_C44630(v157) != 1 )
            goto LABEL_96;
        }
        else if ( !*(_QWORD *)v157 || (*(_QWORD *)v157 & (*(_QWORD *)v157 - 1LL)) != 0 )
        {
LABEL_96:
          if ( a4[4].m128i_i8[0] && (((v154[1] >> 1) & 2) != 0 || (v154[1] & 2) != 0) )
            goto LABEL_188;
          v39 = *v7 == 54;
          v162 = 0;
          if ( v39 )
          {
            if ( (unsigned __int8)sub_993A50(&v162, *((_QWORD *)v7 - 8)) )
              goto LABEL_188;
            v38 = v157;
          }
          v40 = *(_DWORD *)(v38 + 8);
          if ( v40 <= 0x40 )
            v41 = *(_QWORD *)v38 == 0;
          else
            v41 = v40 == (unsigned int)sub_C444A0(v38);
          if ( v41 )
            goto LABEL_103;
LABEL_188:
          if ( (_DWORD)v6 == 32 )
            goto LABEL_208;
          if ( (_DWORD)v6 == 33 )
            goto LABEL_113;
        }
      }
LABEL_103:
      v11 = *v7;
      goto LABEL_104;
    }
  }
LABEL_106:
  if ( *(_BYTE *)v37 == 17 )
  {
    if ( *(_DWORD *)(v37 + 32) > 0x40u )
    {
      v42 = (unsigned int)sub_C44630(v37 + 24) == 1;
      goto LABEL_109;
    }
    v54 = *(_QWORD *)(v37 + 24);
    if ( !v54 )
      goto LABEL_6;
    goto LABEL_140;
  }
  v51 = *(_QWORD *)(v37 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v51 + 8) - 17 <= 1 && *(_BYTE *)v37 <= 0x15u )
  {
    v145 = v37;
    v52 = sub_AD7630(v37, 0, v37);
    v53 = (unsigned __int8 *)v145;
    if ( v52 && *v52 == 17 )
    {
      if ( *((_DWORD *)v52 + 8) > 0x40u )
      {
        v42 = (unsigned int)sub_C44630((__int64)(v52 + 24)) == 1;
LABEL_109:
        if ( !v42 )
          goto LABEL_6;
LABEL_110:
        v162 = 0;
        if ( !(unsigned __int8)sub_993BE0(&v162, (__int64)v5) )
          goto LABEL_6;
        if ( (_DWORD)v6 != 34 )
        {
          if ( (_DWORD)v6 != 37 )
            goto LABEL_6;
          goto LABEL_113;
        }
LABEL_208:
        v76 = sub_1001990(*((_QWORD ***)v5 + 1));
        return sub_AD6450(v76);
      }
      v54 = *((_QWORD *)v52 + 3);
      if ( !v54 )
        goto LABEL_6;
LABEL_140:
      if ( (v54 & (v54 - 1)) != 0 )
        goto LABEL_6;
      goto LABEL_110;
    }
    if ( *(_BYTE *)(v51 + 8) == 17 )
    {
      v149 = *(_DWORD *)(v51 + 32);
      if ( v149 )
      {
        v142 = 0;
        v137 = v5;
        v84 = 0;
        v85 = v53;
        while ( 1 )
        {
          v86 = sub_AD69F0(v85, v84);
          if ( !v86 )
            break;
          if ( *(_BYTE *)v86 != 13 )
          {
            if ( *(_BYTE *)v86 != 17 )
              break;
            if ( *(_DWORD *)(v86 + 32) > 0x40u )
            {
              if ( (unsigned int)sub_C44630(v86 + 24) != 1 )
                break;
            }
            else
            {
              v87 = *(_QWORD *)(v86 + 24);
              if ( !v87 || (v87 & (v87 - 1)) != 0 )
                break;
            }
            v142 = 1;
          }
          if ( v149 == ++v84 )
          {
            v5 = v137;
            if ( v142 )
              goto LABEL_110;
            break;
          }
        }
      }
    }
  }
LABEL_6:
  if ( v154 == 0 || a5 == 0 )
    return 0;
  if ( !v9 )
    return 0;
  v12 = *v154;
  if ( *v9 != *v154 )
    return 0;
  if ( *((_QWORD *)v9 - 8) != *((_QWORD *)v154 - 8) )
    goto LABEL_10;
  if ( v12 == 54 )
  {
    if ( !a4[4].m128i_i8[0] )
      return 0;
    v62 = sub_B448F0((__int64)v154);
    v63 = sub_B44900((__int64)v154);
    if ( v62 )
    {
      v64 = sub_B448F0((__int64)v9);
      if ( v63 && sub_B44900((__int64)v9) )
      {
        if ( v64 )
        {
          sub_B532B0(v6);
          goto LABEL_173;
        }
      }
      else if ( v64 )
      {
        if ( sub_B532B0(v6) )
          goto LABEL_10;
LABEL_173:
        if ( !(unsigned __int8)sub_9B6260(*((_QWORD *)v154 - 8), a4, 0) )
          goto LABEL_10;
        result = sub_1012FB0(v151, *((_BYTE **)v154 - 4), *((_BYTE **)v9 - 4), a4->m128i_i64, a5 - 1);
        if ( !result )
          goto LABEL_10;
        return result;
      }
    }
    if ( *((_QWORD *)v9 - 4) != *((_QWORD *)v154 - 4) )
      return 0;
LABEL_182:
    v66 = sub_B448F0((__int64)v154);
    v67 = sub_B44900((__int64)v154);
    v68 = v67;
    if ( v66 )
    {
      v69 = sub_B448F0((__int64)v9);
      if ( !v68 || !sub_B44900((__int64)v9) )
      {
        if ( !v69 )
          return 0;
        if ( sub_B532B0(v6) )
          return 0;
      }
    }
    else
    {
      if ( !v67 )
        return 0;
      if ( !sub_B44900((__int64)v9) )
        return 0;
    }
    return sub_1012FB0(
             v6 | v151 & 0xFFFFFFFF00000000LL,
             *((_BYTE **)v154 - 8),
             *((_BYTE **)v9 - 8),
             a4->m128i_i64,
             a5 - 1);
  }
  if ( (unsigned __int8)(v12 - 57) > 1u )
    goto LABEL_10;
  if ( (unsigned int)(v6 - 32) <= 1 )
    goto LABEL_10;
  v161 = 0;
  v31 = *((_QWORD *)v154 - 4);
  v160 = &v158;
  if ( !(unsigned __int8)sub_991580((__int64)&v160, v31) )
    goto LABEL_10;
  v32 = *((_QWORD *)v9 - 4);
  LOBYTE(v163) = 0;
  v162 = (__int64 ***)&v159;
  if ( !(unsigned __int8)sub_991580((__int64)&v162, v32) )
    goto LABEL_10;
  v33 = v158;
  v34 = v159;
  if ( *((_DWORD *)v158 + 2) <= 0x40u )
  {
    if ( (*v158 & ~*v159) == 0 )
      goto LABEL_83;
LABEL_305:
    v158 = v34;
    v159 = v33;
    v106 = sub_B52F50(v6);
    v33 = v158;
    v34 = v159;
    v6 = v106;
    v151 &= 0xFFFFFF00FFFFFFFFLL;
    if ( *((_DWORD *)v158 + 2) <= 0x40u )
    {
      if ( (*v158 & ~*v159) != 0 )
        goto LABEL_10;
    }
    else if ( !(unsigned __int8)sub_C446F0(v158, v159) )
    {
      goto LABEL_10;
    }
    goto LABEL_83;
  }
  if ( !(unsigned __int8)sub_C446F0(v158, v159) )
    goto LABEL_305;
LABEL_83:
  if ( (_DWORD)v6 != 37 )
  {
    if ( (_DWORD)v6 == 34 )
    {
LABEL_88:
      v36 = sub_1001990(*((_QWORD ***)v7 + 1));
      return sub_AD6450(v36);
    }
    v35 = sub_1002420(v33);
    if ( v35 != sub_1002420(v34) )
    {
LABEL_10:
      if ( *((_QWORD *)v9 - 4) == *((_QWORD *)v154 - 4) )
      {
        switch ( *v154 )
        {
          case '0':
          case '7':
            if ( sub_B532B0(v6) )
              return 0;
            if ( !a4[4].m128i_i8[0] )
              return 0;
            v60 = *v154;
            if ( (unsigned __int8)(v60 - 55) > 1u && (unsigned int)(v60 - 48) > 1 )
              return 0;
            if ( (v154[1] & 2) == 0 )
              return 0;
            v61 = *v9;
            if ( (unsigned int)(v61 - 48) > 1 && (unsigned __int8)(v61 - 55) > 1u )
              return 0;
            goto LABEL_165;
          case '1':
            if ( (unsigned int)(v6 - 32) > 1 )
              return 0;
            goto LABEL_176;
          case '6':
            if ( a4[4].m128i_i8[0] )
              goto LABEL_182;
            return 0;
          case '8':
LABEL_176:
            if ( !a4[4].m128i_i8[0] || (v154[1] & 2) == 0 )
              return 0;
            v65 = *v9;
            if ( (unsigned __int8)(v65 - 55) > 1u && (unsigned int)(v65 - 48) > 1 )
              return 0;
LABEL_165:
            if ( (v9[1] & 2) == 0 )
              return 0;
            break;
          default:
            return 0;
        }
        return sub_1012FB0(
                 v6 | v151 & 0xFFFFFFFF00000000LL,
                 *((_BYTE **)v154 - 8),
                 *((_BYTE **)v9 - 8),
                 a4->m128i_i64,
                 a5 - 1);
      }
      return 0;
    }
    if ( (_DWORD)v6 != 41 )
    {
      if ( (_DWORD)v6 == 38 )
        goto LABEL_88;
      goto LABEL_10;
    }
  }
  v109 = sub_1001990(*((_QWORD ***)v7 + 1));
  return sub_AD6400(v109);
}
