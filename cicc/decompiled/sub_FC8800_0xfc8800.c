// Function: sub_FC8800
// Address: 0xfc8800
//
__int64 __fastcall sub_FC8800(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r12
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  unsigned int v11; // esi
  __int64 v12; // rcx
  __int64 v13; // r15
  int v15; // ecx
  __int64 v16; // rdi
  unsigned __int8 v17; // al
  _QWORD *v18; // r13
  __int64 v19; // rax
  _QWORD *v20; // r12
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdi
  _QWORD **v24; // r13
  __int64 v25; // rax
  int v26; // r10d
  __int64 v27; // rbx
  __int64 v28; // rsi
  __int64 v29; // rdi
  __int64 v30; // rax
  _BYTE *v31; // r13
  __int64 *v32; // rax
  __int64 v33; // r15
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // r8
  _QWORD **v37; // r15
  _QWORD **v38; // r12
  _QWORD *v39; // rbx
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 *v44; // r15
  __int64 v45; // r14
  __int64 *v46; // rax
  _BYTE *v47; // r14
  __int64 *v48; // rax
  _BYTE *v49; // rsi
  __int64 v50; // rax
  __int64 *v51; // rdi
  __int64 v52; // rsi
  __int64 v53; // rdx
  unsigned __int8 *v54; // rdi
  __int64 *v55; // rax
  __int64 v56; // rdx
  __int64 *v57; // r13
  __int64 *v58; // rax
  __int64 *v59; // rax
  __int64 v60; // r13
  __int64 v61; // rax
  __int64 v62; // r9
  __int64 v63; // rbx
  __int64 v64; // rax
  unsigned __int64 v65; // rcx
  __int64 *v66; // rbx
  __int64 *v67; // rax
  __int64 v68; // r13
  __int64 v69; // rax
  __int64 v70; // rsi
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // r13
  __int64 v74; // r15
  __int64 v75; // rcx
  _BYTE *v76; // r12
  __int64 v77; // rax
  __int64 v78; // r9
  unsigned int v79; // ebx
  unsigned __int8 *v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rdi
  __int64 **v83; // r13
  __int64 v84; // r15
  unsigned __int64 v85; // rax
  _QWORD *v86; // rax
  __int64 v87; // r8
  __int64 v88; // rdi
  __int64 **v89; // r11
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // r15
  unsigned int v93; // r14d
  __int64 v94; // r13
  __int64 v95; // rdi
  __int64 v96; // rbx
  __int64 v97; // rax
  unsigned __int64 v98; // rdx
  unsigned int v99; // ebx
  __int64 v100; // rax
  unsigned int v101; // r12d
  __int64 v102; // rbx
  __int64 v103; // rdx
  __int64 v104; // r8
  __int64 v105; // r9
  __int64 v106; // rax
  unsigned __int64 v107; // rdx
  __int64 v108; // r15
  unsigned __int8 v109; // al
  __int64 v110; // rax
  __int64 v111; // r15
  _QWORD *v112; // rax
  unsigned __int64 v113; // rcx
  __int64 v114; // r13
  __int64 v115; // rsi
  __int64 v116; // rax
  __int64 v117; // r8
  _QWORD *v118; // rdx
  __int64 v119; // rax
  _QWORD *v120; // rcx
  __int64 *v121; // r8
  _QWORD *v122; // rax
  __int64 v123; // rax
  __int64 v124; // r14
  __int64 v125; // r13
  __int64 v126; // rbx
  __int64 v127; // r12
  int v128; // eax
  __int64 (__fastcall *v129)(__int64, _BYTE *); // rbx
  __int64 v130; // rax
  __int64 v131; // [rsp+0h] [rbp-C0h]
  _QWORD *v132; // [rsp+0h] [rbp-C0h]
  __int64 **v133; // [rsp+8h] [rbp-B8h]
  __int64 v134; // [rsp+10h] [rbp-B0h]
  __int64 v135; // [rsp+10h] [rbp-B0h]
  __int64 v136; // [rsp+10h] [rbp-B0h]
  __int64 v137; // [rsp+18h] [rbp-A8h]
  __int64 **v138; // [rsp+18h] [rbp-A8h]
  __int64 **v139; // [rsp+18h] [rbp-A8h]
  __int64 v140; // [rsp+18h] [rbp-A8h]
  int v141; // [rsp+18h] [rbp-A8h]
  __int64 **v142; // [rsp+18h] [rbp-A8h]
  __int64 v143; // [rsp+20h] [rbp-A0h]
  __int64 v144; // [rsp+20h] [rbp-A0h]
  _QWORD *v145; // [rsp+20h] [rbp-A0h]
  __int64 v146; // [rsp+28h] [rbp-98h]
  unsigned int v147; // [rsp+28h] [rbp-98h]
  char v148; // [rsp+28h] [rbp-98h]
  __int64 **v149; // [rsp+28h] [rbp-98h]
  __int64 v150; // [rsp+30h] [rbp-90h] BYREF
  __int64 v151; // [rsp+38h] [rbp-88h]
  __int64 *v152; // [rsp+40h] [rbp-80h] BYREF
  __int64 v153; // [rsp+48h] [rbp-78h]
  _BYTE v154[16]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v155; // [rsp+60h] [rbp-60h]

  v6 = a1;
  v7 = a2;
  v8 = (_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL * *(unsigned int *)(a1 + 16));
  v9 = *(unsigned int *)(*v8 + 24LL);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD *)(*v8 + 8LL);
    v11 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = v10 + ((unsigned __int64)v11 << 6);
    a6 = *(_QWORD *)(v12 + 24);
    if ( a6 == v7 )
    {
LABEL_3:
      if ( v12 != v10 + (v9 << 6) )
        return *(_QWORD *)(v12 + 56);
    }
    else
    {
      v15 = 1;
      while ( a6 != -4096 )
      {
        v26 = v15 + 1;
        v11 = (v9 - 1) & (v15 + v11);
        v12 = v10 + ((unsigned __int64)v11 << 6);
        a6 = *(_QWORD *)(v12 + 24);
        if ( a6 == v7 )
          goto LABEL_3;
        v15 = v26;
      }
    }
  }
  v16 = v8[1];
  if ( v16 )
  {
    v13 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v16 + 8LL))(v16, v7);
    if ( v13 )
    {
      v20 = sub_FC8470(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), v7);
      v21 = v20[2];
      if ( v13 == v21 )
        return v13;
      if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
        sub_BD60C0(v20);
      v20[2] = v13;
      goto LABEL_25;
    }
  }
  v17 = *(_BYTE *)v7;
  if ( *(_BYTE *)v7 <= 3u )
  {
    if ( (*(_BYTE *)v6 & 8) == 0 )
      goto LABEL_12;
    return 0;
  }
  if ( v17 == 25 )
  {
    v22 = sub_B3B7D0(v7);
    v23 = *(_QWORD *)(v6 + 8);
    if ( !v23
      || (v24 = (_QWORD **)(*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v23 + 24LL))(v23, v22),
          v24 == (_QWORD **)sub_B3B7D0(v7)) )
    {
      v13 = v7;
    }
    else
    {
      v13 = sub_B41A60(
              v24,
              *(_QWORD *)(v7 + 24),
              *(_QWORD *)(v7 + 32),
              *(_QWORD *)(v7 + 56),
              *(_QWORD *)(v7 + 64),
              *(_BYTE *)(v7 + 96),
              *(_BYTE *)(v7 + 97),
              *(_DWORD *)(v7 + 100),
              *(_BYTE *)(v7 + 104));
    }
    v20 = sub_FC8470(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), v13);
    v25 = v20[2];
    if ( v25 == v13 )
      return v13;
    goto LABEL_33;
  }
  if ( v17 != 24 )
  {
    if ( v17 > 0x15u )
      return 0;
    switch ( v17 )
    {
      case 4u:
        v33 = sub_FC8800(v6, *(_QWORD *)(v7 - 64));
        if ( v33 + 72 == (*(_QWORD *)(v33 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
        {
          v150 = *(_QWORD *)(v7 - 32);
          v155 = 257;
          v60 = sub_BD5C60(v7);
          v61 = sub_22077B0(80);
          v63 = v61;
          if ( v61 )
            sub_AA4D50(v61, v60, (__int64)&v152, 0, 0);
          v64 = *(unsigned int *)(v6 + 192);
          v65 = *(unsigned int *)(v6 + 196);
          v151 = v63;
          v66 = &v150;
          if ( v64 + 1 > v65 )
          {
            v113 = *(_QWORD *)(v6 + 184);
            if ( v113 > (unsigned __int64)&v150 || (unsigned __int64)&v150 >= v113 + 16 * v64 )
            {
              v148 = 0;
              v114 = -1;
            }
            else
            {
              v148 = 1;
              v114 = (__int64)((__int64)&v150 - v113) >> 4;
            }
            v115 = v6 + 200;
            v144 = v6 + 200;
            v116 = sub_C8D7D0(v6 + 184, v6 + 200, v64 + 1, 0x10u, (unsigned __int64 *)&v152, v62);
            v117 = *(_QWORD *)(v6 + 184);
            v118 = (_QWORD *)v116;
            v119 = 2LL * *(unsigned int *)(v6 + 192);
            if ( v119 * 8 )
            {
              v120 = &v118[v119];
              v121 = (__int64 *)(v117 + 8);
              v122 = v118;
              do
              {
                if ( v122 )
                {
                  *v122 = *(v121 - 1);
                  v115 = *v121;
                  v122[1] = *v121;
                  *v121 = 0;
                }
                v122 += 2;
                v121 += 2;
              }
              while ( v122 != v120 );
              v117 = *(_QWORD *)(v6 + 184);
              v123 = v117 + 16LL * *(unsigned int *)(v6 + 192);
              if ( v117 != v123 )
              {
                v135 = v7;
                v140 = v6;
                v124 = v114;
                v125 = v117;
                v126 = v123;
                do
                {
                  v127 = *(_QWORD *)(v126 - 8);
                  v126 -= 16;
                  if ( v127 )
                  {
                    v132 = v118;
                    sub_AA5290(v127);
                    v115 = 80;
                    j_j___libc_free_0(v127, 80);
                    v118 = v132;
                  }
                }
                while ( v125 != v126 );
                v114 = v124;
                v6 = v140;
                v7 = v135;
                v66 = &v150;
                v117 = *(_QWORD *)(v140 + 184);
              }
            }
            v128 = (int)v152;
            if ( v144 != v117 )
            {
              v141 = (int)v152;
              v145 = v118;
              _libc_free(v117, v115);
              v128 = v141;
              v118 = v145;
            }
            *(_DWORD *)(v6 + 196) = v128;
            *(_QWORD *)(v6 + 184) = v118;
            v64 = *(unsigned int *)(v6 + 192);
            if ( v148 )
              v66 = &v118[2 * v114];
          }
          v67 = (__int64 *)(*(_QWORD *)(v6 + 184) + 16 * v64);
          if ( v67 )
          {
            *v67 = *v66;
            v67[1] = v66[1];
            v66[1] = 0;
          }
          v68 = v151;
          v69 = (unsigned int)(*(_DWORD *)(v6 + 192) + 1);
          *(_DWORD *)(v6 + 192) = v69;
          if ( v68 )
          {
            sub_AA5290(v68);
            j_j___libc_free_0(v68, 80);
            v69 = *(unsigned int *)(v6 + 192);
          }
          v34 = *(_QWORD *)(*(_QWORD *)(v6 + 184) + 16 * v69 - 8);
          if ( v34 )
            goto LABEL_52;
        }
        else
        {
          v34 = sub_FC8800(v6, *(_QWORD *)(v7 - 32));
          if ( v34 )
          {
LABEL_52:
            v35 = sub_ACC1C0(v33, v34);
            goto LABEL_53;
          }
        }
        v34 = *(_QWORD *)(v7 - 32);
        goto LABEL_52;
      case 6u:
        v52 = *(_QWORD *)(v7 - 32);
        v54 = (unsigned __int8 *)sub_FC8800(v6, v52);
        if ( *v54 > 3u )
        {
          v80 = sub_BD3BE0(v54, v52);
          v82 = *(_QWORD *)(v6 + 8);
          v83 = *(__int64 ***)(v7 + 8);
          v84 = (__int64)v80;
          if ( v82 )
          {
            v52 = *(_QWORD *)(v7 + 8);
            v83 = (__int64 **)(*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v82 + 24LL))(v82, v52);
          }
          v85 = sub_ACC6E0(v84, v52, v81);
          v13 = sub_AD4C90(v85, v83, 0);
          v20 = sub_FC8470(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), v7);
          v25 = v20[2];
          if ( v13 == v25 )
            return v13;
          if ( v25 == -4096 || v25 == 0 )
          {
LABEL_36:
            v20[2] = v13;
            if ( !v13 )
              return 0;
LABEL_25:
            if ( v13 != -4096 )
            {
              if ( v13 != -8192 )
                sub_BD73F0((__int64)v20);
              return v13;
            }
            return -4096;
          }
LABEL_34:
          if ( v25 != -8192 )
            sub_BD60C0(v20);
          goto LABEL_36;
        }
        v35 = sub_ACC6E0((__int64)v54, v52, v53);
        goto LABEL_53;
      case 7u:
        v70 = *(_QWORD *)(v7 - 32);
        v71 = sub_FC8800(v6, v70);
        v35 = sub_ACCB80(v71, v70, v72);
LABEL_53:
        v13 = v35;
        v20 = sub_FC8470(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), v7);
        v25 = v20[2];
        if ( v13 == v25 )
          return v13;
LABEL_33:
        if ( v25 == 0 || v25 == -4096 )
          goto LABEL_36;
        goto LABEL_34;
    }
    v147 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
    if ( v147 )
    {
      v73 = 0;
      v74 = v7;
      while ( 1 )
      {
        v79 = v73;
        v75 = (*(_BYTE *)(v74 + 7) & 0x40) != 0 ? *(_QWORD *)(v74 - 8) : v74 - 32LL * (*(_DWORD *)(v74 + 4) & 0x7FFFFFF);
        v76 = *(_BYTE **)(v75 + 32 * v73);
        v49 = v76;
        v77 = sub_FC8800(v6, v76);
        if ( !v77 )
          return 0;
        if ( v76 != (_BYTE *)v77 )
        {
          v88 = *(_QWORD *)(v6 + 8);
          v89 = *(__int64 ***)(v74 + 8);
          v87 = v77;
          v7 = v74;
          if ( v88 )
            goto LABEL_113;
          goto LABEL_114;
        }
        if ( v147 == (_DWORD)++v73 )
        {
          v79 = v147;
          v87 = v77;
          v7 = v74;
          goto LABEL_111;
        }
      }
    }
    v79 = 0;
    v87 = 0;
LABEL_111:
    v88 = *(_QWORD *)(v6 + 8);
    if ( !v88 )
      goto LABEL_108;
    v89 = *(__int64 ***)(v7 + 8);
LABEL_113:
    v143 = v87;
    v49 = v89;
    v90 = (*(__int64 (__fastcall **)(__int64, __int64 **))(*(_QWORD *)v88 + 24LL))(v88, v89);
    v87 = v143;
    v89 = (__int64 **)v90;
    if ( v147 == v79 && v90 == *(_QWORD *)(v7 + 8) )
    {
LABEL_108:
      v86 = sub_FC8470(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), v7);
      return sub_FC7530(v86, v7);
    }
LABEL_114:
    v152 = (__int64 *)v154;
    v153 = 0x800000000LL;
    if ( v147 > 8uLL )
    {
      v49 = v154;
      v134 = v87;
      v139 = v89;
      sub_C8D5F0((__int64)&v152, v154, v147, 8u, v87, v78);
      v87 = v134;
      v89 = v139;
    }
    if ( v79 )
    {
      v137 = v6;
      v49 = v154;
      v91 = (unsigned int)v153;
      v92 = 0;
      v93 = v79;
      v94 = 32LL * v79;
      do
      {
        if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
          v95 = *(_QWORD *)(v7 - 8);
        else
          v95 = v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
        v78 = v91 + 1;
        v96 = *(_QWORD *)(v95 + v92);
        if ( v91 + 1 > (unsigned __int64)HIDWORD(v153) )
        {
          v131 = v87;
          v133 = v89;
          sub_C8D5F0((__int64)&v152, v154, v91 + 1, 8u, v87, v78);
          v91 = (unsigned int)v153;
          v87 = v131;
          v89 = v133;
        }
        v92 += 32;
        v152[v91] = v96;
        v91 = (unsigned int)(v153 + 1);
        LODWORD(v153) = v153 + 1;
      }
      while ( v92 != v94 );
      v79 = v93;
      v6 = v137;
    }
    if ( v147 != v79 )
    {
      v97 = (unsigned int)v153;
      v98 = (unsigned int)v153 + 1LL;
      if ( v98 > HIDWORD(v153) )
      {
        v49 = v154;
        v136 = v87;
        v142 = v89;
        sub_C8D5F0((__int64)&v152, v154, v98, 8u, v87, v78);
        v97 = (unsigned int)v153;
        v87 = v136;
        v89 = v142;
      }
      v99 = v79 + 1;
      v152[v97] = v87;
      LODWORD(v153) = v153 + 1;
      if ( v99 != v147 )
      {
        v100 = v7;
        v138 = v89;
        v101 = v99;
        v102 = v100;
        while ( 1 )
        {
          v103 = (*(_BYTE *)(v102 + 7) & 0x40) != 0
               ? *(_QWORD *)(v102 - 8)
               : v102 - 32LL * (*(_DWORD *)(v102 + 4) & 0x7FFFFFF);
          v49 = *(_BYTE **)(v103 + 32LL * v101);
          v13 = sub_FC8800(v6, v49);
          if ( !v13 )
            break;
          v106 = (unsigned int)v153;
          v107 = (unsigned int)v153 + 1LL;
          if ( v107 > HIDWORD(v153) )
          {
            v49 = v154;
            sub_C8D5F0((__int64)&v152, v154, v107, 8u, v104, v105);
            v106 = (unsigned int)v153;
          }
          ++v101;
          v152[v106] = v13;
          LODWORD(v153) = v153 + 1;
          if ( v101 == v147 )
          {
            v89 = v138;
            v7 = v102;
            goto LABEL_138;
          }
        }
LABEL_143:
        v51 = v152;
        if ( v152 == (__int64 *)v154 )
          return v13;
        goto LABEL_68;
      }
    }
LABEL_138:
    v108 = *(_QWORD *)(v6 + 8);
    if ( v108 )
    {
      v109 = *(_BYTE *)v7;
      if ( *(_BYTE *)v7 <= 0x1Cu )
      {
        if ( v109 != 5 )
          goto LABEL_171;
        if ( *(_WORD *)(v7 + 2) != 34 )
        {
          v108 = 0;
          goto LABEL_167;
        }
      }
      else if ( v109 != 63 )
      {
LABEL_141:
        v110 = sub_AC9EC0(v89);
LABEL_142:
        v111 = v110;
        v112 = sub_FC8470(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), v7);
        v49 = (_BYTE *)v111;
        v13 = sub_FC7530(v112, v111);
        goto LABEL_143;
      }
      v149 = v89;
      v129 = *(__int64 (__fastcall **)(__int64, _BYTE *))(*(_QWORD *)v108 + 24LL);
      v49 = (_BYTE *)sub_BB5290(v7);
      v130 = v129(v108, v49);
      v89 = v149;
      v108 = v130;
    }
    v109 = *(_BYTE *)v7;
    if ( *(_BYTE *)v7 == 5 )
    {
LABEL_167:
      v110 = sub_ADABF0(v7, (__int64)v152, (unsigned int)v153, v89, 0, v108);
      goto LABEL_142;
    }
LABEL_171:
    switch ( v109 )
    {
      case 9u:
        v110 = sub_AD1300(v89, v152, (unsigned int)v153);
        goto LABEL_142;
      case 0xAu:
        v110 = sub_AD24A0(v89, v152, (unsigned int)v153);
        goto LABEL_142;
      case 0xBu:
        v110 = sub_AD3730(v152, (unsigned int)v153);
        goto LABEL_142;
      case 8u:
        v110 = sub_AD0290(*v152, v152[1], v152[2], v152[3]);
        goto LABEL_142;
      case 0xDu:
        v110 = sub_ACADE0(v89);
        goto LABEL_142;
    }
    if ( (unsigned int)v109 - 12 <= 1 )
    {
      v110 = sub_ACA8A0(v89);
      goto LABEL_142;
    }
    if ( v109 == 14 )
    {
      v110 = sub_AC9350(v89);
      goto LABEL_142;
    }
    if ( v109 == 19 )
    {
      v110 = sub_AD6530((__int64)v89, (__int64)v49);
      goto LABEL_142;
    }
    goto LABEL_141;
  }
  v27 = *(_QWORD *)(v7 + 24);
  if ( *(_BYTE *)v27 != 2 )
  {
    if ( *(_BYTE *)v27 == 4 )
    {
      v152 = (__int64 *)v154;
      v153 = 0x400000000LL;
      v36 = *(_QWORD *)(v27 + 136);
      v37 = (_QWORD **)(v36 + 8LL * *(unsigned int *)(v27 + 144));
      if ( (_QWORD **)v36 != v37 )
      {
        v146 = v7;
        v38 = *(_QWORD ***)(v27 + 136);
        while ( 1 )
        {
          v39 = *v38;
          if ( (*(_BYTE *)v6 & 1) != 0 && *(_BYTE *)v39 == 1 )
            goto LABEL_63;
          v40 = v39[17];
          v41 = sub_FC8800(v6, v40);
          if ( !v41 )
            break;
          if ( v41 != v39[17] )
            goto LABEL_62;
LABEL_63:
          v42 = (unsigned int)v153;
          v43 = (unsigned int)v153 + 1LL;
          if ( v43 > HIDWORD(v153) )
          {
            sub_C8D5F0((__int64)&v152, v154, v43, 8u, v36, a6);
            v42 = (unsigned int)v153;
          }
          ++v38;
          v152[v42] = (__int64)v39;
          LODWORD(v153) = v153 + 1;
          if ( v37 == v38 )
          {
            v7 = v146;
            v44 = v152;
            v45 = (unsigned int)v153;
            goto LABEL_67;
          }
        }
        if ( (*(_BYTE *)v6 & 2) != 0 && *(_BYTE *)v39 == 2 )
          goto LABEL_63;
        v41 = sub_ACADE0(*(__int64 ***)(v39[17] + 8LL));
LABEL_62:
        v39 = sub_B98A20(v41, v40);
        goto LABEL_63;
      }
      v45 = 0;
      v44 = (__int64 *)v154;
LABEL_67:
      v46 = (__int64 *)sub_BD5C60(v7);
      v47 = (_BYTE *)sub_B00B60(v46, v44, v45);
      v48 = (__int64 *)sub_BD5C60(v7);
      v49 = v47;
      v50 = sub_B9F6F0(v48, v47);
      v51 = v152;
      v13 = v50;
      if ( v152 == (__int64 *)v154 )
        return v13;
LABEL_68:
      _libc_free(v51, v49);
      return v13;
    }
    if ( (*(_BYTE *)v6 & 1) != 0 )
    {
LABEL_12:
      v18 = sub_FC8470(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), v7);
      v19 = v18[2];
      if ( v19 != v7 )
      {
        if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
          sub_BD60C0(v18);
        v18[2] = v7;
        if ( v7 != -4096 )
        {
          if ( v7 != -8192 )
            sub_BD73F0((__int64)v18);
          return v7;
        }
        return -4096;
      }
      return v7;
    }
    v55 = (__int64 *)sub_FC95E0(v6, *(_QWORD *)(v7 + 24));
    v153 = v56;
    v152 = v55;
    if ( (_BYTE)v56 )
      v57 = v152;
    else
      v57 = (__int64 *)sub_FCBA10(v6, v27);
    if ( (__int64 *)v27 != v57 )
    {
      v58 = (__int64 *)sub_BD5C60(v7);
      v35 = sub_B9F6F0(v58, v57);
      goto LABEL_53;
    }
    goto LABEL_108;
  }
  v28 = *(_QWORD *)(v27 + 136);
  v29 = sub_FC8800(v6, v28);
  if ( v29 )
  {
    if ( *(_QWORD *)(v27 + 136) == v7 )
      return v7;
    v30 = (__int64)sub_B98A20(v29, v28);
  }
  else
  {
    if ( (*(_BYTE *)v6 & 2) != 0 )
      return 0;
    v59 = (__int64 *)sub_BD5C60(v7);
    v30 = sub_B9C770(v59, 0, 0, 0, 1);
  }
  v31 = (_BYTE *)v30;
  v32 = (__int64 *)sub_BD5C60(v7);
  return sub_B9F6F0(v32, v31);
}
