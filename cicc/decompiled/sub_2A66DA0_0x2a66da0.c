// Function: sub_2A66DA0
// Address: 0x2a66da0
//
__int64 __fastcall sub_2A66DA0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 *v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  char v9; // bl
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // rcx
  __int64 *v15; // rdx
  __int64 v16; // rcx
  __int64 *v17; // rdx
  __int64 *v19; // rdx
  __int64 v20; // r10
  unsigned int v21; // edi
  __int64 v22; // rdx
  __int64 v23; // rbx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  _QWORD *v27; // rdx
  _QWORD *v28; // rax
  __int64 v29; // rsi
  __int64 *v30; // r14
  __int64 *v31; // rdx
  __int64 v32; // r10
  unsigned int v33; // edi
  __int64 v34; // rdx
  __int64 *v35; // rdx
  __int64 v36; // r9
  bool v37; // al
  __int64 *v38; // r12
  unsigned int v39; // r12d
  char v40; // al
  __int64 v41; // r9
  _BOOL4 v42; // edi
  __int64 v43; // rcx
  __int64 v44; // rdx
  __int64 *v45; // rbx
  __int64 v46; // rax
  __int64 *v47; // r14
  bool v48; // r15
  __int64 v49; // rsi
  bool v50; // r15
  __int64 v51; // rsi
  bool v52; // r15
  __int64 v53; // rsi
  bool v54; // r15
  _BYTE *v55; // rsi
  _QWORD *v56; // rax
  _QWORD *v57; // rdx
  unsigned int v58; // eax
  __int64 v59; // rsi
  unsigned __int8 *v60; // rsi
  char v61; // al
  __int64 v62; // rax
  __int64 v63; // r10
  __int64 v64; // rdi
  __int64 *v65; // rax
  bool v66; // al
  __int64 v67; // rax
  __int64 v68; // r10
  __int64 v69; // rdi
  __int64 *v70; // rax
  bool v71; // al
  bool v72; // al
  __int64 *v73; // rdx
  bool v74; // bl
  _QWORD *v75; // rax
  _QWORD *v76; // rdx
  unsigned int v77; // eax
  _QWORD *v78; // rax
  _QWORD *v79; // rdx
  unsigned int v80; // eax
  _QWORD *v81; // rax
  _QWORD *v82; // rdx
  unsigned int v83; // eax
  __int64 *v84; // rax
  __int64 v85; // r15
  unsigned int v86; // eax
  char v87; // cl
  unsigned int v88; // esi
  __int64 *v89; // rax
  __int64 v90; // r15
  unsigned int v91; // eax
  char v92; // cl
  unsigned int v93; // esi
  __int64 *v94; // rax
  __int64 v95; // r15
  unsigned int v96; // eax
  char v97; // cl
  unsigned int v98; // esi
  __int64 *v99; // rax
  __int64 v100; // r15
  unsigned int v101; // eax
  char v102; // cl
  unsigned int v103; // esi
  signed __int64 v104; // rax
  int v105; // eax
  __int64 v106; // [rsp+8h] [rbp-108h]
  char v107; // [rsp+8h] [rbp-108h]
  __int64 v108; // [rsp+8h] [rbp-108h]
  char v109; // [rsp+8h] [rbp-108h]
  _BYTE *v110; // [rsp+8h] [rbp-108h]
  char v111; // [rsp+8h] [rbp-108h]
  __int64 v112; // [rsp+8h] [rbp-108h]
  char v113; // [rsp+8h] [rbp-108h]
  __int64 v114; // [rsp+10h] [rbp-100h]
  __int64 v115; // [rsp+18h] [rbp-F8h]
  __int64 v116; // [rsp+28h] [rbp-E8h]
  __int64 *v117; // [rsp+28h] [rbp-E8h]
  __int64 v118; // [rsp+30h] [rbp-E0h]
  __int64 v119; // [rsp+30h] [rbp-E0h]
  __int64 v120; // [rsp+30h] [rbp-E0h]
  __int64 v121; // [rsp+40h] [rbp-D0h]
  __int64 v122; // [rsp+40h] [rbp-D0h]
  __int64 v123; // [rsp+40h] [rbp-D0h]
  __int64 v124; // [rsp+40h] [rbp-D0h]
  __int64 v125; // [rsp+40h] [rbp-D0h]
  __int64 v126; // [rsp+40h] [rbp-D0h]
  unsigned __int8 v127; // [rsp+40h] [rbp-D0h]
  bool v129; // [rsp+56h] [rbp-BAh]
  unsigned __int8 v130; // [rsp+57h] [rbp-B9h]
  __int64 v131; // [rsp+58h] [rbp-B8h]
  __int64 *v132; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v133; // [rsp+68h] [rbp-A8h]
  unsigned __int64 v134; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v135; // [rsp+78h] [rbp-98h]
  unsigned __int64 v136; // [rsp+80h] [rbp-90h]
  unsigned int v137; // [rsp+88h] [rbp-88h]
  unsigned __int64 v138; // [rsp+90h] [rbp-80h] BYREF
  unsigned int v139; // [rsp+98h] [rbp-78h]
  unsigned __int64 v140; // [rsp+A0h] [rbp-70h]
  unsigned int v141; // [rsp+A8h] [rbp-68h]
  unsigned __int64 v142; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v143; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v144; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v145; // [rsp+C8h] [rbp-48h]
  __int16 v146; // [rsp+D0h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 56);
  v131 = a2 + 48;
  v130 = 0;
  if ( a2 + 48 != v4 )
  {
    while ( 1 )
    {
      v5 = v4;
      v4 = *(_QWORD *)(v4 + 8);
      if ( *(_BYTE *)(*(_QWORD *)(v5 - 16) + 8LL) == 7 )
        goto LABEL_5;
      v6 = (__int64 *)(v5 - 24);
      v9 = sub_2A66C70(a1, v5 - 24);
      if ( v9 )
      {
        v130 = sub_F509B0((unsigned __int8 *)(v5 - 24), 0);
        if ( v130 )
        {
          sub_B43D60((_QWORD *)(v5 - 24));
          goto LABEL_5;
        }
        goto LABEL_4;
      }
      v138 = (unsigned __int64)a1;
      switch ( *(_BYTE *)(v5 - 24) )
      {
        case '1':
        case '4':
          if ( (*(_BYTE *)(v5 - 17) & 0x40) != 0 )
          {
            v10 = *(__int64 **)(v5 - 32);
          }
          else
          {
            v7 = 32LL * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF);
            v10 = &v6[v7 / 0xFFFFFFFFFFFFFFF8LL];
          }
          v116 = *v10;
          v118 = v10[4];
          if ( (unsigned __int8)sub_B19060(a3, v116, v7, v116)
            || (unsigned __int8)sub_B19060(a3, v118, v11, v12)
            || !sub_2A65280((__int64 *)&v138, v116)
            || !sub_2A65280((__int64 *)&v138, v118) )
          {
            break;
          }
          v41 = v114;
          v42 = *(_BYTE *)(v5 - 24) != 49;
          v146 = 257;
          LOWORD(v41) = 0;
          v23 = sub_B504D0(3 * v42 + 19, v116, v118, (__int64)&v142, v5, v41);
          if ( *(_BYTE *)(v5 - 24) != 49 )
            goto LABEL_47;
          goto LABEL_68;
        case '8':
          if ( (*(_BYTE *)(v5 - 17) & 0x40) != 0 )
            v31 = *(__int64 **)(v5 - 32);
          else
            v31 = &v6[-4 * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF)];
          v122 = *v31;
          if ( (unsigned __int8)sub_B19060(a3, *v31, (__int64)v31, v8) )
            break;
          v32 = v122;
          if ( *(_BYTE *)v122 > 0x15u )
          {
            v67 = sub_2A64F10(v138, v122);
            v68 = v122;
            v69 = v67 + 8;
            if ( *(_BYTE *)v67 == 4
              || *(_BYTE *)v67 == 5
              && (v120 = v122, v125 = v67 + 8, v70 = sub_9876C0((__int64 *)(v67 + 8)), v69 = v125, v68 = v120, v70) )
            {
              v126 = v68;
              v71 = sub_AB0760(v69);
              v32 = v126;
              if ( v71 )
                goto LABEL_65;
            }
          }
          else if ( *(_BYTE *)v122 == 17 )
          {
            v33 = *(_DWORD *)(v122 + 32);
            v34 = *(_QWORD *)(v122 + 24);
            if ( v33 > 0x40 )
              v34 = *(_QWORD *)(v34 + 8LL * ((v33 - 1) >> 6));
            if ( (v34 & (1LL << ((unsigned __int8)v33 - 1))) == 0 )
            {
LABEL_65:
              v146 = 257;
              if ( (*(_BYTE *)(v5 - 17) & 0x40) != 0 )
                v35 = *(__int64 **)(v5 - 32);
              else
                v35 = &v6[-4 * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF)];
              v36 = v115;
              LOWORD(v36) = 0;
              v23 = sub_B504D0(26, v32, v35[4], (__int64)&v142, v5, v36);
LABEL_68:
              v37 = sub_B44E60(v5 - 24);
              sub_B448B0(v23, v37);
LABEL_47:
              sub_BD6B90((unsigned __int8 *)v23, (unsigned __int8 *)(v5 - 24));
              v27 = (_QWORD *)a3;
              if ( *(_BYTE *)(a3 + 28) )
              {
                v28 = *(_QWORD **)(a3 + 8);
                v24 = *(unsigned int *)(a3 + 20);
                v27 = &v28[v24];
                if ( v28 != v27 )
                {
                  while ( v23 != *v28 )
                  {
                    if ( v27 == ++v28 )
                      goto LABEL_148;
                  }
LABEL_52:
                  sub_BD84D0(v5 - 24, v23);
                  v29 = *(_QWORD *)(v5 + 24);
                  v30 = (__int64 *)(v23 + 48);
                  v142 = v29;
                  if ( v29 )
                  {
                    sub_B96E90((__int64)&v142, v29, 1);
                    if ( v30 == (__int64 *)&v142 )
                    {
                      if ( v142 )
                        sub_B91220((__int64)&v142, v142);
                      goto LABEL_56;
                    }
                    v59 = *(_QWORD *)(v23 + 48);
                    if ( !v59 )
                    {
LABEL_140:
                      v60 = (unsigned __int8 *)v142;
                      *(_QWORD *)(v23 + 48) = v142;
                      if ( v60 )
                        sub_B976B0((__int64)&v142, v60, v23 + 48);
                      goto LABEL_56;
                    }
                  }
                  else if ( v30 == (__int64 *)&v142 || (v59 = *(_QWORD *)(v23 + 48)) == 0 )
                  {
LABEL_56:
                    sub_2A64E30(a1, (__int64)v6);
                    sub_B43D60(v6);
                    goto LABEL_4;
                  }
                  sub_B91220(v23 + 48, v59);
                  goto LABEL_140;
                }
LABEL_148:
                if ( (unsigned int)v24 < *(_DWORD *)(a3 + 16) )
                {
                  *(_DWORD *)(a3 + 20) = v24 + 1;
                  *v27 = v23;
                  ++*(_QWORD *)a3;
                  goto LABEL_52;
                }
              }
              sub_C8CC70(a3, v23, (__int64)v27, v24, v25, v26);
              goto LABEL_52;
            }
          }
          break;
        case 'E':
        case 'I':
          if ( (*(_BYTE *)(v5 - 17) & 0x40) != 0 )
            v19 = *(__int64 **)(v5 - 32);
          else
            v19 = &v6[-4 * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF)];
          v121 = *v19;
          if ( (unsigned __int8)sub_B19060(a3, *v19, (__int64)v19, v8) )
            break;
          v20 = v121;
          if ( *(_BYTE *)v121 > 0x15u )
          {
            v62 = sub_2A64F10(v138, v121);
            v63 = v121;
            v64 = v62 + 8;
            if ( *(_BYTE *)v62 == 4
              || *(_BYTE *)v62 == 5
              && (v119 = v121, v123 = v62 + 8, v65 = sub_9876C0((__int64 *)(v62 + 8)), v64 = v123, v63 = v119, v65) )
            {
              v124 = v63;
              v66 = sub_AB0760(v64);
              v20 = v124;
              if ( v66 )
                goto LABEL_46;
            }
          }
          else if ( *(_BYTE *)v121 == 17 )
          {
            v21 = *(_DWORD *)(v121 + 32);
            v22 = *(_QWORD *)(v121 + 24);
            if ( v21 > 0x40 )
              v22 = *(_QWORD *)(v22 + 8LL * ((v21 - 1) >> 6));
            if ( (v22 & (1LL << ((unsigned __int8)v21 - 1))) == 0 )
            {
LABEL_46:
              v146 = 257;
              v23 = sub_B51D30(
                      4 * (unsigned int)(*(_BYTE *)(v5 - 24) != 69) + 39,
                      v20,
                      *(_QWORD *)(v5 - 16),
                      (__int64)&v142,
                      v5,
                      0);
              sub_B448D0(v23, 1);
              goto LABEL_47;
            }
          }
          break;
        default:
          break;
      }
      v132 = a1;
      v133 = a3;
      v13 = *(unsigned __int8 *)(v5 - 24);
      if ( (unsigned __int8)v13 > 0x36u )
        break;
      v9 = ((0x40540000000000uLL >> v13) & 1) == 0;
      if ( ((0x40540000000000uLL >> v13) & 1) == 0 )
        goto LABEL_88;
      if ( sub_B44900(v5 - 24) && sub_B448F0(v5 - 24) )
        goto LABEL_5;
      if ( (*(_BYTE *)(v5 - 17) & 0x40) != 0 )
        v15 = *(__int64 **)(v5 - 32);
      else
        v15 = &v6[-4 * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF)];
      sub_2A64FA0((__int64)&v134, (__int64 *)&v132, *v15, v14);
      if ( (*(_BYTE *)(v5 - 17) & 0x40) != 0 )
        v17 = *(__int64 **)(v5 - 32);
      else
        v17 = &v6[-4 * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF)];
      sub_2A64FA0((__int64)&v138, (__int64 *)&v132, v17[4], v16);
      if ( !sub_B448F0(v5 - 24) )
      {
        sub_AB28E0((__int64)&v142, *(unsigned __int8 *)(v5 - 24) - 29, (__int64)&v138, 1);
        v9 = sub_AB1BB0((__int64)&v142, (__int64)&v134);
        if ( v9 )
          sub_B447F0((unsigned __int8 *)(v5 - 24), 1);
        sub_969240((__int64 *)&v144);
        sub_969240((__int64 *)&v142);
      }
      if ( !sub_B44900(v5 - 24) )
      {
        sub_AB28E0((__int64)&v142, *(unsigned __int8 *)(v5 - 24) - 29, (__int64)&v138, 2);
        v61 = sub_AB1BB0((__int64)&v142, (__int64)&v134);
        if ( v61 )
        {
          v9 = v61;
          sub_B44850((unsigned __int8 *)v6, 1);
        }
        sub_969240((__int64 *)&v144);
        sub_969240((__int64 *)&v142);
      }
      if ( v141 > 0x40 && v140 )
        j_j___libc_free_0_0(v140);
      if ( v139 > 0x40 && v138 )
        j_j___libc_free_0_0(v138);
      if ( v137 > 0x40 && v136 )
        j_j___libc_free_0_0(v136);
      if ( v135 > 0x40 && v134 )
        j_j___libc_free_0_0(v134);
LABEL_35:
      if ( v9 )
      {
LABEL_4:
        v130 = 1;
LABEL_5:
        if ( v131 == v4 )
          return v130;
      }
      else if ( v131 == v4 )
      {
        return v130;
      }
    }
    if ( (((_BYTE)v13 - 68) & 0xFB) == 0 )
    {
      v127 = *(_BYTE *)(v5 - 24);
      v72 = sub_B44910(v5 - 24);
      v13 = v127;
      if ( !v72 )
      {
        if ( (*(_BYTE *)(v5 - 17) & 0x40) != 0 )
          v73 = *(__int64 **)(v5 - 32);
        else
          v73 = &v6[-4 * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF)];
        sub_2A64FA0((__int64)&v142, (__int64 *)&v132, *v73, v127);
        v74 = sub_AB0760((__int64)&v142);
        if ( v74 )
        {
          sub_B448D0(v5 - 24, 1);
          sub_969240((__int64 *)&v144);
          sub_969240((__int64 *)&v142);
          v130 = v74;
        }
        else
        {
          sub_969240((__int64 *)&v144);
          sub_969240((__int64 *)&v142);
        }
        goto LABEL_5;
      }
    }
    if ( (_BYTE)v13 == 67 )
    {
      if ( ((*(_BYTE *)(v5 - 23) >> 1) & 2) != 0 && (*(_BYTE *)(v5 - 23) & 2) != 0 )
        goto LABEL_5;
      if ( (*(_BYTE *)(v5 - 17) & 0x40) != 0 )
        v38 = *(__int64 **)(v5 - 32);
      else
        v38 = &v6[-4 * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF)];
      sub_2A64FA0((__int64)&v142, (__int64 *)&v132, *v38, v13);
      v39 = sub_BCB060(*(_QWORD *)(v5 - 16));
      v40 = *(_BYTE *)(v5 - 23) >> 1;
      if ( (*(_BYTE *)(v5 - 23) & 2) == 0 )
      {
        if ( v39 >= (unsigned int)sub_AB1CA0((__int64)&v142) )
        {
          v9 = 1;
          v40 = (*(_BYTE *)(v5 - 23) >> 1) & 0xFE | 1;
          *(_BYTE *)(v5 - 23) = (2 * v40) | *(_BYTE *)(v5 - 23) & 1;
        }
        else
        {
          v9 = 0;
          v40 = *(_BYTE *)(v5 - 23) >> 1;
        }
      }
      if ( (v40 & 2) == 0 && v39 >= (unsigned int)sub_AB1D50((__int64)&v142) )
      {
        *(_BYTE *)(v5 - 23) |= 4u;
        sub_969240((__int64 *)&v144);
        sub_969240((__int64 *)&v142);
        v130 = 1;
        goto LABEL_5;
      }
      sub_969240((__int64 *)&v144);
      sub_969240((__int64 *)&v142);
      goto LABEL_35;
    }
LABEL_88:
    if ( (_BYTE)v13 != 63 )
      goto LABEL_5;
    if ( sub_B4DE50(v5 - 24) )
      goto LABEL_5;
    v129 = sub_B4DE40(v5 - 24);
    if ( !v129 )
      goto LABEL_5;
    v44 = *(_DWORD *)(v5 - 20) & 0x7FFFFFF;
    v45 = &v6[4 * (1 - v44)];
    v46 = (-32 * (1 - v44)) >> 7;
    if ( v46 > 0 )
    {
      v47 = &v45[16 * v46];
      v117 = a1;
      while ( 1 )
      {
        v55 = (_BYTE *)*v45;
        if ( *(_BYTE *)*v45 <= 0x15u )
          goto LABEL_93;
        if ( *(_BYTE *)(v133 + 28) )
        {
          v56 = *(_QWORD **)(v133 + 8);
          v57 = &v56[*(unsigned int *)(v133 + 20)];
          if ( v56 != v57 )
          {
            while ( v55 != (_BYTE *)*v56 )
            {
              if ( v57 == ++v56 )
                goto LABEL_205;
            }
LABEL_135:
            v58 = sub_BCB060(*((_QWORD *)v55 + 1));
            sub_AADB10((__int64)&v142, v58, 1);
            goto LABEL_94;
          }
        }
        else
        {
          v110 = (_BYTE *)*v45;
          v94 = sub_C8CA60(v133, (__int64)v55);
          v55 = v110;
          if ( v94 )
            goto LABEL_135;
        }
LABEL_205:
        v95 = sub_2A64F10((__int64)v132, (__int64)v55);
        v111 = *(_BYTE *)v95;
        if ( *(_BYTE *)v95 == 4 )
          break;
        v96 = sub_BCB060(*((_QWORD *)v55 + 1));
        v97 = v111;
        v98 = v96;
        if ( v111 == 5 )
        {
          v98 = v96;
          if ( sub_9876C0((__int64 *)(v95 + 8)) )
            break;
          v97 = *(_BYTE *)v95;
        }
        if ( v97 == 2 )
        {
          v55 = *(_BYTE **)(v95 + 8);
LABEL_93:
          sub_AD8380((__int64)&v142, (__int64)v55);
          goto LABEL_94;
        }
        if ( v97 )
          sub_AADB10((__int64)&v142, v98, 1);
        else
          sub_AADB10((__int64)&v142, v98, 0);
LABEL_94:
        v48 = sub_AB0760((__int64)&v142);
        if ( v145 > 0x40 && v144 )
          j_j___libc_free_0_0(v144);
        if ( v143 > 0x40 && v142 )
          j_j___libc_free_0_0(v142);
        if ( !v48 )
        {
          a1 = v117;
          goto LABEL_248;
        }
        v49 = v45[4];
        if ( *(_BYTE *)v49 <= 0x15u )
          goto LABEL_102;
        if ( *(_BYTE *)(v133 + 28) )
        {
          v75 = *(_QWORD **)(v133 + 8);
          v76 = &v75[*(unsigned int *)(v133 + 20)];
          if ( v75 != v76 )
          {
            while ( v49 != *v75 )
            {
              if ( v76 == ++v75 )
                goto LABEL_199;
            }
LABEL_174:
            v77 = sub_BCB060(*(_QWORD *)(v49 + 8));
            sub_AADB10((__int64)&v142, v77, 1);
            goto LABEL_103;
          }
        }
        else
        {
          v108 = v45[4];
          v89 = sub_C8CA60(v133, v49);
          v49 = v108;
          if ( v89 )
            goto LABEL_174;
        }
LABEL_199:
        v90 = sub_2A64F10((__int64)v132, v49);
        v109 = *(_BYTE *)v90;
        if ( *(_BYTE *)v90 == 4 )
        {
LABEL_222:
          v143 = *(_DWORD *)(v90 + 16);
          if ( v143 > 0x40 )
            sub_C43780((__int64)&v142, (const void **)(v90 + 8));
          else
            v142 = *(_QWORD *)(v90 + 8);
          v145 = *(_DWORD *)(v90 + 32);
          if ( v145 > 0x40 )
            sub_C43780((__int64)&v144, (const void **)(v90 + 24));
          else
            v144 = *(_QWORD *)(v90 + 24);
          goto LABEL_103;
        }
        v91 = sub_BCB060(*(_QWORD *)(v49 + 8));
        v92 = v109;
        v93 = v91;
        if ( v109 == 5 )
        {
          v93 = v91;
          if ( sub_9876C0((__int64 *)(v90 + 8)) )
            goto LABEL_222;
          v92 = *(_BYTE *)v90;
        }
        if ( v92 == 2 )
        {
          v49 = *(_QWORD *)(v90 + 8);
LABEL_102:
          sub_AD8380((__int64)&v142, v49);
          goto LABEL_103;
        }
        if ( v92 )
          sub_AADB10((__int64)&v142, v93, 1);
        else
          sub_AADB10((__int64)&v142, v93, 0);
LABEL_103:
        v50 = sub_AB0760((__int64)&v142);
        if ( v145 > 0x40 && v144 )
          j_j___libc_free_0_0(v144);
        if ( v143 > 0x40 && v142 )
          j_j___libc_free_0_0(v142);
        if ( !v50 )
        {
          v45 += 4;
          a1 = v117;
          goto LABEL_248;
        }
        v51 = v45[8];
        if ( *(_BYTE *)v51 <= 0x15u )
          goto LABEL_111;
        if ( *(_BYTE *)(v133 + 28) )
        {
          v81 = *(_QWORD **)(v133 + 8);
          v82 = &v81[*(unsigned int *)(v133 + 20)];
          if ( v81 != v82 )
          {
            while ( v51 != *v81 )
            {
              if ( v82 == ++v81 )
                goto LABEL_193;
            }
LABEL_186:
            v83 = sub_BCB060(*(_QWORD *)(v51 + 8));
            sub_AADB10((__int64)&v142, v83, 1);
            goto LABEL_112;
          }
        }
        else
        {
          v106 = v45[8];
          v84 = sub_C8CA60(v133, v51);
          v51 = v106;
          if ( v84 )
            goto LABEL_186;
        }
LABEL_193:
        v85 = sub_2A64F10((__int64)v132, v51);
        v107 = *(_BYTE *)v85;
        if ( *(_BYTE *)v85 == 4 )
        {
LABEL_218:
          v143 = *(_DWORD *)(v85 + 16);
          if ( v143 > 0x40 )
            sub_C43780((__int64)&v142, (const void **)(v85 + 8));
          else
            v142 = *(_QWORD *)(v85 + 8);
          v145 = *(_DWORD *)(v85 + 32);
          if ( v145 > 0x40 )
            sub_C43780((__int64)&v144, (const void **)(v85 + 24));
          else
            v144 = *(_QWORD *)(v85 + 24);
          goto LABEL_112;
        }
        v86 = sub_BCB060(*(_QWORD *)(v51 + 8));
        v87 = v107;
        v88 = v86;
        if ( v107 == 5 )
        {
          v88 = v86;
          if ( sub_9876C0((__int64 *)(v85 + 8)) )
            goto LABEL_218;
          v87 = *(_BYTE *)v85;
        }
        if ( v87 == 2 )
        {
          v51 = *(_QWORD *)(v85 + 8);
LABEL_111:
          sub_AD8380((__int64)&v142, v51);
          goto LABEL_112;
        }
        if ( v87 )
          sub_AADB10((__int64)&v142, v88, 1);
        else
          sub_AADB10((__int64)&v142, v88, 0);
LABEL_112:
        v52 = sub_AB0760((__int64)&v142);
        if ( v145 > 0x40 && v144 )
          j_j___libc_free_0_0(v144);
        if ( v143 > 0x40 && v142 )
          j_j___libc_free_0_0(v142);
        if ( !v52 )
        {
          v45 += 8;
          a1 = v117;
          goto LABEL_248;
        }
        v53 = v45[12];
        if ( *(_BYTE *)v53 <= 0x15u )
          goto LABEL_120;
        if ( *(_BYTE *)(v133 + 28) )
        {
          v78 = *(_QWORD **)(v133 + 8);
          v79 = &v78[*(unsigned int *)(v133 + 20)];
          if ( v78 != v79 )
          {
            while ( v53 != *v78 )
            {
              if ( v79 == ++v78 )
                goto LABEL_211;
            }
LABEL_180:
            v80 = sub_BCB060(*(_QWORD *)(v53 + 8));
            sub_AADB10((__int64)&v142, v80, 1);
            goto LABEL_121;
          }
        }
        else
        {
          v112 = v45[12];
          v99 = sub_C8CA60(v133, v53);
          v53 = v112;
          if ( v99 )
            goto LABEL_180;
        }
LABEL_211:
        v100 = sub_2A64F10((__int64)v132, v53);
        v113 = *(_BYTE *)v100;
        if ( *(_BYTE *)v100 == 4 )
        {
LABEL_230:
          v143 = *(_DWORD *)(v100 + 16);
          if ( v143 > 0x40 )
            sub_C43780((__int64)&v142, (const void **)(v100 + 8));
          else
            v142 = *(_QWORD *)(v100 + 8);
          v145 = *(_DWORD *)(v100 + 32);
          if ( v145 > 0x40 )
            sub_C43780((__int64)&v144, (const void **)(v100 + 24));
          else
            v144 = *(_QWORD *)(v100 + 24);
          goto LABEL_121;
        }
        v101 = sub_BCB060(*(_QWORD *)(v53 + 8));
        v102 = v113;
        v103 = v101;
        if ( v113 == 5 )
        {
          v103 = v101;
          if ( sub_9876C0((__int64 *)(v100 + 8)) )
            goto LABEL_230;
          v102 = *(_BYTE *)v100;
        }
        if ( v102 == 2 )
        {
          v53 = *(_QWORD *)(v100 + 8);
LABEL_120:
          sub_AD8380((__int64)&v142, v53);
          goto LABEL_121;
        }
        if ( v102 )
          sub_AADB10((__int64)&v142, v103, 1);
        else
          sub_AADB10((__int64)&v142, v103, 0);
LABEL_121:
        v54 = sub_AB0760((__int64)&v142);
        if ( v145 > 0x40 && v144 )
          j_j___libc_free_0_0(v144);
        if ( v143 > 0x40 && v142 )
          j_j___libc_free_0_0(v142);
        if ( !v54 )
        {
          v45 += 12;
          a1 = v117;
          goto LABEL_248;
        }
        v45 += 16;
        if ( v47 == v45 )
        {
          a1 = v117;
          goto LABEL_243;
        }
      }
      v143 = *(_DWORD *)(v95 + 16);
      if ( v143 > 0x40 )
        sub_C43780((__int64)&v142, (const void **)(v95 + 8));
      else
        v142 = *(_QWORD *)(v95 + 8);
      v145 = *(_DWORD *)(v95 + 32);
      if ( v145 > 0x40 )
        sub_C43780((__int64)&v144, (const void **)(v95 + 24));
      else
        v144 = *(_QWORD *)(v95 + 24);
      goto LABEL_94;
    }
LABEL_243:
    v104 = (char *)v6 - (char *)v45;
    if ( (char *)v6 - (char *)v45 != 64 )
    {
      if ( v104 != 96 )
      {
        if ( v104 != 32 )
        {
LABEL_246:
          v105 = sub_B4DE20((__int64)v6);
          sub_B4DDE0((__int64)v6, v105 | 4);
          v130 = v129;
          goto LABEL_5;
        }
        goto LABEL_257;
      }
      if ( !(unsigned __int8)sub_2A65320((__int64 *)&v132, *v45, v44, v43) )
        goto LABEL_248;
      v45 += 4;
    }
    if ( !(unsigned __int8)sub_2A65320((__int64 *)&v132, *v45, v44, v43) )
      goto LABEL_248;
    v45 += 4;
LABEL_257:
    if ( (unsigned __int8)sub_2A65320((__int64 *)&v132, *v45, v44, v43) )
      goto LABEL_246;
LABEL_248:
    if ( v6 != v45 )
      goto LABEL_5;
    goto LABEL_246;
  }
  return v130;
}
