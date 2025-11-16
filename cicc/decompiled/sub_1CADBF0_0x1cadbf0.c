// Function: sub_1CADBF0
// Address: 0x1cadbf0
//
__int64 __fastcall sub_1CADBF0(_QWORD ***a1, _QWORD *a2, unsigned __int8 a3, char a4)
{
  _QWORD *v5; // r13
  _QWORD *v8; // r15
  _QWORD *v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  _QWORD *v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // r15
  _QWORD *v19; // rsi
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  _BOOL8 v22; // rdi
  __int64 v23; // r15
  unsigned __int8 v25; // al
  _QWORD *v26; // r14
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // r15
  _QWORD *v31; // rsi
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  _QWORD *v34; // rbx
  _BOOL8 v35; // rdi
  __int64 v36; // rax
  unsigned int v37; // edx
  __int64 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  _QWORD *v42; // rax
  unsigned __int64 v43; // rsi
  _QWORD *v44; // r14
  unsigned int v45; // edx
  unsigned __int64 v46; // r14
  __int64 v47; // rax
  __int64 v48; // rdi
  unsigned int v49; // eax
  __int64 v50; // r14
  __int64 v51; // rax
  _QWORD *v52; // rax
  unsigned __int64 v53; // rsi
  _QWORD *v54; // rdi
  unsigned __int64 v55; // rax
  unsigned int v56; // eax
  _QWORD *v57; // rax
  __int64 v58; // rax
  _QWORD *v59; // rsi
  _QWORD *v60; // rax
  __int64 v61; // rax
  unsigned __int64 v62; // rcx
  _QWORD *v63; // rsi
  _QWORD *v64; // rax
  _QWORD *v65; // rdx
  _QWORD *v66; // rbx
  unsigned __int64 v67; // rcx
  _BOOL8 v68; // rdi
  __int64 v69; // rax
  unsigned __int64 v70; // rcx
  _QWORD *v71; // rsi
  _QWORD *v72; // rax
  _QWORD *v73; // rdi
  _QWORD *v74; // rax
  _QWORD *v75; // rdi
  _QWORD *v76; // rax
  unsigned __int64 v77; // rax
  __int64 v78; // rax
  _QWORD *v79; // rax
  _QWORD *v80; // r14
  __int64 v81; // rax
  _QWORD *v82; // rax
  _QWORD *v83; // r14
  __int64 v84; // rax
  _QWORD *v85; // rax
  _QWORD *v86; // r8
  __int64 v87; // rax
  _QWORD *v88; // rax
  _QWORD *v89; // rdx
  _BOOL8 v90; // rdi
  unsigned int v91; // edi
  unsigned __int64 v92; // r14
  _QWORD **v93; // rdi
  __int64 v94; // rdx
  __int64 v95; // r8
  __int64 v96; // r9
  _QWORD ***v97; // rcx
  __int64 v98; // rcx
  int v99; // eax
  __int64 v100; // rax
  int v101; // esi
  __int64 v102; // rsi
  __int64 *v103; // rax
  __int64 v104; // rdi
  unsigned __int64 v105; // r9
  __int64 v106; // rdi
  __int64 v107; // rdx
  __int64 v108; // rsi
  _QWORD *v109; // rax
  __int64 v110; // rax
  _QWORD *v111; // rsi
  _QWORD *v112; // rax
  _QWORD *v113; // rax
  _QWORD *v114; // r14
  __int64 v115; // rax
  unsigned __int64 v116; // r8
  _QWORD *v117; // rsi
  _QWORD *v118; // rax
  _QWORD *v119; // rdx
  bool v120; // al
  __int64 v121; // rdi
  __int64 v122; // rdx
  _QWORD *v123; // rdi
  _QWORD *v124; // rdi
  __int64 v125; // [rsp+8h] [rbp-78h]
  __int64 v126; // [rsp+10h] [rbp-70h]
  __int64 v127; // [rsp+18h] [rbp-68h]
  _QWORD *v128; // [rsp+18h] [rbp-68h]
  __int64 v129; // [rsp+18h] [rbp-68h]
  __int64 *v130; // [rsp+20h] [rbp-60h]
  unsigned __int64 v131; // [rsp+20h] [rbp-60h]
  unsigned __int64 v132; // [rsp+20h] [rbp-60h]
  __int64 v133; // [rsp+20h] [rbp-60h]
  __int64 v134; // [rsp+20h] [rbp-60h]
  int v135; // [rsp+20h] [rbp-60h]
  __int64 v136; // [rsp+20h] [rbp-60h]
  unsigned int v137; // [rsp+20h] [rbp-60h]
  unsigned __int64 v138; // [rsp+20h] [rbp-60h]
  _QWORD *v139; // [rsp+20h] [rbp-60h]
  _QWORD ***v140; // [rsp+28h] [rbp-58h] BYREF
  __int64 v141[2]; // [rsp+30h] [rbp-50h] BYREF
  char v142; // [rsp+40h] [rbp-40h]
  char v143; // [rsp+41h] [rbp-3Fh]

  v5 = a2 + 1;
  v8 = (_QWORD *)a2[2];
  v140 = a1;
  if ( !v8 )
    goto LABEL_21;
  v10 = a2 + 1;
  v11 = v8;
  do
  {
    while ( 1 )
    {
      v12 = v11[2];
      v13 = v11[3];
      if ( v11[4] >= (unsigned __int64)a1 )
        break;
      v11 = (_QWORD *)v11[3];
      if ( !v13 )
        goto LABEL_6;
    }
    v10 = v11;
    v11 = (_QWORD *)v11[2];
  }
  while ( v12 );
LABEL_6:
  if ( v5 == v10 || v10[4] > (unsigned __int64)a1 )
  {
LABEL_21:
    v25 = *((_BYTE *)a1 + 16);
    if ( !a4 )
    {
      if ( v25 != 62 )
        goto LABEL_23;
      if ( !sub_1CAD1F0((__int64)a1) )
      {
        v8 = (_QWORD *)a2[2];
LABEL_23:
        if ( v8 )
        {
          v26 = v5;
          do
          {
            while ( 1 )
            {
              v27 = v8[2];
              v28 = v8[3];
              if ( v8[4] >= (unsigned __int64)v140 )
                break;
              v8 = (_QWORD *)v8[3];
              if ( !v28 )
                goto LABEL_28;
            }
            v26 = v8;
            v8 = (_QWORD *)v8[2];
          }
          while ( v27 );
LABEL_28:
          if ( v5 != v26 && v26[4] <= (unsigned __int64)v140 )
            goto LABEL_35;
        }
        else
        {
          v26 = v5;
        }
        v29 = sub_22077B0(48);
        v30 = (unsigned __int64)v140;
        v31 = v26;
        *(_QWORD *)(v29 + 40) = 0;
        v26 = (_QWORD *)v29;
        *(_QWORD *)(v29 + 32) = v30;
        v32 = sub_1C9E930(a2, v31, (unsigned __int64 *)(v29 + 32));
        v34 = v32;
        if ( v33 )
        {
          if ( v32 || v5 == v33 )
          {
LABEL_33:
            v35 = 1;
LABEL_34:
            sub_220F040(v35, v26, v33, v5);
            ++a2[5];
            goto LABEL_35;
          }
LABEL_94:
          v35 = v30 < v33[4];
          goto LABEL_34;
        }
        goto LABEL_110;
      }
      v57 = (_QWORD *)a2[2];
      v23 = (__int64)*(a1 - 3);
      if ( v57 )
      {
        v53 = (unsigned __int64)v140;
        v44 = v5;
        do
        {
          if ( v57[4] < (unsigned __int64)v140 )
          {
            v57 = (_QWORD *)v57[3];
          }
          else
          {
            v44 = v57;
            v57 = (_QWORD *)v57[2];
          }
        }
        while ( v57 );
        goto LABEL_95;
      }
LABEL_135:
      v44 = v5;
      goto LABEL_97;
    }
    if ( v25 > 0x17u )
    {
      if ( v25 == 62 )
      {
        v23 = (__int64)*(a1 - 3);
        if ( !sub_1642F90(*(_QWORD *)v23, 32) )
        {
          v78 = sub_1644900(**a1, 0x20u);
          v143 = 1;
          v133 = v78;
          v141[0] = (__int64)"newSExt";
          v142 = 3;
          v79 = sub_1648A60(56, 1u);
          v80 = v79;
          if ( v79 )
            sub_15FC810((__int64)v79, v23, v133, (__int64)v141, (__int64)a1);
          v23 = (__int64)v80;
        }
        v74 = (_QWORD *)a2[2];
        if ( v74 )
        {
          v43 = (unsigned __int64)v140;
          v44 = v5;
          do
          {
            if ( v74[4] < (unsigned __int64)v140 )
            {
              v74 = (_QWORD *)v74[3];
            }
            else
            {
              v44 = v74;
              v74 = (_QWORD *)v74[2];
            }
          }
          while ( v74 );
LABEL_103:
          if ( v5 != v44 && v44[4] <= v43 )
            goto LABEL_102;
LABEL_105:
          v69 = sub_22077B0(48);
          v70 = (unsigned __int64)v140;
          v71 = v44;
          v44 = (_QWORD *)v69;
          *(_QWORD *)(v69 + 32) = v140;
          v132 = v70;
          *(_QWORD *)(v69 + 40) = 0;
          v72 = sub_1C9E930(a2, v71, (unsigned __int64 *)(v69 + 32));
          v66 = v72;
          if ( v65 )
          {
            if ( v5 == v65 )
              goto LABEL_100;
            v67 = v132;
            if ( v72 )
              goto LABEL_100;
            goto LABEL_108;
          }
LABEL_118:
          v75 = v44;
          v44 = v66;
          j_j___libc_free_0(v75, 48);
          goto LABEL_102;
        }
LABEL_182:
        v44 = v5;
        goto LABEL_105;
      }
      if ( v25 != 61 )
      {
        if ( (unsigned int)v25 - 35 <= 0x11 )
        {
          if ( dword_4FBE700 == 1 && a3 )
          {
            if ( v25 > 0x2Fu || (v122 = 0x80A800000000LL, !_bittest64(&v122, v25)) )
            {
              if ( (unsigned __int8)(v25 - 47) > 2u )
                goto LABEL_185;
              goto LABEL_43;
            }
            if ( !sub_15F2380((__int64)a1) )
              goto LABEL_185;
            v25 = *((_BYTE *)a1 + 16);
          }
          if ( (unsigned __int8)(v25 - 47) > 2u )
          {
            if ( (v25 & 0xFB) != 0x23 && v25 != 37 )
              goto LABEL_185;
            goto LABEL_47;
          }
LABEL_43:
          v36 = (__int64)*(a1 - 3);
          if ( *(_BYTE *)(v36 + 16) == 13 )
          {
            v37 = *(_DWORD *)(v36 + 32);
            v38 = *(__int64 **)(v36 + 24);
            v39 = v37 > 0x40
                ? *v38
                : (__int64)((_QWORD)v38 << (64 - (unsigned __int8)v37)) >> (64 - (unsigned __int8)v37);
            if ( v39 > 31 )
              goto LABEL_185;
          }
LABEL_47:
          v40 = sub_1CADBF0(*(a1 - 6), a2, a3, 1);
          if ( v40 )
          {
            v130 = (__int64 *)v40;
            v41 = sub_1CADBF0(*(a1 - 3), a2, a3, 1);
            v23 = v41;
            if ( v41 )
            {
              v143 = 1;
              v141[0] = (__int64)"newBI";
              v142 = 3;
              v23 = sub_15FB440((unsigned int)*((unsigned __int8 *)a1 + 16) - 24, v130, v41, (__int64)v141, (__int64)a1);
              v42 = (_QWORD *)a2[2];
              if ( v42 )
              {
                v43 = (unsigned __int64)v140;
                v44 = v5;
                do
                {
                  if ( v42[4] < (unsigned __int64)v140 )
                  {
                    v42 = (_QWORD *)v42[3];
                  }
                  else
                  {
                    v44 = v42;
                    v42 = (_QWORD *)v42[2];
                  }
                }
                while ( v42 );
                goto LABEL_103;
              }
              goto LABEL_182;
            }
            v113 = (_QWORD *)a2[2];
            if ( v113 )
            {
              v114 = v5;
              do
              {
                if ( v113[4] < (unsigned __int64)v140 )
                {
                  v113 = (_QWORD *)v113[3];
                }
                else
                {
                  v114 = v113;
                  v113 = (_QWORD *)v113[2];
                }
              }
              while ( v113 );
              if ( v5 != v114 && v114[4] <= (unsigned __int64)v140 )
              {
LABEL_214:
                v114[5] = 0;
                return v23;
              }
            }
            else
            {
              v114 = v5;
            }
            v115 = sub_22077B0(48);
            v116 = (unsigned __int64)v140;
            v117 = v114;
            *(_QWORD *)(v115 + 40) = 0;
            v114 = (_QWORD *)v115;
            *(_QWORD *)(v115 + 32) = v116;
            v138 = v116;
            v118 = sub_1C9E930(a2, v117, (unsigned __int64 *)(v115 + 32));
            if ( v119 )
            {
              v120 = v5 == v119 || v118 != 0;
              if ( !v120 )
                v120 = v138 < v119[4];
              sub_220F040(v120, v114, v119, v5);
              ++a2[5];
            }
            else
            {
              v124 = v114;
              v114 = v118;
              j_j___libc_free_0(v124, 48);
            }
            goto LABEL_214;
          }
LABEL_185:
          v23 = 0;
          *sub_1C9EA30(a2, (unsigned __int64 *)&v140) = 0;
          return v23;
        }
        if ( v25 == 77 && dword_4FBE700 > 1 )
        {
          v143 = 1;
          v141[0] = (__int64)"newPhi";
          v142 = 3;
          v135 = *((_DWORD *)a1 + 5) & 0xFFFFFFF;
          v127 = sub_1644900(**a1, 0x20u);
          v84 = sub_1648B60(64);
          v23 = v84;
          if ( v84 )
          {
            sub_15F1EA0(v84, v127, 53, 0, 0, (__int64)a1);
            *(_DWORD *)(v23 + 56) = v135;
            sub_164B780(v23, v141);
            sub_1648880(v23, *(_DWORD *)(v23 + 56), 1);
          }
          v85 = (_QWORD *)a2[2];
          v86 = v5;
          if ( !v85 )
            goto LABEL_153;
          do
          {
            if ( v85[4] < (unsigned __int64)a1 )
            {
              v85 = (_QWORD *)v85[3];
            }
            else
            {
              v86 = v85;
              v85 = (_QWORD *)v85[2];
            }
          }
          while ( v85 );
          if ( v5 == v86 || v86[4] > (unsigned __int64)a1 )
          {
LABEL_153:
            v128 = v86;
            v87 = sub_22077B0(48);
            *(_QWORD *)(v87 + 32) = a1;
            *(_QWORD *)(v87 + 40) = 0;
            v136 = v87;
            v88 = sub_1C9E930(a2, v128, (unsigned __int64 *)(v87 + 32));
            if ( v89 )
            {
              v90 = v88 || v5 == v89 || (unsigned __int64)a1 < v89[4];
              sub_220F040(v90, v136, v89, v5);
              ++a2[5];
              v86 = (_QWORD *)v136;
            }
            else
            {
              v121 = v136;
              v139 = v88;
              j_j___libc_free_0(v121, 48);
              v86 = v139;
            }
          }
          v86[5] = v23;
          if ( (*((_DWORD *)a1 + 5) & 0xFFFFFFF) == 0 )
            return v23;
          v91 = a3;
          v92 = 0;
          v137 = v91;
          v129 = 8LL * (*((_DWORD *)a1 + 5) & 0xFFFFFFF);
          while ( 1 )
          {
            v93 = (*((_BYTE *)a1 + 23) & 0x40) != 0 ? *(a1 - 1) : &a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
            v94 = sub_1CADBF0(v93[3 * v92 / 8], a2, v137, 1);
            if ( !v94 )
              break;
            if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
              v97 = (_QWORD ***)*(a1 - 1);
            else
              v97 = &a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
            v98 = (__int64)(&v97[3 * *((unsigned int *)a1 + 14)])[v92 / 8 + 1];
            v99 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
            if ( v99 == *(_DWORD *)(v23 + 56) )
            {
              v125 = v98;
              v126 = v94;
              sub_15F55D0(v23, (__int64)a2, v94, v98, v95, v96);
              v98 = v125;
              v94 = v126;
              v99 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
            }
            v100 = (v99 + 1) & 0xFFFFFFF;
            v101 = v100 | *(_DWORD *)(v23 + 20) & 0xF0000000;
            *(_DWORD *)(v23 + 20) = v101;
            if ( (v101 & 0x40000000) != 0 )
              v102 = *(_QWORD *)(v23 - 8);
            else
              v102 = v23 - 24 * v100;
            v103 = (__int64 *)(v102 + 24LL * (unsigned int)(v100 - 1));
            if ( *v103 )
            {
              v104 = v103[1];
              v105 = v103[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v105 = v104;
              if ( v104 )
                *(_QWORD *)(v104 + 16) = v105 | *(_QWORD *)(v104 + 16) & 3LL;
            }
            *v103 = v94;
            v106 = *(_QWORD *)(v94 + 8);
            v103[1] = v106;
            if ( v106 )
              *(_QWORD *)(v106 + 16) = (unsigned __int64)(v103 + 1) | *(_QWORD *)(v106 + 16) & 3LL;
            v103[2] = v103[2] & 3 | (v94 + 8);
            *(_QWORD *)(v94 + 8) = v103;
            v107 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
            if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
              v108 = *(_QWORD *)(v23 - 8);
            else
              v108 = v23 - 24 * v107;
            v92 += 8LL;
            *(_QWORD *)(v108 + 8LL * (unsigned int)(v107 - 1) + 24LL * *(unsigned int *)(v23 + 56) + 8) = v98;
            if ( v92 == v129 )
              return v23;
          }
          v109 = (_QWORD *)a2[2];
          v26 = v5;
          if ( v109 )
          {
            do
            {
              if ( v109[4] < (unsigned __int64)a1 )
              {
                v109 = (_QWORD *)v109[3];
              }
              else
              {
                v26 = v109;
                v109 = (_QWORD *)v109[2];
              }
            }
            while ( v109 );
            if ( v5 != v26 && v26[4] <= (unsigned __int64)a1 )
            {
LABEL_35:
              v26[5] = 0;
              return 0;
            }
          }
          v110 = sub_22077B0(48);
          v111 = v26;
          *(_QWORD *)(v110 + 32) = a1;
          v26 = (_QWORD *)v110;
          *(_QWORD *)(v110 + 40) = 0;
          v112 = sub_1C9E930(a2, v111, (unsigned __int64 *)(v110 + 32));
          if ( !v33 )
          {
            v123 = v26;
            v26 = v112;
            j_j___libc_free_0(v123, 48);
            goto LABEL_35;
          }
          if ( !v112 && v5 != v33 )
          {
            v35 = (unsigned __int64)a1 < v33[4];
            goto LABEL_34;
          }
          goto LABEL_33;
        }
        goto LABEL_83;
      }
      v23 = (__int64)*(a1 - 3);
      if ( !sub_1642F90(*(_QWORD *)v23, 32) )
      {
        v81 = sub_1644900(**a1, 0x20u);
        v143 = 1;
        v134 = v81;
        v141[0] = (__int64)"newZExt";
        v142 = 3;
        v82 = sub_1648A60(56, 1u);
        v83 = v82;
        if ( v82 )
          sub_15FC690((__int64)v82, v23, v134, (__int64)v141, (__int64)a1);
        v23 = (__int64)v83;
      }
      v76 = (_QWORD *)a2[2];
      if ( !v76 )
        goto LABEL_135;
      v53 = (unsigned __int64)v140;
      v44 = v5;
      do
      {
        if ( v76[4] < (unsigned __int64)v140 )
        {
          v76 = (_QWORD *)v76[3];
        }
        else
        {
          v44 = v76;
          v76 = (_QWORD *)v76[2];
        }
      }
      while ( v76 );
LABEL_95:
      if ( v5 != v44 && v44[4] <= v53 )
        goto LABEL_102;
LABEL_97:
      v61 = sub_22077B0(48);
      v62 = (unsigned __int64)v140;
      v63 = v44;
      v44 = (_QWORD *)v61;
      *(_QWORD *)(v61 + 32) = v140;
      v131 = v62;
      *(_QWORD *)(v61 + 40) = 0;
      v64 = sub_1C9E930(a2, v63, (unsigned __int64 *)(v61 + 32));
      v66 = v64;
      if ( v65 )
      {
        if ( v64 || (v67 = v131, v5 == v65) )
        {
LABEL_100:
          v68 = 1;
LABEL_101:
          sub_220F040(v68, v44, v65, v5);
          ++a2[5];
LABEL_102:
          v44[5] = v23;
          return v23;
        }
LABEL_108:
        v68 = v67 < v65[4];
        goto LABEL_101;
      }
      goto LABEL_118;
    }
    if ( v25 != 13 )
    {
LABEL_83:
      v26 = v5;
      if ( !v8 )
        goto LABEL_91;
      do
      {
        if ( v8[4] < (unsigned __int64)a1 )
        {
          v8 = (_QWORD *)v8[3];
        }
        else
        {
          v26 = v8;
          v8 = (_QWORD *)v8[2];
        }
      }
      while ( v8 );
      goto LABEL_89;
    }
    v45 = *((_DWORD *)a1 + 8);
    v46 = (unsigned __int64)a1[3];
    v47 = 1LL << ((unsigned __int8)v45 - 1);
    if ( v45 > 0x40 )
    {
      v48 = (__int64)(a1 + 3);
      if ( (*(_QWORD *)(v46 + 8LL * ((v45 - 1) >> 6)) & v47) != 0 )
        v49 = sub_16A5810(v48);
      else
        v49 = sub_16A57B0(v48);
      if ( v49 > 0x20 )
      {
        v50 = *(_QWORD *)v46;
LABEL_61:
        v51 = sub_1644900(**a1, 0x20u);
        v23 = sub_159C470(v51, v50, 0);
        v52 = (_QWORD *)a2[2];
        if ( !v52 )
          goto LABEL_135;
        v53 = (unsigned __int64)v140;
        v44 = v5;
        do
        {
          if ( v52[4] < (unsigned __int64)v140 )
          {
            v52 = (_QWORD *)v52[3];
          }
          else
          {
            v44 = v52;
            v52 = (_QWORD *)v52[2];
          }
        }
        while ( v52 );
        goto LABEL_95;
      }
LABEL_129:
      v26 = v5;
      if ( !v8 )
        goto LABEL_91;
      do
      {
        if ( v8[4] < (unsigned __int64)a1 )
        {
          v8 = (_QWORD *)v8[3];
        }
        else
        {
          v26 = v8;
          v8 = (_QWORD *)v8[2];
        }
      }
      while ( v8 );
LABEL_89:
      if ( v5 != v26 && v26[4] <= (unsigned __int64)a1 )
        goto LABEL_35;
LABEL_91:
      v58 = sub_22077B0(48);
      v30 = (unsigned __int64)v140;
      v59 = v26;
      *(_QWORD *)(v58 + 40) = 0;
      v26 = (_QWORD *)v58;
      *(_QWORD *)(v58 + 32) = v30;
      v60 = sub_1C9E930(a2, v59, (unsigned __int64 *)(v58 + 32));
      v34 = v60;
      if ( v33 )
      {
        if ( v5 == v33 || v60 )
          goto LABEL_33;
        goto LABEL_94;
      }
LABEL_110:
      v73 = v26;
      v26 = v34;
      j_j___libc_free_0(v73, 48);
      goto LABEL_35;
    }
    if ( (v47 & v46) != 0 )
    {
      if ( v46 << (64 - (unsigned __int8)v45) == -1 )
        goto LABEL_72;
      _BitScanReverse64(&v55, ~(v46 << (64 - (unsigned __int8)v45)));
      v56 = v55 ^ 0x3F;
    }
    else
    {
      v56 = *((_DWORD *)a1 + 8);
      if ( v46 )
      {
        _BitScanReverse64(&v77, v46);
        v56 = v45 - 64 + (v77 ^ 0x3F);
      }
    }
    if ( v56 <= 0x20 )
      goto LABEL_129;
LABEL_72:
    v50 = (__int64)(v46 << (64 - (unsigned __int8)v45)) >> (64 - (unsigned __int8)v45);
    goto LABEL_61;
  }
  v14 = v5;
  do
  {
    while ( 1 )
    {
      v15 = v8[2];
      v16 = v8[3];
      if ( v8[4] >= (unsigned __int64)a1 )
        break;
      v8 = (_QWORD *)v8[3];
      if ( !v16 )
        goto LABEL_12;
    }
    v14 = v8;
    v8 = (_QWORD *)v8[2];
  }
  while ( v15 );
LABEL_12:
  if ( v5 == v14 || v14[4] > (unsigned __int64)a1 )
  {
    v17 = sub_22077B0(48);
    v18 = (unsigned __int64)v140;
    v19 = v14;
    *(_QWORD *)(v17 + 40) = 0;
    v14 = (_QWORD *)v17;
    *(_QWORD *)(v17 + 32) = v18;
    v20 = sub_1C9E930(a2, v19, (unsigned __int64 *)(v17 + 32));
    if ( v21 )
    {
      v22 = v5 == v21 || v20 || v18 < v21[4];
      sub_220F040(v22, v14, v21, v5);
      ++a2[5];
    }
    else
    {
      v54 = v14;
      v14 = v20;
      j_j___libc_free_0(v54, 48);
    }
  }
  return v14[5];
}
