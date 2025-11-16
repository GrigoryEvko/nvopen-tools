// Function: sub_2CF65B0
// Address: 0x2cf65b0
//
__int64 __fastcall sub_2CF65B0(unsigned __int8 *a1, _QWORD *a2, unsigned __int8 a3, char a4)
{
  _QWORD *v5; // r13
  _QWORD *v8; // r15
  _QWORD *v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // r15
  __int64 v19; // rsi
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  char v22; // di
  __int64 v23; // r15
  unsigned __int64 v25; // rdx
  __int64 v26; // r14
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // r15
  __int64 v31; // rsi
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  _QWORD *v34; // rbx
  char v35; // di
  __int64 v36; // rdx
  __int64 *v37; // rax
  unsigned int v38; // edx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  _QWORD *v42; // rax
  unsigned __int64 v43; // rsi
  __int64 v44; // r14
  unsigned int v45; // edx
  unsigned __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rdi
  unsigned int v49; // eax
  __int64 v50; // r14
  __int64 v51; // rax
  _QWORD *v52; // rax
  unsigned __int64 v53; // rdi
  char v54; // cl
  unsigned __int64 v55; // rax
  __int64 v56; // rax
  _QWORD *v57; // rax
  unsigned __int64 v58; // rsi
  __int64 v59; // rax
  __int64 v60; // rsi
  _QWORD *v61; // rax
  __int64 v62; // rax
  unsigned __int64 v63; // rcx
  __int64 v64; // rsi
  _QWORD *v65; // rax
  _QWORD *v66; // rdx
  _QWORD *v67; // rbx
  unsigned __int64 v68; // rcx
  char v69; // di
  __int64 v70; // rax
  unsigned __int64 v71; // rcx
  __int64 v72; // rsi
  _QWORD *v73; // rax
  unsigned __int64 v74; // rdi
  _QWORD *v75; // rax
  _QWORD *v76; // rax
  unsigned __int64 v77; // rdi
  unsigned int v78; // eax
  unsigned __int64 v79; // rax
  _QWORD **v80; // rax
  __int64 v81; // rbx
  __int64 v82; // rax
  _QWORD *v83; // rax
  _QWORD *v84; // r14
  _QWORD **v85; // rax
  __int64 v86; // rbx
  __int64 v87; // rax
  _QWORD *v88; // rax
  _QWORD *v89; // r14
  __int64 v90; // rax
  _QWORD *v91; // rax
  _QWORD *v92; // r8
  __int64 v93; // rax
  _QWORD *v94; // rax
  _QWORD *v95; // rdx
  char v96; // di
  unsigned int v97; // edi
  __int64 v98; // r14
  __int64 v99; // rdx
  __int64 v100; // rcx
  int v101; // eax
  int v102; // eax
  unsigned int v103; // esi
  __int64 v104; // rax
  __int64 v105; // rsi
  __int64 v106; // rsi
  _QWORD *v107; // rax
  __int64 v108; // rax
  __int64 v109; // rsi
  _QWORD *v110; // rax
  _QWORD *v111; // rax
  __int64 v112; // r14
  __int64 v113; // rax
  unsigned __int64 v114; // r8
  __int64 v115; // rsi
  _QWORD *v116; // rax
  _QWORD *v117; // rdx
  _QWORD *v118; // rbx
  char v119; // al
  unsigned __int64 v120; // rdi
  __int64 v121; // rax
  bool v122; // al
  unsigned __int64 v123; // rdi
  unsigned __int64 v124; // rdi
  __int64 v125; // [rsp+8h] [rbp-88h]
  __int64 v126; // [rsp+10h] [rbp-80h]
  __int64 v127; // [rsp+18h] [rbp-78h]
  __int64 v128; // [rsp+18h] [rbp-78h]
  __int64 v129; // [rsp+18h] [rbp-78h]
  __int64 v130; // [rsp+20h] [rbp-70h]
  __int64 *v131; // [rsp+20h] [rbp-70h]
  unsigned __int64 v132; // [rsp+20h] [rbp-70h]
  unsigned __int64 v133; // [rsp+20h] [rbp-70h]
  __int64 v134; // [rsp+20h] [rbp-70h]
  __int64 v135; // [rsp+20h] [rbp-70h]
  int v136; // [rsp+20h] [rbp-70h]
  __int64 v137; // [rsp+20h] [rbp-70h]
  unsigned __int64 v138; // [rsp+20h] [rbp-70h]
  _QWORD *v139; // [rsp+20h] [rbp-70h]
  unsigned __int8 v140; // [rsp+20h] [rbp-70h]
  unsigned __int8 *v141; // [rsp+28h] [rbp-68h] BYREF
  const char *v142[4]; // [rsp+30h] [rbp-60h] BYREF
  char v143; // [rsp+50h] [rbp-40h]
  char v144; // [rsp+51h] [rbp-3Fh]

  v5 = a2 + 1;
  v8 = (_QWORD *)a2[2];
  v141 = a1;
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
    v25 = *a1;
    if ( !a4 )
    {
      if ( (_BYTE)v25 != 69 )
        goto LABEL_23;
      if ( !sub_2CF5CA0((__int64)a1) )
      {
        v8 = (_QWORD *)a2[2];
LABEL_23:
        if ( v8 )
        {
          v26 = (__int64)v5;
          do
          {
            while ( 1 )
            {
              v27 = v8[2];
              v28 = v8[3];
              if ( v8[4] >= (unsigned __int64)v141 )
                break;
              v8 = (_QWORD *)v8[3];
              if ( !v28 )
                goto LABEL_28;
            }
            v26 = (__int64)v8;
            v8 = (_QWORD *)v8[2];
          }
          while ( v27 );
LABEL_28:
          if ( v5 != (_QWORD *)v26 && *(_QWORD *)(v26 + 32) <= (unsigned __int64)v141 )
            goto LABEL_35;
        }
        else
        {
          v26 = (__int64)v5;
        }
        goto LABEL_30;
      }
      v57 = (_QWORD *)a2[2];
      v23 = *((_QWORD *)a1 - 4);
      if ( v57 )
      {
        v58 = (unsigned __int64)v141;
        v44 = (__int64)v5;
        do
        {
          if ( v57[4] < (unsigned __int64)v141 )
          {
            v57 = (_QWORD *)v57[3];
          }
          else
          {
            v44 = (__int64)v57;
            v57 = (_QWORD *)v57[2];
          }
        }
        while ( v57 );
        goto LABEL_105;
      }
      goto LABEL_159;
    }
    if ( (unsigned __int8)v25 > 0x1Cu )
    {
      if ( (_BYTE)v25 == 69 )
      {
        v23 = *((_QWORD *)a1 - 4);
        if ( !sub_BCAC40(*(_QWORD *)(v23 + 8), 32) )
        {
          v80 = (_QWORD **)*((_QWORD *)a1 + 1);
          v81 = (__int64)(a1 + 24);
          v82 = sub_BCCE00(*v80, 0x20u);
          v144 = 1;
          v134 = v82;
          v142[0] = "newSExt";
          v143 = 3;
          v83 = sub_BD2C40(72, 1u);
          v84 = v83;
          if ( v83 )
            sub_B51650((__int64)v83, v23, v134, (__int64)v142, v81, 0);
          v23 = (__int64)v84;
        }
        v75 = (_QWORD *)a2[2];
        if ( v75 )
        {
          v58 = (unsigned __int64)v141;
          v44 = (__int64)v5;
          do
          {
            if ( v75[4] < (unsigned __int64)v141 )
            {
              v75 = (_QWORD *)v75[3];
            }
            else
            {
              v44 = (__int64)v75;
              v75 = (_QWORD *)v75[2];
            }
          }
          while ( v75 );
LABEL_105:
          if ( v5 != (_QWORD *)v44 && *(_QWORD *)(v44 + 32) <= v58 )
            goto LABEL_104;
LABEL_107:
          v70 = sub_22077B0(0x30u);
          v71 = (unsigned __int64)v141;
          v72 = v44;
          v44 = v70;
          *(_QWORD *)(v70 + 32) = v141;
          v133 = v71;
          *(_QWORD *)(v70 + 40) = 0;
          v73 = sub_2CE6960(a2, v72, (unsigned __int64 *)(v70 + 32));
          v67 = v73;
          if ( v66 )
          {
            if ( v5 != v66 )
            {
              v68 = v133;
              if ( !v73 )
              {
LABEL_110:
                v69 = v68 < v66[4];
                goto LABEL_103;
              }
            }
LABEL_102:
            v69 = 1;
LABEL_103:
            sub_220F040(v69, v44, v66, v5);
            ++a2[5];
LABEL_104:
            *(_QWORD *)(v44 + 40) = v23;
            return v23;
          }
          goto LABEL_127;
        }
LABEL_159:
        v44 = (__int64)v5;
        goto LABEL_107;
      }
      if ( (_BYTE)v25 != 68 )
      {
        if ( (unsigned int)(unsigned __int8)v25 - 42 <= 0x11 )
        {
          if ( (_DWORD)qword_50147A8 == 1 && a3 )
          {
            if ( (unsigned __int8)v25 > 0x36u || (v121 = 0x40540000000000LL, !_bittest64(&v121, v25)) )
            {
              if ( (unsigned __int8)(v25 - 54) > 2u )
                goto LABEL_182;
              goto LABEL_43;
            }
            v140 = *a1;
            v122 = sub_B44900((__int64)a1);
            LOBYTE(v25) = v140;
            if ( !v122 )
              goto LABEL_182;
          }
          if ( (unsigned __int8)(v25 - 54) > 2u )
          {
            if ( (v25 & 0xFB) != 0x2A && (_BYTE)v25 != 44 )
              goto LABEL_182;
            goto LABEL_48;
          }
LABEL_43:
          v36 = *((_QWORD *)a1 - 4);
          if ( *(_BYTE *)v36 == 17 )
          {
            v37 = *(__int64 **)(v36 + 24);
            v38 = *(_DWORD *)(v36 + 32);
            if ( v38 > 0x40 )
            {
              v39 = *v37;
LABEL_47:
              if ( v39 > 31 )
                goto LABEL_182;
              goto LABEL_48;
            }
            if ( v38 )
            {
              v39 = (__int64)((_QWORD)v37 << (64 - (unsigned __int8)v38)) >> (64 - (unsigned __int8)v38);
              goto LABEL_47;
            }
          }
LABEL_48:
          v40 = sub_2CF65B0(*((_QWORD *)a1 - 8), a2, a3, 1);
          if ( v40 )
          {
            v130 = v40;
            v41 = sub_2CF65B0(*((_QWORD *)a1 - 4), a2, a3, 1);
            v23 = v41;
            if ( v41 )
            {
              v144 = 1;
              v143 = 3;
              v142[0] = "newBI";
              v23 = sub_B504D0((unsigned int)*a1 - 29, v130, v41, (__int64)v142, (__int64)(a1 + 24), 0);
              v42 = (_QWORD *)a2[2];
              if ( v42 )
              {
                v43 = (unsigned __int64)v141;
                v44 = (__int64)v5;
                do
                {
                  if ( v42[4] < (unsigned __int64)v141 )
                  {
                    v42 = (_QWORD *)v42[3];
                  }
                  else
                  {
                    v44 = (__int64)v42;
                    v42 = (_QWORD *)v42[2];
                  }
                }
                while ( v42 );
                goto LABEL_97;
              }
LABEL_146:
              v44 = (__int64)v5;
              goto LABEL_99;
            }
            v111 = (_QWORD *)a2[2];
            if ( v111 )
            {
              v112 = (__int64)v5;
              do
              {
                if ( v111[4] < (unsigned __int64)v141 )
                {
                  v111 = (_QWORD *)v111[3];
                }
                else
                {
                  v112 = (__int64)v111;
                  v111 = (_QWORD *)v111[2];
                }
              }
              while ( v111 );
              if ( v5 != (_QWORD *)v112 && *(_QWORD *)(v112 + 32) <= (unsigned __int64)v141 )
              {
LABEL_211:
                *(_QWORD *)(v112 + 40) = 0;
                return v23;
              }
            }
            else
            {
              v112 = (__int64)v5;
            }
            v113 = sub_22077B0(0x30u);
            v114 = (unsigned __int64)v141;
            v115 = v112;
            *(_QWORD *)(v113 + 40) = 0;
            v112 = v113;
            *(_QWORD *)(v113 + 32) = v114;
            v138 = v114;
            v116 = sub_2CE6960(a2, v115, (unsigned __int64 *)(v113 + 32));
            v118 = v116;
            if ( v117 )
            {
              v119 = v116 != 0 || v5 == v117;
              if ( v118 == 0 && v5 != v117 )
                v119 = v138 < v117[4];
              sub_220F040(v119, v112, v117, v5);
              ++a2[5];
            }
            else
            {
              v124 = v112;
              v112 = (__int64)v116;
              j_j___libc_free_0(v124);
            }
            goto LABEL_211;
          }
LABEL_182:
          v23 = 0;
          *(_QWORD *)sub_2CE6A60(a2, (unsigned __int64 *)&v141) = 0;
          return v23;
        }
        if ( (int)qword_50147A8 > 1 && (_BYTE)v25 == 84 )
        {
          v144 = 1;
          v142[0] = "newPhi";
          v143 = 3;
          v136 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
          v127 = sub_BCCE00(**((_QWORD ***)a1 + 1), 0x20u);
          v90 = sub_BD2DA0(80);
          v23 = v90;
          if ( v90 )
          {
            sub_B44260(v90, v127, 55, 0x8000000u, (__int64)(a1 + 24), 0);
            *(_DWORD *)(v23 + 72) = v136;
            sub_BD6B50((unsigned __int8 *)v23, v142);
            sub_BD2A10(v23, *(_DWORD *)(v23 + 72), 1);
          }
          v91 = (_QWORD *)a2[2];
          v92 = v5;
          if ( !v91 )
            goto LABEL_162;
          do
          {
            if ( v91[4] < (unsigned __int64)a1 )
            {
              v91 = (_QWORD *)v91[3];
            }
            else
            {
              v92 = v91;
              v91 = (_QWORD *)v91[2];
            }
          }
          while ( v91 );
          if ( v5 == v92 || v92[4] > (unsigned __int64)a1 )
          {
LABEL_162:
            v128 = (__int64)v92;
            v93 = sub_22077B0(0x30u);
            *(_QWORD *)(v93 + 32) = a1;
            *(_QWORD *)(v93 + 40) = 0;
            v137 = v93;
            v94 = sub_2CE6960(a2, v128, (unsigned __int64 *)(v93 + 32));
            if ( v95 )
            {
              v96 = v94 || v5 == v95 || (unsigned __int64)a1 < v95[4];
              sub_220F040(v96, v137, v95, v5);
              ++a2[5];
              v92 = (_QWORD *)v137;
            }
            else
            {
              v120 = v137;
              v139 = v94;
              j_j___libc_free_0(v120);
              v92 = v139;
            }
          }
          v92[5] = v23;
          if ( (*((_DWORD *)a1 + 1) & 0x7FFFFFF) == 0 )
            return v23;
          v97 = a3;
          v98 = 0;
          v129 = 8LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
          while ( 1 )
          {
            v99 = sub_2CF65B0(*(_QWORD *)(*((_QWORD *)a1 - 1) + 4 * v98), a2, v97, 1);
            if ( !v99 )
              break;
            v100 = *(_QWORD *)(*((_QWORD *)a1 - 1) + 32LL * *((unsigned int *)a1 + 18) + v98);
            v101 = *(_DWORD *)(v23 + 4) & 0x7FFFFFF;
            if ( v101 == *(_DWORD *)(v23 + 72) )
            {
              v125 = *(_QWORD *)(*((_QWORD *)a1 - 1) + 32LL * *((unsigned int *)a1 + 18) + v98);
              v126 = v99;
              sub_B48D90(v23);
              v100 = v125;
              v99 = v126;
              v101 = *(_DWORD *)(v23 + 4) & 0x7FFFFFF;
            }
            v102 = (v101 + 1) & 0x7FFFFFF;
            v103 = v102 | *(_DWORD *)(v23 + 4) & 0xF8000000;
            v104 = *(_QWORD *)(v23 - 8) + 32LL * (unsigned int)(v102 - 1);
            *(_DWORD *)(v23 + 4) = v103;
            if ( *(_QWORD *)v104 )
            {
              v105 = *(_QWORD *)(v104 + 8);
              **(_QWORD **)(v104 + 16) = v105;
              if ( v105 )
                *(_QWORD *)(v105 + 16) = *(_QWORD *)(v104 + 16);
            }
            *(_QWORD *)v104 = v99;
            v106 = *(_QWORD *)(v99 + 16);
            *(_QWORD *)(v104 + 8) = v106;
            if ( v106 )
              *(_QWORD *)(v106 + 16) = v104 + 8;
            *(_QWORD *)(v104 + 16) = v99 + 16;
            v98 += 8;
            *(_QWORD *)(v99 + 16) = v104;
            *(_QWORD *)(*(_QWORD *)(v23 - 8)
                      + 32LL * *(unsigned int *)(v23 + 72)
                      + 8LL * ((*(_DWORD *)(v23 + 4) & 0x7FFFFFFu) - 1)) = v100;
            if ( v98 == v129 )
              return v23;
          }
          v107 = (_QWORD *)a2[2];
          v26 = (__int64)v5;
          if ( v107 )
          {
            do
            {
              if ( v107[4] < (unsigned __int64)a1 )
              {
                v107 = (_QWORD *)v107[3];
              }
              else
              {
                v26 = (__int64)v107;
                v107 = (_QWORD *)v107[2];
              }
            }
            while ( v107 );
            if ( v5 != (_QWORD *)v26 && *(_QWORD *)(v26 + 32) <= (unsigned __int64)a1 )
            {
LABEL_35:
              *(_QWORD *)(v26 + 40) = 0;
              return 0;
            }
          }
          v108 = sub_22077B0(0x30u);
          v109 = v26;
          *(_QWORD *)(v108 + 32) = a1;
          v26 = v108;
          *(_QWORD *)(v108 + 40) = 0;
          v110 = sub_2CE6960(a2, v109, (unsigned __int64 *)(v108 + 32));
          if ( !v33 )
          {
            v123 = v26;
            v26 = (__int64)v110;
            j_j___libc_free_0(v123);
            goto LABEL_35;
          }
          if ( !v110 && v5 != v33 )
          {
            v35 = (unsigned __int64)a1 < v33[4];
            goto LABEL_34;
          }
          goto LABEL_33;
        }
LABEL_91:
        v26 = (__int64)v5;
        if ( v8 )
        {
          do
          {
            if ( v8[4] < (unsigned __int64)a1 )
            {
              v8 = (_QWORD *)v8[3];
            }
            else
            {
              v26 = (__int64)v8;
              v8 = (_QWORD *)v8[2];
            }
          }
          while ( v8 );
          if ( v5 != (_QWORD *)v26 && *(_QWORD *)(v26 + 32) <= (unsigned __int64)a1 )
            goto LABEL_35;
        }
        v59 = sub_22077B0(0x30u);
        v30 = (unsigned __int64)v141;
        v60 = v26;
        *(_QWORD *)(v59 + 40) = 0;
        v26 = v59;
        *(_QWORD *)(v59 + 32) = v30;
        v61 = sub_2CE6960(a2, v60, (unsigned __int64 *)(v59 + 32));
        v34 = v61;
        if ( v33 )
        {
          if ( v5 == v33 || v61 )
            goto LABEL_33;
          goto LABEL_88;
        }
LABEL_111:
        v74 = v26;
        v26 = (__int64)v34;
        j_j___libc_free_0(v74);
        goto LABEL_35;
      }
      v23 = *((_QWORD *)a1 - 4);
      if ( !sub_BCAC40(*(_QWORD *)(v23 + 8), 32) )
      {
        v85 = (_QWORD **)*((_QWORD *)a1 + 1);
        v86 = (__int64)(a1 + 24);
        v87 = sub_BCCE00(*v85, 0x20u);
        v144 = 1;
        v135 = v87;
        v142[0] = "newZExt";
        v143 = 3;
        v88 = sub_BD2C40(72, 1u);
        v89 = v88;
        if ( v88 )
          sub_B515B0((__int64)v88, v23, v135, (__int64)v142, v86, 0);
        v23 = (__int64)v89;
      }
      v76 = (_QWORD *)a2[2];
      if ( !v76 )
        goto LABEL_146;
      v43 = (unsigned __int64)v141;
      v44 = (__int64)v5;
      do
      {
        if ( v76[4] < (unsigned __int64)v141 )
        {
          v76 = (_QWORD *)v76[3];
        }
        else
        {
          v44 = (__int64)v76;
          v76 = (_QWORD *)v76[2];
        }
      }
      while ( v76 );
LABEL_97:
      if ( v5 != (_QWORD *)v44 && *(_QWORD *)(v44 + 32) <= v43 )
        goto LABEL_104;
LABEL_99:
      v62 = sub_22077B0(0x30u);
      v63 = (unsigned __int64)v141;
      v64 = v44;
      v44 = v62;
      *(_QWORD *)(v62 + 32) = v141;
      v132 = v63;
      *(_QWORD *)(v62 + 40) = 0;
      v65 = sub_2CE6960(a2, v64, (unsigned __int64 *)(v62 + 32));
      v67 = v65;
      if ( v66 )
      {
        if ( !v65 )
        {
          v68 = v132;
          if ( v5 != v66 )
            goto LABEL_110;
        }
        goto LABEL_102;
      }
LABEL_127:
      v77 = v44;
      v44 = (__int64)v67;
      j_j___libc_free_0(v77);
      goto LABEL_104;
    }
    if ( (_BYTE)v25 != 17 )
      goto LABEL_91;
    v45 = *((_DWORD *)a1 + 8);
    v46 = *((_QWORD *)a1 + 3);
    v47 = 1LL << ((unsigned __int8)v45 - 1);
    if ( v45 > 0x40 )
    {
      v48 = (__int64)(a1 + 24);
      v131 = (__int64 *)*((_QWORD *)a1 + 3);
      if ( (*(_QWORD *)(v46 + 8LL * ((v45 - 1) >> 6)) & v47) != 0 )
        v49 = sub_C44500(v48);
      else
        v49 = sub_C444A0(v48);
      if ( v49 > 0x20 )
      {
        v50 = *v131;
LABEL_62:
        v51 = sub_BCCE00(**((_QWORD ***)a1 + 1), 0x20u);
        v23 = sub_ACD640(v51, v50, 0);
        v52 = (_QWORD *)a2[2];
        if ( !v52 )
          goto LABEL_146;
        v43 = (unsigned __int64)v141;
        v44 = (__int64)v5;
        do
        {
          if ( v52[4] < (unsigned __int64)v141 )
          {
            v52 = (_QWORD *)v52[3];
          }
          else
          {
            v44 = (__int64)v52;
            v52 = (_QWORD *)v52[2];
          }
        }
        while ( v52 );
        goto LABEL_97;
      }
LABEL_128:
      v26 = (__int64)v5;
      if ( v8 )
      {
        do
        {
          if ( v8[4] < (unsigned __int64)a1 )
          {
            v8 = (_QWORD *)v8[3];
          }
          else
          {
            v26 = (__int64)v8;
            v8 = (_QWORD *)v8[2];
          }
        }
        while ( v8 );
        if ( v5 != (_QWORD *)v26 && *(_QWORD *)(v26 + 32) <= (unsigned __int64)a1 )
          goto LABEL_35;
      }
LABEL_30:
      v29 = sub_22077B0(0x30u);
      v30 = (unsigned __int64)v141;
      v31 = v26;
      *(_QWORD *)(v29 + 40) = 0;
      v26 = v29;
      *(_QWORD *)(v29 + 32) = v30;
      v32 = sub_2CE6960(a2, v31, (unsigned __int64 *)(v29 + 32));
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
LABEL_88:
        v35 = v30 < v33[4];
        goto LABEL_34;
      }
      goto LABEL_111;
    }
    v50 = v46 & v47;
    if ( (v46 & v47) != 0 )
    {
      if ( !v45 )
        goto LABEL_128;
      v54 = 64 - v45;
      if ( v46 << (64 - (unsigned __int8)v45) == -1 )
      {
        v56 = -1;
        goto LABEL_74;
      }
      _BitScanReverse64(&v55, ~(v46 << (64 - (unsigned __int8)v45)));
      if ( (int)(v55 ^ 0x3F) <= 32 )
        goto LABEL_128;
    }
    else
    {
      v78 = *((_DWORD *)a1 + 8);
      if ( v46 )
      {
        _BitScanReverse64(&v79, v46);
        v78 = v45 - 64 + (v79 ^ 0x3F);
      }
      if ( v78 <= 0x20 )
        goto LABEL_128;
      if ( !v45 )
        goto LABEL_62;
    }
    v54 = 64 - v45;
    v56 = v46 << (64 - (unsigned __int8)v45);
LABEL_74:
    v50 = v56 >> v54;
    goto LABEL_62;
  }
  v14 = (__int64)v5;
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
    v14 = (__int64)v8;
    v8 = (_QWORD *)v8[2];
  }
  while ( v15 );
LABEL_12:
  if ( v5 == (_QWORD *)v14 || *(_QWORD *)(v14 + 32) > (unsigned __int64)a1 )
  {
    v17 = sub_22077B0(0x30u);
    v18 = (unsigned __int64)v141;
    v19 = v14;
    *(_QWORD *)(v17 + 40) = 0;
    v14 = v17;
    *(_QWORD *)(v17 + 32) = v18;
    v20 = sub_2CE6960(a2, v19, (unsigned __int64 *)(v17 + 32));
    if ( v21 )
    {
      v22 = v5 == v21 || v20 || v18 < v21[4];
      sub_220F040(v22, v14, v21, v5);
      ++a2[5];
    }
    else
    {
      v53 = v14;
      v14 = (__int64)v20;
      j_j___libc_free_0(v53);
    }
  }
  return *(_QWORD *)(v14 + 40);
}
