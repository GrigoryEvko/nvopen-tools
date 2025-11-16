// Function: sub_19C3980
// Address: 0x19c3980
//
__int64 __fastcall sub_19C3980(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v14; // r12
  double v16; // xmm4_8
  double v17; // xmm5_8
  __int64 v18; // rbx
  _QWORD *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r13
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 *v25; // rdi
  int v26; // eax
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rcx
  int v31; // r8d
  __int64 v32; // r9
  int v33; // r8d
  unsigned int v34; // r10d
  __int64 *v35; // rcx
  __int64 v36; // r13
  _QWORD *v37; // rdx
  unsigned int v38; // r10d
  __int64 *v39; // rcx
  __int64 v40; // r13
  _QWORD *v41; // rax
  __int64 *v42; // rsi
  __int64 v43; // r10
  __int64 v44; // rdx
  __int64 result; // rax
  __int64 v46; // r13
  __int64 v47; // rdx
  __int64 v48; // r9
  __int64 v49; // r10
  __int64 v50; // r14
  __int64 v51; // rax
  __int64 v52; // r13
  __int64 v53; // rax
  __int64 v54; // r9
  __int64 v55; // r11
  __int64 v56; // rdi
  unsigned int v57; // eax
  __int64 v58; // r8
  unsigned int v59; // ecx
  __int64 v60; // rax
  __int64 v61; // r10
  __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 v64; // rsi
  __int64 v65; // rdx
  __int64 v66; // rdi
  __int64 v67; // rdx
  unsigned __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // r13
  _QWORD *v72; // rdi
  _QWORD *v73; // rax
  _QWORD *v74; // rdi
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // r13
  __int64 v78; // rbx
  __int64 v79; // r14
  __int64 v80; // r10
  char v81; // di
  unsigned int v82; // esi
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // rcx
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 *v88; // rax
  __int64 v89; // rcx
  unsigned __int64 v90; // rdx
  __int64 v91; // rdx
  __int64 v92; // rax
  __int64 v93; // r11
  __int64 v94; // rax
  __int64 v95; // rsi
  unsigned int v96; // ecx
  __int64 *v97; // rdx
  __int64 v98; // r8
  __int64 v99; // r14
  _QWORD *v100; // rax
  __int64 v101; // r11
  _QWORD *v102; // r13
  int v103; // eax
  _BYTE *v104; // rsi
  unsigned int v105; // esi
  __int64 v106; // rdi
  unsigned int v107; // r10d
  _QWORD *v108; // rax
  __int64 v109; // rcx
  __int64 v110; // r14
  __int64 v111; // rdi
  _QWORD *v112; // rbx
  _QWORD *v113; // rax
  __int64 v114; // rax
  __int64 v115; // r14
  __int64 i; // rbx
  _QWORD *v117; // rax
  __int64 *v118; // rsi
  __int64 *v119; // rbx
  __int64 *j; // r14
  __int64 v121; // rdi
  int v122; // ecx
  int v123; // r11d
  int v124; // ecx
  int v125; // r11d
  _QWORD *v126; // rax
  int v127; // edx
  int v128; // r9d
  _QWORD *v129; // rdx
  int v130; // eax
  int v131; // eax
  int v132; // r14d
  int v133; // r14d
  __int64 v134; // r10
  _QWORD *v135; // rsi
  int v136; // edi
  unsigned int v137; // ecx
  __int64 v138; // r8
  int v139; // r14d
  int v140; // r14d
  __int64 v141; // r10
  unsigned int v142; // ecx
  _QWORD *v143; // r8
  int v144; // edi
  __int64 v145; // [rsp+8h] [rbp-B8h]
  __int64 v146; // [rsp+8h] [rbp-B8h]
  __int64 v147; // [rsp+8h] [rbp-B8h]
  __int64 v148; // [rsp+8h] [rbp-B8h]
  __int64 v149; // [rsp+8h] [rbp-B8h]
  __int64 v150; // [rsp+10h] [rbp-B0h]
  __int64 v151; // [rsp+10h] [rbp-B0h]
  __int64 v152; // [rsp+10h] [rbp-B0h]
  __int64 v153; // [rsp+10h] [rbp-B0h]
  __int64 v154; // [rsp+10h] [rbp-B0h]
  __int64 v155; // [rsp+10h] [rbp-B0h]
  int v156; // [rsp+10h] [rbp-B0h]
  unsigned int v157; // [rsp+10h] [rbp-B0h]
  __int64 v158; // [rsp+10h] [rbp-B0h]
  __int64 v159; // [rsp+18h] [rbp-A8h]
  __int64 v160; // [rsp+18h] [rbp-A8h]
  _QWORD *v161; // [rsp+18h] [rbp-A8h]
  __int64 *v162; // [rsp+20h] [rbp-A0h]
  _QWORD *v165; // [rsp+48h] [rbp-78h] BYREF
  __int64 *v166; // [rsp+50h] [rbp-70h] BYREF
  __int64 *v167; // [rsp+58h] [rbp-68h]
  __int64 *v168; // [rsp+60h] [rbp-60h]
  _QWORD v169[2]; // [rsp+70h] [rbp-50h] BYREF
  char v170; // [rsp+80h] [rbp-40h]
  char v171; // [rsp+81h] [rbp-3Fh]

  v14 = a4;
  v166 = 0;
  v167 = 0;
  v168 = 0;
  v162 = (__int64 *)sub_16498A0(a4);
  if ( !a5 )
  {
    if ( *(_BYTE *)(v14 + 16) == 13 && sub_1642F90(*(_QWORD *)v14, 1) )
    {
      v112 = *(_QWORD **)(v14 + 24);
      if ( *(_DWORD *)(v14 + 32) > 0x40u )
        v112 = (_QWORD *)*v112;
      v113 = (_QWORD *)sub_16498A0(v14);
      v114 = sub_1643320(v113);
      v14 = sub_159C470(v114, v112 == 0, 0);
      goto LABEL_110;
    }
    v18 = *(_QWORD *)(a3 + 8);
    if ( !v18 )
    {
LABEL_31:
      v44 = a2;
      goto LABEL_32;
    }
    while ( 1 )
    {
      v19 = sub_1648700(v18);
      if ( *((_BYTE *)v19 + 16) <= 0x17u )
        goto LABEL_30;
      v20 = v19[5];
      v165 = v19;
      if ( !sub_1377F70(a2 + 56, v20) )
        goto LABEL_30;
      v21 = (__int64)v165;
      if ( *((_BYTE *)v165 + 16) == 75 )
      {
        v22 = *((unsigned __int16 *)v165 + 9);
        BYTE1(v22) &= ~0x80u;
        if ( (unsigned int)(v22 - 32) <= 1 )
        {
          v23 = *(v165 - 3);
          v24 = *(v165 - 6);
          if ( v23 )
          {
            if ( a3 == v24 && v14 == v23 || v14 == v24 && a3 == v23 )
            {
              v25 = (__int64 *)sub_16498A0((__int64)v165);
              v26 = *(unsigned __int16 *)(v21 + 18);
              BYTE1(v26) &= ~0x80u;
              v27 = v26 == 32 ? sub_159C540(v25) : sub_159C4F0(v25);
              if ( v27 )
              {
                if ( *(_BYTE *)(v27 + 16) <= 0x17u )
                  goto LABEL_23;
                v28 = *(_QWORD *)(v27 + 40);
                v29 = v165[5];
                if ( v28 == v29 )
                  goto LABEL_23;
                v30 = *(_QWORD *)(a1 + 160);
                v31 = *(_DWORD *)(v30 + 24);
                if ( !v31 )
                  goto LABEL_23;
                v32 = *(_QWORD *)(v30 + 8);
                v33 = v31 - 1;
                v34 = v33 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
                v35 = (__int64 *)(v32 + 16LL * v34);
                v36 = *v35;
                if ( v28 != *v35 )
                {
                  v124 = 1;
                  while ( v36 != -8 )
                  {
                    v125 = v124 + 1;
                    v34 = v33 & (v124 + v34);
                    v35 = (__int64 *)(v32 + 16LL * v34);
                    v36 = *v35;
                    if ( v28 == *v35 )
                      goto LABEL_18;
                    v124 = v125;
                  }
LABEL_23:
                  sub_164D160((__int64)v165, v27, a6, a7, a8, a9, v16, v17, a12, a13);
                  goto LABEL_24;
                }
LABEL_18:
                v37 = (_QWORD *)v35[1];
                if ( !v37 )
                  goto LABEL_23;
                v38 = v33 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                v39 = (__int64 *)(v32 + 16LL * v38);
                v40 = *v39;
                if ( v29 == *v39 )
                {
LABEL_20:
                  v41 = (_QWORD *)v39[1];
                  if ( v37 == v41 )
                    goto LABEL_23;
                  while ( v41 )
                  {
                    v41 = (_QWORD *)*v41;
                    if ( v37 == v41 )
                      goto LABEL_23;
                  }
                }
                else
                {
                  v122 = 1;
                  while ( v40 != -8 )
                  {
                    v123 = v122 + 1;
                    v38 = v33 & (v122 + v38);
                    v39 = (__int64 *)(v32 + 16LL * v38);
                    v40 = *v39;
                    if ( v29 == *v39 )
                      goto LABEL_20;
                    v122 = v123;
                  }
                }
              }
            }
          }
        }
      }
LABEL_24:
      v42 = v167;
      if ( v167 == v168 )
      {
        sub_170B610((__int64)&v166, v167, &v165);
        v43 = (__int64)v165;
      }
      else
      {
        v43 = (__int64)v165;
        if ( v167 )
        {
          *v167 = (__int64)v165;
          v42 = v167;
        }
        v167 = v42 + 1;
      }
      if ( *(_BYTE *)(v43 + 16) != 27 )
        goto LABEL_30;
      if ( *(_BYTE *)(v14 + 16) != 13 )
        goto LABEL_30;
      v159 = v43;
      v46 = ((*(_DWORD *)(v43 + 20) & 0xFFFFFFFu) >> 1) - 1;
      v48 = sub_19C1150(v43, 0, v43, v46, v14);
      if ( v46 == v47 || v47 == 4294967294LL )
        goto LABEL_30;
      v49 = v159;
      v160 = 24;
      v50 = *(_QWORD *)(v49 + 40);
      if ( (_DWORD)v47 != -2 )
        v160 = 24LL * (unsigned int)(2 * v47 + 3);
      v51 = (*(_BYTE *)(v48 + 23) & 0x40) != 0 ? *(_QWORD *)(v48 - 8) : v48 - 24LL * (*(_DWORD *)(v48 + 20) & 0xFFFFFFF);
      v145 = v48;
      v150 = v49;
      v52 = *(_QWORD *)(v51 + v160);
      v53 = sub_13FCB50(a2);
      v54 = v145;
      v55 = v53;
      v56 = (*(_BYTE *)(v150 + 23) & 0x40) != 0
          ? *(_QWORD *)(v150 - 8)
          : v150 - 24LL * (*(_DWORD *)(v150 + 20) & 0xFFFFFFF);
      if ( *(_QWORD *)(v56 + 24) == v52 )
        goto LABEL_30;
      v57 = (*(_DWORD *)(v150 + 20) & 0xFFFFFFFu) >> 1;
      v58 = v57 - 1;
      if ( v57 == 1 )
        goto LABEL_30;
      v59 = 2;
      v60 = 1;
      v61 = 0;
      do
      {
        while ( 1 )
        {
          v63 = 24;
          if ( (_DWORD)v60 != -1 )
            v63 = 24LL * (v59 + 1);
          if ( v52 == *(_QWORD *)(v56 + v63) )
            break;
          v62 = v60;
          v59 += 2;
          ++v60;
          if ( v62 == v58 )
            goto LABEL_52;
        }
        if ( v61 )
          goto LABEL_30;
        v64 = v59;
        v65 = v60;
        v59 += 2;
        ++v60;
        v61 = *(_QWORD *)(v56 + 24 * v64);
      }
      while ( v65 != v58 );
LABEL_52:
      if ( !v61 )
        goto LABEL_30;
      v66 = *(_QWORD *)(a1 + 304);
      if ( v55 )
      {
        if ( sub_15CC8F0(v66, v52, v55) )
          goto LABEL_30;
        v54 = v145;
        v66 = *(_QWORD *)(a1 + 304);
      }
      v151 = v54;
      sub_1AA91E0(v50, v52, v66, *(_QWORD *)(a1 + 160));
      if ( (*(_BYTE *)(v151 + 23) & 0x40) != 0 )
        v67 = *(_QWORD *)(v151 - 8);
      else
        v67 = v151 - 24LL * (*(_DWORD *)(v151 + 20) & 0xFFFFFFF);
      v152 = *(_QWORD *)(v67 + v160);
      v68 = sub_157EBA0(v152);
      v69 = sub_15F4DF0(v68, 0);
      v70 = *(_QWORD *)(v50 + 56);
      v171 = 1;
      v71 = v69;
      v170 = 3;
      v146 = v70;
      v169[0] = "us-unreachable";
      v161 = (_QWORD *)sub_22077B0(64);
      if ( v161 )
        sub_157FB60(v161, (__int64)v162, (__int64)v169, v146, v71);
      v72 = sub_1648A60(56, 0);
      if ( v72 )
        sub_15F82E0((__int64)v72, (__int64)v162, (__int64)v161);
      v73 = (_QWORD *)sub_157EBA0(v152);
      sub_15F20C0(v73);
      v147 = sub_159C4F0(v162);
      v74 = sub_1648A60(56, 3u);
      if ( v74 )
        sub_15F8650((__int64)v74, (__int64)v161, v71, v147, v152);
      v75 = sub_157F280(v152);
      if ( v76 != v75 )
      {
        v148 = v18;
        v77 = v76;
        v78 = v50;
        v79 = v75;
        while ( 1 )
        {
          v80 = sub_1599EF0(*(__int64 ***)v79);
          v81 = *(_BYTE *)(v79 + 23) & 0x40;
          v82 = *(_DWORD *)(v79 + 20) & 0xFFFFFFF;
          if ( v82 )
          {
            v83 = 24LL * *(unsigned int *)(v79 + 56) + 8;
            v84 = 0;
            while ( 1 )
            {
              v85 = v79 - 24LL * v82;
              if ( v81 )
                v85 = *(_QWORD *)(v79 - 8);
              if ( v78 == *(_QWORD *)(v85 + v83) )
                break;
              ++v84;
              v83 += 8;
              if ( v82 == (_DWORD)v84 )
                goto LABEL_127;
            }
            v86 = 24 * v84;
            if ( v81 )
            {
LABEL_73:
              v87 = *(_QWORD *)(v79 - 8);
              goto LABEL_74;
            }
          }
          else
          {
LABEL_127:
            v86 = 0x17FFFFFFE8LL;
            if ( v81 )
              goto LABEL_73;
          }
          v87 = v79 - 24LL * v82;
LABEL_74:
          v88 = (__int64 *)(v87 + v86);
          if ( *v88 )
          {
            v89 = v88[1];
            v90 = v88[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v90 = v89;
            if ( v89 )
              *(_QWORD *)(v89 + 16) = *(_QWORD *)(v89 + 16) & 3LL | v90;
          }
          *v88 = v80;
          if ( v80 )
          {
            v91 = *(_QWORD *)(v80 + 8);
            v88[1] = v91;
            if ( v91 )
              *(_QWORD *)(v91 + 16) = (unsigned __int64)(v88 + 1) | *(_QWORD *)(v91 + 16) & 3LL;
            v88[2] = (v80 + 8) | v88[2] & 3;
            *(_QWORD *)(v80 + 8) = v88;
          }
          v92 = *(_QWORD *)(v79 + 32);
          if ( !v92 )
            BUG();
          v79 = 0;
          if ( *(_BYTE *)(v92 - 8) == 77 )
            v79 = v92 - 24;
          if ( v77 == v79 )
          {
            v18 = v148;
            break;
          }
        }
      }
      v93 = *(_QWORD *)(a1 + 304);
      v94 = *(unsigned int *)(v93 + 48);
      if ( !(_DWORD)v94 )
        goto LABEL_138;
      v95 = *(_QWORD *)(v93 + 32);
      v96 = (v94 - 1) & (((unsigned int)v152 >> 9) ^ ((unsigned int)v152 >> 4));
      v97 = (__int64 *)(v95 + 16LL * v96);
      v98 = *v97;
      if ( v152 != *v97 )
      {
        v127 = 1;
        while ( v98 != -8 )
        {
          v128 = v127 + 1;
          v96 = (v94 - 1) & (v127 + v96);
          v97 = (__int64 *)(v95 + 16LL * v96);
          v98 = *v97;
          if ( v152 == *v97 )
            goto LABEL_88;
          v127 = v128;
        }
LABEL_138:
        *(_BYTE *)(v93 + 72) = 0;
        v155 = v93;
        v126 = (_QWORD *)sub_22077B0(56);
        v101 = v155;
        v102 = v126;
        if ( !v126 )
        {
          v169[0] = 0;
          BUG();
        }
        v126[1] = 0;
        v99 = 0;
        *v126 = v161;
        v103 = 0;
LABEL_92:
        *((_DWORD *)v102 + 4) = v103;
        v102[3] = 0;
        v102[4] = 0;
        v102[5] = 0;
        v102[6] = -1;
        goto LABEL_93;
      }
LABEL_88:
      if ( v97 == (__int64 *)(v95 + 16 * v94) )
        goto LABEL_138;
      v99 = v97[1];
      *(_BYTE *)(v93 + 72) = 0;
      v153 = v93;
      v100 = (_QWORD *)sub_22077B0(56);
      v101 = v153;
      v102 = v100;
      if ( v100 )
      {
        v100[1] = v99;
        *v100 = v161;
        if ( v99 )
          v103 = *(_DWORD *)(v99 + 16) + 1;
        else
          v103 = 0;
        goto LABEL_92;
      }
LABEL_93:
      v169[0] = v102;
      v104 = *(_BYTE **)(v99 + 32);
      if ( v104 == *(_BYTE **)(v99 + 40) )
      {
        v154 = v101;
        sub_15CE310(v99 + 24, v104, v169);
        v101 = v154;
      }
      else
      {
        if ( v104 )
        {
          *(_QWORD *)v104 = v102;
          v104 = *(_BYTE **)(v99 + 32);
        }
        *(_QWORD *)(v99 + 32) = v104 + 8;
      }
      v105 = *(_DWORD *)(v101 + 48);
      if ( !v105 )
      {
        ++*(_QWORD *)(v101 + 24);
        goto LABEL_161;
      }
      v106 = *(_QWORD *)(v101 + 32);
      v107 = (v105 - 1) & (((unsigned int)v161 >> 9) ^ ((unsigned int)v161 >> 4));
      v108 = (_QWORD *)(v106 + 16LL * v107);
      v109 = *v108;
      if ( v161 != (_QWORD *)*v108 )
      {
        v156 = 1;
        v129 = 0;
        while ( v109 != -8 )
        {
          if ( !v129 && v109 == -16 )
            v129 = v108;
          v107 = (v105 - 1) & (v156 + v107);
          v108 = (_QWORD *)(v106 + 16LL * v107);
          v109 = *v108;
          if ( v161 == (_QWORD *)*v108 )
            goto LABEL_99;
          ++v156;
        }
        if ( !v129 )
          v129 = v108;
        v130 = *(_DWORD *)(v101 + 40);
        ++*(_QWORD *)(v101 + 24);
        v131 = v130 + 1;
        if ( 4 * v131 >= 3 * v105 )
        {
LABEL_161:
          v158 = v101;
          sub_15CFCF0(v101 + 24, 2 * v105);
          v101 = v158;
          v139 = *(_DWORD *)(v158 + 48);
          if ( !v139 )
          {
LABEL_182:
            ++*(_DWORD *)(v101 + 40);
            BUG();
          }
          v140 = v139 - 1;
          v141 = *(_QWORD *)(v158 + 32);
          v142 = v140 & (((unsigned int)v161 >> 9) ^ ((unsigned int)v161 >> 4));
          v131 = *(_DWORD *)(v158 + 40) + 1;
          v129 = (_QWORD *)(v141 + 16LL * v142);
          v143 = (_QWORD *)*v129;
          if ( v161 != (_QWORD *)*v129 )
          {
            v144 = 1;
            v135 = 0;
            while ( v143 != (_QWORD *)-8LL )
            {
              if ( !v135 && v143 == (_QWORD *)-16LL )
                v135 = v129;
              v142 = v140 & (v144 + v142);
              v129 = (_QWORD *)(v141 + 16LL * v142);
              v143 = (_QWORD *)*v129;
              if ( v161 == (_QWORD *)*v129 )
                goto LABEL_151;
              ++v144;
            }
LABEL_157:
            if ( v135 )
              v129 = v135;
          }
        }
        else if ( v105 - *(_DWORD *)(v101 + 44) - v131 <= v105 >> 3 )
        {
          v149 = v101;
          v157 = ((unsigned int)v161 >> 9) ^ ((unsigned int)v161 >> 4);
          sub_15CFCF0(v101 + 24, v105);
          v101 = v149;
          v132 = *(_DWORD *)(v149 + 48);
          if ( !v132 )
            goto LABEL_182;
          v133 = v132 - 1;
          v134 = *(_QWORD *)(v149 + 32);
          v135 = 0;
          v136 = 1;
          v131 = *(_DWORD *)(v149 + 40) + 1;
          v137 = v133 & v157;
          v129 = (_QWORD *)(v134 + 16LL * (v133 & v157));
          v138 = *v129;
          if ( v161 != (_QWORD *)*v129 )
          {
            while ( v138 != -8 )
            {
              if ( v138 == -16 && !v135 )
                v135 = v129;
              v137 = v133 & (v136 + v137);
              v129 = (_QWORD *)(v134 + 16LL * v137);
              v138 = *v129;
              if ( v161 == (_QWORD *)*v129 )
                goto LABEL_151;
              ++v136;
            }
            goto LABEL_157;
          }
        }
LABEL_151:
        *(_DWORD *)(v101 + 40) = v131;
        if ( *v129 != -8 )
          --*(_DWORD *)(v101 + 44);
        v129[1] = v102;
        *v129 = v161;
        goto LABEL_30;
      }
LABEL_99:
      v110 = v108[1];
      v108[1] = v102;
      if ( v110 )
      {
        v111 = *(_QWORD *)(v110 + 24);
        if ( v111 )
          j_j___libc_free_0(v111, *(_QWORD *)(v110 + 40) - v111);
        j_j___libc_free_0(v110, 56);
      }
LABEL_30:
      v18 = *(_QWORD *)(v18 + 8);
      if ( !v18 )
        goto LABEL_31;
    }
  }
LABEL_110:
  v115 = *(_QWORD *)(a3 + 8);
  for ( i = a2 + 56; v115; v115 = *(_QWORD *)(v115 + 8) )
  {
    v117 = sub_1648700(v115);
    if ( *((_BYTE *)v117 + 16) > 0x17u )
    {
      v169[0] = v117;
      if ( sub_1377F70(i, v117[5]) )
      {
        v118 = v167;
        if ( v167 == v168 )
        {
          sub_170B610((__int64)&v166, v167, v169);
        }
        else
        {
          if ( v167 )
          {
            *v167 = v169[0];
            v118 = v167;
          }
          v167 = v118 + 1;
        }
      }
    }
  }
  v119 = v167;
  for ( j = v166; v119 != j; ++j )
  {
    v121 = *j;
    sub_1648780(v121, a3, v14);
  }
  v44 = a2;
LABEL_32:
  result = sub_19C1DC0(a1, (__int64)&v166, v44, a6, a7, a8, a9, v16, v17, a12, a13);
  if ( v166 )
    return j_j___libc_free_0(v166, (char *)v168 - (char *)v166);
  return result;
}
