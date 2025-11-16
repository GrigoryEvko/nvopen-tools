// Function: sub_1D765D0
// Address: 0x1d765d0
//
__int64 __fastcall sub_1D765D0(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        __m128 a5,
        __m128 a6,
        __m128 a7,
        __m128 a8,
        double a9,
        double a10,
        __m128i a11,
        __m128 a12)
{
  __int64 v12; // r14
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 v16; // r13
  unsigned int v17; // eax
  __int64 v18; // rsi
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r13
  int v25; // r13d
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // rbx
  _BYTE *v31; // r15
  unsigned __int8 v32; // al
  unsigned int v33; // r12d
  int v34; // eax
  int v35; // eax
  __int64 (*v36)(); // rax
  __int64 **v37; // r12
  _QWORD *v38; // rbx
  _QWORD *v39; // rax
  __int64 *v40; // rsi
  __int64 *v41; // rax
  unsigned __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rdx
  unsigned __int8 *v45; // rsi
  unsigned __int8 *v46; // rsi
  unsigned __int8 *v47; // rsi
  __int64 v48; // rax
  __int64 v49; // rbx
  _QWORD *v50; // rax
  __int64 v51; // r9
  __int64 *v52; // rbx
  __int64 v53; // rcx
  __int64 v54; // rax
  _QWORD *v55; // rax
  __int64 v56; // rax
  __int64 v57; // rdx
  unsigned __int8 *v58; // rsi
  __int64 v59; // rsi
  __int64 v60; // rax
  __int64 v61; // r10
  __int64 v62; // rbx
  __int64 *v63; // r12
  __int64 v64; // rcx
  __int64 v65; // rax
  double v66; // xmm4_8
  double v67; // xmm5_8
  __int64 v68; // r10
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // r10
  __int64 v74; // rsi
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rax
  __int64 v79; // rsi
  __int64 v80; // rdi
  __int64 (*v81)(); // rax
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // r13
  int v86; // r13d
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // r13
  __int64 *v90; // r13
  __int64 *v91; // r15
  __int64 v92; // rax
  __int64 v93; // rax
  unsigned int v94; // edx
  __int64 v95; // rbx
  __int64 v96; // r14
  char v97; // al
  __int64 v98; // rdi
  int v99; // eax
  __int64 v100; // rax
  __int64 *v101; // rbx
  __int64 v102; // r13
  __int64 v103; // r13
  __int64 v104; // rax
  unsigned int v105; // eax
  unsigned __int64 v106; // r9
  unsigned __int64 v107; // rax
  __int64 ****v108; // rdx
  __int64 ****v109; // rax
  __int64 v110; // rax
  unsigned int v111; // r12d
  int v112; // eax
  bool v113; // al
  __int64 v114; // rax
  _QWORD *v115; // rbx
  _QWORD *v116; // r13
  __int64 v117; // rsi
  __int64 v118; // r13
  __int64 *v119; // rax
  unsigned int v120; // r13d
  __int64 v121; // rax
  __int64 v122; // r13
  __int64 *v123; // rax
  unsigned int v124; // r13d
  __int64 *v125; // rax
  __int64 v126; // rax
  __int64 v127; // rbx
  __int64 *v128; // rax
  __int64 v129; // rax
  char v130; // al
  double v131; // xmm4_8
  double v132; // xmm5_8
  __int64 **v133; // rdi
  __int64 ***v134; // rdx
  __int64 **v135; // rax
  _QWORD *v136; // rax
  __int64 v137; // r9
  _QWORD **v138; // rax
  __int64 *v139; // rax
  __int64 v140; // rax
  __int64 v141; // r9
  __int64 v142; // r8
  __int64 v143; // rsi
  __int64 v144; // rax
  unsigned int v145; // r13d
  __int64 v146; // r12
  __int64 v147; // rax
  char v148; // dl
  bool v149; // al
  __int64 v150; // [rsp+8h] [rbp-118h]
  __int64 v151; // [rsp+10h] [rbp-110h]
  __int64 v152; // [rsp+10h] [rbp-110h]
  __int64 v153; // [rsp+18h] [rbp-108h]
  _QWORD *v154; // [rsp+18h] [rbp-108h]
  __int64 *v155; // [rsp+18h] [rbp-108h]
  __int64 v156; // [rsp+18h] [rbp-108h]
  _QWORD *v157; // [rsp+20h] [rbp-100h]
  __int64 v158; // [rsp+20h] [rbp-100h]
  __int64 v159; // [rsp+20h] [rbp-100h]
  __int64 v160; // [rsp+20h] [rbp-100h]
  __int64 v161; // [rsp+20h] [rbp-100h]
  __int64 v162; // [rsp+28h] [rbp-F8h]
  __int64 v163; // [rsp+28h] [rbp-F8h]
  __int64 v164; // [rsp+28h] [rbp-F8h]
  __int64 v165; // [rsp+28h] [rbp-F8h]
  __int64 v166; // [rsp+28h] [rbp-F8h]
  __int64 v167; // [rsp+30h] [rbp-F0h]
  __int64 v168; // [rsp+30h] [rbp-F0h]
  __int64 v169; // [rsp+38h] [rbp-E8h]
  unsigned int v170; // [rsp+38h] [rbp-E8h]
  __int64 v171; // [rsp+38h] [rbp-E8h]
  int v172; // [rsp+38h] [rbp-E8h]
  __int64 v174; // [rsp+48h] [rbp-D8h]
  __int64 v175; // [rsp+48h] [rbp-D8h]
  unsigned int v176; // [rsp+48h] [rbp-D8h]
  __int64 v177; // [rsp+48h] [rbp-D8h]
  __int64 v178; // [rsp+48h] [rbp-D8h]
  __int64 v179; // [rsp+48h] [rbp-D8h]
  __int64 v180; // [rsp+48h] [rbp-D8h]
  unsigned __int8 v181; // [rsp+48h] [rbp-D8h]
  int v182; // [rsp+48h] [rbp-D8h]
  unsigned int v183; // [rsp+58h] [rbp-C8h] BYREF
  unsigned int v184; // [rsp+5Ch] [rbp-C4h] BYREF
  __int64 v185[2]; // [rsp+60h] [rbp-C0h] BYREF
  char v186; // [rsp+70h] [rbp-B0h]
  char v187; // [rsp+71h] [rbp-AFh]
  unsigned __int8 *v188; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v189; // [rsp+88h] [rbp-98h]
  __int16 v190; // [rsp+90h] [rbp-90h]
  unsigned __int64 v191; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v192; // [rsp+A8h] [rbp-78h]
  __int64 *v193; // [rsp+B0h] [rbp-70h] BYREF
  __int64 *v194; // [rsp+B8h] [rbp-68h]
  __int64 v195; // [rsp+C0h] [rbp-60h]
  int v196; // [rsp+C8h] [rbp-58h]
  __int64 v197; // [rsp+D0h] [rbp-50h]
  __int64 v198; // [rsp+D8h] [rbp-48h]

  v12 = a2;
  v14 = *(_QWORD *)(a1 + 176);
  v174 = *(_QWORD *)(a2 + 40);
  if ( !v14 )
    goto LABEL_4;
  v15 = *(__int64 (**)())(*(_QWORD *)v14 + 592LL);
  if ( v15 == sub_1D5A3C0 )
    goto LABEL_3;
  if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, unsigned int *, unsigned int *))v15)(v14, a2, &v183, &v184) )
    goto LABEL_105;
  if ( *(char *)(a2 + 23) >= 0 )
    goto LABEL_115;
  v83 = sub_1648A40(a2);
  v85 = v83 + v84;
  if ( *(char *)(a2 + 23) >= 0 )
  {
    if ( (unsigned int)(v85 >> 4) )
LABEL_218:
      BUG();
LABEL_115:
    v89 = -24;
    goto LABEL_88;
  }
  if ( !(unsigned int)((v85 - sub_1648A40(a2)) >> 4) )
    goto LABEL_115;
  if ( *(char *)(a2 + 23) >= 0 )
    goto LABEL_218;
  v86 = *(_DWORD *)(sub_1648A40(a2) + 8);
  if ( *(char *)(a2 + 23) >= 0 )
    BUG();
  v87 = sub_1648A40(a2);
  v89 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v87 + v88 - 4) - v86);
LABEL_88:
  v90 = (__int64 *)(a2 + v89);
  v91 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( v90 != v91 )
  {
    v168 = a2;
    while ( 1 )
    {
      while ( 1 )
      {
        v92 = *(_QWORD *)*v91;
        if ( *(_BYTE *)(v92 + 8) != 15 )
          goto LABEL_90;
        LODWORD(v192) = 8 * sub_15A95A0(*(_QWORD *)(a1 + 904), *(_DWORD *)(v92 + 8) >> 8);
        if ( (unsigned int)v192 > 0x40 )
          sub_16A4EF0((__int64)&v191, 0, 0);
        else
          v191 = 0;
        v93 = sub_164A410(*v91, *(_QWORD *)(a1 + 904), (__int64)&v191);
        v94 = v192;
        v95 = v93;
        if ( (unsigned int)v192 > 0x40 )
          break;
        a2 = v184;
        v96 = v191;
        if ( ((unsigned int)v191 & (v184 - 1)) != 0 )
          goto LABEL_90;
LABEL_96:
        v97 = *(_BYTE *)(v95 + 16);
        if ( v97 == 53 )
        {
          a4 = *(unsigned __int16 *)(v95 + 18);
          if ( (unsigned int)(1 << a4) >> 1 >= (unsigned int)a2 )
            goto LABEL_100;
          a2 = *(_QWORD *)(v95 + 56);
          v107 = sub_12BE0A0(*(_QWORD *)(a1 + 904), a2);
          if ( v107 >= v96 + (unsigned __int64)v183 )
          {
            a2 = v184;
            sub_15F8A20(v95, v184);
          }
          v97 = *(_BYTE *)(v95 + 16);
        }
        if ( v97 == 3 && sub_15E6530(v95) )
        {
          a2 = *(_QWORD *)(a1 + 904);
          v105 = sub_1649510(v95, a2);
          if ( v105 < v184 )
          {
            a2 = *(_QWORD *)(v95 + 24);
            v106 = sub_12BE0A0(*(_QWORD *)(a1 + 904), a2);
            if ( v106 >= (unsigned __int64)v183 + v96 )
            {
              a2 = v184;
              sub_15E4CC0(v95, v184);
            }
          }
        }
        v94 = v192;
LABEL_100:
        if ( v94 > 0x40 )
        {
          v98 = v191;
          if ( v191 )
            goto LABEL_102;
        }
LABEL_90:
        v91 += 3;
        if ( v90 == v91 )
          goto LABEL_103;
      }
      v170 = v192;
      v99 = sub_16A57B0((__int64)&v191);
      v94 = v170;
      a2 = v184;
      a4 = v170 - v99;
      if ( (unsigned int)a4 <= 0x40 )
      {
        v98 = v191;
        v96 = *(_QWORD *)v191;
        if ( ((unsigned int)*(_QWORD *)v191 & (v184 - 1)) == 0 )
          goto LABEL_96;
      }
      else
      {
        if ( v184 == 1 )
        {
          v96 = -1;
          a2 = 1;
          goto LABEL_96;
        }
        v98 = v191;
        if ( !v191 )
          goto LABEL_90;
      }
LABEL_102:
      j_j___libc_free_0_0(v98);
      v91 += 3;
      if ( v90 == v91 )
      {
LABEL_103:
        v12 = v168;
        break;
      }
    }
  }
  if ( *(_BYTE *)(v12 + 16) == 78 )
  {
    v20 = *(_QWORD *)(v12 - 24);
    if ( !*(_BYTE *)(v20 + 16) )
    {
      if ( (*(_BYTE *)(v20 + 33) & 0x20) == 0 )
      {
        if ( !*(_QWORD *)(a1 + 176) )
          goto LABEL_15;
        goto LABEL_3;
      }
      a4 = (unsigned int)(*(_DWORD *)(v20 + 36) - 133);
      if ( (unsigned int)a4 <= 4 && ((1LL << (*(_BYTE *)(v20 + 36) + 123)) & 0x15) != 0 )
      {
        v118 = *(_QWORD *)(a1 + 904);
        v119 = (__int64 *)sub_1649C60(*(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)));
        a2 = 0;
        v120 = sub_1AE99B0(v119, 0, v118, 0, 0, 0);
        if ( v120 > (unsigned int)sub_15603A0((_QWORD *)(v12 + 56), 0) )
        {
          v191 = *(_QWORD *)(v12 + 56);
          a2 = sub_16498A0(v12);
          *(_QWORD *)(v12 + 56) = sub_1563C10((__int64 *)&v191, (__int64 *)a2, 1, 1);
          if ( v120 )
          {
            v128 = (__int64 *)sub_16498A0(v12);
            v129 = sub_155D330(v128, v120);
            LODWORD(v188) = 0;
            v171 = v129;
            v191 = *(_QWORD *)(v12 + 56);
            a2 = sub_16498A0(v12);
            v191 = sub_1563E10((__int64 *)&v191, (__int64 *)a2, (int *)&v188, 1, v171);
            *(_QWORD *)(v12 + 56) = v191;
          }
        }
        v121 = *(_QWORD *)(v12 - 24);
        if ( *(_BYTE *)(v121 + 16) )
          BUG();
        if ( (*(_DWORD *)(v121 + 36) & 0xFFFFFFFD) == 0x85 )
        {
          v122 = *(_QWORD *)(a1 + 904);
          v123 = (__int64 *)sub_1649C60(*(_QWORD *)(v12 + 24 * (1LL - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF))));
          a2 = 1;
          v124 = sub_1AE99B0(v123, 0, v122, 0, 0, 0);
          if ( v124 > (unsigned int)sub_15603A0((_QWORD *)(v12 + 56), 1) )
          {
            v191 = *(_QWORD *)(v12 + 56);
            a2 = sub_16498A0(v12);
            *(_QWORD *)(v12 + 56) = sub_1563C10((__int64 *)&v191, (__int64 *)a2, 2, 1);
            if ( v124 )
            {
              v125 = (__int64 *)sub_16498A0(v12);
              v126 = sub_155D330(v125, v124);
              LODWORD(v188) = 1;
              v127 = v126;
              v191 = *(_QWORD *)(v12 + 56);
              a2 = sub_16498A0(v12);
              v191 = sub_1563E10((__int64 *)&v191, (__int64 *)a2, (int *)&v188, 1, v127);
              *(_QWORD *)(v12 + 56) = v191;
            }
          }
        }
      }
    }
  }
LABEL_105:
  if ( !*(_QWORD *)(a1 + 176) )
    goto LABEL_4;
LABEL_3:
  if ( *(_BYTE *)(a1 + 897) )
    goto LABEL_4;
  a2 = 0xFFFFFFFFLL;
  if ( !(unsigned __int8)sub_1560260((_QWORD *)(v12 + 56), -1, 7) )
  {
    v82 = *(_QWORD *)(v12 - 24);
    if ( *(_BYTE *)(v82 + 16) )
      return 0;
    a2 = 0xFFFFFFFFLL;
    v191 = *(_QWORD *)(v82 + 112);
    if ( !(unsigned __int8)sub_1560260(&v191, -1, 7) )
    {
LABEL_4:
      v16 = *(_QWORD *)(v12 - 24);
      if ( *(_BYTE *)(v16 + 16) )
        return 0;
      if ( (*(_BYTE *)(v16 + 33) & 0x20) != 0 )
      {
        v17 = *(_DWORD *)(v16 + 36);
        if ( v17 == 144 )
        {
          v100 = sub_140EAC0((__int64 *)v12, *(_QWORD *)(a1 + 904), *(_QWORD *)(a1 + 200), 1);
          v101 = *(__int64 **)(a1 + 232);
          v102 = v100;
          if ( v101 )
          {
            v101 -= 3;
            v191 = 6;
            v192 = 0;
            v193 = v101;
            if ( v101 != (__int64 *)-8LL && v101 != (__int64 *)-16LL )
              sub_164C220((__int64)&v191);
          }
          else
          {
            v191 = 6;
            v192 = 0;
            v193 = 0;
          }
          sub_13E4A40(v12, v102, *(_QWORD *)(a1 + 200), 0, 0);
          if ( v193 != v101 )
          {
            *(_QWORD *)(a1 + 232) = *(_QWORD *)(v174 + 48);
            sub_1D672E0(a1 + 240);
            if ( *(_BYTE *)(a1 + 304) )
            {
              v114 = *(unsigned int *)(a1 + 296);
              if ( (_DWORD)v114 )
              {
                v115 = *(_QWORD **)(a1 + 280);
                v116 = &v115[2 * v114];
                do
                {
                  if ( *v115 != -4 && *v115 != -8 )
                  {
                    v117 = v115[1];
                    if ( v117 )
                      sub_161E7C0((__int64)(v115 + 1), v117);
                  }
                  v115 += 2;
                }
                while ( v116 != v115 );
              }
              j___libc_free_0(*(_QWORD *)(a1 + 280));
              *(_BYTE *)(a1 + 304) = 0;
            }
          }
          sub_1455FA0((__int64)&v191);
          return 1;
        }
        if ( v17 > 0x90 )
        {
          if ( ((v17 - 407) & 0xFFFFFFFD) == 0 )
          {
            v103 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
            if ( *(_BYTE *)(v103 + 16) == 61 )
            {
              v104 = *(_QWORD *)(v103 + 8);
              if ( v104 )
              {
                if ( !*(_QWORD *)(v104 + 8) && *(_QWORD *)(v12 + 40) != *(_QWORD *)(v103 + 40) )
                {
                  sub_15F22F0((_QWORD *)v103, v12);
                  sub_165A590((__int64)&v191, a1 + 320, v103);
                  return 1;
                }
              }
            }
            return 0;
          }
          if ( v17 == 203 )
            goto LABEL_10;
        }
        else
        {
          switch ( v17 )
          {
            case 0x21u:
              goto LABEL_26;
            case 0x73u:
LABEL_10:
              v18 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
LABEL_11:
              sub_164D160(
                v12,
                v18,
                a5,
                *(double *)a6.m128_u64,
                *(double *)a7.m128_u64,
                *(double *)a8.m128_u64,
                a9,
                a10,
                *(double *)a11.m128i_i64,
                a12);
              sub_15F20C0((_QWORD *)v12);
              return 1;
            case 0x1Fu:
LABEL_26:
              v29 = *(_QWORD *)(a1 + 176);
              v30 = *(_QWORD *)(a1 + 904);
              if ( !v29 || !v30 )
                return 0;
              v31 = *(_BYTE **)(v12 + 24 * (1LL - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)));
              v32 = v31[16];
              if ( v32 == 13 )
              {
                v33 = *((_DWORD *)v31 + 8);
                if ( v33 <= 0x40 )
                {
                  if ( *((_QWORD *)v31 + 3) == 1 )
                    return 0;
LABEL_32:
                  v35 = *(_DWORD *)(v16 + 36);
                  if ( v35 == 33 )
                  {
                    v36 = *(__int64 (**)())(*(_QWORD *)v29 + 152LL);
                    if ( v36 == sub_1D5A370 )
                      goto LABEL_35;
                  }
                  else if ( v35 != 31 || (v36 = *(__int64 (**)())(*(_QWORD *)v29 + 160LL), v36 == sub_1D5A380) )
                  {
LABEL_35:
                    v37 = *(__int64 ***)v12;
                    if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) != 16 )
                    {
                      v176 = sub_1643030(*(_QWORD *)v12);
                      if ( v176 <= (unsigned int)sub_15A96E0(v30) )
                      {
                        v38 = *(_QWORD **)(v12 + 40);
                        LOWORD(v193) = 259;
                        v169 = (__int64)v38;
                        v191 = (unsigned __int64)"cond.false";
                        v39 = (_QWORD *)sub_157FBF0(v38, (__int64 *)(v12 + 24), (__int64)&v191);
                        v40 = *(__int64 **)(v12 + 32);
                        v167 = (__int64)v39;
                        v191 = (unsigned __int64)"cond.end";
                        LOWORD(v193) = 259;
                        v162 = sub_157FBF0(v39, v40, (__int64)&v191);
                        v41 = (__int64 *)sub_16498A0(v12);
                        v191 = 0;
                        v193 = 0;
                        v194 = v41;
                        v195 = 0;
                        v196 = 0;
                        v197 = 0;
                        v198 = 0;
                        v192 = 0;
                        v42 = sub_157EBA0((__int64)v38);
                        v192 = *(_QWORD *)(v42 + 40);
                        v44 = v42 + 24;
                        v193 = (__int64 *)(v42 + 24);
                        v45 = *(unsigned __int8 **)(v42 + 48);
                        v188 = v45;
                        if ( v45 )
                        {
                          sub_1623A60((__int64)&v188, (__int64)v45, 2);
                          if ( v191 )
                            sub_161E7C0((__int64)&v191, v191);
                          v191 = (unsigned __int64)v188;
                          if ( v188 )
                            sub_1623210((__int64)&v188, v188, (__int64)&v191);
                        }
                        v46 = *(unsigned __int8 **)(v12 + 48);
                        v188 = v46;
                        if ( v46 )
                        {
                          sub_1623A60((__int64)&v188, (__int64)v46, 2);
                          v47 = (unsigned __int8 *)v191;
                          if ( !v191 )
                            goto LABEL_45;
                        }
                        else
                        {
                          v47 = (unsigned __int8 *)v191;
                          if ( !v191 )
                            goto LABEL_47;
                        }
                        sub_161E7C0((__int64)&v191, (__int64)v47);
LABEL_45:
                        v47 = v188;
                        v191 = (unsigned __int64)v188;
                        if ( v188 )
                          sub_1623210((__int64)&v188, v188, (__int64)&v191);
LABEL_47:
                        v48 = sub_15A06D0(v37, (__int64)v47, v44, v43);
                        v187 = 1;
                        v186 = 3;
                        v185[0] = (__int64)"cmpz";
                        if ( *(_BYTE *)(*(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)) + 16LL) > 0x10u
                          || *(_BYTE *)(v48 + 16) > 0x10u )
                        {
                          v153 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
                          v190 = 257;
                          v160 = v48;
                          v136 = sub_1648A60(56, 2u);
                          v137 = v160;
                          v49 = (__int64)v136;
                          if ( v136 )
                          {
                            v161 = (__int64)v136;
                            v138 = *(_QWORD ***)v153;
                            if ( *(_BYTE *)(*(_QWORD *)v153 + 8LL) == 16 )
                            {
                              v150 = v153;
                              v151 = v137;
                              v154 = v138[4];
                              v139 = (__int64 *)sub_1643320(*v138);
                              v140 = (__int64)sub_16463B0(v139, (unsigned int)v154);
                              v141 = v151;
                              v142 = v150;
                            }
                            else
                            {
                              v152 = v153;
                              v156 = v137;
                              v140 = sub_1643320(*v138);
                              v142 = v152;
                              v141 = v156;
                            }
                            sub_15FEC10(v49, v140, 51, 32, v142, v141, (__int64)&v188, 0);
                          }
                          else
                          {
                            v161 = 0;
                          }
                          if ( v192 )
                          {
                            v155 = v193;
                            sub_157E9D0(v192 + 40, v49);
                            v143 = *v155;
                            v144 = *(_QWORD *)(v49 + 24) & 7LL;
                            *(_QWORD *)(v49 + 32) = v155;
                            v143 &= 0xFFFFFFFFFFFFFFF8LL;
                            *(_QWORD *)(v49 + 24) = v143 | v144;
                            *(_QWORD *)(v143 + 8) = v49 + 24;
                            *v155 = *v155 & 7 | (v49 + 24);
                          }
                          sub_164B780(v161, v185);
                          sub_12A86E0((__int64 *)&v191, v49);
                        }
                        else
                        {
                          v49 = sub_15A37B0(
                                  0x20u,
                                  *(_QWORD **)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)),
                                  (_QWORD *)v48,
                                  0);
                        }
                        v190 = 257;
                        v50 = sub_1648A60(56, 3u);
                        v51 = (__int64)v50;
                        if ( v50 )
                        {
                          v157 = v50;
                          sub_15F83E0((__int64)v50, v162, v167, v49, 0);
                          v51 = (__int64)v157;
                        }
                        if ( v192 )
                        {
                          v52 = v193;
                          v158 = v51;
                          sub_157E9D0(v192 + 40, v51);
                          v51 = v158;
                          v53 = *v52;
                          v54 = *(_QWORD *)(v158 + 24);
                          *(_QWORD *)(v158 + 32) = v52;
                          v53 &= 0xFFFFFFFFFFFFFFF8LL;
                          *(_QWORD *)(v158 + 24) = v53 | v54 & 7;
                          *(_QWORD *)(v53 + 8) = v158 + 24;
                          *v52 = *v52 & 7 | (v158 + 24);
                        }
                        v159 = v51;
                        sub_164B780(v51, (__int64 *)&v188);
                        sub_12A86E0((__int64 *)&v191, v159);
                        v55 = (_QWORD *)sub_157EBA0(v169);
                        sub_15F20C0(v55);
                        v56 = *(_QWORD *)(v162 + 48);
                        if ( !v56 )
                          BUG();
                        v57 = *(_QWORD *)(v56 + 16);
                        v193 = *(__int64 **)(v162 + 48);
                        v192 = v57;
                        v58 = *(unsigned __int8 **)(v56 + 24);
                        v188 = v58;
                        if ( v58 )
                        {
                          sub_1623A60((__int64)&v188, (__int64)v58, 2);
                          v59 = v191;
                          if ( !v191 )
                            goto LABEL_58;
                        }
                        else
                        {
                          v59 = v191;
                          if ( !v191 )
                          {
LABEL_60:
                            v187 = 1;
                            v185[0] = (__int64)"ctz";
                            v186 = 3;
                            v190 = 257;
                            v60 = sub_1648B60(64);
                            v61 = v60;
                            if ( v60 )
                            {
                              v163 = v60;
                              v62 = v60;
                              sub_15F1EA0(v60, (__int64)v37, 53, 0, 0, 0);
                              *(_DWORD *)(v163 + 56) = 2;
                              sub_164B780(v163, (__int64 *)&v188);
                              sub_1648880(v163, *(_DWORD *)(v163 + 56), 1);
                              v61 = v163;
                            }
                            else
                            {
                              v62 = 0;
                            }
                            if ( v192 )
                            {
                              v63 = v193;
                              v164 = v61;
                              sub_157E9D0(v192 + 40, v61);
                              v61 = v164;
                              v64 = *v63;
                              v65 = *(_QWORD *)(v164 + 24);
                              *(_QWORD *)(v164 + 32) = v63;
                              v64 &= 0xFFFFFFFFFFFFFFF8LL;
                              *(_QWORD *)(v164 + 24) = v64 | v65 & 7;
                              *(_QWORD *)(v64 + 8) = v164 + 24;
                              *v63 = *v63 & 7 | (v164 + 24);
                            }
                            v165 = v61;
                            sub_164B780(v62, v185);
                            sub_12A86E0((__int64 *)&v191, v165);
                            sub_164D160(
                              v12,
                              v165,
                              a5,
                              *(double *)a6.m128_u64,
                              *(double *)a7.m128_u64,
                              *(double *)a8.m128_u64,
                              v66,
                              v67,
                              *(double *)a11.m128i_i64,
                              a12);
                            v68 = v165;
                            v189 = v176;
                            if ( v176 > 0x40 )
                            {
                              sub_16A4EF0((__int64)&v188, v176, 0);
                              v68 = v165;
                            }
                            else
                            {
                              v188 = (unsigned __int8 *)(v176 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v176));
                            }
                            v177 = v68;
                            v69 = sub_159C0E0(v194, (__int64)&v188);
                            v73 = v177;
                            v74 = v69;
                            if ( v189 > 0x40 && v188 )
                            {
                              v166 = v177;
                              v178 = v69;
                              j_j___libc_free_0_0(v188);
                              v73 = v166;
                              v74 = v178;
                            }
                            v179 = v73;
                            sub_1704F80(v73, v74, v169, v70, v71, v72);
                            sub_1704F80(v179, v12, v167, v75, v76, v77);
                            v78 = sub_159C4F0(v194);
                            sub_1593B40((_QWORD *)(v12 + 24 * (1LL - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF))), v78);
                            v79 = v191;
                            *a3 = 1;
                            result = 1;
                            if ( v79 )
                            {
                              sub_161E7C0((__int64)&v191, v79);
                              return 1;
                            }
                            return result;
                          }
                        }
                        sub_161E7C0((__int64)&v191, v59);
LABEL_58:
                        v191 = (unsigned __int64)v188;
                        if ( v188 )
                          sub_1623210((__int64)&v188, v188, (__int64)&v191);
                        goto LABEL_60;
                      }
                    }
                    return 0;
                  }
                  if ( ((unsigned __int8 (__fastcall *)(__int64))v36)(v29) )
                    return 0;
                  goto LABEL_35;
                }
                v175 = v29;
                v34 = sub_16A57B0((__int64)(v31 + 24));
                v29 = v175;
                if ( v34 == v33 - 1 )
                  return 0;
              }
              else
              {
                if ( *(_BYTE *)(*(_QWORD *)v31 + 8LL) != 16 || v32 > 0x10u )
                  goto LABEL_32;
                v180 = *(_QWORD *)(a1 + 176);
                v110 = sub_15A1020(v31, a2, *(_QWORD *)v31, a4);
                v29 = v180;
                if ( v110 && *(_BYTE *)(v110 + 16) == 13 )
                {
                  v111 = *(_DWORD *)(v110 + 32);
                  if ( v111 <= 0x40 )
                  {
                    v113 = *(_QWORD *)(v110 + 24) == 1;
                  }
                  else
                  {
                    v112 = sub_16A57B0(v110 + 24);
                    v29 = v180;
                    v113 = v111 - 1 == v112;
                  }
                  if ( v113 )
                    return 0;
                }
                else
                {
                  if ( !(unsigned int)*(_QWORD *)(*(_QWORD *)v31 + 32LL) )
                    return 0;
                  v182 = *(_QWORD *)(*(_QWORD *)v31 + 32LL);
                  v145 = 0;
                  v146 = v29;
                  while ( 1 )
                  {
                    v147 = sub_15A0A60((__int64)v31, v145);
                    if ( !v147 )
                      break;
                    v148 = *(_BYTE *)(v147 + 16);
                    if ( v148 != 9 )
                    {
                      if ( v148 != 13 )
                        break;
                      if ( *(_DWORD *)(v147 + 32) <= 0x40u )
                      {
                        v149 = *(_QWORD *)(v147 + 24) == 1;
                      }
                      else
                      {
                        v172 = *(_DWORD *)(v147 + 32);
                        v149 = v172 - 1 == (unsigned int)sub_16A57B0(v147 + 24);
                      }
                      if ( !v149 )
                        break;
                    }
                    if ( v182 == ++v145 )
                      return 0;
                  }
                  v29 = v146;
                }
                v16 = *(_QWORD *)(v12 - 24);
              }
              if ( *(_BYTE *)(v16 + 16) )
                BUG();
              goto LABEL_32;
          }
        }
        v80 = *(_QWORD *)(a1 + 176);
        if ( v80 )
        {
          v191 = (unsigned __int64)&v193;
          v192 = 0x200000000LL;
          v81 = *(__int64 (**)())(*(_QWORD *)v80 + 728LL);
          if ( v81 != sub_1D5A3D0 )
          {
            v130 = ((__int64 (__fastcall *)(__int64, __int64, unsigned __int64 *, unsigned __int8 **))v81)(
                     v80,
                     v12,
                     &v191,
                     &v188);
            v133 = (__int64 **)v191;
            if ( v130 )
            {
              while ( 1 )
              {
                v133 = (__int64 **)v191;
                if ( !(_DWORD)v192 )
                  break;
                v134 = *(__int64 ****)(v191 + 8LL * (unsigned int)v192 - 8);
                LODWORD(v192) = v192 - 1;
                v135 = *v134;
                if ( *((_BYTE *)*v134 + 8) == 16 )
                  v135 = (__int64 **)*v135[2];
                result = sub_1D73760(
                           a1,
                           v12,
                           v134,
                           (__int64)v188,
                           *((_DWORD *)v135 + 2) >> 8,
                           *(double *)a5.m128_u64,
                           a6,
                           a7,
                           a8,
                           v131,
                           v132,
                           a11,
                           a12);
                if ( (_BYTE)result )
                {
                  if ( (__int64 **)v191 != &v193 )
                  {
                    v181 = result;
                    _libc_free(v191);
                    return v181;
                  }
                  return result;
                }
              }
            }
            if ( v133 != &v193 )
              _libc_free((unsigned __int64)v133);
            v16 = *(_QWORD *)(v12 - 24);
          }
          if ( *(_BYTE *)(v16 + 16) )
            return 0;
        }
      }
LABEL_15:
      sub_3950240(&v191, *(_QWORD *)(a1 + 200), 1);
      v21 = sub_3950490(&v191, v12);
      if ( v21 )
      {
        v18 = v21;
        goto LABEL_11;
      }
      return 0;
    }
  }
  if ( *(char *)(v12 + 23) >= 0 )
    goto LABEL_136;
  v22 = sub_1648A40(v12);
  v24 = v22 + v23;
  if ( *(char *)(v12 + 23) >= 0 )
  {
    if ( (unsigned int)(v24 >> 4) )
LABEL_220:
      BUG();
LABEL_136:
    v28 = -24;
    goto LABEL_137;
  }
  if ( !(unsigned int)((v24 - sub_1648A40(v12)) >> 4) )
    goto LABEL_136;
  if ( *(char *)(v12 + 23) >= 0 )
    goto LABEL_220;
  v25 = *(_DWORD *)(sub_1648A40(v12) + 8);
  if ( *(char *)(v12 + 23) >= 0 )
    BUG();
  v26 = sub_1648A40(v12);
  v28 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v26 + v27 - 4) - v25);
LABEL_137:
  v108 = (__int64 ****)(v12 + v28);
  a4 = 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
  v109 = (__int64 ****)(v12 - a4);
  if ( v108 == (__int64 ****)(v12 - a4) )
    goto LABEL_4;
  while ( 1 )
  {
    a4 = (__int64)**v109;
    if ( *(_BYTE *)(a4 + 8) == 15 )
      return sub_1D73760(
               a1,
               v12,
               *v109,
               a4,
               *(_DWORD *)(a4 + 8) >> 8,
               *(double *)a5.m128_u64,
               a6,
               a7,
               a8,
               a9,
               a10,
               a11,
               a12);
    v109 += 3;
    if ( v108 == v109 )
      goto LABEL_4;
  }
}
