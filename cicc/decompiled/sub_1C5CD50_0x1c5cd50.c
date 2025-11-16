// Function: sub_1C5CD50
// Address: 0x1c5cd50
//
__int64 __fastcall sub_1C5CD50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __m128i a7,
        __m128i a8)
{
  unsigned __int64 v10; // r15
  int v11; // eax
  __int64 v12; // rcx
  int v13; // r14d
  __int64 v14; // rax
  __int64 *v15; // r8
  __int64 v16; // r12
  __int64 *v17; // rbx
  __int64 v18; // r13
  char v19; // al
  bool v20; // zf
  __int64 *v21; // r12
  char *v22; // r14
  int v23; // r13d
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 ***v26; // r14
  __int64 v27; // r15
  __int64 v28; // r12
  __int64 v29; // rax
  unsigned __int8 *v30; // rsi
  __int64 **v31; // rdx
  __int64 v32; // rax
  __int64 *v33; // rsi
  __int64 *v34; // r12
  bool v35; // al
  int v36; // r13d
  __int64 v37; // rax
  __int64 *v38; // r14
  __int64 v39; // rsi
  unsigned int v40; // eax
  unsigned int v41; // r14d
  char v42; // al
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdx
  unsigned __int8 *v50; // rsi
  const char *v52; // rax
  __int64 *v53; // rdi
  __int64 v54; // r13
  unsigned __int64 v55; // r14
  __int64 v56; // rsi
  __int64 v57; // r13
  char v58; // al
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // r13
  __int64 v62; // rax
  __int64 v63; // r14
  unsigned __int8 v64; // al
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // r13
  __int64 v68; // r12
  __int64 *v69; // rbx
  __int64 v70; // rax
  _BYTE *v71; // r15
  _BYTE *v72; // r14
  char *v73; // r12
  __int64 *v74; // rax
  __int64 v75; // rbx
  unsigned int v76; // r13d
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // r12
  int v80; // r12d
  __int64 v81; // rax
  __int64 v82; // rdx
  int v83; // eax
  __int64 v84; // r12
  __int64 v85; // r13
  char *v86; // rsi
  unsigned int v87; // eax
  char **v88; // rdi
  char *v89; // r10
  __int64 v90; // rax
  char v91; // al
  __int64 v92; // rsi
  __int64 v93; // rcx
  char v94; // al
  int v95; // r8d
  int v96; // r9d
  __int64 *v97; // r12
  __int64 v98; // rax
  _QWORD *v99; // rax
  char *v100; // r12
  __int64 *v101; // rax
  int v102; // edi
  int v103; // ecx
  int v104; // eax
  unsigned int v105; // esi
  int v106; // eax
  char *v107; // rax
  __int64 v108; // [rsp+8h] [rbp-148h]
  __int64 *v109; // [rsp+10h] [rbp-140h]
  __int64 v113; // [rsp+40h] [rbp-110h]
  __int64 *v114; // [rsp+40h] [rbp-110h]
  __int64 v115; // [rsp+48h] [rbp-108h]
  __int64 *v116; // [rsp+50h] [rbp-100h]
  __int64 v117; // [rsp+50h] [rbp-100h]
  __int64 v118; // [rsp+58h] [rbp-F8h]
  _BYTE *v119; // [rsp+58h] [rbp-F8h]
  __int64 *v120; // [rsp+58h] [rbp-F8h]
  char *v121; // [rsp+60h] [rbp-F0h] BYREF
  char *v122; // [rsp+68h] [rbp-E8h] BYREF
  __int64 *v123; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v124; // [rsp+78h] [rbp-D8h]
  __int64 v125; // [rsp+80h] [rbp-D0h]
  const char *v126; // [rsp+90h] [rbp-C0h] BYREF
  const char *v127; // [rsp+98h] [rbp-B8h]
  __int64 v128; // [rsp+A0h] [rbp-B0h]
  unsigned __int8 *v129; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v130; // [rsp+B8h] [rbp-98h]
  __int64 v131; // [rsp+C0h] [rbp-90h]
  __int64 v132; // [rsp+C8h] [rbp-88h]
  __int64 *v133; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v134; // [rsp+D8h] [rbp-78h]
  __int64 v135; // [rsp+E0h] [rbp-70h]
  __int64 v136; // [rsp+E8h] [rbp-68h]
  __int64 v137; // [rsp+F0h] [rbp-60h]
  int v138; // [rsp+F8h] [rbp-58h]
  __int64 v139; // [rsp+100h] [rbp-50h]
  __int64 v140; // [rsp+108h] [rbp-48h]

  v10 = sub_157EBA0(a2);
  v11 = *(_DWORD *)(a3 + 8);
  if ( a6 )
  {
    if ( v11 )
    {
      v12 = v10;
      v13 = 0;
      v14 = 0;
      v15 = 0;
      v16 = a3;
      do
      {
        while ( 1 )
        {
          v17 = *(__int64 **)(*(_QWORD *)v16 + 8 * v14);
          v18 = v17[2];
          if ( a2 == *(_QWORD *)(v18 + 40) )
            break;
          v14 = (unsigned int)(v13 + 1);
          v13 = v14;
          if ( *(_DWORD *)(v16 + 8) == (_DWORD)v14 )
            goto LABEL_9;
        }
        v116 = v15;
        v118 = v12;
        v19 = sub_15CCEE0(a4, v17[2], v12);
        v15 = v116;
        v12 = v118;
        v20 = v19 == 0;
        v14 = (unsigned int)(v13 + 1);
        if ( !v20 )
        {
          v15 = v17;
          v12 = v18;
        }
        ++v13;
      }
      while ( *(_DWORD *)(v16 + 8) != (_DWORD)v14 );
LABEL_9:
      a3 = v16;
      v10 = v12;
      v21 = v15;
      if ( v15 )
      {
LABEL_10:
        v121 = 0;
        if ( sub_14560B0(*v21) )
        {
          v22 = v121;
          v119 = (_BYTE *)v21[3];
LABEL_12:
          if ( v22 && !sub_14560B0((__int64)v22) )
            v117 = sub_1480620(*(_QWORD *)(a1 + 184), (__int64)v121, 0);
          else
            v117 = 0;
          v23 = 0;
          v24 = 0;
          if ( !*(_DWORD *)(a3 + 8) )
            return 1;
          while ( 1 )
          {
            v34 = *(__int64 **)(*(_QWORD *)a3 + 8 * v24);
            if ( !sub_14560B0(*v34) )
              break;
            if ( !v117 )
            {
              v26 = (__int64 ***)v34[3];
              v27 = (__int64)v119;
              v28 = v34[2];
              goto LABEL_22;
            }
            if ( !sub_14560B0(*v34) )
              goto LABEL_19;
            v26 = (__int64 ***)v34[3];
            v25 = v117;
            v28 = v34[2];
LABEL_21:
            v113 = v25;
            v27 = (__int64)v119;
            if ( !sub_14560B0(v25) )
            {
              v43 = sub_145DC80(*(_QWORD *)(a1 + 184), (__int64)v119);
              v44 = sub_13A5B00(*(_QWORD *)(a1 + 184), v43, v113, 0, 0);
              v27 = sub_38767A0(a5, v44, 0, v28);
              if ( *(_BYTE *)(v27 + 16) > 0x17u
                && (*(_QWORD *)(v27 + 40) != *(_QWORD *)(v28 + 40) || !sub_15CCEE0(a4, v27, v28)) )
              {
                v27 = sub_15F4880(v27);
                LOWORD(v135) = 259;
                v133 = (__int64 *)"scevcgp_";
                sub_164B780(v27, (__int64 *)&v133);
                sub_15F2120(v27, v28);
              }
            }
LABEL_22:
            if ( *(__int64 ***)v27 != *v26 )
            {
              v29 = sub_16498A0(v28);
              v133 = 0;
              v136 = v29;
              v137 = 0;
              v138 = 0;
              v139 = 0;
              v140 = 0;
              v134 = *(_QWORD *)(v28 + 40);
              v135 = v28 + 24;
              v30 = *(unsigned __int8 **)(v28 + 48);
              v129 = v30;
              if ( v30 )
              {
                sub_1623A60((__int64)&v129, (__int64)v30, 2);
                if ( v133 )
                  sub_161E7C0((__int64)&v133, (__int64)v133);
                v133 = (__int64 *)v129;
                if ( v129 )
                  sub_1623210((__int64)&v129, v129, (__int64)&v133);
              }
              v126 = "scevcgptmp_";
              LOWORD(v128) = 259;
              v31 = *v26;
              if ( *v26 != *(__int64 ***)v27 )
              {
                if ( *(_BYTE *)(v27 + 16) <= 0x10u )
                {
                  v32 = sub_15A46C0(47, (__int64 ***)v27, v31, 0);
                  v33 = v133;
                  v27 = v32;
                  goto LABEL_31;
                }
                LOWORD(v131) = 257;
                v45 = sub_15FDBD0(47, v27, (__int64)v31, (__int64)&v129, 0);
                v27 = v45;
                if ( v134 )
                {
                  v114 = (__int64 *)v135;
                  sub_157E9D0(v134 + 40, v45);
                  v46 = *v114;
                  v47 = *(_QWORD *)(v27 + 24) & 7LL;
                  *(_QWORD *)(v27 + 32) = v114;
                  v46 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v27 + 24) = v46 | v47;
                  *(_QWORD *)(v46 + 8) = v27 + 24;
                  *v114 = *v114 & 7 | (v27 + 24);
                }
                sub_164B780(v27, (__int64 *)&v126);
                if ( !v133 )
                  goto LABEL_33;
                v123 = v133;
                sub_1623A60((__int64)&v123, (__int64)v133, 2);
                v48 = *(_QWORD *)(v27 + 48);
                v49 = v27 + 48;
                if ( v48 )
                {
                  sub_161E7C0(v27 + 48, v48);
                  v49 = v27 + 48;
                }
                v50 = (unsigned __int8 *)v123;
                *(_QWORD *)(v27 + 48) = v123;
                if ( v50 )
                  sub_1623210((__int64)&v123, v50, v49);
              }
              v33 = v133;
LABEL_31:
              if ( v33 )
                sub_161E7C0((__int64)&v133, (__int64)v33);
            }
LABEL_33:
            sub_1648780(v28, (__int64)v26, v27);
            v24 = (unsigned int)(v23 + 1);
            v23 = v24;
            if ( (_DWORD)v24 == *(_DWORD *)(a3 + 8) )
              return 1;
          }
          v35 = sub_14560B0(*v34);
          v25 = v117;
          if ( !v35 )
          {
            if ( v117 )
LABEL_19:
              v25 = sub_13A5B00(*(_QWORD *)(a1 + 184), *v34, v117, 0, 0);
            else
              v25 = *v34;
          }
          v26 = (__int64 ***)v34[3];
          v27 = (__int64)v119;
          v28 = v34[2];
          if ( !v25 )
            goto LABEL_22;
          goto LABEL_21;
        }
        if ( !dword_4FBC3E0 )
        {
          v22 = (char *)*v21;
          v121 = (char *)*v21;
          v119 = (_BYTE *)v21[3];
          goto LABEL_12;
        }
        v129 = (unsigned __int8 *)v21[1];
        v58 = sub_1C56280(a1, (__int64 *)&v129, &v133);
        v59 = v21[3];
        if ( v58 )
        {
          if ( v133 != (__int64 *)(*(_QWORD *)(a1 + 8) + 32LL * *(unsigned int *)(a1 + 24)) )
          {
            v66 = *((unsigned int *)v133 + 4);
            if ( (_DWORD)v66 )
            {
              v120 = v21;
              v67 = 0;
              v68 = 8 * v66;
              v115 = a3;
              v69 = v133;
              while ( 1 )
              {
                v70 = v69[1];
                v71 = *(_BYTE **)(v70 + v67);
                if ( v71[16] > 0x17u && *(_BYTE *)(v59 + 16) > 0x17u )
                {
                  if ( sub_15CCEE0(*(_QWORD *)(a1 + 200), *(_QWORD *)(v70 + v67), v59) )
                  {
                    v119 = v71;
                    a3 = v115;
                    v22 = v121;
                    goto LABEL_12;
                  }
                  v59 = v120[3];
                }
                v67 += 8;
                if ( v67 == v68 )
                {
                  v21 = v120;
                  a3 = v115;
                  break;
                }
              }
            }
          }
        }
        v60 = *v21;
        v126 = (const char *)v59;
        v127 = (const char *)v60;
        v61 = a1 + 112;
        if ( (unsigned __int8)sub_1C56330(a1 + 112, (__int64 *)&v126, &v133) )
        {
          if ( v133 != (__int64 *)(*(_QWORD *)(a1 + 120) + 32LL * *(unsigned int *)(a1 + 136)) )
          {
            v62 = v133[2];
            v119 = (_BYTE *)v62;
            if ( v62 )
            {
              v22 = (char *)v133[3];
              if ( *(_BYTE *)(v62 + 16) <= 0x17u
                || *(_BYTE *)(v21[3] + 16) <= 0x17u
                || sub_15CCEE0(*(_QWORD *)(a1 + 200), v62, v21[3]) )
              {
                v121 = v22;
                goto LABEL_12;
              }
            }
          }
        }
        v63 = sub_1649C60(v21[3]);
        v64 = *(_BYTE *)(v63 + 16);
        if ( v64 <= 0x17u )
        {
          if ( v64 != 5 || *(_WORD *)(v63 + 18) != 32 )
            goto LABEL_106;
        }
        else if ( v64 != 56 )
        {
          goto LABEL_106;
        }
        v65 = *v21;
        v129 = (unsigned __int8 *)v63;
        v130 = v65;
        sub_1C55950(&v133, (__int64 *)(a1 + 144), (__int64 *)&v129);
        if ( v135 != *(_QWORD *)(a1 + 152) + 32LL * *(unsigned int *)(a1 + 168) )
        {
          v22 = *(char **)(v135 + 24);
          v119 = *(_BYTE **)(v135 + 16);
          v121 = v22;
          goto LABEL_12;
        }
        v92 = v21[3];
        v93 = *(_QWORD *)v92;
        if ( *(_BYTE *)(v92 + 16) <= 0x17u )
          v92 = 0;
        v119 = sub_1C5BCD0(a1, v92, a5, v93, v63, *v21, a7, a8, (__int64 *)&v121);
        if ( v119 )
        {
          if ( v121 && !sub_14560B0((__int64)v121) )
            goto LABEL_144;
          v122 = (char *)v21[1];
          v94 = sub_1C56280(a1, (__int64 *)&v122, &v123);
          v97 = v123;
          if ( v94 )
          {
            v98 = *((unsigned int *)v123 + 4);
            if ( (unsigned int)v98 >= *((_DWORD *)v123 + 5) )
            {
              sub_16CD150((__int64)(v123 + 1), v123 + 3, 0, 8, v95, v96);
              v99 = (_QWORD *)(v97[1] + 8LL * *((unsigned int *)v97 + 4));
            }
            else
            {
              v99 = (_QWORD *)(v123[1] + 8 * v98);
            }
LABEL_143:
            *v99 = v119;
            ++*((_DWORD *)v97 + 4);
LABEL_144:
            v100 = v121;
            v101 = sub_1C5B220(v61, (__int64 *)&v126);
            v101[3] = (__int64)v100;
            v101[2] = (__int64)v119;
            v22 = v121;
            goto LABEL_12;
          }
          v104 = *(_DWORD *)(a1 + 16);
          v105 = *(_DWORD *)(a1 + 24);
          ++*(_QWORD *)a1;
          v106 = v104 + 1;
          if ( 4 * v106 >= 3 * v105 )
          {
            v105 *= 2;
          }
          else if ( v105 - *(_DWORD *)(a1 + 20) - v106 > v105 >> 3 )
          {
LABEL_155:
            *(_DWORD *)(a1 + 16) = v106;
            if ( *v97 != -8 )
              --*(_DWORD *)(a1 + 20);
            v107 = v122;
            v97[2] = 0x100000000LL;
            *v97 = (__int64)v107;
            v99 = v97 + 3;
            v97[1] = (__int64)(v97 + 3);
            goto LABEL_143;
          }
          sub_1C5ACF0(a1, v105);
          sub_1C56280(a1, (__int64 *)&v122, &v123);
          v97 = v123;
          v106 = *(_DWORD *)(a1 + 16) + 1;
          goto LABEL_155;
        }
LABEL_106:
        v72 = (_BYTE *)v21[3];
        v73 = (char *)*v21;
        v119 = v72;
        v121 = v73;
        v74 = sub_1C5B220(v61, (__int64 *)&v126);
        v74[2] = (__int64)v72;
        v74[3] = (__int64)v73;
        v22 = v121;
        goto LABEL_12;
      }
    }
    goto LABEL_81;
  }
  if ( !v11 )
    goto LABEL_81;
  v36 = 0;
  v21 = 0;
  v37 = 0;
  do
  {
    v38 = *(__int64 **)(*(_QWORD *)a3 + 8 * v37);
    v20 = !sub_14560B0(*v38);
    v37 = (unsigned int)(v36 + 1);
    if ( !v20 )
      v21 = v38;
    ++v36;
  }
  while ( (_DWORD)v37 != *(_DWORD *)(a3 + 8) );
  if ( !v21 )
LABEL_81:
    v21 = **(__int64 ***)a3;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v39 = v21[3];
  if ( *(_BYTE *)(v39 + 16) <= 0x17u )
    goto LABEL_10;
  v121 = (char *)v21[3];
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  LOBYTE(v40) = sub_15CCEE0(a4, v39, v10);
  v41 = v40;
  if ( (_BYTE)v40 )
    goto LABEL_74;
  v42 = v121[16];
  if ( (unsigned __int8)(v42 - 77) > 1u && v42 != 54 )
  {
    sub_1C55D10((__int64)&v126, &v121);
    sub_1A64820((__int64)&v133, (__int64)&v129, (__int64 *)&v121);
    sub_1C55D10((__int64)&v123, &v121);
    v52 = v127;
    if ( v126 == v127 )
      goto LABEL_73;
    v109 = v21;
    v108 = a3;
    while ( 1 )
    {
      v75 = *((_QWORD *)v52 - 1);
      v127 = v52 - 8;
      v76 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
      if ( *(_BYTE *)(v75 + 16) != 78 )
        goto LABEL_118;
      if ( *(char *)(v75 + 23) < 0 )
      {
        v77 = sub_1648A40(v75);
        v79 = v77 + v78;
        if ( *(char *)(v75 + 23) >= 0 )
        {
          if ( (unsigned int)(v79 >> 4) )
LABEL_164:
            BUG();
        }
        else if ( (unsigned int)((v79 - sub_1648A40(v75)) >> 4) )
        {
          if ( *(char *)(v75 + 23) >= 0 )
            goto LABEL_164;
          v80 = *(_DWORD *)(sub_1648A40(v75) + 8);
          if ( *(char *)(v75 + 23) >= 0 )
            BUG();
          v81 = sub_1648A40(v75);
          v83 = *(_DWORD *)(v81 + v82 - 4) - v80;
          goto LABEL_117;
        }
      }
      v83 = 0;
LABEL_117:
      v76 = v76 - 1 - v83;
LABEL_118:
      v84 = v76;
      v85 = 0;
      if ( (_DWORD)v84 )
      {
        do
        {
          if ( *(_BYTE *)(v75 + 16) == 78 )
          {
            v86 = *(char **)(v75 + 24 * (v85 - (*(_DWORD *)(v75 + 20) & 0xFFFFFFF)));
            if ( !v86 )
              BUG();
          }
          else
          {
            if ( (*(_BYTE *)(v75 + 23) & 0x40) != 0 )
              v90 = *(_QWORD *)(v75 - 8);
            else
              v90 = v75 - 24LL * (*(_DWORD *)(v75 + 20) & 0xFFFFFFF);
            v86 = *(char **)(v90 + 24 * v85);
          }
          if ( (unsigned __int8)v86[16] <= 0x17u )
            goto LABEL_126;
          v122 = v86;
          if ( sub_15CCEE0(a4, (__int64)v86, v10) )
            goto LABEL_126;
          if ( (_DWORD)v132 )
          {
            v87 = (v132 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
            v88 = (char **)(v130 + 8LL * v87);
            v89 = *v88;
            if ( *v88 == v122 )
            {
LABEL_125:
              if ( v88 != (char **)(v130 + 8LL * (unsigned int)v132) )
                goto LABEL_126;
            }
            else
            {
              v102 = 1;
              while ( v89 != (char *)-8LL )
              {
                v103 = v102 + 1;
                v87 = (v132 - 1) & (v87 + v102);
                v88 = (char **)(v130 + 8LL * v87);
                v89 = *v88;
                if ( v122 == *v88 )
                  goto LABEL_125;
                v102 = v103;
              }
            }
          }
          v91 = v122[16];
          if ( (unsigned __int8)(v91 - 77) <= 1u || v91 == 54 )
          {
            v41 = (unsigned __int8)v41;
            goto LABEL_48;
          }
          sub_1A64820((__int64)&v133, (__int64)&v129, (__int64 *)&v122);
          sub_1C55D10((__int64)&v123, &v122);
          sub_1C55D10((__int64)&v126, &v122);
LABEL_126:
          ++v85;
        }
        while ( v84 != v85 );
      }
      v52 = v127;
      if ( v127 == v126 )
      {
        v21 = v109;
        a3 = v108;
LABEL_73:
        sub_1CCBEA0(&v129, &v123);
LABEL_74:
        if ( v126 )
          j_j___libc_free_0(v126, v128 - (_QWORD)v126);
        j___libc_free_0(v130);
        v53 = v123;
        v54 = (v124 - (__int64)v123) >> 3;
        if ( (_DWORD)v54 )
        {
          v55 = 0;
          v56 = v10;
          v57 = 8LL * (unsigned int)v54;
          do
          {
            sub_15F22F0((_QWORD *)v53[v55 / 8], v56);
            v53 = v123;
            v56 = v123[v55 / 8];
            v55 += 8LL;
          }
          while ( v55 != v57 );
        }
        if ( v53 )
          j_j___libc_free_0(v53, v125 - (_QWORD)v53);
        goto LABEL_10;
      }
    }
  }
LABEL_48:
  if ( v126 )
    j_j___libc_free_0(v126, v128 - (_QWORD)v126);
  j___libc_free_0(v130);
  if ( v123 )
    j_j___libc_free_0(v123, v125 - (_QWORD)v123);
  return v41;
}
