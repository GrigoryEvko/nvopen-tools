// Function: sub_33C00D0
// Address: 0x33c00d0
//
void __fastcall sub_33C00D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rax
  char v13; // si
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // rdi
  unsigned int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 *v31; // r15
  __int64 *v32; // r12
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // r12
  __int64 v36; // rdx
  __int64 v37; // r13
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // rdx
  int v43; // r9d
  __int64 v44; // r10
  __int64 v45; // r11
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rax
  int v49; // edx
  __int64 v50; // rbx
  int v51; // r12d
  unsigned int v52; // eax
  __int64 v53; // r14
  __int64 v54; // rax
  __int64 v55; // r15
  __int64 v56; // rdx
  __int64 v57; // rax
  _DWORD *v58; // rax
  unsigned __int16 v59; // r8
  int v60; // eax
  int v61; // edx
  __int64 v62; // rax
  __int64 v63; // rsi
  int v64; // edx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rdi
  __int64 v69; // rdx
  int v70; // eax
  int v71; // edx
  __int64 v72; // r15
  int v73; // ecx
  int v74; // r11d
  __int64 v75; // rax
  __int64 v76; // rsi
  __int64 v77; // rax
  int v78; // edx
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // r14
  __int64 v83; // r14
  __int64 v84; // rbx
  __int64 v85; // rax
  int v86; // r13d
  __int64 v87; // r14
  __int64 v88; // r12
  __int64 v89; // rax
  bool v90; // al
  int v91; // r14d
  __int64 v92; // r14
  __int64 v93; // rax
  __int64 v94; // r15
  __int64 v95; // rdx
  __int64 v96; // rax
  _DWORD *v97; // rax
  unsigned __int16 v98; // r8
  int v99; // eax
  int v100; // edx
  __int64 v101; // rax
  __int64 v102; // rsi
  int v103; // edx
  __int64 v104; // rdi
  int v105; // eax
  int v106; // edx
  int v107; // ecx
  int v108; // r11d
  __int64 v109; // rax
  __int64 v110; // rsi
  __int64 v111; // rax
  __int64 v112; // rdx
  __int128 v113; // [rsp-20h] [rbp-140h]
  __int128 v114; // [rsp-10h] [rbp-130h]
  __int128 v115; // [rsp-10h] [rbp-130h]
  __int128 v116; // [rsp-10h] [rbp-130h]
  int v117; // [rsp+0h] [rbp-120h]
  int v118; // [rsp+0h] [rbp-120h]
  int v119; // [rsp+8h] [rbp-118h]
  int v120; // [rsp+8h] [rbp-118h]
  int v121; // [rsp+8h] [rbp-118h]
  __int64 (__fastcall *v122)(__int64, __int64, unsigned int); // [rsp+10h] [rbp-110h]
  unsigned __int16 v123; // [rsp+10h] [rbp-110h]
  __int64 v124; // [rsp+10h] [rbp-110h]
  __int64 v125; // [rsp+10h] [rbp-110h]
  __int64 (__fastcall *v126)(__int64, __int64, unsigned int); // [rsp+10h] [rbp-110h]
  unsigned __int16 v127; // [rsp+10h] [rbp-110h]
  __int64 v128; // [rsp+20h] [rbp-100h]
  __int64 v129; // [rsp+28h] [rbp-F8h]
  __int64 v130; // [rsp+30h] [rbp-F0h]
  __int64 v131; // [rsp+30h] [rbp-F0h]
  __int64 v132; // [rsp+38h] [rbp-E8h]
  __int64 v133; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v134; // [rsp+A8h] [rbp-78h]
  __int64 *v135; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v136; // [rsp+B8h] [rbp-68h]
  __int64 v137; // [rsp+C0h] [rbp-60h] BYREF
  int v138; // [rsp+C8h] [rbp-58h]
  __int64 v139; // [rsp+D0h] [rbp-50h]
  __int64 v140; // [rsp+D8h] [rbp-48h]
  __int64 v141; // [rsp+E0h] [rbp-40h]
  __int64 v142; // [rsp+E8h] [rbp-38h]

  v6 = a2;
  v7 = a1;
  v8 = *(_QWORD *)(a1 + 960);
  v9 = *(_QWORD *)(a2 - 32);
  v10 = *(_QWORD *)(v8 + 56);
  v11 = *(_QWORD *)(v8 + 744);
  v129 = *(_QWORD *)(v10 + 8LL * *(unsigned int *)(*(_QWORD *)(a2 - 96) + 44LL));
  v130 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)v9 )
  {
    if ( *(_BYTE *)v9 != 25 )
    {
LABEL_3:
      if ( *(char *)(a2 + 7) >= 0 )
        goto LABEL_103;
      v12 = sub_BD2BC0(a2);
      v13 = *(_BYTE *)(a2 + 7);
      v15 = v12 + v14;
      if ( v13 >= 0 )
      {
        v17 = v15 >> 4;
        if ( !v17 )
          goto LABEL_103;
      }
      else
      {
        v16 = sub_BD2BC0(v6);
        v13 = *(_BYTE *)(v6 + 7);
        v17 = (v15 - v16) >> 4;
        if ( !v17 )
          goto LABEL_69;
      }
      v18 = v6;
      v19 = 0;
      v20 = 16LL * v17;
      do
      {
        v21 = 0;
        if ( v13 < 0 )
          v21 = sub_BD2BC0(v18);
        if ( !*(_DWORD *)(*(_QWORD *)(v21 + v19) + 8LL) )
        {
          v6 = v18;
          v7 = a1;
          v22 = sub_338B750(a1, v9);
          sub_343D050(a1, v6, v22, v23, v130);
          goto LABEL_12;
        }
        v19 += 16;
        v13 = *(_BYTE *)(v18 + 7);
      }
      while ( v20 != v19 );
      v6 = v18;
      v7 = a1;
LABEL_69:
      if ( v13 < 0 )
      {
        v80 = sub_BD2BC0(v6);
        v82 = v80 + v81;
        if ( *(char *)(v6 + 7) < 0 )
          v82 -= sub_BD2BC0(v6);
        v83 = v82 >> 4;
        if ( (_DWORD)v83 )
        {
          v125 = v11;
          v128 = v7;
          v84 = 0;
          v85 = 16LL * (unsigned int)v83;
          v86 = 0;
          v87 = v6;
          v88 = v85;
          do
          {
            v89 = 0;
            if ( *(char *)(v87 + 7) < 0 )
              v89 = sub_BD2BC0(v87);
            v90 = *(_DWORD *)(*(_QWORD *)(v89 + v84) + 8LL) == 7;
            v84 += 16;
            v86 += v90;
          }
          while ( v88 != v84 );
          v6 = v87;
          v91 = v86;
          v7 = v128;
          v11 = v125;
          if ( v91 )
          {
            sub_33AB4F0(v128, v6, v130);
            goto LABEL_12;
          }
        }
      }
LABEL_103:
      v111 = sub_338B750(v7, v9);
      sub_33A7C00(v7, (unsigned __int8 *)v6, v111, v112, 0, 0, v130, 0);
      goto LABEL_12;
    }
    sub_338BA40((__int64 *)a1, (int *)a2, v130);
  }
  else
  {
    if ( (*(_BYTE *)(v9 + 33) & 0x20) == 0 )
      goto LABEL_3;
    v52 = *(_DWORD *)(v9 + 36);
    if ( v52 <= 0x9D )
    {
      if ( v52 > 0x9B )
      {
        sub_33AEC60(a1, a2, v130);
        goto LABEL_12;
      }
      if ( v52 != 73 )
      {
        if ( v52 == 151 )
        {
          sub_343D170(a1, a2, v130);
          goto LABEL_12;
        }
        goto LABEL_110;
      }
    }
    else
    {
      if ( v52 == 14230 )
      {
        v92 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
        v93 = sub_3373A60(a1, a2, v10, a4, a5, a6);
        v94 = *(_QWORD *)(a1 + 864);
        v135 = (__int64 *)v93;
        v136 = v95;
        v126 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v92 + 32LL);
        v96 = sub_2E79000(*(__int64 **)(v94 + 40));
        if ( v126 == sub_2D42F30 )
        {
          v97 = sub_AE2980(v96, 0);
          v98 = 2;
          v99 = v97[1];
          if ( v99 != 1 )
          {
            v98 = 3;
            if ( v99 != 2 )
            {
              v98 = 4;
              if ( v99 != 4 )
              {
                v98 = 5;
                if ( v99 != 8 )
                {
                  v98 = 6;
                  if ( v99 != 16 )
                  {
                    v98 = 7;
                    if ( v99 != 32 )
                    {
                      v98 = 8;
                      if ( v99 != 64 )
                        v98 = 9 * (v99 == 128);
                    }
                  }
                }
              }
            }
          }
        }
        else
        {
          v98 = v126(v92, v96, 0);
        }
        v100 = *(_DWORD *)(a1 + 848);
        v101 = *(_QWORD *)a1;
        v133 = 0;
        LODWORD(v134) = v100;
        if ( v101 )
        {
          if ( &v133 != (__int64 *)(v101 + 48) )
          {
            v102 = *(_QWORD *)(v101 + 48);
            v133 = v102;
            if ( v102 )
            {
              v127 = v98;
              sub_B96E90((__int64)&v133, v102, 1);
              v98 = v127;
            }
          }
        }
        v137 = sub_3400BD0(v94, 14230, (unsigned int)&v133, v98, 0, 1, 0);
        v138 = v103;
        if ( v133 )
          sub_B91220((__int64)&v133, v133);
        v104 = *(_QWORD *)(a1 + 864);
        LOWORD(v133) = 1;
        v134 = 0;
        v105 = sub_33E5830(v104, &v133);
        v133 = 0;
        v72 = *(_QWORD *)(v7 + 864);
        v107 = v105;
        v108 = v106;
        v109 = *(_QWORD *)v7;
        LODWORD(v134) = *(_DWORD *)(v7 + 848);
        if ( v109 )
        {
          if ( &v133 != (__int64 *)(v109 + 48) )
          {
            v110 = *(_QWORD *)(v109 + 48);
            v133 = v110;
            if ( v110 )
            {
              v118 = v106;
              v121 = v107;
              sub_B96E90((__int64)&v133, v110, 1);
              v108 = v118;
              v107 = v121;
            }
          }
        }
        *((_QWORD *)&v116 + 1) = 2;
        *(_QWORD *)&v116 = &v135;
        v77 = sub_3411630(v72, 48, (unsigned int)&v133, v107, v108, 2, v116);
        if ( v77 )
          goto LABEL_55;
        *(_QWORD *)(v72 + 384) = 0;
        *(_DWORD *)(v72 + 392) = v78;
        goto LABEL_56;
      }
      if ( v52 > 0x3796 )
      {
        if ( v52 == 14249 )
        {
          v53 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
          v54 = sub_3373A60(a1, a2, v10, a4, a5, a6);
          v55 = *(_QWORD *)(a1 + 864);
          v135 = (__int64 *)v54;
          v136 = v56;
          v122 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v53 + 32LL);
          v57 = sub_2E79000(*(__int64 **)(v55 + 40));
          if ( v122 == sub_2D42F30 )
          {
            v58 = sub_AE2980(v57, 0);
            v59 = 2;
            v60 = v58[1];
            if ( v60 != 1 )
            {
              v59 = 3;
              if ( v60 != 2 )
              {
                v59 = 4;
                if ( v60 != 4 )
                {
                  v59 = 5;
                  if ( v60 != 8 )
                  {
                    v59 = 6;
                    if ( v60 != 16 )
                    {
                      v59 = 7;
                      if ( v60 != 32 )
                      {
                        v59 = 8;
                        if ( v60 != 64 )
                          v59 = 9 * (v60 == 128);
                      }
                    }
                  }
                }
              }
            }
          }
          else
          {
            v59 = v122(v53, v57, 0);
          }
          v61 = *(_DWORD *)(a1 + 848);
          v62 = *(_QWORD *)a1;
          v133 = 0;
          LODWORD(v134) = v61;
          if ( v62 )
          {
            if ( &v133 != (__int64 *)(v62 + 48) )
            {
              v63 = *(_QWORD *)(v62 + 48);
              v133 = v63;
              if ( v63 )
              {
                v123 = v59;
                sub_B96E90((__int64)&v133, v63, 1);
                v59 = v123;
              }
            }
          }
          v137 = sub_3400BD0(v55, 14249, (unsigned int)&v133, v59, 0, 1, 0);
          v138 = v64;
          if ( v133 )
            sub_B91220((__int64)&v133, v133);
          v65 = sub_338B750(a1, *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
          v140 = v66;
          LODWORD(v66) = *(_DWORD *)(v6 + 4);
          v139 = v65;
          v67 = sub_338B750(a1, *(_QWORD *)(v6 + 32 * (1 - (v66 & 0x7FFFFFF))));
          v68 = *(_QWORD *)(a1 + 864);
          v142 = v69;
          LOWORD(v133) = 1;
          v141 = v67;
          v134 = 0;
          v70 = sub_33E5830(v68, &v133);
          v133 = 0;
          v72 = *(_QWORD *)(v7 + 864);
          v73 = v70;
          v74 = v71;
          v75 = *(_QWORD *)v7;
          LODWORD(v134) = *(_DWORD *)(v7 + 848);
          if ( v75 )
          {
            if ( &v133 != (__int64 *)(v75 + 48) )
            {
              v76 = *(_QWORD *)(v75 + 48);
              v133 = v76;
              if ( v76 )
              {
                v117 = v71;
                v119 = v73;
                sub_B96E90((__int64)&v133, v76, 1);
                v74 = v117;
                v73 = v119;
              }
            }
          }
          *((_QWORD *)&v115 + 1) = 4;
          *(_QWORD *)&v115 = &v135;
          v77 = sub_3411630(v72, 48, (unsigned int)&v133, v73, v74, 4, v115);
          if ( v77 )
          {
LABEL_55:
            v120 = v78;
            v124 = v77;
            nullsub_1875(v77, v72, 0);
            *(_QWORD *)(v72 + 384) = v124;
            *(_DWORD *)(v72 + 392) = v120;
            sub_33E2B60(v72, 0);
            goto LABEL_56;
          }
          *(_QWORD *)(v72 + 384) = 0;
          *(_DWORD *)(v72 + 392) = v78;
LABEL_56:
          if ( v133 )
            sub_B91220((__int64)&v133, v133);
          goto LABEL_12;
        }
LABEL_110:
        BUG();
      }
      if ( v52 - 316 > 3 )
        goto LABEL_110;
    }
    v79 = *(_QWORD *)(v10 + 8LL * *(unsigned int *)(v130 + 44));
    if ( !v79 )
      goto LABEL_14;
    *(_BYTE *)(v79 + 217) = 1;
  }
LABEL_12:
  v9 = *(_QWORD *)(v6 - 32);
  if ( !v9 || *(_BYTE *)v9 )
    goto LABEL_15;
LABEL_14:
  if ( *(_QWORD *)(v9 + 24) != *(_QWORD *)(v6 + 80) || *(_DWORD *)(v9 + 36) != 151 )
LABEL_15:
    sub_33BFCB0(v7, v6);
  v24 = *(__int64 **)(v7 + 960);
  v25 = 0;
  v135 = &v137;
  v136 = 0x100000000LL;
  if ( v24[4] )
  {
    v26 = sub_FF0430(v24[4], *(_QWORD *)(v11 + 16), v130);
    v24 = *(__int64 **)(v7 + 960);
    v25 = v26;
  }
  sub_33669C0(v24, v130, v25, (__int64)&v135);
  sub_3373E10(v7, v11, v129, 0xFFFFFFFFLL, v27, v28);
  v31 = v135;
  v32 = &v135[2 * (unsigned int)v136];
  if ( v32 != v135 )
  {
    do
    {
      v33 = *v31;
      v31 += 2;
      *(_BYTE *)(v33 + 216) = 1;
      sub_3373E10(v7, v11, *(v31 - 2), *((unsigned int *)v31 - 2), v29, v30);
    }
    while ( v32 != v31 );
  }
  sub_2E33470(*(unsigned int **)(v11 + 144), *(unsigned int **)(v11 + 152));
  v34 = *(_QWORD *)(v7 + 864);
  v35 = sub_33EEAD0(v34, v129);
  v37 = v36;
  v41 = sub_3373A60(v7, v129, v36, v38, v39, v40);
  v133 = 0;
  v44 = v41;
  v45 = v42;
  v46 = *(_QWORD *)v7;
  LODWORD(v134) = *(_DWORD *)(v7 + 848);
  if ( v46 )
  {
    if ( &v133 != (__int64 *)(v46 + 48) )
    {
      v47 = *(_QWORD *)(v46 + 48);
      v133 = v47;
      if ( v47 )
      {
        v131 = v44;
        v132 = v42;
        sub_B96E90((__int64)&v133, v47, 1);
        v44 = v131;
        v45 = v132;
      }
    }
  }
  *((_QWORD *)&v114 + 1) = v37;
  *(_QWORD *)&v114 = v35;
  *((_QWORD *)&v113 + 1) = v45;
  *(_QWORD *)&v113 = v44;
  v48 = sub_3406EB0(v34, 301, (unsigned int)&v133, 1, 0, v43, v113, v114);
  v50 = v48;
  v51 = v49;
  if ( v48 )
  {
    nullsub_1875(v48, v34, 0);
    *(_QWORD *)(v34 + 384) = v50;
    *(_DWORD *)(v34 + 392) = v51;
    sub_33E2B60(v34, 0);
  }
  else
  {
    *(_QWORD *)(v34 + 384) = 0;
    *(_DWORD *)(v34 + 392) = v49;
  }
  if ( v133 )
    sub_B91220((__int64)&v133, v133);
  if ( v135 != &v137 )
    _libc_free((unsigned __int64)v135);
}
