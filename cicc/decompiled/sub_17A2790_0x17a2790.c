// Function: sub_17A2790
// Address: 0x17a2790
//
unsigned __int64 __fastcall sub_17A2790(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4, unsigned int a5, __int64 a6)
{
  __int64 *v6; // rax
  unsigned int v11; // r15d
  __int64 v12; // rcx
  int v13; // edi
  __int64 v14; // rsi
  __int64 v15; // r8
  __int64 v16; // rdx
  unsigned int v17; // ecx
  __int64 v18; // r14
  unsigned int v19; // r15d
  __int64 v20; // r12
  __int64 v21; // r12
  bool v22; // cc
  char v23; // bl
  unsigned __int64 v24; // r12
  __int64 v26; // rdi
  __int64 *v27; // rdi
  unsigned int v28; // edx
  __int64 v29; // rax
  const void *v30; // rax
  unsigned int v31; // eax
  __int64 v32; // r8
  unsigned __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // r8
  char v36; // al
  unsigned int v37; // edx
  __int64 v38; // r15
  unsigned __int64 v39; // r15
  char v40; // al
  unsigned __int64 *v41; // r12
  const void *v42; // rdi
  __int64 v43; // rdi
  __int64 *v44; // rdi
  unsigned int v45; // eax
  __int64 v46; // rdx
  unsigned __int64 v47; // rdx
  unsigned int v48; // ecx
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // rdx
  const void *v51; // rdx
  unsigned int v52; // eax
  __int64 v53; // rdx
  unsigned __int64 v54; // rdx
  unsigned int v55; // ecx
  unsigned __int64 v56; // rdx
  unsigned __int64 v57; // rdx
  __int64 v58; // rdx
  unsigned int v59; // edx
  __int64 v60; // r8
  __int64 v61; // r8
  char v62; // al
  __int64 v63; // r12
  __int64 v64; // rdi
  __int64 *v65; // rdi
  unsigned int v66; // edx
  __int64 v67; // rax
  const void *v68; // rax
  unsigned int v69; // eax
  __int64 v70; // r8
  unsigned __int64 v71; // r8
  __int64 v72; // rax
  __int64 v73; // r8
  char v74; // al
  unsigned int v75; // edx
  __int64 v76; // r14
  unsigned __int64 v77; // r14
  char v78; // al
  unsigned int v79; // r14d
  __int64 v80; // r15
  unsigned __int64 v81; // r15
  char v82; // r13
  __int64 v83; // r12
  unsigned __int64 *v84; // r12
  unsigned int v85; // eax
  __int64 v86; // rdi
  const void *v87; // rdi
  unsigned int v88; // r15d
  __int64 v89; // r14
  unsigned __int64 v90; // r14
  unsigned int v91; // eax
  __int64 v92; // rdi
  __int64 *v93; // [rsp+0h] [rbp-F0h]
  __int64 *v94; // [rsp+0h] [rbp-F0h]
  __int64 *v95; // [rsp+0h] [rbp-F0h]
  __int64 v96; // [rsp+8h] [rbp-E8h]
  __int64 v97; // [rsp+8h] [rbp-E8h]
  unsigned int v98; // [rsp+8h] [rbp-E8h]
  __int64 v99; // [rsp+8h] [rbp-E8h]
  unsigned int v100; // [rsp+8h] [rbp-E8h]
  __int64 v101; // [rsp+8h] [rbp-E8h]
  __int64 v102; // [rsp+8h] [rbp-E8h]
  __int64 v103; // [rsp+10h] [rbp-E0h]
  char v104; // [rsp+10h] [rbp-E0h]
  __int64 v105; // [rsp+10h] [rbp-E0h]
  char v106; // [rsp+10h] [rbp-E0h]
  unsigned int v107; // [rsp+10h] [rbp-E0h]
  char v108; // [rsp+10h] [rbp-E0h]
  __int64 v109; // [rsp+18h] [rbp-D8h]
  unsigned int v110; // [rsp+18h] [rbp-D8h]
  char v111; // [rsp+18h] [rbp-D8h]
  unsigned int v112; // [rsp+18h] [rbp-D8h]
  char v113; // [rsp+18h] [rbp-D8h]
  const void *v114; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v115; // [rsp+28h] [rbp-C8h]
  __int64 v116; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v117; // [rsp+38h] [rbp-B8h]
  const void *v118; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v119; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v120; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v121; // [rsp+58h] [rbp-98h]
  unsigned __int64 v122; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v123; // [rsp+68h] [rbp-88h]
  unsigned __int64 v124; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v125; // [rsp+78h] [rbp-78h]
  unsigned __int64 v126; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v127; // [rsp+88h] [rbp-68h]
  unsigned __int64 v128; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v129; // [rsp+98h] [rbp-58h]
  unsigned __int64 v130; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v131; // [rsp+A8h] [rbp-48h]
  unsigned __int64 v132; // [rsp+B0h] [rbp-40h] BYREF
  unsigned int v133; // [rsp+B8h] [rbp-38h]

  v6 = a1;
  v11 = *(_DWORD *)(a3 + 8);
  v12 = *(_QWORD *)a2;
  v127 = v11;
  v109 = v12;
  if ( v11 > 0x40 )
  {
    v96 = a6;
    sub_16A4EF0((__int64)&v126, 0, 0);
    v129 = v11;
    sub_16A4EF0((__int64)&v128, 0, 0);
    v131 = v11;
    sub_16A4EF0((__int64)&v130, 0, 0);
    v133 = v11;
    sub_16A4EF0((__int64)&v132, 0, 0);
    a6 = v96;
    v6 = a1;
  }
  else
  {
    v126 = 0;
    v129 = v11;
    v128 = 0;
    v131 = v11;
    v130 = 0;
    v133 = v11;
    v132 = 0;
  }
  v13 = *(unsigned __int8 *)(a2 + 16);
  v14 = v6[332];
  v15 = v6[330];
  v16 = v6[333];
  if ( v13 == 51 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v64 = *(_QWORD *)(a2 - 8);
    else
      v64 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v95 = v6;
    v101 = a6;
    sub_14BB090(*(_QWORD *)(v64 + 24), (__int64)&v130, v16, a5 + 1, v15, a6, v14, 0);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v65 = *(__int64 **)(a2 - 8);
    else
      v65 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    sub_14BB090(*v65, (__int64)&v126, v95[333], a5 + 1, v95[330], v101, v95[332], 0);
    v66 = v131;
    v125 = v131;
    if ( v131 > 0x40 )
    {
      sub_16A4FD0((__int64)&v124, (const void **)&v130);
      v66 = v125;
      if ( v125 > 0x40 )
      {
        sub_16A8890((__int64 *)&v124, (__int64 *)&v126);
        v66 = v125;
        v68 = (const void *)v124;
LABEL_141:
        v118 = v68;
        v69 = v133;
        v119 = v66;
        v125 = v133;
        if ( v133 > 0x40 )
        {
          sub_16A4FD0((__int64)&v124, (const void **)&v132);
          v69 = v125;
          if ( v125 > 0x40 )
          {
            sub_16A89F0((__int64 *)&v124, (__int64 *)&v128);
            v69 = v125;
            v71 = v124;
            v66 = v119;
LABEL_144:
            v121 = v69;
            v120 = v71;
            v123 = v66;
            if ( v66 > 0x40 )
            {
              sub_16A4FD0((__int64)&v122, &v118);
              v66 = v123;
              if ( v123 > 0x40 )
              {
                sub_16A89F0((__int64 *)&v122, (__int64 *)&v120);
                v66 = v123;
                v73 = v122;
LABEL_147:
                v22 = *(_DWORD *)(a3 + 8) <= 0x40u;
                v125 = v66;
                v124 = v73;
                v123 = 0;
                if ( v22 )
                {
                  v74 = (*(_QWORD *)a3 & ~v73) == 0;
                }
                else
                {
                  v102 = v73;
                  v107 = v66;
                  v74 = sub_16A5A00((__int64 *)a3, (__int64 *)&v124);
                  v73 = v102;
                  v66 = v107;
                }
                if ( v66 > 0x40 )
                {
                  if ( v73 )
                  {
                    v108 = v74;
                    j_j___libc_free_0_0(v73);
                    v74 = v108;
                    if ( v123 > 0x40 )
                    {
                      if ( v122 )
                      {
                        j_j___libc_free_0_0(v122);
                        v74 = v108;
                      }
                    }
                  }
                }
                if ( v74 )
                  goto LABEL_185;
                v75 = v129;
                v123 = v129;
                if ( v129 > 0x40 )
                {
                  sub_16A4FD0((__int64)&v122, (const void **)&v128);
                  v75 = v123;
                  if ( v123 > 0x40 )
                  {
                    sub_16A89F0((__int64 *)&v122, (__int64 *)&v130);
                    v75 = v123;
                    v77 = v122;
LABEL_158:
                    v22 = *(_DWORD *)(a3 + 8) <= 0x40u;
                    v125 = v75;
                    v124 = v77;
                    v123 = 0;
                    if ( v22 )
                    {
                      v78 = (*(_QWORD *)a3 & ~v77) == 0;
                    }
                    else
                    {
                      v112 = v75;
                      v78 = sub_16A5A00((__int64 *)a3, (__int64 *)&v124);
                      v75 = v112;
                    }
                    if ( v75 > 0x40 )
                    {
                      if ( v77 )
                      {
                        v113 = v78;
                        j_j___libc_free_0_0(v77);
                        v78 = v113;
                        if ( v123 > 0x40 )
                        {
                          if ( v122 )
                          {
                            j_j___libc_free_0_0(v122);
                            v78 = v113;
                          }
                        }
                      }
                    }
                    if ( v78 )
                      goto LABEL_64;
                    v79 = v133;
                    v123 = v133;
                    if ( v133 > 0x40 )
                    {
                      sub_16A4FD0((__int64)&v122, (const void **)&v132);
                      v79 = v123;
                      if ( v123 > 0x40 )
                      {
                        sub_16A89F0((__int64 *)&v122, (__int64 *)&v126);
                        v79 = v123;
                        v81 = v122;
LABEL_169:
                        v22 = *(_DWORD *)(a3 + 8) <= 0x40u;
                        v125 = v79;
                        v124 = v81;
                        v123 = 0;
                        if ( v22 )
                          v82 = (*(_QWORD *)a3 & ~v81) == 0;
                        else
                          v82 = sub_16A5A00((__int64 *)a3, (__int64 *)&v124);
                        if ( v79 <= 0x40 )
                          goto LABEL_176;
                        if ( !v81 )
                          goto LABEL_176;
                        j_j___libc_free_0_0(v81);
                        if ( v123 <= 0x40 )
                          goto LABEL_176;
                        goto LABEL_174;
                      }
                      v80 = v122;
                    }
                    else
                    {
                      v80 = v132;
                    }
                    v81 = v126 | v80;
                    v122 = v81;
                    goto LABEL_169;
                  }
                  v76 = v122;
                }
                else
                {
                  v76 = v128;
                }
                v77 = v130 | v76;
                v122 = v77;
                goto LABEL_158;
              }
              v72 = v122;
              v71 = v120;
            }
            else
            {
              v72 = (__int64)v118;
            }
            v73 = v72 | v71;
            v122 = v73;
            goto LABEL_147;
          }
          v70 = v124;
          v66 = v119;
        }
        else
        {
          v70 = v132;
        }
        v71 = v128 | v70;
        goto LABEL_144;
      }
      v67 = v124;
    }
    else
    {
      v67 = v130;
    }
    v68 = (const void *)(v126 & v67);
    goto LABEL_141;
  }
  if ( v13 == 52 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v43 = *(_QWORD *)(a2 - 8);
    else
      v43 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v94 = v6;
    v99 = a6;
    sub_14BB090(*(_QWORD *)(v43 + 24), (__int64)&v130, v16, a5 + 1, v15, a6, v14, 0);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v44 = *(__int64 **)(a2 - 8);
    else
      v44 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    sub_14BB090(*v44, (__int64)&v126, v94[333], a5 + 1, v94[330], v99, v94[332], 0);
    v45 = v133;
    v123 = v133;
    if ( v133 > 0x40 )
    {
      sub_16A4FD0((__int64)&v122, (const void **)&v132);
      v45 = v123;
      if ( v123 > 0x40 )
      {
        sub_16A8890((__int64 *)&v122, (__int64 *)&v128);
        v45 = v123;
        v47 = v122;
LABEL_80:
        v48 = v131;
        v125 = v45;
        v124 = v47;
        v123 = 0;
        v119 = v131;
        if ( v131 > 0x40 )
        {
          sub_16A4FD0((__int64)&v118, (const void **)&v130);
          v48 = v119;
          if ( v119 > 0x40 )
          {
            sub_16A8890((__int64 *)&v118, (__int64 *)&v126);
            v48 = v119;
            v50 = (unsigned __int64)v118;
            v45 = v125;
LABEL_83:
            v121 = v48;
            v120 = v50;
            v119 = 0;
            if ( v45 > 0x40 )
            {
              sub_16A89F0((__int64 *)&v124, (__int64 *)&v120);
              v45 = v125;
              v51 = (const void *)v124;
              v48 = v121;
            }
            else
            {
              v51 = (const void *)(v124 | v50);
              v124 = (unsigned __int64)v51;
            }
            v115 = v45;
            v114 = v51;
            v125 = 0;
            if ( v48 > 0x40 && v120 )
              j_j___libc_free_0_0(v120);
            if ( v119 > 0x40 && v118 )
              j_j___libc_free_0_0(v118);
            if ( v125 > 0x40 && v124 )
              j_j___libc_free_0_0(v124);
            if ( v123 > 0x40 && v122 )
              j_j___libc_free_0_0(v122);
            v52 = v133;
            v123 = v133;
            if ( v133 > 0x40 )
            {
              sub_16A4FD0((__int64)&v122, (const void **)&v132);
              v52 = v123;
              if ( v123 > 0x40 )
              {
                sub_16A8890((__int64 *)&v122, (__int64 *)&v126);
                v52 = v123;
                v54 = v122;
LABEL_100:
                v55 = v131;
                v125 = v52;
                v124 = v54;
                v123 = 0;
                v119 = v131;
                if ( v131 > 0x40 )
                {
                  sub_16A4FD0((__int64)&v118, (const void **)&v130);
                  v55 = v119;
                  if ( v119 > 0x40 )
                  {
                    sub_16A8890((__int64 *)&v118, (__int64 *)&v128);
                    v55 = v119;
                    v57 = (unsigned __int64)v118;
                    v52 = v125;
LABEL_103:
                    v121 = v55;
                    v120 = v57;
                    v119 = 0;
                    if ( v52 > 0x40 )
                    {
                      sub_16A89F0((__int64 *)&v124, (__int64 *)&v120);
                      v52 = v125;
                      v58 = v124;
                      v55 = v121;
                    }
                    else
                    {
                      v58 = v124 | v57;
                      v124 = v58;
                    }
                    v117 = v52;
                    v116 = v58;
                    v125 = 0;
                    if ( v55 > 0x40 && v120 )
                      j_j___libc_free_0_0(v120);
                    if ( v119 > 0x40 && v118 )
                      j_j___libc_free_0_0(v118);
                    if ( v125 > 0x40 && v124 )
                      j_j___libc_free_0_0(v124);
                    if ( v123 > 0x40 && v122 )
                      j_j___libc_free_0_0(v122);
                    v59 = v115;
                    v123 = v115;
                    if ( v115 > 0x40 )
                    {
                      sub_16A4FD0((__int64)&v122, &v114);
                      v59 = v123;
                      if ( v123 > 0x40 )
                      {
                        sub_16A89F0((__int64 *)&v122, &v116);
                        v59 = v123;
                        v61 = v122;
LABEL_120:
                        v22 = *(_DWORD *)(a3 + 8) <= 0x40u;
                        v125 = v59;
                        v124 = v61;
                        v123 = 0;
                        if ( v22 )
                        {
                          v62 = (*(_QWORD *)a3 & ~v61) == 0;
                        }
                        else
                        {
                          v100 = v59;
                          v105 = v61;
                          v62 = sub_16A5A00((__int64 *)a3, (__int64 *)&v124);
                          v59 = v100;
                          v61 = v105;
                        }
                        if ( v59 > 0x40 )
                        {
                          if ( v61 )
                          {
                            v106 = v62;
                            j_j___libc_free_0_0(v61);
                            v62 = v106;
                            if ( v123 > 0x40 )
                            {
                              if ( v122 )
                              {
                                j_j___libc_free_0_0(v122);
                                v62 = v106;
                              }
                            }
                          }
                        }
                        if ( v62 )
                        {
                          v24 = sub_15A3C50(v109, (__int64)&v116);
                          goto LABEL_212;
                        }
                        if ( *(_DWORD *)(a3 + 8) <= 0x40u )
                        {
                          if ( (*(_QWORD *)a3 & ~v130) != 0 )
                          {
                            if ( (*(_QWORD *)a3 & ~v126) == 0 )
                              goto LABEL_131;
                            goto LABEL_232;
                          }
                        }
                        else if ( !(unsigned __int8)sub_16A5A00((__int64 *)a3, (__int64 *)&v130) )
                        {
                          if ( (unsigned __int8)sub_16A5A00((__int64 *)a3, (__int64 *)&v126) )
                          {
LABEL_131:
                            if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                              v63 = *(_QWORD *)(a2 - 8);
                            else
                              v63 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
                            v24 = *(_QWORD *)(v63 + 24);
                            goto LABEL_212;
                          }
LABEL_232:
                          if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
                            j_j___libc_free_0_0(*a4);
                          v22 = *((_DWORD *)a4 + 6) <= 0x40u;
                          *a4 = v114;
                          v85 = v115;
                          v115 = 0;
                          *((_DWORD *)a4 + 2) = v85;
                          if ( v22 || (v86 = a4[2]) == 0 )
                          {
                            a4[2] = v116;
                            *((_DWORD *)a4 + 6) = v117;
                            goto LABEL_240;
                          }
                          j_j___libc_free_0_0(v86);
                          v22 = v115 <= 0x40;
                          a4[2] = v116;
                          *((_DWORD *)a4 + 6) = v117;
                          if ( v22 )
                            goto LABEL_240;
                          v87 = v114;
                          if ( !v114 )
                            goto LABEL_240;
LABEL_239:
                          j_j___libc_free_0_0(v87);
LABEL_240:
                          v24 = 0;
                          goto LABEL_18;
                        }
                        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                          v84 = *(unsigned __int64 **)(a2 - 8);
                        else
                          v84 = (unsigned __int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
                        v24 = *v84;
LABEL_212:
                        if ( v117 > 0x40 && v116 )
                          j_j___libc_free_0_0(v116);
                        if ( v115 > 0x40 )
                        {
                          v42 = v114;
                          if ( v114 )
                            goto LABEL_72;
                        }
                        goto LABEL_18;
                      }
                      v60 = v122;
                    }
                    else
                    {
                      v60 = (__int64)v114;
                    }
                    v61 = v116 | v60;
                    v122 = v61;
                    goto LABEL_120;
                  }
                  v56 = (unsigned __int64)v118;
                  v52 = v125;
                }
                else
                {
                  v56 = v130;
                }
                v57 = v128 & v56;
                v118 = (const void *)v57;
                goto LABEL_103;
              }
              v53 = v122;
            }
            else
            {
              v53 = v132;
            }
            v54 = v126 & v53;
            v122 = v54;
            goto LABEL_100;
          }
          v49 = (unsigned __int64)v118;
          v45 = v125;
        }
        else
        {
          v49 = v130;
        }
        v50 = v126 & v49;
        v118 = (const void *)v50;
        goto LABEL_83;
      }
      v46 = v122;
    }
    else
    {
      v46 = v132;
    }
    v47 = v128 & v46;
    v122 = v47;
    goto LABEL_80;
  }
  if ( v13 != 50 )
  {
    v17 = a5;
    v18 = (__int64)(a4 + 2);
    sub_14BB090(a2, (__int64)a4, v16, v17, v15, a6, v14, 0);
    v19 = *((_DWORD *)a4 + 2);
    v123 = v19;
    if ( v19 > 0x40 )
    {
      sub_16A4FD0((__int64)&v122, (const void **)a4);
      v19 = v123;
      if ( v123 > 0x40 )
      {
        sub_16A89F0((__int64 *)&v122, a4 + 2);
        v19 = v123;
        v21 = v122;
        goto LABEL_9;
      }
      v20 = v122;
    }
    else
    {
      v20 = *a4;
    }
    v21 = a4[2] | v20;
    v122 = v21;
LABEL_9:
    v22 = *(_DWORD *)(a3 + 8) <= 0x40u;
    v125 = v19;
    v124 = v21;
    v123 = 0;
    if ( v22 )
      v23 = (*(_QWORD *)a3 & ~v21) == 0;
    else
      v23 = sub_16A5A00((__int64 *)a3, (__int64 *)&v124);
    if ( v19 > 0x40 )
    {
      if ( v21 )
      {
        j_j___libc_free_0_0(v21);
        if ( v123 > 0x40 )
        {
          if ( v122 )
            j_j___libc_free_0_0(v122);
        }
      }
    }
    v24 = 0;
    if ( v23 )
      v24 = sub_15A3C50(v109, v18);
    goto LABEL_18;
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v26 = *(_QWORD *)(a2 - 8);
  else
    v26 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v93 = v6;
  v97 = a6;
  sub_14BB090(*(_QWORD *)(v26 + 24), (__int64)&v130, v16, a5 + 1, v15, a6, v14, 0);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v27 = *(__int64 **)(a2 - 8);
  else
    v27 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  sub_14BB090(*v27, (__int64)&v126, v93[333], a5 + 1, v93[330], v97, v93[332], 0);
  v28 = v131;
  v125 = v131;
  if ( v131 <= 0x40 )
  {
    v29 = v130;
LABEL_38:
    v30 = (const void *)(v126 | v29);
    goto LABEL_39;
  }
  sub_16A4FD0((__int64)&v124, (const void **)&v130);
  v28 = v125;
  if ( v125 <= 0x40 )
  {
    v29 = v124;
    goto LABEL_38;
  }
  sub_16A89F0((__int64 *)&v124, (__int64 *)&v126);
  v30 = (const void *)v124;
  v28 = v125;
LABEL_39:
  v118 = v30;
  v31 = v133;
  v119 = v28;
  v125 = v133;
  if ( v133 <= 0x40 )
  {
    v32 = v132;
LABEL_41:
    v33 = v128 & v32;
    goto LABEL_42;
  }
  sub_16A4FD0((__int64)&v124, (const void **)&v132);
  v31 = v125;
  if ( v125 <= 0x40 )
  {
    v32 = v124;
    v28 = v119;
    goto LABEL_41;
  }
  sub_16A8890((__int64 *)&v124, (__int64 *)&v128);
  v31 = v125;
  v33 = v124;
  v28 = v119;
LABEL_42:
  v121 = v31;
  v120 = v33;
  v123 = v28;
  if ( v28 <= 0x40 )
  {
    v34 = (__int64)v118;
LABEL_44:
    v35 = v34 | v33;
    v122 = v35;
    goto LABEL_45;
  }
  sub_16A4FD0((__int64)&v122, &v118);
  v28 = v123;
  if ( v123 <= 0x40 )
  {
    v34 = v122;
    v33 = v120;
    goto LABEL_44;
  }
  sub_16A89F0((__int64 *)&v122, (__int64 *)&v120);
  v35 = v122;
  v28 = v123;
LABEL_45:
  v22 = *(_DWORD *)(a3 + 8) <= 0x40u;
  v125 = v28;
  v124 = v35;
  v123 = 0;
  if ( v22 )
  {
    v36 = (*(_QWORD *)a3 & ~v35) == 0;
  }
  else
  {
    v98 = v28;
    v103 = v35;
    v36 = sub_16A5A00((__int64 *)a3, (__int64 *)&v124);
    v28 = v98;
    v35 = v103;
  }
  if ( v28 > 0x40 )
  {
    if ( v35 )
    {
      v104 = v36;
      j_j___libc_free_0_0(v35);
      v36 = v104;
      if ( v123 > 0x40 )
      {
        if ( v122 )
        {
          j_j___libc_free_0_0(v122);
          v36 = v104;
        }
      }
    }
  }
  if ( v36 )
  {
LABEL_185:
    v24 = sub_15A3C50(v109, (__int64)&v120);
    goto LABEL_67;
  }
  v37 = v127;
  v123 = v127;
  if ( v127 <= 0x40 )
  {
    v38 = v126;
LABEL_55:
    v39 = v132 | v38;
    v122 = v39;
    goto LABEL_56;
  }
  sub_16A4FD0((__int64)&v122, (const void **)&v126);
  v37 = v123;
  if ( v123 <= 0x40 )
  {
    v38 = v122;
    goto LABEL_55;
  }
  sub_16A89F0((__int64 *)&v122, (__int64 *)&v132);
  v39 = v122;
  v37 = v123;
LABEL_56:
  v22 = *(_DWORD *)(a3 + 8) <= 0x40u;
  v125 = v37;
  v124 = v39;
  v123 = 0;
  if ( v22 )
  {
    v40 = (*(_QWORD *)a3 & ~v39) == 0;
  }
  else
  {
    v110 = v37;
    v40 = sub_16A5A00((__int64 *)a3, (__int64 *)&v124);
    v37 = v110;
  }
  if ( v37 > 0x40 )
  {
    if ( v39 )
    {
      v111 = v40;
      j_j___libc_free_0_0(v39);
      v40 = v111;
      if ( v123 > 0x40 )
      {
        if ( v122 )
        {
          j_j___libc_free_0_0(v122);
          v40 = v111;
        }
      }
    }
  }
  if ( v40 )
  {
LABEL_64:
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v41 = *(unsigned __int64 **)(a2 - 8);
    else
      v41 = (unsigned __int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v24 = *v41;
    goto LABEL_67;
  }
  v88 = v131;
  v123 = v131;
  if ( v131 <= 0x40 )
  {
    v89 = v130;
LABEL_249:
    v90 = v128 | v89;
    v122 = v90;
    goto LABEL_250;
  }
  sub_16A4FD0((__int64)&v122, (const void **)&v130);
  v88 = v123;
  if ( v123 <= 0x40 )
  {
    v89 = v122;
    goto LABEL_249;
  }
  sub_16A89F0((__int64 *)&v122, (__int64 *)&v128);
  v88 = v123;
  v90 = v122;
LABEL_250:
  v22 = *(_DWORD *)(a3 + 8) <= 0x40u;
  v125 = v88;
  v124 = v90;
  v123 = 0;
  if ( v22 )
    v82 = (*(_QWORD *)a3 & ~v90) == 0;
  else
    v82 = sub_16A5A00((__int64 *)a3, (__int64 *)&v124);
  if ( v88 <= 0x40 )
    goto LABEL_176;
  if ( !v90 )
    goto LABEL_176;
  j_j___libc_free_0_0(v90);
  if ( v123 <= 0x40 )
    goto LABEL_176;
LABEL_174:
  if ( v122 )
    j_j___libc_free_0_0(v122);
LABEL_176:
  if ( !v82 )
  {
    if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
      j_j___libc_free_0_0(*a4);
    v22 = *((_DWORD *)a4 + 6) <= 0x40u;
    *a4 = v118;
    v91 = v119;
    v119 = 0;
    *((_DWORD *)a4 + 2) = v91;
    if ( v22 || (v92 = a4[2]) == 0 )
    {
      a4[2] = v120;
      *((_DWORD *)a4 + 6) = v121;
      goto LABEL_240;
    }
    j_j___libc_free_0_0(v92);
    v22 = v119 <= 0x40;
    a4[2] = v120;
    *((_DWORD *)a4 + 6) = v121;
    if ( v22 )
      goto LABEL_240;
    v87 = v118;
    if ( !v118 )
      goto LABEL_240;
    goto LABEL_239;
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v83 = *(_QWORD *)(a2 - 8);
  else
    v83 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v24 = *(_QWORD *)(v83 + 24);
LABEL_67:
  if ( v121 > 0x40 && v120 )
    j_j___libc_free_0_0(v120);
  if ( v119 > 0x40 )
  {
    v42 = v118;
    if ( v118 )
LABEL_72:
      j_j___libc_free_0_0(v42);
  }
LABEL_18:
  if ( v133 > 0x40 && v132 )
    j_j___libc_free_0_0(v132);
  if ( v131 > 0x40 && v130 )
    j_j___libc_free_0_0(v130);
  if ( v129 > 0x40 && v128 )
    j_j___libc_free_0_0(v128);
  if ( v127 > 0x40 && v126 )
    j_j___libc_free_0_0(v126);
  return v24;
}
