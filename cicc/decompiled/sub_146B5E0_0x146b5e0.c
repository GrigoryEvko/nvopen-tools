// Function: sub_146B5E0
// Address: 0x146b5e0
//
__int64 __fastcall sub_146B5E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 **v8; // rdx
  __int64 *v9; // r9
  __int64 v10; // r12
  int v12; // edx
  unsigned int v13; // ebx
  unsigned __int64 v14; // r13
  __int64 v15; // r12
  _QWORD *v16; // rax
  char v17; // al
  int v18; // eax
  unsigned int v19; // esi
  int v20; // eax
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 v27; // r13
  int v28; // esi
  __int64 v29; // rcx
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rax
  unsigned int v34; // eax
  __int64 *v35; // rcx
  int v36; // edx
  __int64 v37; // rax
  _QWORD *v38; // rbx
  __int64 v39; // rax
  __int64 v40; // r9
  __int64 v41; // rbx
  unsigned __int8 v42; // al
  unsigned int v43; // r8d
  int v44; // esi
  __int64 *v45; // rdx
  __int64 v46; // rsi
  int v47; // eax
  __int64 v48; // rax
  char v49; // al
  __int64 *v50; // rdx
  __int64 v51; // rax
  unsigned int v52; // ecx
  int v53; // esi
  unsigned int v54; // ecx
  __int64 *v55; // rdx
  __int64 v56; // r8
  __int64 v57; // rax
  bool v58; // r12
  __int64 *v59; // rdi
  int v60; // eax
  int v61; // eax
  unsigned int v62; // eax
  __int64 *v63; // rax
  __int64 *v64; // r14
  __int64 v65; // r13
  __int64 *v66; // rbx
  __int64 v67; // rax
  __int64 v68; // r15
  __int64 *v69; // r14
  __int64 v70; // rax
  __int64 *v71; // r15
  __int64 v72; // rax
  __int64 *v73; // r9
  __int64 v74; // r15
  unsigned int v75; // eax
  __int64 *v76; // rbx
  __int64 v77; // rdx
  __int64 v78; // rdi
  __int64 v79; // r13
  int v80; // esi
  __int64 v81; // rcx
  int v82; // edx
  __int64 v83; // rax
  char v84; // di
  unsigned int v85; // esi
  __int64 v86; // rdx
  __int64 v87; // rax
  __int64 v88; // rcx
  unsigned __int8 v89; // al
  __int64 *v90; // rdx
  __int64 v91; // rax
  int v92; // r10d
  __int64 *v93; // r8
  __int64 *v94; // rdi
  int v95; // eax
  unsigned int v96; // esi
  int v97; // eax
  __int64 v98; // rax
  int v99; // r10d
  int v100; // r13d
  __int64 *v101; // rdi
  int v102; // edx
  int v103; // r13d
  __int64 *v104; // r12
  int v105; // r11d
  __int64 *v106; // r10
  int v107; // edx
  __int64 *v108; // r11
  __int64 v109; // [rsp+8h] [rbp-178h]
  __int64 *v110; // [rsp+10h] [rbp-170h]
  __int64 *v111; // [rsp+10h] [rbp-170h]
  __int64 *v112; // [rsp+18h] [rbp-168h]
  int v113; // [rsp+2Ch] [rbp-154h]
  __int64 v114; // [rsp+38h] [rbp-148h]
  __int64 v115; // [rsp+40h] [rbp-140h]
  __int64 v116; // [rsp+40h] [rbp-140h]
  __int64 v117; // [rsp+40h] [rbp-140h]
  __int64 v118; // [rsp+40h] [rbp-140h]
  int v120; // [rsp+50h] [rbp-130h]
  __int64 v122; // [rsp+60h] [rbp-120h]
  __int64 *v123; // [rsp+68h] [rbp-118h] BYREF
  __int64 v124; // [rsp+70h] [rbp-110h] BYREF
  __int64 *v125; // [rsp+78h] [rbp-108h] BYREF
  __int64 v126; // [rsp+80h] [rbp-100h] BYREF
  __int64 *v127; // [rsp+88h] [rbp-F8h]
  __int64 v128; // [rsp+90h] [rbp-F0h]
  unsigned int v129; // [rsp+98h] [rbp-E8h]
  __int64 v130; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 *v131; // [rsp+A8h] [rbp-D8h]
  __int64 v132; // [rsp+B0h] [rbp-D0h]
  unsigned int v133; // [rsp+B8h] [rbp-C8h]
  __int64 *v134; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v135; // [rsp+C8h] [rbp-B8h]
  _BYTE v136[176]; // [rsp+D0h] [rbp-B0h] BYREF

  v5 = *(unsigned int *)(a1 + 616);
  v123 = (__int64 *)a2;
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(a1 + 600);
    v7 = (v5 - 1) & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
    v8 = (__int64 **)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v123 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 **)(v6 + 16 * v5) )
        return (__int64)v8[1];
    }
    else
    {
      v12 = 1;
      while ( v9 != (__int64 *)-8LL )
      {
        v99 = v12 + 1;
        v7 = (v5 - 1) & (v12 + v7);
        v8 = (__int64 **)(v6 + 16LL * v7);
        v9 = *v8;
        if ( v123 == *v8 )
          goto LABEL_3;
        v12 = v99;
      }
    }
  }
  v13 = *(_DWORD *)(a3 + 8);
  v14 = (unsigned int)dword_4F9B5E0;
  v15 = a1 + 592;
  if ( v13 > 0x40 )
  {
    if ( v13 - (unsigned int)sub_1455840(a3) > 0x40 )
    {
LABEL_42:
      v49 = sub_145FD20(v15, (__int64 *)&v123, &v134);
      v50 = v134;
      if ( v49 )
      {
LABEL_43:
        v50[1] = 0;
        return 0;
      }
      v95 = *(_DWORD *)(a1 + 608);
      v96 = *(_DWORD *)(a1 + 616);
      ++*(_QWORD *)(a1 + 592);
      v97 = v95 + 1;
      if ( 4 * v97 >= 3 * v96 )
      {
        v96 *= 2;
      }
      else if ( v96 - *(_DWORD *)(a1 + 612) - v97 > v96 >> 3 )
      {
LABEL_122:
        *(_DWORD *)(a1 + 608) = v97;
        if ( *v50 != -8 )
          --*(_DWORD *)(a1 + 612);
        v98 = (__int64)v123;
        v50[1] = 0;
        *v50 = v98;
        goto LABEL_43;
      }
      sub_146B420(v15, v96);
      sub_145FD20(v15, (__int64 *)&v123, &v134);
      v50 = v134;
      v97 = *(_DWORD *)(a1 + 608) + 1;
      goto LABEL_122;
    }
    v16 = **(_QWORD ***)a3;
  }
  else
  {
    v16 = *(_QWORD **)a3;
  }
  if ( v14 < (unsigned __int64)v16 )
    goto LABEL_42;
  v17 = sub_145FD20(v15, (__int64 *)&v123, &v134);
  v112 = v134;
  if ( !v17 )
  {
    v18 = *(_DWORD *)(a1 + 608);
    ++*(_QWORD *)(a1 + 592);
    v19 = *(_DWORD *)(a1 + 616);
    v20 = v18 + 1;
    if ( 4 * v20 >= 3 * v19 )
    {
      v19 *= 2;
    }
    else if ( v19 - *(_DWORD *)(a1 + 612) - v20 > v19 >> 3 )
    {
LABEL_14:
      *(_DWORD *)(a1 + 608) = v20;
      if ( *v112 != -8 )
        --*(_DWORD *)(a1 + 612);
      v21 = (__int64)v123;
      v112[1] = 0;
      *v112 = v21;
      goto LABEL_17;
    }
    sub_146B420(v15, v19);
    sub_145FD20(v15, (__int64 *)&v123, &v134);
    v112 = v134;
    v20 = *(_DWORD *)(a1 + 608) + 1;
    goto LABEL_14;
  }
LABEL_17:
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v22 = *(__int64 **)(a4 + 32);
  v129 = 0;
  v23 = *v22;
  v122 = *v22;
  v10 = sub_13FCB50(a4);
  if ( !v10 )
    goto LABEL_45;
  v24 = sub_157F280(v23);
  v26 = v25;
LABEL_19:
  if ( v26 != v24 )
  {
    while ( 1 )
    {
      v27 = sub_1454270(v24, v10);
      if ( !v27 )
        goto LABEL_24;
      v28 = v129;
      v130 = v24;
      v29 = v24;
      if ( !v129 )
        break;
      v30 = (v129 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v31 = &v127[2 * v30];
      v32 = *v31;
      if ( v24 != *v31 )
      {
        v105 = 1;
        v106 = 0;
        while ( v32 != -8 )
        {
          if ( !v106 && v32 == -16 )
            v106 = v31;
          v30 = (v129 - 1) & (v105 + v30);
          v31 = &v127[2 * v30];
          v32 = *v31;
          if ( v24 == *v31 )
            goto LABEL_23;
          ++v105;
        }
        if ( v106 )
          v31 = v106;
        ++v126;
        v107 = v128 + 1;
        if ( 4 * ((int)v128 + 1) < 3 * v129 )
        {
          if ( v129 - HIDWORD(v128) - v107 > v129 >> 3 )
          {
LABEL_149:
            LODWORD(v128) = v107;
            if ( *v31 != -8 )
              --HIDWORD(v128);
            *v31 = v29;
            v31[1] = 0;
            goto LABEL_23;
          }
LABEL_154:
          sub_146A3C0((__int64)&v126, v28);
          sub_1463C30((__int64)&v126, &v130, &v134);
          v31 = v134;
          v29 = v130;
          v107 = v128 + 1;
          goto LABEL_149;
        }
LABEL_153:
        v28 = 2 * v129;
        goto LABEL_154;
      }
LABEL_23:
      v31[1] = v27;
LABEL_24:
      if ( !v24 )
        BUG();
      v33 = *(_QWORD *)(v24 + 32);
      if ( !v33 )
        BUG();
      v24 = 0;
      if ( *(_BYTE *)(v33 - 8) != 77 )
        goto LABEL_19;
      v24 = v33 - 24;
      if ( v26 == v33 - 24 )
        goto LABEL_28;
    }
    ++v126;
    goto LABEL_153;
  }
LABEL_28:
  if ( !v129 )
  {
LABEL_44:
    v10 = 0;
    v112[1] = 0;
    goto LABEL_45;
  }
  v34 = (v129 - 1) & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
  v35 = (__int64 *)v127[2 * v34];
  v36 = 1;
  if ( v123 != v35 )
  {
    while ( v35 != (__int64 *)-8LL )
    {
      v34 = (v129 - 1) & (v36 + v34);
      v35 = (__int64 *)v127[2 * v34];
      if ( v123 == v35 )
        goto LABEL_30;
      ++v36;
    }
    goto LABEL_44;
  }
LABEL_30:
  v37 = sub_1455EB0((__int64)v123, v10);
  v38 = *(_QWORD **)a3;
  v114 = v37;
  if ( *(_DWORD *)(a3 + 8) > 0x40u )
    v38 = (_QWORD *)*v38;
  v113 = (int)v38;
  v39 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 40LL));
  v40 = v10;
  v120 = 0;
  v109 = v39;
  if ( !(_DWORD)v38 )
  {
LABEL_58:
    v134 = v123;
    v10 = sub_146A9A0((__int64)&v126, (__int64 *)&v134)[1];
    v112[1] = v10;
    goto LABEL_45;
  }
  while ( 1 )
  {
    v41 = v114;
    v130 = 0;
    v131 = 0;
    v132 = 0;
    v133 = 0;
    v42 = *(_BYTE *)(v114 + 16);
    if ( v42 <= 0x10u )
    {
      v125 = v123;
LABEL_35:
      ++v130;
      v43 = 0;
LABEL_36:
      v115 = v40;
      v44 = 2 * v43;
      goto LABEL_37;
    }
    v115 = v40;
    if ( v42 <= 0x17u )
    {
      v94 = 0;
      v10 = 0;
      goto LABEL_119;
    }
    v51 = sub_146A580(v114, a4, (__int64)&v126, v109, *(_QWORD *)(a1 + 40));
    v40 = v115;
    v41 = v51;
    if ( !v51 )
    {
      v94 = v131;
      v10 = 0;
      goto LABEL_119;
    }
    v46 = (__int64)v123;
    v43 = v133;
    v125 = v123;
    if ( !v133 )
      goto LABEL_35;
    v52 = (v133 - 1) & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
    v45 = &v131[2 * v52];
    v48 = *v45;
    if ( v123 == (__int64 *)*v45 )
      goto LABEL_50;
    v103 = 1;
    v104 = 0;
    while ( v48 != -8 )
    {
      if ( !v104 && v48 == -16 )
        v104 = v45;
      v52 = (v133 - 1) & (v103 + v52);
      v45 = &v131[2 * v52];
      v48 = *v45;
      if ( v123 == (__int64 *)*v45 )
        goto LABEL_50;
      ++v103;
    }
    if ( v104 )
      v45 = v104;
    ++v130;
    v47 = v132 + 1;
    if ( 4 * ((int)v132 + 1) >= 3 * v133 )
      goto LABEL_36;
    if ( v133 - (v47 + HIDWORD(v132)) > v133 >> 3 )
      goto LABEL_38;
    v44 = v133;
LABEL_37:
    sub_146A3C0((__int64)&v130, v44);
    sub_1463C30((__int64)&v130, (__int64 *)&v125, &v134);
    v45 = v134;
    v46 = (__int64)v125;
    v40 = v115;
    v47 = v132 + 1;
LABEL_38:
    LODWORD(v132) = v47;
    if ( *v45 != -8 )
      --HIDWORD(v132);
    *v45 = v46;
    v48 = (__int64)v123;
    v45[1] = 0;
LABEL_50:
    v45[1] = v41;
    v53 = v129;
    v125 = (__int64 *)v48;
    if ( !v129 )
    {
      ++v126;
      goto LABEL_160;
    }
    v54 = (v129 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
    v55 = &v127[2 * v54];
    v56 = *v55;
    if ( *v55 != v48 )
    {
      v100 = 1;
      v101 = 0;
      while ( v56 != -8 )
      {
        if ( !v101 && v56 == -16 )
          v101 = v55;
        v54 = (v129 - 1) & (v100 + v54);
        v55 = &v127[2 * v54];
        v56 = *v55;
        if ( *v55 == v48 )
          goto LABEL_52;
        ++v100;
      }
      if ( !v101 )
        v101 = v55;
      ++v126;
      v102 = v128 + 1;
      if ( 4 * ((int)v128 + 1) < 3 * v129 )
      {
        if ( v129 - HIDWORD(v128) - v102 > v129 >> 3 )
        {
LABEL_133:
          LODWORD(v128) = v102;
          if ( *v101 != -8 )
            --HIDWORD(v128);
          *v101 = v48;
          v57 = 0;
          v101[1] = 0;
          goto LABEL_53;
        }
        v118 = v40;
LABEL_161:
        sub_146A3C0((__int64)&v126, v53);
        sub_1463C30((__int64)&v126, (__int64 *)&v125, &v134);
        v101 = v134;
        v48 = (__int64)v125;
        v40 = v118;
        v102 = v128 + 1;
        goto LABEL_133;
      }
LABEL_160:
      v118 = v40;
      v53 = 2 * v129;
      goto LABEL_161;
    }
LABEL_52:
    v57 = v55[1];
LABEL_53:
    v58 = v57 == v41;
    v134 = (__int64 *)v136;
    v135 = 0x800000000LL;
    if ( (_DWORD)v128 )
    {
      v63 = v127;
      v64 = &v127[2 * v129];
      if ( v127 != v64 )
      {
        while ( 1 )
        {
          v65 = *v63;
          v66 = v63;
          if ( *v63 != -16 && *v63 != -8 )
            break;
          v63 += 2;
          if ( v64 == v63 )
            goto LABEL_54;
        }
        if ( v64 != v63 )
        {
          v67 = 0;
          v68 = v40;
          do
          {
            if ( *(_BYTE *)(v65 + 16) == 77 && v123 != (__int64 *)v65 && v122 == *(_QWORD *)(v65 + 40) )
            {
              if ( (unsigned int)v67 >= HIDWORD(v135) )
              {
                sub_16CD150(&v134, v136, 0, 16);
                LODWORD(v67) = v135;
              }
              v90 = &v134[2 * (unsigned int)v67];
              if ( v90 )
              {
                v67 = v66[1];
                *v90 = v65;
                v90[1] = v67;
                LODWORD(v67) = v135;
              }
              v67 = (unsigned int)(v67 + 1);
              LODWORD(v135) = v67;
            }
            v66 += 2;
            if ( v66 == v64 )
              break;
            while ( 1 )
            {
              v65 = *v66;
              if ( *v66 != -8 && v65 != -16 )
                break;
              v66 += 2;
              if ( v64 == v66 )
                goto LABEL_72;
            }
          }
          while ( v64 != v66 );
LABEL_72:
          v69 = v134;
          v70 = 2 * v67;
          v40 = v68;
          v71 = &v134[v70];
          if ( &v134[v70] == v134 )
            goto LABEL_54;
          v72 = v40;
          v73 = v71;
          v74 = v72;
          while ( 2 )
          {
            v79 = *v69;
            v80 = v133;
            v124 = *v69;
            if ( !v133 )
            {
              ++v130;
              goto LABEL_82;
            }
            v75 = (v133 - 1) & (((unsigned int)v79 >> 9) ^ ((unsigned int)v79 >> 4));
            v76 = &v131[2 * v75];
            v77 = *v76;
            if ( v79 == *v76 )
            {
              v78 = v76[1];
LABEL_76:
              if ( v78 )
              {
LABEL_77:
                if ( v69[1] != v78 )
                  v58 = 0;
                v69 += 2;
                if ( v73 == v69 )
                {
                  v40 = v74;
                  goto LABEL_54;
                }
                continue;
              }
            }
            else
            {
              v92 = 1;
              v93 = 0;
              while ( v77 != -8 )
              {
                if ( v93 || v77 != -16 )
                  v76 = v93;
                v75 = (v133 - 1) & (v92 + v75);
                v108 = &v131[2 * v75];
                v77 = *v108;
                if ( v79 == *v108 )
                {
                  v78 = v108[1];
                  v76 = &v131[2 * v75];
                  goto LABEL_76;
                }
                ++v92;
                v93 = v76;
                v76 = &v131[2 * v75];
              }
              if ( v93 )
                v76 = v93;
              ++v130;
              v82 = v132 + 1;
              if ( 4 * ((int)v132 + 1) >= 3 * v133 )
              {
LABEL_82:
                v110 = v73;
                v80 = 2 * v133;
              }
              else
              {
                v81 = v79;
                if ( v133 - HIDWORD(v132) - v82 > v133 >> 3 )
                  goto LABEL_84;
                v110 = v73;
              }
              sub_146A3C0((__int64)&v130, v80);
              sub_1463C30((__int64)&v130, &v124, &v125);
              v76 = v125;
              v81 = v124;
              v73 = v110;
              v82 = v132 + 1;
LABEL_84:
              LODWORD(v132) = v82;
              if ( *v76 != -8 )
                --HIDWORD(v132);
              *v76 = v81;
              v76[1] = 0;
            }
            break;
          }
          v83 = 0x17FFFFFFE8LL;
          v84 = *(_BYTE *)(v79 + 23) & 0x40;
          v85 = *(_DWORD *)(v79 + 20) & 0xFFFFFFF;
          if ( v85 )
          {
            v86 = 24LL * *(unsigned int *)(v79 + 56) + 8;
            v87 = 0;
            do
            {
              v88 = v79 - 24LL * v85;
              if ( v84 )
                v88 = *(_QWORD *)(v79 - 8);
              if ( v74 == *(_QWORD *)(v88 + v86) )
              {
                v83 = 24 * v87;
                goto LABEL_94;
              }
              ++v87;
              v86 += 8;
            }
            while ( v85 != (_DWORD)v87 );
            v83 = 0x17FFFFFFE8LL;
          }
LABEL_94:
          if ( v84 )
          {
            v78 = *(_QWORD *)(*(_QWORD *)(v79 - 8) + v83);
            v89 = *(_BYTE *)(v78 + 16);
            if ( v89 <= 0x10u )
            {
LABEL_96:
              v76[1] = v78;
              goto LABEL_77;
            }
          }
          else
          {
            v78 = *(_QWORD *)(v79 - 24LL * v85 + v83);
            v89 = *(_BYTE *)(v78 + 16);
            if ( v89 <= 0x10u )
              goto LABEL_96;
          }
          if ( v89 <= 0x17u )
          {
            v78 = 0;
          }
          else
          {
            v111 = v73;
            v91 = sub_146A580(v78, a4, (__int64)&v126, v109, *(_QWORD *)(a1 + 40));
            v73 = v111;
            v78 = v91;
          }
          goto LABEL_96;
        }
      }
    }
LABEL_54:
    if ( v58 )
      break;
    v59 = v127;
    ++v126;
    v127 = v131;
    v60 = v128;
    LODWORD(v128) = v132;
    LODWORD(v132) = v60;
    v61 = HIDWORD(v128);
    HIDWORD(v128) = HIDWORD(v132);
    HIDWORD(v132) = v61;
    v62 = v129;
    ++v130;
    v131 = v59;
    v129 = v133;
    v133 = v62;
    if ( v134 != (__int64 *)v136 )
    {
      v116 = v40;
      _libc_free((unsigned __int64)v134);
      v59 = v131;
      v40 = v116;
    }
    v117 = v40;
    j___libc_free_0(v59);
    ++v120;
    v40 = v117;
    if ( v113 == v120 )
      goto LABEL_58;
  }
  v125 = v123;
  v10 = sub_146A9A0((__int64)&v126, (__int64 *)&v125)[1];
  v112[1] = v10;
  if ( v134 != (__int64 *)v136 )
    _libc_free((unsigned __int64)v134);
  v94 = v131;
LABEL_119:
  j___libc_free_0(v94);
LABEL_45:
  j___libc_free_0(v127);
  return v10;
}
