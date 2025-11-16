// Function: sub_27F4830
// Address: 0x27f4830
//
__int64 __fastcall sub_27F4830(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 *v8; // rdx
  __int64 v9; // r9
  __int64 *v10; // r13
  __int64 *v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned int v14; // esi
  __int64 v15; // r13
  __int64 v16; // r14
  __int64 v17; // rdi
  __int64 v18; // r9
  _QWORD *v19; // r8
  __int64 v20; // rdx
  _QWORD *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r10
  __int64 *v24; // rax
  __int64 v25; // r13
  __int64 v26; // r14
  __int64 v27; // r10
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  __int64 v31; // r15
  unsigned __int8 *v32; // rax
  unsigned __int8 *v33; // r12
  unsigned __int64 v34; // rax
  int v35; // edx
  __int64 v36; // rdi
  __int64 v37; // rax
  unsigned int v38; // esi
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 *v41; // r11
  int v42; // r10d
  unsigned int v43; // edx
  __int64 *v44; // rax
  __int64 v45; // rdi
  int v46; // eax
  __int64 v47; // r10
  int v48; // edx
  int v49; // r10d
  __int64 v50; // rax
  unsigned int v51; // esi
  __int64 v52; // r8
  __int64 v53; // r10
  __int64 *v54; // r9
  int v55; // edx
  unsigned int v56; // edi
  __int64 *v57; // rax
  __int64 v58; // rcx
  _QWORD *v59; // rax
  _QWORD *v60; // rax
  __int64 v61; // rax
  __int64 v62; // r10
  __int64 v63; // rdx
  __int64 v64; // r9
  unsigned int v65; // ecx
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rsi
  unsigned int v69; // eax
  __int64 v70; // r8
  __int64 v71; // rcx
  char *v72; // rdi
  __int64 v73; // r11
  char *v74; // rax
  char *v75; // rdx
  char *v76; // rsi
  __int64 v77; // rax
  unsigned int v78; // edx
  __int64 *v79; // rdi
  __int64 v80; // rax
  __int64 v81; // rcx
  __int64 *v82; // r13
  __int64 *v83; // r12
  __int64 v84; // r14
  __int64 v85; // rax
  unsigned __int64 v86; // rdx
  _QWORD *v87; // rdx
  _QWORD *v88; // rcx
  _QWORD *v89; // rax
  _QWORD *v90; // rax
  int v91; // edi
  int v92; // edx
  int v93; // ecx
  int v94; // eax
  int v95; // edi
  _QWORD *v96; // rsi
  __int64 v97; // r10
  int v98; // r10d
  int v99; // r10d
  int v100; // r8d
  __int64 v101; // r11
  __int64 *v102; // rdi
  unsigned int v103; // esi
  __int64 v104; // r9
  int v105; // r10d
  int v106; // r10d
  __int64 v107; // r11
  unsigned int v108; // esi
  __int64 v109; // r9
  int v110; // r8d
  signed __int64 v111; // rdx
  char *v112; // rdx
  int v113; // ecx
  int v114; // edx
  int v115; // edi
  __int64 v116; // [rsp+10h] [rbp-2B0h]
  __int64 v117; // [rsp+10h] [rbp-2B0h]
  __int64 v118; // [rsp+18h] [rbp-2A8h]
  __int64 v119; // [rsp+18h] [rbp-2A8h]
  __int64 v120; // [rsp+18h] [rbp-2A8h]
  __int64 v121; // [rsp+18h] [rbp-2A8h]
  __int64 *v122; // [rsp+20h] [rbp-2A0h]
  __int64 v123; // [rsp+20h] [rbp-2A0h]
  __int64 *v124; // [rsp+20h] [rbp-2A0h]
  __int64 v125; // [rsp+20h] [rbp-2A0h]
  __int64 v126; // [rsp+20h] [rbp-2A0h]
  __int64 v127; // [rsp+20h] [rbp-2A0h]
  __int64 v128; // [rsp+20h] [rbp-2A0h]
  __int64 v129; // [rsp+28h] [rbp-298h]
  __int64 *v130; // [rsp+30h] [rbp-290h]
  __int64 v131; // [rsp+30h] [rbp-290h]
  unsigned __int16 v132; // [rsp+30h] [rbp-290h]
  unsigned __int16 v133; // [rsp+30h] [rbp-290h]
  __int64 v134; // [rsp+40h] [rbp-280h]
  __int64 v135; // [rsp+40h] [rbp-280h]
  unsigned __int16 v136; // [rsp+40h] [rbp-280h]
  __int64 *v137; // [rsp+40h] [rbp-280h]
  __int64 v138; // [rsp+40h] [rbp-280h]
  __int64 v139; // [rsp+40h] [rbp-280h]
  __int64 *v140; // [rsp+40h] [rbp-280h]
  __int64 v141; // [rsp+40h] [rbp-280h]
  __int64 v142; // [rsp+48h] [rbp-278h] BYREF
  __int64 *v143; // [rsp+58h] [rbp-268h] BYREF
  __int64 *v144[4]; // [rsp+60h] [rbp-260h] BYREF
  __int64 *v145; // [rsp+80h] [rbp-240h] BYREF
  __int64 v146; // [rsp+88h] [rbp-238h]
  _QWORD v147[70]; // [rsp+90h] [rbp-230h] BYREF

  v142 = a2;
  if ( !(_BYTE)qword_4FFE5C8 )
    return sub_D4B130(*(_QWORD *)(a1 + 16));
  v5 = *(unsigned int *)(a1 + 56);
  v6 = *(_QWORD *)(a1 + 40);
  if ( (_DWORD)v5 )
  {
    v7 = (v5 - 1) & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v142 == *v8 )
    {
LABEL_6:
      if ( v8 != (__int64 *)(v6 + 16 * v5) )
        return v8[1];
    }
    else
    {
      v48 = 1;
      while ( v9 != -4096 )
      {
        v49 = v48 + 1;
        v7 = (v5 - 1) & (v48 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( v142 == *v8 )
          goto LABEL_6;
        v48 = v49;
      }
    }
  }
  v10 = *(__int64 **)(a1 + 72);
  v134 = a1 + 64;
  v11 = &v10[2 * *(unsigned int *)(a1 + 88)];
  if ( *(_DWORD *)(a1 + 80) && v10 != v11 )
  {
    while ( 1 )
    {
      v50 = *v10;
      if ( *v10 != -4096 && v50 != -8192 )
        break;
      v10 += 2;
      if ( v11 == v10 )
        goto LABEL_9;
    }
LABEL_49:
    if ( v11 != v10 && (v142 == v10[1] || v142 != *(_QWORD *)(v50 - 32) && v142 != *(_QWORD *)(v50 - 64)) )
    {
      while ( 1 )
      {
        v10 += 2;
        if ( v11 == v10 )
          break;
        v50 = *v10;
        if ( *v10 != -8192 && v50 != -4096 )
          goto LABEL_49;
      }
    }
  }
  else
  {
LABEL_9:
    v10 = v11;
  }
  v129 = a1 + 32;
  v3 = sub_D4B130(*(_QWORD *)(a1 + 16));
  if ( v10 == (__int64 *)(*(_QWORD *)(a1 + 72) + 16LL * *(unsigned int *)(a1 + 88)) )
  {
    v51 = *(_DWORD *)(a1 + 56);
    if ( v51 )
    {
      v52 = v142;
      v53 = *(_QWORD *)(a1 + 40);
      v54 = 0;
      v55 = 1;
      v56 = (v51 - 1) & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
      v57 = (__int64 *)(v53 + 16LL * v56);
      v58 = *v57;
      if ( *v57 == v142 )
      {
LABEL_59:
        v57[1] = v3;
        return v3;
      }
      while ( v58 != -4096 )
      {
        if ( v58 == -8192 && !v54 )
          v54 = v57;
        v56 = (v51 - 1) & (v55 + v56);
        v57 = (__int64 *)(v53 + 16LL * v56);
        v58 = *v57;
        if ( v142 == *v57 )
          goto LABEL_59;
        ++v55;
      }
      v113 = *(_DWORD *)(a1 + 48);
      if ( v54 )
        v57 = v54;
      ++*(_QWORD *)(a1 + 32);
      v114 = v113 + 1;
      v145 = v57;
      if ( 4 * (v113 + 1) < 3 * v51 )
      {
        if ( v51 - *(_DWORD *)(a1 + 52) - v114 > v51 >> 3 )
        {
LABEL_183:
          *(_DWORD *)(a1 + 48) = v114;
          if ( *v57 != -4096 )
            --*(_DWORD *)(a1 + 52);
          *v57 = v52;
          v57[1] = 0;
          goto LABEL_59;
        }
LABEL_188:
        sub_22E02D0(v129, v51);
        sub_27EFA30(v129, &v142, &v145);
        v52 = v142;
        v114 = *(_DWORD *)(a1 + 48) + 1;
        v57 = v145;
        goto LABEL_183;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 32);
      v145 = 0;
    }
    v51 *= 2;
    goto LABEL_188;
  }
  v12 = *v10;
  v13 = sub_AA48A0(v142);
  v14 = *(_DWORD *)(a1 + 88);
  v15 = *(_QWORD *)(v12 - 32);
  v130 = (__int64 *)v13;
  v16 = *(_QWORD *)(v12 - 64);
  if ( v14 )
  {
    v17 = *(_QWORD *)(a1 + 72);
    v18 = 1;
    v19 = 0;
    v20 = (v14 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v21 = (_QWORD *)(v17 + 16 * v20);
    v22 = *v21;
    if ( v12 == *v21 )
    {
LABEL_13:
      v23 = v21[1];
      goto LABEL_14;
    }
    while ( v22 != -4096 )
    {
      if ( !v19 && v22 == -8192 )
        v19 = v21;
      v20 = (v14 - 1) & ((_DWORD)v18 + (_DWORD)v20);
      v21 = (_QWORD *)(v17 + 16LL * (unsigned int)v20);
      v22 = *v21;
      if ( v12 == *v21 )
        goto LABEL_13;
      v18 = (unsigned int)(v18 + 1);
    }
    v93 = *(_DWORD *)(a1 + 80);
    if ( v19 )
      v21 = v19;
    ++*(_QWORD *)(a1 + 64);
    v20 = (unsigned int)(v93 + 1);
    if ( 4 * (int)v20 < 3 * v14 )
    {
      v22 = v14 - *(_DWORD *)(a1 + 84) - (unsigned int)v20;
      if ( (unsigned int)v22 > v14 >> 3 )
        goto LABEL_36;
      sub_27F3350(v134, v14);
      v94 = *(_DWORD *)(a1 + 88);
      if ( v94 )
      {
        v18 = (unsigned int)(v94 - 1);
        v19 = *(_QWORD **)(a1 + 72);
        v95 = 1;
        v20 = (unsigned int)(*(_DWORD *)(a1 + 80) + 1);
        v96 = 0;
        v22 = (unsigned int)v18 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v21 = &v19[2 * v22];
        v97 = *v21;
        if ( v12 != *v21 )
        {
          while ( v97 != -4096 )
          {
            if ( !v96 && v97 == -8192 )
              v96 = v21;
            v22 = (unsigned int)v18 & (v95 + (_DWORD)v22);
            v21 = &v19[2 * (unsigned int)v22];
            v97 = *v21;
            if ( v12 == *v21 )
              goto LABEL_36;
            ++v95;
          }
LABEL_140:
          if ( v96 )
            v21 = v96;
          goto LABEL_36;
        }
        goto LABEL_36;
      }
LABEL_209:
      ++*(_DWORD *)(a1 + 80);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 64);
  }
  sub_27F3350(v134, 2 * v14);
  v46 = *(_DWORD *)(a1 + 88);
  if ( !v46 )
    goto LABEL_209;
  v18 = (unsigned int)(v46 - 1);
  v19 = *(_QWORD **)(a1 + 72);
  v22 = (unsigned int)v18 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v20 = (unsigned int)(*(_DWORD *)(a1 + 80) + 1);
  v21 = &v19[2 * v22];
  v47 = *v21;
  if ( v12 != *v21 )
  {
    v115 = 1;
    v96 = 0;
    while ( v47 != -4096 )
    {
      if ( !v96 && v47 == -8192 )
        v96 = v21;
      v22 = (unsigned int)v18 & (v115 + (_DWORD)v22);
      v21 = &v19[2 * (unsigned int)v22];
      v47 = *v21;
      if ( v12 == *v21 )
        goto LABEL_36;
      ++v115;
    }
    goto LABEL_140;
  }
LABEL_36:
  *(_DWORD *)(a1 + 80) = v20;
  if ( *v21 != -4096 )
    --*(_DWORD *)(a1 + 84);
  *v21 = v12;
  v23 = 0;
  v21[1] = 0;
LABEL_14:
  v135 = v23;
  v24 = (__int64 *)sub_27F4830(a1, *(_QWORD *)(v12 + 40), v20, v22, v19, v18);
  v144[0] = (__int64 *)a1;
  v143 = v24;
  v144[1] = v130;
  v144[2] = (__int64 *)&v143;
  v25 = sub_27F4290(v144, v15);
  v26 = sub_27F4290(v144, v16);
  v27 = sub_27F4290(v144, v135);
  v28 = *(_QWORD *)(v27 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v28 == v27 + 48 )
    goto LABEL_60;
  if ( !v28 )
    goto LABEL_43;
  if ( (unsigned int)*(unsigned __int8 *)(v28 - 24) - 30 > 0xA )
  {
LABEL_60:
    v131 = v27;
    v118 = sub_AA56F0((__int64)v143);
    sub_AA4AC0(v131, v118 + 24);
    sub_B43C20((__int64)&v145, v131);
    v122 = v145;
    v136 = v146;
    v59 = sub_BD2C40(72, 1u);
    v27 = v131;
    if ( v59 )
    {
      sub_B4C8F0((__int64)v59, v118, 1u, (__int64)v122, v136);
      v27 = v131;
    }
  }
  v29 = *(_QWORD *)(v25 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v29 == v25 + 48 )
    goto LABEL_62;
  if ( !v29 )
    goto LABEL_43;
  if ( (unsigned int)*(unsigned __int8 *)(v29 - 24) - 30 > 0xA )
  {
LABEL_62:
    v123 = v27;
    sub_AA4AC0(v25, v27 + 24);
    sub_B43C20((__int64)&v145, v25);
    v137 = v145;
    v132 = v146;
    v60 = sub_BD2C40(72, 1u);
    v27 = v123;
    if ( v60 )
    {
      sub_B4C8F0((__int64)v60, v123, 1u, (__int64)v137, v132);
      v27 = v123;
    }
  }
  v30 = *(_QWORD *)(v26 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v30 != v26 + 48 )
  {
    if ( v30 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(v30 - 24) - 30 <= 0xA )
        goto LABEL_23;
      goto LABEL_111;
    }
LABEL_43:
    BUG();
  }
LABEL_111:
  v127 = v27;
  sub_AA4AC0(v26, v27 + 24);
  sub_B43C20((__int64)&v145, v26);
  v140 = v145;
  v133 = v146;
  v90 = sub_BD2C40(72, 1u);
  v27 = v127;
  if ( v90 )
  {
    sub_B4C8F0((__int64)v90, v127, 1u, (__int64)v140, v133);
    v27 = v127;
  }
LABEL_23:
  if ( v143 != (__int64 *)v3 )
    goto LABEL_24;
  v138 = v27;
  sub_AA5E80(v3, v27);
  v124 = *(__int64 **)(a1 + 24);
  v145 = v143;
  v61 = sub_AA56F0((__int64)v143);
  sub_D6E1B0(v124, v61, v138, &v145, 1, 1);
  v62 = v138;
  v63 = *(_QWORD *)(a1 + 8);
  v64 = 0;
  v65 = *(_DWORD *)(v63 + 32);
  v66 = (unsigned int)(*(_DWORD *)(v138 + 44) + 1);
  if ( (unsigned int)v66 < v65 )
    v64 = *(_QWORD *)(*(_QWORD *)(v63 + 24) + 8 * v66);
  v67 = **(_QWORD **)(*(_QWORD *)(a1 + 16) + 32LL);
  if ( v67 )
  {
    v68 = (unsigned int)(*(_DWORD *)(v67 + 44) + 1);
    v69 = *(_DWORD *)(v67 + 44) + 1;
  }
  else
  {
    v68 = 0;
    v69 = 0;
  }
  if ( v69 >= v65 )
  {
    *(_BYTE *)(v63 + 112) = 0;
    BUG();
  }
  v70 = *(_QWORD *)(*(_QWORD *)(v63 + 24) + 8 * v68);
  *(_BYTE *)(v63 + 112) = 0;
  v71 = *(_QWORD *)(v70 + 8);
  if ( v64 != v71 )
  {
    v72 = *(char **)(v71 + 24);
    v73 = *(unsigned int *)(v71 + 32);
    v74 = &v72[8 * v73];
    if ( (8 * v73) >> 5 )
    {
      v75 = &v72[32 * ((8 * v73) >> 5)];
      while ( v70 != *(_QWORD *)v72 )
      {
        if ( v70 == *((_QWORD *)v72 + 1) )
        {
          v72 += 8;
          v76 = v72 + 8;
          goto LABEL_78;
        }
        if ( v70 == *((_QWORD *)v72 + 2) )
        {
          v72 += 16;
          v76 = v72 + 8;
          goto LABEL_78;
        }
        if ( v70 == *((_QWORD *)v72 + 3) )
        {
          v72 += 24;
          v76 = v72 + 8;
          goto LABEL_78;
        }
        v72 += 32;
        if ( v75 == v72 )
          goto LABEL_161;
      }
      goto LABEL_77;
    }
LABEL_161:
    v111 = v74 - v72;
    if ( v74 - v72 == 16 )
    {
      v112 = v72;
    }
    else
    {
      if ( v111 != 24 )
      {
        if ( v111 != 8 )
        {
LABEL_164:
          v72 = v74;
          v76 = v74 + 8;
          goto LABEL_78;
        }
        goto LABEL_167;
      }
      v76 = v72 + 8;
      v112 = v72 + 8;
      if ( v70 == *(_QWORD *)v72 )
        goto LABEL_78;
    }
    v72 = v112 + 8;
    if ( v70 == *(_QWORD *)v112 )
    {
      v72 = v112;
      goto LABEL_77;
    }
LABEL_167:
    if ( v70 != *(_QWORD *)v72 )
      goto LABEL_164;
LABEL_77:
    v76 = v72 + 8;
LABEL_78:
    if ( v76 != v74 )
    {
      v116 = *(_QWORD *)(v70 + 8);
      v119 = v64;
      v125 = v70;
      memmove(v72, v76, v74 - v76);
      v71 = v116;
      v64 = v119;
      v70 = v125;
      v62 = v138;
      LODWORD(v73) = *(_DWORD *)(v116 + 32);
    }
    *(_DWORD *)(v71 + 32) = v73 - 1;
    *(_QWORD *)(v70 + 8) = v64;
    v77 = *(unsigned int *)(v64 + 32);
    if ( v77 + 1 > (unsigned __int64)*(unsigned int *)(v64 + 36) )
    {
      v121 = v70;
      v128 = v62;
      v141 = v64;
      sub_C8D5F0(v64 + 24, (const void *)(v64 + 40), v77 + 1, 8u, v70, v64);
      v64 = v141;
      v70 = v121;
      v62 = v128;
      v77 = *(unsigned int *)(v141 + 32);
    }
    *(_QWORD *)(*(_QWORD *)(v64 + 24) + 8 * v77) = v70;
    ++*(_DWORD *)(v64 + 32);
    if ( *(_DWORD *)(v70 + 16) != *(_DWORD *)(*(_QWORD *)(v70 + 8) + 16LL) + 1 )
    {
      v147[0] = v70;
      v78 = 1;
      v145 = v147;
      v79 = v147;
      v146 = 0x4000000001LL;
      v139 = v25;
      v126 = v26;
      v120 = v62;
      v117 = v3;
      do
      {
        v80 = v78--;
        v81 = v79[v80 - 1];
        LODWORD(v146) = v78;
        *(_DWORD *)(v81 + 16) = *(_DWORD *)(*(_QWORD *)(v81 + 8) + 16LL) + 1;
        v82 = *(__int64 **)(v81 + 24);
        v83 = &v82[*(unsigned int *)(v81 + 32)];
        if ( v82 != v83 )
        {
          do
          {
            v84 = *v82;
            if ( *(_DWORD *)(*v82 + 16) != *(_DWORD *)(*(_QWORD *)(*v82 + 8) + 16LL) + 1 )
            {
              v85 = v78;
              v86 = v78 + 1LL;
              if ( v86 > HIDWORD(v146) )
              {
                sub_C8D5F0((__int64)&v145, v147, v86, 8u, v70, v64);
                v85 = (unsigned int)v146;
              }
              v145[v85] = v84;
              v78 = v146 + 1;
              LODWORD(v146) = v146 + 1;
            }
            ++v82;
          }
          while ( v83 != v82 );
          v79 = v145;
        }
      }
      while ( v78 );
      v25 = v139;
      v26 = v126;
      v62 = v120;
      v3 = v117;
      if ( v79 != v147 )
      {
        _libc_free((unsigned __int64)v79);
        v62 = v120;
      }
    }
  }
  if ( *(_DWORD *)(a1 + 48) )
  {
    v87 = *(_QWORD **)(a1 + 40);
    v88 = &v87[2 * *(unsigned int *)(a1 + 56)];
    if ( v87 != v88 )
    {
      while ( 1 )
      {
        v89 = v87;
        if ( *v87 != -4096 && *v87 != -8192 )
          break;
        v87 += 2;
        if ( v88 == v87 )
          goto LABEL_24;
      }
      if ( v88 != v87 )
      {
        do
        {
          if ( v89[1] == v3 && *v89 != *(_QWORD *)(v12 + 40) )
            v89[1] = v62;
          v89 += 2;
          if ( v89 == v88 )
            break;
          while ( *v89 == -8192 || *v89 == -4096 )
          {
            v89 += 2;
            if ( v88 == v89 )
              goto LABEL_24;
          }
        }
        while ( v89 != v88 );
      }
    }
  }
LABEL_24:
  v31 = *(_QWORD *)(v12 - 96);
  v32 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v33 = v32;
  if ( v32 )
    sub_B4C9A0((__int64)v32, v25, v26, v31, 3u, 0, 0, 0);
  v34 = v143[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (__int64 *)v34 == v143 + 6 )
  {
    v36 = 0;
  }
  else
  {
    if ( !v34 )
      BUG();
    v35 = *(unsigned __int8 *)(v34 - 24);
    v36 = 0;
    v37 = v34 - 24;
    if ( (unsigned int)(v35 - 30) < 0xB )
      v36 = v37;
  }
  sub_F34910(v36, v33);
  v38 = *(_DWORD *)(a1 + 56);
  if ( !v38 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_151;
  }
  v39 = v142;
  v40 = *(_QWORD *)(a1 + 40);
  v41 = 0;
  v42 = 1;
  v43 = (v38 - 1) & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
  v44 = (__int64 *)(v40 + 16LL * v43);
  v45 = *v44;
  if ( v142 == *v44 )
    return v44[1];
  while ( v45 != -4096 )
  {
    if ( v45 == -8192 && !v41 )
      v41 = v44;
    v43 = (v38 - 1) & (v42 + v43);
    v44 = (__int64 *)(v40 + 16LL * v43);
    v45 = *v44;
    if ( v142 == *v44 )
      return v44[1];
    ++v42;
  }
  v91 = *(_DWORD *)(a1 + 48);
  if ( v41 )
    v44 = v41;
  ++*(_QWORD *)(a1 + 32);
  v92 = v91 + 1;
  if ( 4 * (v91 + 1) >= 3 * v38 )
  {
LABEL_151:
    sub_22E02D0(v129, 2 * v38);
    v105 = *(_DWORD *)(a1 + 56);
    if ( v105 )
    {
      v39 = v142;
      v106 = v105 - 1;
      v107 = *(_QWORD *)(a1 + 40);
      v92 = *(_DWORD *)(a1 + 48) + 1;
      v108 = v106 & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
      v44 = (__int64 *)(v107 + 16LL * v108);
      v109 = *v44;
      if ( *v44 == v142 )
        goto LABEL_124;
      v110 = 1;
      v102 = 0;
      while ( v109 != -4096 )
      {
        if ( v109 == -8192 && !v102 )
          v102 = v44;
        v108 = v106 & (v110 + v108);
        v44 = (__int64 *)(v107 + 16LL * v108);
        v109 = *v44;
        if ( v142 == *v44 )
          goto LABEL_124;
        ++v110;
      }
      goto LABEL_147;
    }
LABEL_207:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
  if ( v38 - *(_DWORD *)(a1 + 52) - v92 <= v38 >> 3 )
  {
    sub_22E02D0(v129, v38);
    v98 = *(_DWORD *)(a1 + 56);
    if ( v98 )
    {
      v39 = v142;
      v99 = v98 - 1;
      v100 = 1;
      v101 = *(_QWORD *)(a1 + 40);
      v92 = *(_DWORD *)(a1 + 48) + 1;
      v102 = 0;
      v103 = v99 & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
      v44 = (__int64 *)(v101 + 16LL * v103);
      v104 = *v44;
      if ( *v44 == v142 )
        goto LABEL_124;
      while ( v104 != -4096 )
      {
        if ( !v102 && v104 == -8192 )
          v102 = v44;
        v103 = v99 & (v100 + v103);
        v44 = (__int64 *)(v101 + 16LL * v103);
        v104 = *v44;
        if ( v142 == *v44 )
          goto LABEL_124;
        ++v100;
      }
LABEL_147:
      if ( v102 )
        v44 = v102;
      goto LABEL_124;
    }
    goto LABEL_207;
  }
LABEL_124:
  *(_DWORD *)(a1 + 48) = v92;
  if ( *v44 != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v44 = v39;
  v3 = 0;
  v44[1] = 0;
  return v3;
}
