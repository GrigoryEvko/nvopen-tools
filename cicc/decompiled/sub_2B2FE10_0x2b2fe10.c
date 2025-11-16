// Function: sub_2B2FE10
// Address: 0x2b2fe10
//
unsigned __int8 *__fastcall sub_2B2FE10(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  __int64 v3; // r13
  _QWORD *v5; // rsi
  __int64 v6; // r9
  _QWORD *v7; // r14
  unsigned __int64 v8; // rsi
  __int64 v9; // rax
  int v10; // ecx
  _QWORD *v11; // rdx
  unsigned __int8 ****v12; // r13
  unsigned __int8 *v13; // r10
  unsigned __int8 *v14; // rcx
  __int64 v15; // r11
  char v16; // r9
  __int64 v17; // rax
  unsigned int v18; // r14d
  unsigned __int8 **v19; // rdi
  unsigned __int8 **v20; // rsi
  unsigned __int8 *v21; // rax
  unsigned __int8 *v22; // r12
  unsigned __int64 v24; // rax
  char v25; // dl
  __int64 v26; // rsi
  unsigned __int8 ****v27; // rax
  int v28; // edi
  unsigned int v29; // esi
  unsigned __int8 ****v30; // r15
  unsigned __int8 *v31; // r8
  int v32; // edi
  unsigned int v33; // esi
  unsigned __int8 *v34; // r8
  unsigned __int8 ****v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 *v40; // r14
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 *v44; // r13
  __int64 v45; // r15
  __int64 v46; // r15
  __int64 v47; // r15
  _BYTE *v48; // r15
  __int64 v49; // rdi
  __int64 v50; // rdi
  __int64 v51; // rdx
  __int64 v52; // r8
  __int64 v53; // rcx
  _BYTE *v54; // rdi
  _QWORD *v55; // rcx
  __int64 v56; // rdi
  __int64 v57; // rdi
  __int64 v58; // rdx
  __int64 v59; // r8
  __int64 v60; // rcx
  _BYTE *v61; // rdi
  _QWORD *v62; // rcx
  __int64 v63; // rdi
  __int64 v64; // rdi
  __int64 v65; // rdx
  __int64 v66; // r8
  __int64 v67; // rcx
  _BYTE *v68; // rdi
  _QWORD *v69; // rcx
  __int64 v70; // rdi
  __int64 v71; // rdi
  __int64 v72; // rdx
  __int64 v73; // r8
  __int64 v74; // rcx
  _BYTE *v75; // rdi
  _QWORD *v76; // rcx
  unsigned __int64 v77; // rsi
  __int64 v78; // r14
  __int64 v79; // r15
  int v80; // r9d
  int v81; // r9d
  __int64 v82; // rdx
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rax
  __int64 *v86; // r13
  __int64 *v87; // rcx
  __int64 v88; // rax
  __int64 *v89; // r14
  __int64 v90; // r15
  __int64 v91; // r15
  __int64 v92; // r15
  _BYTE *v93; // r15
  __int64 v94; // rdi
  __int64 v95; // rdi
  __int64 v96; // rdx
  __int64 v97; // r8
  __int64 v98; // rcx
  _BYTE *v99; // rdi
  _QWORD *v100; // rdx
  __int64 *v101; // rdi
  unsigned __int8 *v102; // rax
  __int64 v103; // rdi
  __int64 v104; // rdi
  __int64 v105; // rdx
  __int64 v106; // r8
  __int64 v107; // rcx
  _BYTE *v108; // rdi
  _QWORD *v109; // rdx
  __int64 v110; // rdi
  __int64 v111; // rdi
  __int64 v112; // rdx
  __int64 v113; // r8
  __int64 v114; // rcx
  _BYTE *v115; // rdi
  _QWORD *v116; // rdx
  __int64 v117; // rdi
  __int64 v118; // rdi
  __int64 v119; // rdx
  __int64 v120; // r8
  __int64 v121; // rcx
  _BYTE *v122; // rdi
  _QWORD *v123; // rdx
  signed __int64 v124; // rax
  __int64 *v125; // [rsp+0h] [rbp-80h]
  unsigned __int8 ****v126; // [rsp+8h] [rbp-78h]
  __int64 *v127; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v128; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int8 *v129; // [rsp+18h] [rbp-68h] BYREF
  _QWORD *v130[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v131; // [rsp+40h] [rbp-40h]

  v129 = a2;
  v128 = a3;
  if ( !a2 )
    return v128;
  v3 = *(_QWORD *)a1;
  v5 = *(_QWORD **)(****(_QWORD ****)(a1 + 8) + 48LL);
  v130[0] = v5;
  if ( v5 && (sub_B96E90((__int64)v130, (__int64)v5, 1), (v7 = v130[0]) != 0) )
  {
    v8 = *(unsigned int *)(v3 + 8);
    v9 = *(_QWORD *)v3;
    v10 = *(_DWORD *)(v3 + 8);
    v11 = (_QWORD *)(*(_QWORD *)v3 + 16 * v8);
    if ( *(_QWORD **)v3 != v11 )
    {
      while ( *(_DWORD *)v9 )
      {
        v9 += 16;
        if ( v11 == (_QWORD *)v9 )
          goto LABEL_20;
      }
      *(_QWORD **)(v9 + 8) = v130[0];
      goto LABEL_9;
    }
LABEL_20:
    v24 = *(unsigned int *)(v3 + 12);
    if ( v8 >= v24 )
    {
      v77 = v8 + 1;
      if ( v24 < v77 )
      {
        sub_C8D5F0(v3, (const void *)(v3 + 16), v77, 0x10u, v3 + 16, v6);
        v11 = (_QWORD *)(*(_QWORD *)v3 + 16LL * *(unsigned int *)(v3 + 8));
      }
      *v11 = 0;
      v11[1] = v7;
      ++*(_DWORD *)(v3 + 8);
      v7 = v130[0];
    }
    else
    {
      if ( v11 )
      {
        *(_DWORD *)v11 = 0;
        v11[1] = v7;
        v10 = *(_DWORD *)(v3 + 8);
        v7 = v130[0];
      }
      *(_DWORD *)(v3 + 8) = v10 + 1;
    }
  }
  else
  {
    sub_93FB40(v3, 0);
    v7 = v130[0];
  }
  if ( v7 )
LABEL_9:
    sub_B91220((__int64)v130, (__int64)v7);
  v12 = *(unsigned __int8 *****)(a1 + 8);
  v13 = v129;
  v14 = v128;
  if ( !**(_BYTE **)(a1 + 16) )
    goto LABEL_13;
  v25 = (_BYTE)v12[49] & 1;
  if ( v25 )
  {
    v27 = v12 + 50;
    v28 = 15;
  }
  else
  {
    v26 = *((unsigned int *)v12 + 102);
    v27 = (unsigned __int8 ****)v12[50];
    if ( !(_DWORD)v26 )
      goto LABEL_145;
    v28 = v26 - 1;
  }
  v29 = v28 & (((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4));
  v30 = &v27[9 * v29];
  v31 = (unsigned __int8 *)*v30;
  if ( v129 == (unsigned __int8 *)*v30 )
    goto LABEL_29;
  v81 = 1;
  while ( v31 != (unsigned __int8 *)-4096LL )
  {
    v29 = v28 & (v81 + v29);
    v30 = &v27[9 * v29];
    v31 = (unsigned __int8 *)*v30;
    if ( *v30 == (unsigned __int8 ***)v129 )
      goto LABEL_29;
    ++v81;
  }
  if ( v25 )
  {
    v79 = 144;
    goto LABEL_146;
  }
  v26 = *((unsigned int *)v12 + 102);
LABEL_145:
  v79 = 9 * v26;
LABEL_146:
  v30 = &v27[v79];
LABEL_29:
  if ( v25 )
  {
    v32 = 15;
  }
  else
  {
    v36 = *((unsigned int *)v12 + 102);
    if ( !(_DWORD)v36 )
      goto LABEL_142;
    v32 = v36 - 1;
  }
  v33 = v32 & (((unsigned int)v128 >> 9) ^ ((unsigned int)v128 >> 4));
  v34 = (unsigned __int8 *)v27[9 * v33];
  v126 = &v27[9 * v33];
  if ( v34 == v128 )
    goto LABEL_32;
  v80 = 1;
  while ( v34 != (unsigned __int8 *)-4096LL )
  {
    v33 = v32 & (v80 + v33);
    v34 = (unsigned __int8 *)v27[9 * v33];
    v126 = &v27[9 * v33];
    if ( v34 == v128 )
      goto LABEL_32;
    ++v80;
  }
  if ( v25 )
  {
    v78 = 144;
    goto LABEL_143;
  }
  v36 = *((unsigned int *)v12 + 102);
LABEL_142:
  v78 = 9 * v36;
LABEL_143:
  v126 = &v27[v78];
LABEL_32:
  if ( !v25 )
  {
    v35 = &v27[9 * *((unsigned int *)v12 + 102)];
    if ( v30 != v35 )
      goto LABEL_34;
LABEL_38:
    if ( v126 == v35 )
      goto LABEL_13;
    goto LABEL_34;
  }
  v35 = v27 + 144;
  if ( v30 == v35 )
    goto LABEL_38;
LABEL_34:
  if ( sub_98ED70(v129, **(_QWORD **)(a1 + 24), 0, 0, 0) )
  {
    v12 = *(unsigned __int8 *****)(a1 + 8);
    v14 = v128;
    v13 = v129;
    goto LABEL_13;
  }
  v37 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(v37 + 392) & 1) != 0 )
  {
    v38 = v37 + 400;
    v39 = 1152;
  }
  else
  {
    v38 = *(_QWORD *)(v37 + 400);
    v39 = 72LL * *(unsigned int *)(v37 + 408);
  }
  if ( v30 == (unsigned __int8 ****)(v39 + v38) )
    goto LABEL_136;
  v40 = (__int64 *)v30[1];
  v41 = 8LL * *((unsigned int *)v30 + 4);
  v125 = &v40[(unsigned __int64)v41 / 8];
  v130[0] = &v129;
  v42 = v41 >> 3;
  v43 = v41 >> 5;
  if ( !v43 )
    goto LABEL_132;
  v44 = &v40[4 * v43];
  do
  {
    v48 = (_BYTE *)*v40;
    if ( *(_BYTE *)*v40 != 86 )
      goto LABEL_48;
    v49 = *((_QWORD *)v48 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v49 + 8) - 17 <= 1 )
      v49 = **(_QWORD **)(v49 + 16);
    if ( !sub_BCAC40(v49, 1) )
    {
LABEL_149:
      v50 = *((_QWORD *)v48 + 1);
LABEL_59:
      if ( (unsigned int)*(unsigned __int8 *)(v50 + 8) - 17 <= 1 )
        v50 = **(_QWORD **)(v50 + 16);
      if ( !sub_BCAC40(v50, 1) )
        goto LABEL_48;
      if ( *v48 != 58 )
      {
        if ( *v48 != 86 )
          goto LABEL_48;
        v53 = *((_QWORD *)v48 + 1);
        if ( *(_QWORD *)(*((_QWORD *)v48 - 12) + 8LL) != v53 )
          goto LABEL_48;
        v54 = (_BYTE *)*((_QWORD *)v48 - 8);
        if ( *v54 > 0x15u || !sub_AD7A80(v54, 1, v51, v53, v52) )
          goto LABEL_48;
      }
      goto LABEL_67;
    }
    if ( *v48 != 57 )
    {
      v50 = *((_QWORD *)v48 + 1);
      if ( *v48 != 86 || *(_QWORD *)(*((_QWORD *)v48 - 12) + 8LL) != v50 || **((_BYTE **)v48 - 4) > 0x15u )
        goto LABEL_59;
      if ( !sub_AC30F0(*((_QWORD *)v48 - 4)) )
        goto LABEL_149;
    }
LABEL_67:
    sub_2B27770((__int64)v48);
    if ( (v48[7] & 0x40) != 0 )
      v55 = (_QWORD *)*((_QWORD *)v48 - 1);
    else
      v55 = &v48[-32 * (*((_DWORD *)v48 + 1) & 0x7FFFFFF)];
    if ( *v130[0] == *v55 )
      goto LABEL_70;
LABEL_48:
    v45 = v40[1];
    if ( *(_BYTE *)v45 != 86 )
      goto LABEL_49;
    v56 = *(_QWORD *)(v45 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v56 + 8) - 17 <= 1 )
      v56 = **(_QWORD **)(v56 + 16);
    if ( !sub_BCAC40(v56, 1) )
    {
LABEL_152:
      v57 = *(_QWORD *)(v45 + 8);
LABEL_80:
      if ( (unsigned int)*(unsigned __int8 *)(v57 + 8) - 17 <= 1 )
        v57 = **(_QWORD **)(v57 + 16);
      if ( !sub_BCAC40(v57, 1) )
        goto LABEL_49;
      if ( *(_BYTE *)v45 != 58 )
      {
        if ( *(_BYTE *)v45 != 86 )
          goto LABEL_49;
        v60 = *(_QWORD *)(v45 + 8);
        if ( *(_QWORD *)(*(_QWORD *)(v45 - 96) + 8LL) != v60 )
          goto LABEL_49;
        v61 = *(_BYTE **)(v45 - 64);
        if ( *v61 > 0x15u || !sub_AD7A80(v61, 1, v58, v60, v59) )
          goto LABEL_49;
      }
      goto LABEL_88;
    }
    if ( *(_BYTE *)v45 != 57 )
    {
      v57 = *(_QWORD *)(v45 + 8);
      if ( *(_BYTE *)v45 != 86 || *(_QWORD *)(*(_QWORD *)(v45 - 96) + 8LL) != v57 || **(_BYTE **)(v45 - 32) > 0x15u )
        goto LABEL_80;
      if ( !sub_AC30F0(*(_QWORD *)(v45 - 32)) )
        goto LABEL_152;
    }
LABEL_88:
    sub_2B27770(v45);
    if ( (*(_BYTE *)(v45 + 7) & 0x40) != 0 )
      v62 = *(_QWORD **)(v45 - 8);
    else
      v62 = (_QWORD *)(v45 - 32LL * (*(_DWORD *)(v45 + 4) & 0x7FFFFFF));
    if ( *v130[0] == *v62 )
    {
      ++v40;
      goto LABEL_70;
    }
LABEL_49:
    v46 = v40[2];
    if ( *(_BYTE *)v46 != 86 )
      goto LABEL_50;
    v63 = *(_QWORD *)(v46 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v63 + 8) - 17 <= 1 )
      v63 = **(_QWORD **)(v63 + 16);
    if ( !sub_BCAC40(v63, 1) )
    {
LABEL_155:
      v64 = *(_QWORD *)(v46 + 8);
LABEL_99:
      if ( (unsigned int)*(unsigned __int8 *)(v64 + 8) - 17 <= 1 )
        v64 = **(_QWORD **)(v64 + 16);
      if ( !sub_BCAC40(v64, 1) )
        goto LABEL_50;
      if ( *(_BYTE *)v46 != 58 )
      {
        if ( *(_BYTE *)v46 != 86 )
          goto LABEL_50;
        v67 = *(_QWORD *)(v46 + 8);
        if ( *(_QWORD *)(*(_QWORD *)(v46 - 96) + 8LL) != v67 )
          goto LABEL_50;
        v68 = *(_BYTE **)(v46 - 64);
        if ( *v68 > 0x15u || !sub_AD7A80(v68, 1, v65, v67, v66) )
          goto LABEL_50;
      }
      goto LABEL_107;
    }
    if ( *(_BYTE *)v46 != 57 )
    {
      v64 = *(_QWORD *)(v46 + 8);
      if ( *(_BYTE *)v46 != 86 || v64 != *(_QWORD *)(*(_QWORD *)(v46 - 96) + 8LL) || **(_BYTE **)(v46 - 32) > 0x15u )
        goto LABEL_99;
      if ( !sub_AC30F0(*(_QWORD *)(v46 - 32)) )
        goto LABEL_155;
    }
LABEL_107:
    sub_2B27770(v46);
    if ( (*(_BYTE *)(v46 + 7) & 0x40) != 0 )
      v69 = *(_QWORD **)(v46 - 8);
    else
      v69 = (_QWORD *)(v46 - 32LL * (*(_DWORD *)(v46 + 4) & 0x7FFFFFF));
    if ( *v130[0] == *v69 )
    {
      v40 += 2;
      goto LABEL_70;
    }
LABEL_50:
    v47 = v40[3];
    if ( *(_BYTE *)v47 != 86 )
      goto LABEL_51;
    v70 = *(_QWORD *)(v47 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v70 + 8) - 17 <= 1 )
      v70 = **(_QWORD **)(v70 + 16);
    if ( !sub_BCAC40(v70, 1) )
    {
LABEL_158:
      v71 = *(_QWORD *)(v47 + 8);
LABEL_118:
      if ( (unsigned int)*(unsigned __int8 *)(v71 + 8) - 17 <= 1 )
        v71 = **(_QWORD **)(v71 + 16);
      if ( !sub_BCAC40(v71, 1) )
        goto LABEL_51;
      if ( *(_BYTE *)v47 != 58 )
      {
        if ( *(_BYTE *)v47 != 86 )
          goto LABEL_51;
        v74 = *(_QWORD *)(v47 + 8);
        if ( *(_QWORD *)(*(_QWORD *)(v47 - 96) + 8LL) != v74 )
          goto LABEL_51;
        v75 = *(_BYTE **)(v47 - 64);
        if ( *v75 > 0x15u || !sub_AD7A80(v75, 1, v72, v74, v73) )
          goto LABEL_51;
      }
      goto LABEL_126;
    }
    if ( *(_BYTE *)v47 != 57 )
    {
      v71 = *(_QWORD *)(v47 + 8);
      if ( *(_BYTE *)v47 != 86 || v71 != *(_QWORD *)(*(_QWORD *)(v47 - 96) + 8LL) || **(_BYTE **)(v47 - 32) > 0x15u )
        goto LABEL_118;
      if ( !sub_AC30F0(*(_QWORD *)(v47 - 32)) )
        goto LABEL_158;
    }
LABEL_126:
    sub_2B27770(v47);
    if ( (*(_BYTE *)(v47 + 7) & 0x40) != 0 )
      v76 = *(_QWORD **)(v47 - 8);
    else
      v76 = (_QWORD *)(v47 - 32LL * (*(_DWORD *)(v47 + 4) & 0x7FFFFFF));
    if ( *v130[0] == *v76 )
    {
      v40 += 3;
      goto LABEL_70;
    }
LABEL_51:
    v40 += 4;
  }
  while ( v40 != v44 );
  v42 = v125 - v40;
LABEL_132:
  if ( v42 == 2 )
    goto LABEL_269;
  if ( v42 != 3 )
  {
    if ( v42 == 1 )
      goto LABEL_135;
    goto LABEL_136;
  }
  if ( sub_2B280B0(v130, *v40) )
    goto LABEL_70;
  ++v40;
LABEL_269:
  if ( sub_2B280B0(v130, *v40) )
    goto LABEL_70;
  ++v40;
LABEL_135:
  if ( sub_2B280B0(v130, *v40) )
  {
LABEL_70:
    if ( v125 == v40 )
      goto LABEL_136;
    v14 = v128;
    v12 = *(unsigned __int8 *****)(a1 + 8);
    v13 = v129;
    goto LABEL_13;
  }
LABEL_136:
  if ( sub_98ED70(v128, **(_QWORD **)(a1 + 24), 0, 0, 0) )
    goto LABEL_137;
  v82 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(v82 + 392) & 1) != 0 )
  {
    v83 = v82 + 400;
    v84 = 1152;
  }
  else
  {
    v83 = *(_QWORD *)(v82 + 400);
    v84 = 72LL * *(unsigned int *)(v82 + 408);
  }
  if ( v126 == (unsigned __int8 ****)(v84 + v83) )
    goto LABEL_195;
  v85 = *((unsigned int *)v126 + 4);
  v86 = (__int64 *)v126[1];
  v130[0] = &v128;
  v85 *= 8;
  v87 = (__int64 *)((char *)v86 + v85);
  v88 = v85 >> 5;
  v127 = v87;
  if ( v88 )
  {
    v89 = &v86[4 * v88];
    while ( 1 )
    {
      v93 = (_BYTE *)*v86;
      if ( *(_BYTE *)*v86 != 86 )
        goto LABEL_172;
      v94 = *((_QWORD *)v93 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v94 + 8) - 17 <= 1 )
        v94 = **(_QWORD **)(v94 + 16);
      if ( !sub_BCAC40(v94, 1) )
        break;
      if ( *v93 != 57 )
      {
        v95 = *((_QWORD *)v93 + 1);
        if ( *v93 != 86 || v95 != *(_QWORD *)(*((_QWORD *)v93 - 12) + 8LL) || **((_BYTE **)v93 - 4) > 0x15u )
          goto LABEL_183;
        if ( !sub_AC30F0(*((_QWORD *)v93 - 4)) )
          break;
      }
LABEL_191:
      sub_2B27770((__int64)v93);
      if ( (v93[7] & 0x40) != 0 )
        v100 = (_QWORD *)*((_QWORD *)v93 - 1);
      else
        v100 = &v93[-32 * (*((_DWORD *)v93 + 1) & 0x7FFFFFF)];
      if ( *v130[0] == *v100 )
        goto LABEL_194;
LABEL_172:
      v90 = v86[1];
      if ( *(_BYTE *)v90 != 86 )
        goto LABEL_173;
      v110 = *(_QWORD *)(v90 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v110 + 8) - 17 <= 1 )
        v110 = **(_QWORD **)(v110 + 16);
      if ( !sub_BCAC40(v110, 1) )
      {
LABEL_263:
        v111 = *(_QWORD *)(v90 + 8);
LABEL_222:
        if ( (unsigned int)*(unsigned __int8 *)(v111 + 8) - 17 <= 1 )
          v111 = **(_QWORD **)(v111 + 16);
        if ( !sub_BCAC40(v111, 1) )
          goto LABEL_173;
        if ( *(_BYTE *)v90 != 58 )
        {
          if ( *(_BYTE *)v90 != 86 )
            goto LABEL_173;
          v114 = *(_QWORD *)(v90 + 8);
          if ( *(_QWORD *)(*(_QWORD *)(v90 - 96) + 8LL) != v114 )
            goto LABEL_173;
          v115 = *(_BYTE **)(v90 - 64);
          if ( *v115 > 0x15u || !sub_AD7A80(v115, 1, v112, v114, v113) )
            goto LABEL_173;
        }
        goto LABEL_230;
      }
      if ( *(_BYTE *)v90 != 57 )
      {
        v111 = *(_QWORD *)(v90 + 8);
        if ( *(_BYTE *)v90 != 86 || *(_QWORD *)(*(_QWORD *)(v90 - 96) + 8LL) != v111 || **(_BYTE **)(v90 - 32) > 0x15u )
          goto LABEL_222;
        if ( !sub_AC30F0(*(_QWORD *)(v90 - 32)) )
          goto LABEL_263;
      }
LABEL_230:
      sub_2B27770(v90);
      if ( (*(_BYTE *)(v90 + 7) & 0x40) != 0 )
        v116 = *(_QWORD **)(v90 - 8);
      else
        v116 = (_QWORD *)(v90 - 32LL * (*(_DWORD *)(v90 + 4) & 0x7FFFFFF));
      if ( *v130[0] == *v116 )
      {
        ++v86;
        goto LABEL_194;
      }
LABEL_173:
      v91 = v86[2];
      if ( *(_BYTE *)v91 != 86 )
        goto LABEL_174;
      v103 = *(_QWORD *)(v91 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v103 + 8) - 17 <= 1 )
        v103 = **(_QWORD **)(v103 + 16);
      if ( !sub_BCAC40(v103, 1) )
      {
LABEL_266:
        v104 = *(_QWORD *)(v91 + 8);
LABEL_203:
        if ( (unsigned int)*(unsigned __int8 *)(v104 + 8) - 17 <= 1 )
          v104 = **(_QWORD **)(v104 + 16);
        if ( !sub_BCAC40(v104, 1) )
          goto LABEL_174;
        if ( *(_BYTE *)v91 != 58 )
        {
          if ( *(_BYTE *)v91 != 86 )
            goto LABEL_174;
          v107 = *(_QWORD *)(v91 + 8);
          if ( *(_QWORD *)(*(_QWORD *)(v91 - 96) + 8LL) != v107 )
            goto LABEL_174;
          v108 = *(_BYTE **)(v91 - 64);
          if ( *v108 > 0x15u || !sub_AD7A80(v108, 1, v105, v107, v106) )
            goto LABEL_174;
        }
        goto LABEL_211;
      }
      if ( *(_BYTE *)v91 != 57 )
      {
        v104 = *(_QWORD *)(v91 + 8);
        if ( *(_BYTE *)v91 != 86 || *(_QWORD *)(*(_QWORD *)(v91 - 96) + 8LL) != v104 || **(_BYTE **)(v91 - 32) > 0x15u )
          goto LABEL_203;
        if ( !sub_AC30F0(*(_QWORD *)(v91 - 32)) )
          goto LABEL_266;
      }
LABEL_211:
      sub_2B27770(v91);
      if ( (*(_BYTE *)(v91 + 7) & 0x40) != 0 )
        v109 = *(_QWORD **)(v91 - 8);
      else
        v109 = (_QWORD *)(v91 - 32LL * (*(_DWORD *)(v91 + 4) & 0x7FFFFFF));
      if ( *v130[0] == *v109 )
      {
        v86 += 2;
        goto LABEL_194;
      }
LABEL_174:
      v92 = v86[3];
      if ( *(_BYTE *)v92 != 86 )
        goto LABEL_175;
      v117 = *(_QWORD *)(v92 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v117 + 8) - 17 <= 1 )
        v117 = **(_QWORD **)(v117 + 16);
      if ( !sub_BCAC40(v117, 1) )
      {
LABEL_260:
        v118 = *(_QWORD *)(v92 + 8);
LABEL_241:
        if ( (unsigned int)*(unsigned __int8 *)(v118 + 8) - 17 <= 1 )
          v118 = **(_QWORD **)(v118 + 16);
        if ( !sub_BCAC40(v118, 1) )
          goto LABEL_175;
        if ( *(_BYTE *)v92 != 58 )
        {
          if ( *(_BYTE *)v92 != 86 )
            goto LABEL_175;
          v121 = *(_QWORD *)(v92 + 8);
          if ( *(_QWORD *)(*(_QWORD *)(v92 - 96) + 8LL) != v121 )
            goto LABEL_175;
          v122 = *(_BYTE **)(v92 - 64);
          if ( *v122 > 0x15u || !sub_AD7A80(v122, 1, v119, v121, v120) )
            goto LABEL_175;
        }
        goto LABEL_249;
      }
      if ( *(_BYTE *)v92 != 57 )
      {
        v118 = *(_QWORD *)(v92 + 8);
        if ( *(_BYTE *)v92 != 86 || *(_QWORD *)(*(_QWORD *)(v92 - 96) + 8LL) != v118 || **(_BYTE **)(v92 - 32) > 0x15u )
          goto LABEL_241;
        if ( !sub_AC30F0(*(_QWORD *)(v92 - 32)) )
          goto LABEL_260;
      }
LABEL_249:
      sub_2B27770(v92);
      if ( (*(_BYTE *)(v92 + 7) & 0x40) != 0 )
        v123 = *(_QWORD **)(v92 - 8);
      else
        v123 = (_QWORD *)(v92 - 32LL * (*(_DWORD *)(v92 + 4) & 0x7FFFFFF));
      if ( *v130[0] == *v123 )
      {
        v86 += 3;
        goto LABEL_194;
      }
LABEL_175:
      v86 += 4;
      if ( v89 == v86 )
        goto LABEL_273;
    }
    v95 = *((_QWORD *)v93 + 1);
LABEL_183:
    if ( (unsigned int)*(unsigned __int8 *)(v95 + 8) - 17 <= 1 )
      v95 = **(_QWORD **)(v95 + 16);
    if ( !sub_BCAC40(v95, 1) )
      goto LABEL_172;
    if ( *v93 != 58 )
    {
      if ( *v93 != 86 )
        goto LABEL_172;
      v98 = *((_QWORD *)v93 + 1);
      if ( *(_QWORD *)(*((_QWORD *)v93 - 12) + 8LL) != v98 )
        goto LABEL_172;
      v99 = (_BYTE *)*((_QWORD *)v93 - 8);
      if ( *v99 > 0x15u || !sub_AD7A80(v99, 1, v96, v98, v97) )
        goto LABEL_172;
    }
    goto LABEL_191;
  }
LABEL_273:
  v124 = (char *)v127 - (char *)v86;
  if ( (char *)v127 - (char *)v86 == 16 )
  {
LABEL_282:
    if ( sub_2B27F80(v130, *v86) )
      goto LABEL_194;
    ++v86;
LABEL_276:
    if ( sub_2B27F80(v130, *v86) )
      goto LABEL_194;
LABEL_195:
    v101 = *(__int64 **)a1;
    v131 = 257;
    v102 = (unsigned __int8 *)sub_1156690(v101, (__int64)v129, (__int64)v130);
    v12 = *(unsigned __int8 *****)(a1 + 8);
    v14 = v128;
    v129 = v102;
    v13 = v102;
  }
  else
  {
    if ( v124 != 24 )
    {
      if ( v124 == 8 )
        goto LABEL_276;
      goto LABEL_195;
    }
    if ( !sub_2B27F80(v130, *v86) )
    {
      ++v86;
      goto LABEL_282;
    }
LABEL_194:
    if ( v127 == v86 )
      goto LABEL_195;
LABEL_137:
    v14 = v129;
    v13 = v128;
    v12 = *(unsigned __int8 *****)(a1 + 8);
    v129 = v128;
    v128 = v14;
  }
LABEL_13:
  v15 = *(_QWORD *)a1;
  v16 = 1;
  v130[0] = "op.rdx";
  v131 = 259;
  v17 = *((unsigned int *)v12 + 2);
  v18 = *((_DWORD *)v12 + 394);
  if ( v17 != 2 )
  {
    v16 = 0;
    if ( v17 == 1 )
    {
      v19 = **v12;
      v20 = &v19[*((unsigned int *)*v12 + 2)];
      v16 = v20 != sub_2B0AAE0(v19, (__int64)v20);
    }
  }
  v21 = (unsigned __int8 *)sub_2B21610(v15, v18, (__int64)v13, (__int64)v14, (__int64)v130, v16);
  v22 = v21;
  if ( v18 - 6 <= 3 && *v21 == 86 )
  {
    sub_F70480(*((unsigned __int8 **)v21 - 12), **v12, *((unsigned int *)*v12 + 2), 0, 0);
    sub_F70480(v22, (*v12)[18], *((unsigned int *)*v12 + 38), 0, 0);
  }
  else
  {
    sub_F70480(v21, **v12, *((unsigned int *)*v12 + 2), 0, 0);
  }
  return v22;
}
