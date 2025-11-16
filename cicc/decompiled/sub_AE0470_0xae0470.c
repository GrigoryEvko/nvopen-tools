// Function: sub_AE0470
// Address: 0xae0470
//
__int64 __fastcall sub_AE0470(__int64 a1, __int64 *a2, char a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 *v5; // r13
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 result; // rax
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rax
  unsigned __int8 v13; // si
  __int64 v14; // r11
  __int64 v15; // rbx
  __int64 *v16; // r14
  signed __int64 v17; // r12
  __int64 v18; // rsi
  int v19; // r12d
  bool v20; // al
  __int64 v21; // rdi
  __int64 v22; // rsi
  unsigned __int8 v23; // al
  __int64 v24; // r12
  __int64 v25; // r13
  __int64 v26; // rbx
  __int64 v27; // rsi
  int v28; // eax
  int v29; // r13d
  unsigned __int64 v30; // rcx
  __int64 *v31; // r14
  __int64 *v32; // rbx
  __int64 v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // rsi
  unsigned __int8 v36; // al
  __int64 v37; // r12
  __int64 v38; // rbx
  int v39; // eax
  unsigned __int64 v40; // r14
  __int64 v41; // rcx
  __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rsi
  unsigned __int8 v46; // al
  __int64 v47; // r14
  __int64 v48; // r12
  __int64 v49; // rbx
  __int64 v50; // rsi
  int v51; // eax
  unsigned __int64 v52; // rbx
  __int64 *v53; // r13
  __int64 *v54; // r12
  __int64 v55; // rsi
  __int64 v56; // rdx
  unsigned __int8 v57; // dl
  __int64 *v58; // rbx
  __int64 v59; // rax
  __int64 *v60; // r9
  __int64 v61; // r8
  unsigned int v62; // esi
  _QWORD *v63; // rax
  __int64 *v64; // r13
  int v65; // r11d
  char *v66; // r10
  unsigned int v67; // edi
  char *v68; // rcx
  __int64 v69; // rdx
  __int64 v70; // r12
  __int64 v71; // rdx
  __int64 v72; // rsi
  int v73; // eax
  __int64 v74; // rax
  unsigned __int64 v75; // rdx
  __int64 v76; // rsi
  char *v77; // rdi
  unsigned int v78; // r14d
  int v79; // r9d
  __int64 v80; // rcx
  __int64 *v81; // rsi
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // r8
  __int64 v85; // r9
  __int64 v86; // rdi
  int v87; // edx
  __int64 *v88; // rax
  __int64 v89; // rdx
  __int64 *v90; // rsi
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  __int64 *v95; // rdi
  int v96; // edx
  __int64 *v97; // rax
  __int64 v98; // rdx
  __int64 *v99; // rsi
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // rdi
  int v105; // eax
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rsi
  __int64 v109; // rax
  __int64 v110; // rax
  int v111; // r11d
  char *v112; // r9
  int v113; // r11d
  unsigned int v114; // r9d
  int v115; // [rsp+8h] [rbp-C8h]
  int v116; // [rsp+8h] [rbp-C8h]
  __int64 v117; // [rsp+10h] [rbp-C0h]
  __int64 v118; // [rsp+18h] [rbp-B8h]
  __int64 v119; // [rsp+20h] [rbp-B0h]
  __int64 v120; // [rsp+20h] [rbp-B0h]
  int v121; // [rsp+20h] [rbp-B0h]
  __int64 v122; // [rsp+20h] [rbp-B0h]
  __int64 v123; // [rsp+28h] [rbp-A8h]
  __int64 v124; // [rsp+28h] [rbp-A8h]
  __int64 v125; // [rsp+30h] [rbp-A0h] BYREF
  void *src; // [rsp+38h] [rbp-98h]
  __int64 v127; // [rsp+40h] [rbp-90h]
  __int64 v128; // [rsp+48h] [rbp-88h]
  __int64 *v129; // [rsp+50h] [rbp-80h] BYREF
  __int64 v130; // [rsp+58h] [rbp-78h]
  __int64 v131; // [rsp+60h] [rbp-70h] BYREF
  __int64 v132; // [rsp+68h] [rbp-68h] BYREF
  void *v133; // [rsp+70h] [rbp-60h]
  __int64 v134; // [rsp+78h] [rbp-58h]
  __int64 v135; // [rsp+80h] [rbp-50h]
  _QWORD v136[2]; // [rsp+88h] [rbp-48h] BYREF
  _BYTE v137[56]; // [rsp+98h] [rbp-38h] BYREF

  v4 = a1;
  v5 = (__int64 *)(a1 + 72);
  v6 = a1 + 120;
  v7 = *a2;
  *(_QWORD *)(v6 - 120) = a2;
  *(_QWORD *)(v6 - 104) = a4;
  *(_QWORD *)(v6 - 112) = v7;
  result = 0x400000000LL;
  *(_QWORD *)(v6 - 96) = 0;
  *(_QWORD *)(v6 - 88) = 0;
  *(_QWORD *)(v6 - 80) = 0;
  *(_QWORD *)(v6 - 72) = 0;
  *(_QWORD *)(v6 - 64) = v5;
  *(_QWORD *)(v6 - 56) = 0x400000000LL;
  v117 = v6;
  *(_QWORD *)(v4 + 104) = v6;
  *(_QWORD *)(v4 + 152) = v4 + 168;
  *(_QWORD *)(v4 + 200) = v4 + 216;
  *(_QWORD *)(v4 + 112) = 0x400000000LL;
  *(_QWORD *)(v4 + 160) = 0x400000000LL;
  *(_QWORD *)(v4 + 208) = 0x400000000LL;
  v118 = v4 + 264;
  *(_QWORD *)(v4 + 248) = v4 + 264;
  *(_QWORD *)(v4 + 256) = 0x400000000LL;
  *(_QWORD *)(v4 + 296) = 0;
  *(_QWORD *)(v4 + 304) = 0;
  *(_QWORD *)(v4 + 312) = 0;
  *(_DWORD *)(v4 + 320) = 0;
  *(_QWORD *)(v4 + 328) = v4 + 344;
  *(_QWORD *)(v4 + 336) = 0;
  *(_QWORD *)(v4 + 344) = v4 + 360;
  *(_QWORD *)(v4 + 352) = 0x400000000LL;
  *(_BYTE *)(v4 + 392) = a3;
  *(_QWORD *)(v4 + 400) = 0;
  *(_QWORD *)(v4 + 408) = 0;
  *(_QWORD *)(v4 + 416) = 0;
  *(_DWORD *)(v4 + 424) = 0;
  if ( !a4 )
    return result;
  v9 = *(_BYTE *)(a4 - 16);
  v10 = a4 - 16;
  if ( (v9 & 2) != 0 )
    v11 = *(_QWORD *)(a4 - 32);
  else
    v11 = v10 - 8LL * ((v9 >> 2) & 0xF);
  v12 = *(_QWORD *)(v11 + 32);
  if ( v12 )
  {
    v13 = *(_BYTE *)(v12 - 16);
    if ( (v13 & 2) != 0 )
    {
      v14 = *(_QWORD *)(v12 - 32);
      v15 = v14 + 8LL * *(unsigned int *)(v12 - 24);
    }
    else
    {
      v108 = (v13 >> 2) & 0xF;
      v14 = v12 - 16 - 8 * v108;
      v15 = v12 - 16 + 8 * (((*(_WORD *)(v12 - 16) >> 6) & 0xF) - v108);
    }
    if ( v15 != v14 )
    {
      v16 = (__int64 *)v14;
      v17 = ((unsigned __int64)(v15 - 8 - v14) >> 3) + 1;
      if ( v17 > 4 )
      {
        v90 = (__int64 *)sub_C8D7D0(v4 + 56, v5, ((unsigned __int64)(v15 - 8 - v14) >> 3) + 1, 8, &v131);
        sub_ADDB20(v4 + 56, v90, v91, v92, v93, v94);
        v95 = *(__int64 **)(v4 + 56);
        v96 = v131;
        v97 = v90;
        if ( v5 != v95 )
        {
          v116 = v131;
          _libc_free(v95, v90);
          v96 = v116;
          v97 = v90;
        }
        *(_DWORD *)(v4 + 68) = v96;
        v98 = *(unsigned int *)(v4 + 64);
        *(_QWORD *)(v4 + 56) = v97;
        v5 = &v97[v98];
      }
      while ( 1 )
      {
        v18 = *v16;
        if ( *v16 )
          break;
        if ( v5 )
        {
          ++v16;
          *v5++ = 0;
          if ( (__int64 *)v15 == v16 )
          {
LABEL_16:
            a4 = *(_QWORD *)(v4 + 16);
            v19 = *(_DWORD *)(v4 + 64) + v17;
            v10 = a4 - 16;
            goto LABEL_17;
          }
        }
        else
        {
LABEL_12:
          ++v16;
          ++v5;
          if ( (__int64 *)v15 == v16 )
            goto LABEL_16;
        }
      }
      if ( v5 )
      {
        *v5 = v18;
        sub_B96E90(v5, v18, 1);
      }
      goto LABEL_12;
    }
    v19 = 0;
LABEL_17:
    *(_DWORD *)(v4 + 64) = v19;
  }
  v20 = (*(_BYTE *)(a4 - 16) & 2) != 0;
  if ( (*(_BYTE *)(a4 - 16) & 2) != 0 )
    v21 = *(_QWORD *)(a4 - 32);
  else
    v21 = v10 - 8LL * ((*(_BYTE *)(a4 - 16) >> 2) & 0xF);
  v22 = *(_QWORD *)(v21 + 40);
  if ( !v22 )
    goto LABEL_39;
  v23 = *(_BYTE *)(v22 - 16);
  if ( (v23 & 2) != 0 )
  {
    v123 = *(_QWORD *)(v22 - 32);
    v24 = v123 + 8LL * *(unsigned int *)(v22 - 24);
  }
  else
  {
    v107 = (v23 >> 2) & 0xF;
    v123 = v22 - 16 - 8 * v107;
    v24 = v22 - 16 + 8 * (((*(_WORD *)(v22 - 16) >> 6) & 0xF) - v107);
  }
  v25 = *(_QWORD *)(v4 + 104);
  v26 = v25 + 8LL * *(unsigned int *)(v4 + 112);
  while ( v25 != v26 )
  {
    while ( 1 )
    {
      v27 = *(_QWORD *)(v26 - 8);
      v26 -= 8;
      if ( !v27 )
        break;
      sub_B91220(v26);
      if ( v25 == v26 )
        goto LABEL_27;
    }
  }
LABEL_27:
  *(_DWORD *)(v4 + 112) = 0;
  v28 = 0;
  v29 = 0;
  if ( v24 == v123 )
    goto LABEL_38;
  v30 = ((unsigned __int64)(v24 - 8 - v123) >> 3) + 1;
  v29 = ((unsigned __int64)(v24 - 8 - v123) >> 3) + 1;
  if ( v30 > *(unsigned int *)(v4 + 116) )
  {
    v81 = (__int64 *)sub_C8D7D0(v4 + 104, v117, v30, 8, &v131);
    sub_ADDB20(v4 + 104, v81, v82, v83, v84, v85);
    v86 = *(_QWORD *)(v4 + 104);
    v87 = v131;
    v88 = v81;
    if ( v117 != v86 )
    {
      v115 = v131;
      _libc_free(v86, v81);
      v87 = v115;
      v88 = v81;
    }
    *(_DWORD *)(v4 + 116) = v87;
    v89 = *(unsigned int *)(v4 + 112);
    *(_QWORD *)(v4 + 104) = v88;
    v31 = &v88[v89];
  }
  else
  {
    v31 = *(__int64 **)(v4 + 104);
  }
  v32 = (__int64 *)v123;
  do
  {
    while ( 1 )
    {
      v33 = *v32;
      if ( *v32 )
      {
        if ( v31 )
        {
          *v31 = v33;
          sub_B96E90(v31, v33, 1);
        }
        goto LABEL_33;
      }
      if ( v31 )
        break;
LABEL_33:
      ++v32;
      ++v31;
      if ( (__int64 *)v24 == v32 )
        goto LABEL_37;
    }
    ++v32;
    *v31++ = 0;
  }
  while ( (__int64 *)v24 != v32 );
LABEL_37:
  v28 = *(_DWORD *)(v4 + 112);
LABEL_38:
  a4 = *(_QWORD *)(v4 + 16);
  *(_DWORD *)(v4 + 112) = v29 + v28;
  v10 = a4 - 16;
  v20 = (*(_BYTE *)(a4 - 16) & 2) != 0;
LABEL_39:
  if ( v20 )
    v34 = *(_QWORD *)(a4 - 32);
  else
    v34 = v10 - 8LL * ((*(_BYTE *)(a4 - 16) >> 2) & 0xF);
  v35 = *(_QWORD *)(v34 + 48);
  if ( v35 )
  {
    v36 = *(_BYTE *)(v35 - 16);
    if ( (v36 & 2) != 0 )
    {
      v37 = *(_QWORD *)(v35 - 32);
      v38 = v37 + 8LL * *(unsigned int *)(v35 - 24);
    }
    else
    {
      v109 = (v36 >> 2) & 0xF;
      v37 = v35 - 16 - 8 * v109;
      v38 = v35 - 16 + 8 * (((*(_WORD *)(v35 - 16) >> 6) & 0xF) - v109);
    }
    *(_DWORD *)(v4 + 208) = 0;
    v39 = 0;
    LODWORD(v40) = 0;
    if ( v38 != v37 )
    {
      v40 = ((unsigned __int64)(v38 - 8 - v37) >> 3) + 1;
      if ( *(unsigned int *)(v4 + 212) < v40 )
      {
        sub_C8D5F0(v4 + 200, v4 + 216, ((unsigned __int64)(v38 - 8 - v37) >> 3) + 1, 8);
        v41 = *(_QWORD *)(v4 + 200) + 8LL * *(unsigned int *)(v4 + 208);
      }
      else
      {
        v41 = *(_QWORD *)(v4 + 200);
      }
      v42 = v38 - v37;
      v43 = 0;
      do
      {
        *(_QWORD *)(v41 + v43) = *(_QWORD *)(v37 + v43);
        v43 += 8;
      }
      while ( v43 != v42 );
      v39 = *(_DWORD *)(v4 + 208);
    }
    a4 = *(_QWORD *)(v4 + 16);
    *(_DWORD *)(v4 + 208) = v39 + v40;
    v10 = a4 - 16;
    v20 = (*(_BYTE *)(a4 - 16) & 2) != 0;
  }
  if ( v20 )
    v44 = *(_QWORD *)(a4 - 32);
  else
    v44 = v10 - 8LL * ((*(_BYTE *)(a4 - 16) >> 2) & 0xF);
  v45 = *(_QWORD *)(v44 + 56);
  if ( !v45 )
    goto LABEL_72;
  v46 = *(_BYTE *)(v45 - 16);
  if ( (v46 & 2) != 0 )
  {
    v124 = *(_QWORD *)(v45 - 32);
    v47 = v124 + 8LL * *(unsigned int *)(v45 - 24);
  }
  else
  {
    v110 = (v46 >> 2) & 0xF;
    v124 = v45 - 16 - 8 * v110;
    v47 = v45 - 16 + 8 * (((*(_WORD *)(v45 - 16) >> 6) & 0xF) - v110);
  }
  v48 = *(_QWORD *)(v4 + 248);
  v49 = v48 + 8LL * *(unsigned int *)(v4 + 256);
  while ( v48 != v49 )
  {
    while ( 1 )
    {
      v50 = *(_QWORD *)(v49 - 8);
      v49 -= 8;
      if ( !v50 )
        break;
      sub_B91220(v49);
      if ( v48 == v49 )
        goto LABEL_60;
    }
  }
LABEL_60:
  *(_DWORD *)(v4 + 256) = 0;
  v51 = 0;
  LODWORD(v52) = 0;
  if ( v47 == v124 )
    goto LABEL_71;
  v52 = ((unsigned __int64)(v47 - 8 - v124) >> 3) + 1;
  if ( v52 > *(unsigned int *)(v4 + 260) )
  {
    v99 = (__int64 *)sub_C8D7D0(v4 + 248, v118, ((unsigned __int64)(v47 - 8 - v124) >> 3) + 1, 8, &v131);
    sub_ADDB20(v4 + 248, v99, v100, v101, v102, v103);
    v104 = *(_QWORD *)(v4 + 248);
    v105 = v131;
    if ( v118 != v104 )
    {
      v121 = v131;
      _libc_free(v104, v99);
      v105 = v121;
    }
    *(_DWORD *)(v4 + 260) = v105;
    v106 = *(unsigned int *)(v4 + 256);
    *(_QWORD *)(v4 + 248) = v99;
    v53 = &v99[v106];
  }
  else
  {
    v53 = *(__int64 **)(v4 + 248);
  }
  v54 = (__int64 *)v124;
  while ( 2 )
  {
    while ( 2 )
    {
      v55 = *v54;
      if ( *v54 )
      {
        if ( v53 )
        {
          *v53 = v55;
          sub_B96E90(v53, v55, 1);
        }
LABEL_66:
        ++v54;
        ++v53;
        if ( (__int64 *)v47 == v54 )
          goto LABEL_70;
        continue;
      }
      break;
    }
    if ( !v53 )
      goto LABEL_66;
    ++v54;
    *v53++ = 0;
    if ( (__int64 *)v47 != v54 )
      continue;
    break;
  }
LABEL_70:
  v51 = *(_DWORD *)(v4 + 256);
LABEL_71:
  a4 = *(_QWORD *)(v4 + 16);
  *(_DWORD *)(v4 + 256) = v51 + v52;
  v10 = a4 - 16;
  v20 = (*(_BYTE *)(a4 - 16) & 2) != 0;
LABEL_72:
  if ( v20 )
    v56 = *(_QWORD *)(a4 - 32);
  else
    v56 = v10 - 8LL * ((*(_BYTE *)(a4 - 16) >> 2) & 0xF);
  result = *(_QWORD *)(v56 + 64);
  if ( result )
  {
    v57 = *(_BYTE *)(result - 16);
    if ( (v57 & 2) != 0 )
    {
      v58 = *(__int64 **)(result - 32);
      v59 = *(unsigned int *)(result - 24);
    }
    else
    {
      v58 = (__int64 *)(result - 16 - 8LL * ((v57 >> 2) & 0xF));
      v59 = (*(_WORD *)(result - 16) >> 6) & 0xF;
    }
    v60 = &v58[v59];
    v125 = 0;
    src = 0;
    v127 = 0;
    v128 = 0;
    v129 = &v131;
    v130 = 0;
    if ( v60 != v58 )
    {
      v61 = v4;
      v62 = 0;
      v63 = 0;
      v64 = v60;
      while ( 1 )
      {
        v70 = *v58;
        if ( !v62 )
          break;
        v65 = 1;
        v66 = 0;
        v67 = (v62 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
        v68 = (char *)&v63[v67];
        v69 = *(_QWORD *)v68;
        if ( *(_QWORD *)v68 == v70 )
        {
LABEL_80:
          if ( v64 == ++v58 )
            goto LABEL_91;
          goto LABEL_81;
        }
        while ( v69 != -4096 )
        {
          if ( v66 || v69 != -8192 )
            v68 = v66;
          v67 = (v62 - 1) & (v65 + v67);
          v69 = v63[v67];
          if ( v70 == v69 )
            goto LABEL_80;
          ++v65;
          v66 = v68;
          v68 = (char *)&v63[v67];
        }
        if ( !v66 )
          v66 = v68;
        ++v125;
        v73 = v127 + 1;
        if ( 4 * ((int)v127 + 1) >= 3 * v62 )
          goto LABEL_84;
        if ( v62 - (v73 + HIDWORD(v127)) <= v62 >> 3 )
        {
          v120 = v61;
          sub_9C0C30((__int64)&v125, v62);
          if ( !(_DWORD)v128 )
          {
LABEL_155:
            LODWORD(v127) = v127 + 1;
            BUG();
          }
          v77 = 0;
          v61 = v120;
          v78 = (v128 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
          v79 = 1;
          v66 = (char *)src + 8 * v78;
          v80 = *(_QWORD *)v66;
          v73 = v127 + 1;
          if ( v70 != *(_QWORD *)v66 )
          {
            while ( v80 != -4096 )
            {
              if ( v80 == -8192 && !v77 )
                v77 = v66;
              v113 = v79 + 1;
              v114 = (v128 - 1) & (v78 + v79);
              v66 = (char *)src + 8 * v114;
              v78 = v114;
              v80 = *(_QWORD *)v66;
              if ( v70 == *(_QWORD *)v66 )
                goto LABEL_86;
              v79 = v113;
            }
            if ( v77 )
              v66 = v77;
          }
        }
LABEL_86:
        LODWORD(v127) = v73;
        if ( *(_QWORD *)v66 != -4096 )
          --HIDWORD(v127);
        *(_QWORD *)v66 = v70;
        v74 = (unsigned int)v130;
        v75 = (unsigned int)v130 + 1LL;
        if ( v75 > HIDWORD(v130) )
        {
          v122 = v61;
          sub_C8D5F0(&v129, &v131, v75, 8);
          v74 = (unsigned int)v130;
          v61 = v122;
        }
        ++v58;
        v129[v74] = v70;
        LODWORD(v130) = v130 + 1;
        if ( v64 == v58 )
        {
LABEL_91:
          v4 = v61;
          goto LABEL_92;
        }
LABEL_81:
        v63 = src;
        v62 = v128;
      }
      ++v125;
LABEL_84:
      v119 = v61;
      sub_9C0C30((__int64)&v125, 2 * v62);
      if ( !(_DWORD)v128 )
        goto LABEL_155;
      v61 = v119;
      LODWORD(v71) = (v128 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
      v66 = (char *)src + 8 * (unsigned int)v71;
      v72 = *(_QWORD *)v66;
      v73 = v127 + 1;
      if ( *(_QWORD *)v66 != v70 )
      {
        v111 = 1;
        v112 = 0;
        while ( v72 != -4096 )
        {
          if ( v72 == -8192 && !v112 )
            v112 = v66;
          v71 = ((_DWORD)v128 - 1) & (unsigned int)(v71 + v111);
          v66 = (char *)src + 8 * v71;
          v72 = *(_QWORD *)v66;
          if ( v70 == *(_QWORD *)v66 )
            goto LABEL_86;
          ++v111;
        }
        if ( v112 )
          v66 = v112;
      }
      goto LABEL_86;
    }
LABEL_92:
    v131 = 0;
    v132 = 0;
    v133 = 0;
    v134 = 0;
    v135 = 0;
    sub_C7D6A0(0, 0, 8);
    LODWORD(v135) = v128;
    if ( (_DWORD)v128 )
    {
      v133 = (void *)sub_C7D670(8LL * (unsigned int)v128, 8);
      v134 = v127;
      memcpy(v133, src, 8LL * (unsigned int)v135);
    }
    else
    {
      v133 = 0;
      v134 = 0;
    }
    v136[1] = 0;
    v136[0] = v137;
    if ( (_DWORD)v130 )
      sub_ADBF30((__int64)v136, (__int64)&v129);
    sub_AE0110(v4 + 296, &v131, (__int64)&v132);
    if ( (_BYTE *)v136[0] != v137 )
      _libc_free(v136[0], &v131);
    v76 = 8LL * (unsigned int)v135;
    sub_C7D6A0(v133, v76, 8);
    if ( v129 != &v131 )
      _libc_free(v129, v76);
    return sub_C7D6A0(src, 8LL * (unsigned int)v128, 8);
  }
  return result;
}
