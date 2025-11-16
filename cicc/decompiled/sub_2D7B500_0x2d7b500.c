// Function: sub_2D7B500
// Address: 0x2d7b500
//
__int64 __fastcall sub_2D7B500(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 *v6; // rsi
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // r12
  _BYTE *v11; // r13
  char v12; // r14
  unsigned __int64 v13; // r12
  __int64 v14; // rdx
  char v15; // bl
  _BYTE *v16; // r13
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rax
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rbx
  _BYTE *v26; // r12
  __int64 v27; // rdi
  unsigned __int16 v29; // bx
  __int64 v30; // rdi
  unsigned __int16 v31; // r14
  __int64 v32; // rax
  __int64 (*v33)(); // rax
  __int64 v34; // rbx
  _BYTE *v35; // r14
  __int64 v36; // rdi
  unsigned __int64 v37; // rbx
  _BYTE *v38; // r13
  __int64 *v39; // rax
  __int64 v40; // rdx
  unsigned int v41; // esi
  __int64 v42; // r12
  __int64 v43; // r9
  int v44; // r11d
  __int64 *v45; // rdi
  unsigned int v46; // ecx
  __int64 *v47; // rax
  __int64 v48; // r8
  __int64 v49; // rax
  unsigned int v50; // r10d
  __int64 v51; // rbx
  _BYTE *v52; // r12
  __int64 v53; // rdi
  unsigned __int64 v54; // rbx
  __int64 *v55; // rdx
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 v58; // r9
  int v59; // r11d
  __int64 *v60; // rdi
  unsigned int v61; // ecx
  __int64 *v62; // rax
  __int64 v63; // r8
  _QWORD *v64; // rax
  __int64 v65; // rax
  __int64 v66; // r9
  __int64 v67; // rdx
  __int64 v68; // r13
  __int64 v69; // rax
  unsigned __int64 v70; // rax
  unsigned __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 *v73; // rax
  __int64 v74; // r12
  int v75; // eax
  int v76; // ecx
  int v77; // eax
  int v78; // ecx
  unsigned __int8 v79; // al
  __int64 *v80; // rbx
  unsigned __int8 v81; // r12
  __int64 *v82; // r13
  __int64 v83; // rdi
  unsigned __int8 v84; // al
  __int64 v85; // rbx
  __int64 *v86; // rdx
  __int64 v87; // rdx
  unsigned int v88; // esi
  __int64 v89; // r9
  int v90; // r11d
  __int64 *v91; // rdi
  unsigned int v92; // ecx
  __int64 *v93; // rax
  __int64 v94; // r8
  _QWORD *v95; // rax
  __int64 v96; // rax
  __int64 v97; // r13
  int v98; // eax
  int v99; // ecx
  __int64 *v100; // rbx
  __int64 *v101; // r12
  __int64 v102; // rdi
  __int64 v103; // [rsp+10h] [rbp-2E0h]
  __int64 v104; // [rsp+20h] [rbp-2D0h]
  unsigned __int64 v106; // [rsp+58h] [rbp-298h]
  __int64 v107; // [rsp+58h] [rbp-298h]
  __int64 v108; // [rsp+58h] [rbp-298h]
  char v109; // [rsp+60h] [rbp-290h]
  char v110; // [rsp+60h] [rbp-290h]
  _BYTE *v111; // [rsp+60h] [rbp-290h]
  unsigned __int8 v112; // [rsp+68h] [rbp-288h]
  __int64 v113; // [rsp+68h] [rbp-288h]
  char v114; // [rsp+7Fh] [rbp-271h] BYREF
  __int64 v115; // [rsp+80h] [rbp-270h] BYREF
  __int64 *v116; // [rsp+88h] [rbp-268h] BYREF
  unsigned __int64 v117[2]; // [rsp+90h] [rbp-260h] BYREF
  _QWORD v118[2]; // [rsp+A0h] [rbp-250h] BYREF
  unsigned __int64 v119[2]; // [rsp+B0h] [rbp-240h] BYREF
  _BYTE v120[16]; // [rsp+C0h] [rbp-230h] BYREF
  _BYTE *v121; // [rsp+D0h] [rbp-220h] BYREF
  __int64 v122; // [rsp+D8h] [rbp-218h]
  _BYTE v123[16]; // [rsp+E0h] [rbp-210h] BYREF
  unsigned __int64 v124; // [rsp+F0h] [rbp-200h] BYREF
  __int64 *v125; // [rsp+F8h] [rbp-1F8h]
  __int64 *v126; // [rsp+100h] [rbp-1F0h]
  __int64 v127; // [rsp+108h] [rbp-1E8h]
  _QWORD v128[4]; // [rsp+110h] [rbp-1E0h] BYREF
  _BYTE *v129; // [rsp+130h] [rbp-1C0h] BYREF
  __int64 v130; // [rsp+138h] [rbp-1B8h]
  _BYTE v131[16]; // [rsp+140h] [rbp-1B0h] BYREF
  __int64 v132; // [rsp+150h] [rbp-1A0h] BYREF
  char *v133; // [rsp+158h] [rbp-198h]
  __int64 v134; // [rsp+160h] [rbp-190h]
  int v135; // [rsp+168h] [rbp-188h]
  char v136; // [rsp+16Ch] [rbp-184h]
  char v137; // [rsp+170h] [rbp-180h] BYREF
  _BYTE *v138; // [rsp+180h] [rbp-170h] BYREF
  __int64 v139; // [rsp+188h] [rbp-168h]
  _BYTE v140[128]; // [rsp+190h] [rbp-160h] BYREF
  __int64 v141; // [rsp+210h] [rbp-E0h]
  __int64 *v142; // [rsp+220h] [rbp-D0h] BYREF
  __int64 v143; // [rsp+228h] [rbp-C8h]
  _BYTE v144[128]; // [rsp+230h] [rbp-C0h] BYREF
  __int64 v145; // [rsp+2B0h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 32);
  v5 = *a2;
  v114 = 0;
  v109 = sub_DFB3A0(v4, v5, &v114);
  v6 = (__int64 *)&v138;
  v141 = a1 + 376;
  v117[0] = (unsigned __int64)v118;
  v121 = v123;
  v122 = 0x200000000LL;
  v7 = *a2;
  v138 = v140;
  v139 = 0x1000000000LL;
  v118[0] = v7;
  v117[1] = 0x100000001LL;
  v112 = sub_2D61E30(a1, (__int64)&v138, (__int64)v117, (__int64)&v121, 0);
  v8 = (unsigned __int64)v121;
  v9 = (unsigned __int64)&v121[8 * (unsigned int)v122];
  if ( v121 == (_BYTE *)v9 )
    goto LABEL_5;
  while ( 1 )
  {
    v10 = *(_QWORD *)v8;
    if ( (*(_BYTE *)(*(_QWORD *)v8 + 7LL) & 0x40) != 0 )
    {
      v6 = *(__int64 **)(v10 - 8);
      v11 = (_BYTE *)*v6;
      if ( *(_BYTE *)*v6 == 61 )
        break;
      goto LABEL_4;
    }
    v6 = (__int64 *)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF));
    v11 = (_BYTE *)*v6;
    if ( *(_BYTE *)*v6 == 61 )
      break;
LABEL_4:
    v8 += 8LL;
    if ( v9 == v8 )
      goto LABEL_5;
  }
  if ( !v112 && *((_QWORD *)v11 + 5) == *(_QWORD *)(v10 + 40) )
    goto LABEL_5;
  v107 = *(_QWORD *)(a1 + 16);
  v6 = *(__int64 **)(a1 + 816);
  v29 = sub_2D5BAE0(v107, (__int64)v6, *(__int64 **)(v10 + 8), 0);
  v30 = v107;
  v31 = sub_2D5BAE0(v107, (__int64)v6, *((__int64 **)v11 + 1), 0);
  v32 = *((_QWORD *)v11 + 2);
  if ( !v32 )
  {
    if ( v31 )
      goto LABEL_40;
    goto LABEL_51;
  }
  if ( !*(_QWORD *)(v32 + 8) )
    goto LABEL_43;
  if ( !v31 )
    goto LABEL_51;
LABEL_40:
  if ( !*(_QWORD *)(v107 + 8LL * v31 + 112) )
  {
LABEL_51:
    if ( !v29 || !*(_QWORD *)(v107 + 8LL * v29 + 112) )
      goto LABEL_41;
LABEL_43:
    v9 = (unsigned int)(*(_BYTE *)v10 == 68) + 2;
    if ( !v31 )
      goto LABEL_5;
    if ( !v29 )
      goto LABEL_5;
    v9 = (unsigned int)(4 * v9);
    if ( (((int)*(unsigned __int16 *)(v30 + 2 * (v31 + 274LL * v29 + 71704) + 6) >> v9) & 0xF) != 0 )
      goto LABEL_5;
    v34 = (__int64)v138;
    v35 = &v138[8 * (unsigned int)v139];
    while ( (_BYTE *)v34 != v35 )
    {
      while ( 1 )
      {
        v36 = *((_QWORD *)v35 - 1);
        v35 -= 8;
        if ( !v36 )
          break;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v36 + 8LL))(v36);
        if ( (_BYTE *)v34 == v35 )
          goto LABEL_50;
      }
    }
LABEL_50:
    LODWORD(v139) = 0;
    sub_B44530((_QWORD *)v10, (__int64)v11);
    v112 = 1;
    *a2 = v10;
    goto LABEL_23;
  }
LABEL_41:
  v33 = *(__int64 (**)())(*(_QWORD *)v107 + 1376LL);
  if ( v33 != sub_2D56660 )
  {
    v6 = *(__int64 **)(v10 + 8);
    v30 = v107;
    if ( ((unsigned __int8 (__fastcall *)(__int64, __int64 *, _QWORD))v33)(v107, v6, *((_QWORD *)v11 + 1)) )
      goto LABEL_43;
  }
LABEL_5:
  v12 = v109;
  if ( !v109 )
    goto LABEL_22;
  v13 = (unsigned __int64)v121;
  v14 = (unsigned int)v122;
  v132 = 0;
  v133 = &v137;
  v15 = v114;
  v16 = &v121[8 * (unsigned int)v122];
  v136 = 1;
  v134 = 1;
  v135 = 0;
  if ( v121 != v16 )
  {
    do
    {
      v21 = *(_QWORD *)v13;
      if ( (*(_BYTE *)(*(_QWORD *)v13 + 7LL) & 0x40) != 0 )
      {
        v17 = *(__int64 **)(v21 - 8);
      }
      else
      {
        v14 = 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF);
        v17 = (__int64 *)(v21 - v14);
      }
      v18 = *v17;
      v19 = *(unsigned int *)(a1 + 560);
      v6 = *(__int64 **)(a1 + 544);
      if ( (_DWORD)v19 )
      {
        v9 = ((_DWORD)v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v14 = (__int64)&v6[2 * v9];
        v20 = *(_QWORD *)v14;
        if ( v18 == *(_QWORD *)v14 )
        {
LABEL_11:
          if ( (__int64 *)v14 != &v6[2 * v19] )
          {
            v14 = *(_QWORD *)(v14 + 8);
            v12 = 0;
            if ( v14 )
            {
              v6 = &v132;
              sub_BED950((__int64)&v142, (__int64)&v132, v14);
            }
          }
        }
        else
        {
          v14 = 1;
          while ( v20 != -4096 )
          {
            v50 = v14 + 1;
            v9 = ((_DWORD)v19 - 1) & (unsigned int)(v14 + v9);
            v14 = (__int64)&v6[2 * (unsigned int)v9];
            v20 = *(_QWORD *)v14;
            if ( v18 == *(_QWORD *)v14 )
              goto LABEL_11;
            v14 = v50;
          }
        }
      }
      v13 += 8LL;
    }
    while ( v16 != (_BYTE *)v13 );
    v110 = v12;
    if ( !v12 )
      goto LABEL_73;
    if ( v15 && (_DWORD)v122 == 1 )
    {
      v110 = v15;
      goto LABEL_73;
    }
    v37 = (unsigned __int64)v121;
    v38 = &v121[8 * (unsigned int)v122];
    if ( v121 != v38 )
    {
      v113 = a1 + 536;
      while ( 1 )
      {
        v49 = *(_QWORD *)v37;
        v39 = (*(_BYTE *)(*(_QWORD *)v37 + 7LL) & 0x40) != 0
            ? *(__int64 **)(v49 - 8)
            : (__int64 *)(v49 - 32LL * (*(_DWORD *)(v49 + 4) & 0x7FFFFFF));
        v40 = *v39;
        v41 = *(_DWORD *)(a1 + 560);
        v42 = *a2;
        v129 = (_BYTE *)*v39;
        if ( !v41 )
          break;
        v43 = *(_QWORD *)(a1 + 544);
        v44 = 1;
        v45 = 0;
        v46 = (v41 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v47 = (__int64 *)(v43 + 16LL * v46);
        v48 = *v47;
        if ( v40 == *v47 )
        {
LABEL_65:
          v37 += 8LL;
          v47[1] = v42;
          if ( v38 == (_BYTE *)v37 )
            goto LABEL_126;
        }
        else
        {
          while ( v48 != -4096 )
          {
            if ( v48 == -8192 && !v45 )
              v45 = v47;
            v46 = (v41 - 1) & (v44 + v46);
            v47 = (__int64 *)(v43 + 16LL * v46);
            v48 = *v47;
            if ( v40 == *v47 )
              goto LABEL_65;
            ++v44;
          }
          if ( !v45 )
            v45 = v47;
          v77 = *(_DWORD *)(a1 + 552);
          ++*(_QWORD *)(a1 + 536);
          v78 = v77 + 1;
          v142 = v45;
          if ( 4 * (v77 + 1) >= 3 * v41 )
            goto LABEL_132;
          if ( v41 - *(_DWORD *)(a1 + 556) - v78 <= v41 >> 3 )
            goto LABEL_133;
LABEL_123:
          *(_DWORD *)(a1 + 552) = v78;
          if ( *v45 != -4096 )
            --*(_DWORD *)(a1 + 556);
          v37 += 8LL;
          *v45 = v40;
          v45[1] = 0;
          v45[1] = v42;
          if ( v38 == (_BYTE *)v37 )
            goto LABEL_126;
        }
      }
      ++*(_QWORD *)(a1 + 536);
      v142 = 0;
LABEL_132:
      v41 *= 2;
LABEL_133:
      sub_CE25F0(v113, v41);
      sub_2D67C70(v113, (__int64 *)&v129, &v142);
      v40 = (__int64)v129;
      v45 = v142;
      v78 = *(_DWORD *)(a1 + 552) + 1;
      goto LABEL_123;
    }
LABEL_126:
    if ( !v136 )
      _libc_free((unsigned __int64)v133);
LABEL_22:
    sub_2D57BD0((__int64 *)&v138, 0);
    v112 = 0;
    goto LABEL_23;
  }
  if ( !v114 )
    goto LABEL_22;
  v110 = v114;
  if ( (_DWORD)v122 != 1 )
    goto LABEL_126;
LABEL_73:
  v51 = (__int64)v138;
  v52 = &v138[8 * (unsigned int)v139];
  while ( (_BYTE *)v51 != v52 )
  {
    while ( 1 )
    {
      v53 = *((_QWORD *)v52 - 1);
      v52 -= 8;
      if ( !v53 )
        break;
      (*(void (__fastcall **)(__int64, __int64 *, __int64, unsigned __int64))(*(_QWORD *)v53 + 8LL))(v53, v6, v14, v9);
      if ( (_BYTE *)v51 == v52 )
        goto LABEL_77;
    }
  }
LABEL_77:
  v54 = (unsigned __int64)v121;
  LODWORD(v139) = 0;
  v22 = v122;
  v23 = 8LL * (unsigned int)v122;
  v106 = (unsigned __int64)&v121[v23];
  if ( v121 != &v121[v23] )
  {
    v104 = a1 + 536;
    while ( 1 )
    {
      v68 = *(_QWORD *)v54;
      v55 = (*(_BYTE *)(*(_QWORD *)v54 + 7LL) & 0x40) != 0
          ? *(__int64 **)(v68 - 8)
          : (__int64 *)(v68 - 32LL * (*(_DWORD *)(v68 + 4) & 0x7FFFFFF));
      v56 = *v55;
      v57 = *(_DWORD *)(a1 + 560);
      v129 = (_BYTE *)v56;
      if ( !v57 )
        break;
      v58 = *(_QWORD *)(a1 + 544);
      v59 = 1;
      v60 = 0;
      v61 = (v57 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
      v62 = (__int64 *)(v58 + 16LL * v61);
      v63 = *v62;
      if ( v56 != *v62 )
      {
        while ( v63 != -4096 )
        {
          if ( !v60 && v63 == -8192 )
            v60 = v62;
          v61 = (v57 - 1) & (v59 + v61);
          v62 = (__int64 *)(v58 + 16LL * v61);
          v63 = *v62;
          if ( v56 == *v62 )
            goto LABEL_82;
          ++v59;
        }
        if ( !v60 )
          v60 = v62;
        v75 = *(_DWORD *)(a1 + 552);
        ++*(_QWORD *)(a1 + 536);
        v76 = v75 + 1;
        v142 = v60;
        if ( 4 * (v75 + 1) < 3 * v57 )
        {
          if ( v57 - *(_DWORD *)(a1 + 556) - v76 > v57 >> 3 )
          {
LABEL_110:
            *(_DWORD *)(a1 + 552) = v76;
            if ( *v60 != -4096 )
              --*(_DWORD *)(a1 + 556);
            *v60 = v56;
            v64 = v60 + 1;
            v60[1] = 0;
            goto LABEL_83;
          }
LABEL_130:
          sub_CE25F0(v104, v57);
          sub_2D67C70(v104, (__int64 *)&v129, &v142);
          v56 = (__int64)v129;
          v60 = v142;
          v76 = *(_DWORD *)(a1 + 552) + 1;
          goto LABEL_110;
        }
LABEL_129:
        v57 *= 2;
        goto LABEL_130;
      }
LABEL_82:
      v64 = v62 + 1;
LABEL_83:
      *v64 = 0;
      v65 = sub_2D7B240(a1 + 760, (__int64 *)&v129);
      v67 = *(unsigned int *)(v65 + 8);
      if ( v67 + 1 > (unsigned __int64)*(unsigned int *)(v65 + 12) )
      {
        v103 = v65;
        sub_C8D5F0(v65, (const void *)(v65 + 16), v67 + 1, 8u, v67 + 1, v66);
        v65 = v103;
        v67 = *(unsigned int *)(v103 + 8);
      }
      v54 += 8LL;
      *(_QWORD *)(*(_QWORD *)v65 + 8 * v67) = v68;
      ++*(_DWORD *)(v65 + 8);
      if ( v106 == v54 )
      {
        v106 = (unsigned __int64)v121;
        v22 = v122;
        v23 = 8LL * (unsigned int)v122;
        goto LABEL_18;
      }
    }
    ++*(_QWORD *)(a1 + 536);
    v142 = 0;
    goto LABEL_129;
  }
LABEL_18:
  v24 = *(_QWORD *)(v106 + v23 - 8);
  LODWORD(v122) = v22 - 1;
  *a2 = v24;
  if ( !v110 )
  {
    v69 = HIDWORD(v134);
    if ( HIDWORD(v134) != v135 )
    {
      if ( !v136 )
        v69 = (unsigned int)v134;
      v124 = (unsigned __int64)v133;
      v125 = (__int64 *)&v133[8 * v69];
      sub_254BBF0((__int64)&v124);
      v126 = &v132;
      v127 = v132;
      v70 = (unsigned __int64)(v136 ? &v133[8 * HIDWORD(v134)] : &v133[8 * (unsigned int)v134]);
      v128[0] = v70;
      v128[1] = v70;
      sub_254BBF0((__int64)v128);
      v128[2] = &v132;
      v128[3] = v132;
      v73 = (__int64 *)v124;
      if ( v124 != v128[0] )
      {
        while ( 1 )
        {
          v74 = *v73;
          if ( !(unsigned __int8)sub_B19060(a1 + 376, *v73, v71, v72) )
            break;
LABEL_95:
          v72 = (__int64)v125;
          v73 = (__int64 *)(v124 + 8);
          v124 = (unsigned __int64)v73;
          if ( v73 == v125 )
          {
LABEL_98:
            if ( (__int64 *)v128[0] == v73 )
              goto LABEL_19;
          }
          else
          {
            while ( 1 )
            {
              v71 = *v73 + 2;
              if ( v71 > 1 )
                break;
              v124 = (unsigned __int64)++v73;
              if ( v125 == v73 )
                goto LABEL_98;
            }
            v73 = (__int64 *)v124;
            if ( v128[0] == v124 )
              goto LABEL_19;
          }
        }
        v142 = (__int64 *)v144;
        v143 = 0x1000000000LL;
        v145 = a1 + 376;
        v119[0] = (unsigned __int64)v120;
        v119[1] = 0x100000000LL;
        v129 = v131;
        v130 = 0x200000000LL;
        sub_9C95B0((__int64)v119, v74);
        v79 = sub_2D61E30(a1, (__int64)&v142, (__int64)v119, (__int64)&v129, 0);
        v80 = v142;
        v81 = v79;
        v82 = &v142[(unsigned int)v143];
        while ( v80 != v82 )
        {
          while ( 1 )
          {
            v83 = *--v82;
            if ( !v83 )
              break;
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v83 + 8LL))(v83);
            if ( v80 == v82 )
              goto LABEL_141;
          }
        }
LABEL_141:
        v84 = v112;
        v85 = (__int64)v129;
        LODWORD(v143) = 0;
        if ( v81 )
          v84 = v81;
        v112 = v84;
        v111 = &v129[8 * (unsigned int)v130];
        if ( v129 == v111 )
        {
LABEL_166:
          if ( v111 != v131 )
            _libc_free((unsigned __int64)v111);
          if ( (_BYTE *)v119[0] != v120 )
            _libc_free(v119[0]);
          v100 = v142;
          v101 = &v142[(unsigned int)v143];
          if ( v142 != v101 )
          {
            do
            {
              v102 = *--v101;
              if ( v102 )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v102 + 8LL))(v102);
            }
            while ( v100 != v101 );
            v101 = v142;
          }
          if ( v101 != (__int64 *)v144 )
            _libc_free((unsigned __int64)v101);
          goto LABEL_95;
        }
        v108 = a1 + 536;
        while ( 1 )
        {
          v97 = *(_QWORD *)v85;
          v86 = (*(_BYTE *)(*(_QWORD *)v85 + 7LL) & 0x40) != 0
              ? *(__int64 **)(v97 - 8)
              : (__int64 *)(v97 - 32LL * (*(_DWORD *)(v97 + 4) & 0x7FFFFFF));
          v87 = *v86;
          v88 = *(_DWORD *)(a1 + 560);
          v115 = v87;
          if ( !v88 )
            break;
          v89 = *(_QWORD *)(a1 + 544);
          v90 = 1;
          v91 = 0;
          v92 = (v88 - 1) & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
          v93 = (__int64 *)(v89 + 16LL * v92);
          v94 = *v93;
          if ( v87 != *v93 )
          {
            while ( v94 != -4096 )
            {
              if ( v94 == -8192 && !v91 )
                v91 = v93;
              v92 = (v88 - 1) & (v90 + v92);
              v93 = (__int64 *)(v89 + 16LL * v92);
              v94 = *v93;
              if ( v87 == *v93 )
                goto LABEL_148;
              ++v90;
            }
            if ( !v91 )
              v91 = v93;
            v98 = *(_DWORD *)(a1 + 552);
            ++*(_QWORD *)(a1 + 536);
            v99 = v98 + 1;
            v116 = v91;
            if ( 4 * (v98 + 1) < 3 * v88 )
            {
              if ( v88 - *(_DWORD *)(a1 + 556) - v99 > v88 >> 3 )
              {
LABEL_162:
                *(_DWORD *)(a1 + 552) = v99;
                if ( *v91 != -4096 )
                  --*(_DWORD *)(a1 + 556);
                *v91 = v87;
                v95 = v91 + 1;
                v91[1] = 0;
                goto LABEL_149;
              }
LABEL_179:
              sub_CE25F0(v108, v88);
              sub_2D67C70(v108, &v115, &v116);
              v87 = v115;
              v91 = v116;
              v99 = *(_DWORD *)(a1 + 552) + 1;
              goto LABEL_162;
            }
LABEL_178:
            v88 *= 2;
            goto LABEL_179;
          }
LABEL_148:
          v95 = v93 + 1;
LABEL_149:
          *v95 = 0;
          v85 += 8;
          v96 = sub_2D7B240(a1 + 760, &v115);
          sub_9C95B0(v96, v97);
          if ( v111 == (_BYTE *)v85 )
          {
            v111 = v129;
            goto LABEL_166;
          }
        }
        ++*(_QWORD *)(a1 + 536);
        v116 = 0;
        goto LABEL_178;
      }
    }
  }
LABEL_19:
  if ( !v136 )
    _libc_free((unsigned __int64)v133);
  if ( !v112 )
    goto LABEL_22;
LABEL_23:
  if ( v121 != v123 )
    _libc_free((unsigned __int64)v121);
  if ( (_QWORD *)v117[0] != v118 )
    _libc_free(v117[0]);
  v25 = (__int64)v138;
  v26 = &v138[8 * (unsigned int)v139];
  if ( v138 != v26 )
  {
    do
    {
      v27 = *((_QWORD *)v26 - 1);
      v26 -= 8;
      if ( v27 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
    }
    while ( (_BYTE *)v25 != v26 );
    v26 = v138;
  }
  if ( v26 != v140 )
    _libc_free((unsigned __int64)v26);
  return v112;
}
