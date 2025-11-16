// Function: sub_288C870
// Address: 0x288c870
//
__int64 __fastcall sub_288C870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 *v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 *v12; // r12
  __int64 *v13; // r15
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 *v16; // rax
  __int64 v17; // rdx
  char *v18; // rsi
  __int64 v19; // rdi
  __int64 *v20; // rcx
  __int64 v21; // rdx
  unsigned __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  __int64 v25; // r9
  _BYTE *v26; // r14
  _BYTE *v27; // r12
  size_t v28; // r8
  __int64 v29; // rbx
  unsigned __int64 *v30; // r15
  int v31; // eax
  char *v32; // rdi
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // r12
  unsigned __int64 *v36; // r12
  char v37; // bl
  __int64 *v38; // rax
  unsigned __int64 *i; // r12
  __int64 *v40; // rax
  _BYTE *v41; // r12
  char *v42; // r14
  __int64 v43; // r14
  bool v44; // zf
  __int64 v45; // r15
  __int64 v46; // r14
  __int64 v47; // r8
  __int64 v48; // rax
  __int64 v49; // r9
  unsigned __int64 v50; // rdx
  __int64 v51; // r14
  unsigned int v52; // eax
  char v53; // r13
  __int64 v54; // rbx
  __int64 v55; // r10
  unsigned int v56; // edx
  __int64 *v57; // rax
  __int64 v58; // r11
  __int64 v59; // r12
  __int64 v60; // rax
  __int64 *v61; // rax
  _QWORD *v62; // rax
  __int64 *v63; // rax
  __int64 *v64; // rax
  __int64 *v65; // r12
  unsigned int v67; // eax
  __int64 *v68; // rdi
  int v69; // edx
  char v70; // bl
  __int64 v71; // rax
  __int64 v72; // r9
  __int64 *v73; // r14
  __int64 *v74; // rcx
  __int64 *v75; // r15
  __int64 v76; // rsi
  __int64 *v77; // rax
  __int64 v78; // rcx
  __int64 *v79; // rdx
  _BYTE *v80; // r15
  _BYTE *v81; // r13
  __int64 v82; // r12
  _BYTE *v83; // rdi
  int v84; // eax
  _BYTE *v85; // rdx
  __int64 v86; // r12
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rdx
  unsigned __int64 v90; // rcx
  __int64 v91; // r13
  __int64 v92; // rax
  _BYTE *v93; // rdi
  __int64 *v94; // rax
  int v95; // edx
  int v96; // edx
  unsigned int v97; // eax
  __int64 v98; // r10
  int v99; // edx
  int v100; // edx
  unsigned int v101; // eax
  __int64 v102; // r10
  __int64 *v103; // rax
  __int64 *v104; // rax
  __int64 v105; // [rsp+8h] [rbp-198h]
  int v106; // [rsp+8h] [rbp-198h]
  unsigned __int64 *src; // [rsp+20h] [rbp-180h]
  int srca; // [rsp+20h] [rbp-180h]
  __int64 srcb; // [rsp+20h] [rbp-180h]
  __int64 v112; // [rsp+30h] [rbp-170h]
  __int64 v113; // [rsp+30h] [rbp-170h]
  __int64 v114; // [rsp+30h] [rbp-170h]
  _QWORD v116[2]; // [rsp+40h] [rbp-160h] BYREF
  __int64 v117; // [rsp+50h] [rbp-150h] BYREF
  __int64 v118; // [rsp+58h] [rbp-148h] BYREF
  __int64 v119[2]; // [rsp+60h] [rbp-140h] BYREF
  __int64 *v120; // [rsp+70h] [rbp-130h]
  __int64 v121[2]; // [rsp+80h] [rbp-120h] BYREF
  _QWORD v122[2]; // [rsp+90h] [rbp-110h] BYREF
  __int64 v123; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v124; // [rsp+B0h] [rbp-F0h]
  char v125; // [rsp+C0h] [rbp-E0h]
  void *v126; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v127; // [rsp+D8h] [rbp-C8h]
  _BYTE v128[32]; // [rsp+E0h] [rbp-C0h] BYREF
  _BYTE *v129; // [rsp+100h] [rbp-A0h] BYREF
  __int64 v130; // [rsp+108h] [rbp-98h]
  _BYTE v131[32]; // [rsp+110h] [rbp-90h] BYREF
  __int64 v132; // [rsp+130h] [rbp-70h] BYREF
  __int64 *v133; // [rsp+138h] [rbp-68h]
  __int64 v134; // [rsp+140h] [rbp-60h]
  int v135; // [rsp+148h] [rbp-58h]
  unsigned __int8 v136; // [rsp+14Ch] [rbp-54h]
  char v137; // [rsp+150h] [rbp-50h] BYREF

  v6 = a3;
  sub_1049690(v119, *(_QWORD *)(**(_QWORD **)(a3 + 32) + 72LL));
  v11 = *(_QWORD *)v6;
  v136 = 1;
  v132 = 0;
  v105 = v11;
  v133 = (__int64 *)&v137;
  v134 = 4;
  v135 = 0;
  if ( v11 )
  {
    v12 = *(__int64 **)(v11 + 16);
    v13 = *(__int64 **)(v11 + 8);
    v14 = 1;
    if ( v13 != v12 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v15 = *v13;
          if ( (_BYTE)v14 )
            break;
LABEL_53:
          ++v13;
          sub_C8CC70((__int64)&v132, v15, (__int64)v8, v9, v10, v14);
          v14 = v136;
          if ( v12 == v13 )
            goto LABEL_9;
        }
        v16 = v133;
        v9 = HIDWORD(v134);
        v8 = &v133[HIDWORD(v134)];
        if ( v133 == v8 )
        {
LABEL_55:
          if ( HIDWORD(v134) >= (unsigned int)v134 )
            goto LABEL_53;
          v9 = (unsigned int)(HIDWORD(v134) + 1);
          ++v13;
          ++HIDWORD(v134);
          *v8 = v15;
          v14 = v136;
          ++v132;
          if ( v12 == v13 )
            break;
        }
        else
        {
          while ( v15 != *v16 )
          {
            if ( v8 == ++v16 )
              goto LABEL_55;
          }
          if ( v12 == ++v13 )
            break;
        }
      }
    }
  }
  else
  {
    v71 = *(_QWORD *)(a5 + 24);
    v72 = 1;
    v73 = *(__int64 **)(v71 + 40);
    v74 = *(__int64 **)(v71 + 32);
    if ( v74 != v73 )
    {
      v75 = *(__int64 **)(v71 + 32);
      v76 = *v74;
LABEL_133:
      v77 = v133;
      v78 = HIDWORD(v134);
      v79 = &v133[HIDWORD(v134)];
      if ( v133 == v79 )
      {
LABEL_141:
        if ( HIDWORD(v134) >= (unsigned int)v134 )
        {
          while ( 1 )
          {
            ++v75;
            sub_C8CC70((__int64)&v132, v76, (__int64)v79, v78, v10, v72);
            v72 = v136;
            if ( v73 == v75 )
              break;
LABEL_138:
            v76 = *v75;
            if ( (_BYTE)v72 )
              goto LABEL_133;
          }
        }
        else
        {
          v78 = (unsigned int)(HIDWORD(v134) + 1);
          ++v75;
          ++HIDWORD(v134);
          *v79 = v76;
          v72 = v136;
          ++v132;
          if ( v73 != v75 )
            goto LABEL_138;
        }
      }
      else
      {
        while ( v76 != *v77 )
        {
          if ( v79 == ++v77 )
            goto LABEL_141;
        }
        if ( v73 != ++v75 )
          goto LABEL_138;
      }
    }
  }
LABEL_9:
  v17 = 14;
  v18 = "<unnamed loop>";
  v19 = **(_QWORD **)(v6 + 32);
  if ( v19 && (*(_BYTE *)(v19 + 7) & 0x10) != 0 )
    v18 = (char *)sub_BD5D20(v19);
  v121[0] = (__int64)v122;
  sub_287ECD0(v121, v18, (__int64)&v18[v17]);
  v20 = *(__int64 **)(a5 + 32);
  BYTE4(v129) = 0;
  v21 = *(_QWORD *)(a5 + 24);
  v22 = *(_QWORD *)(a5 + 16);
  BYTE4(v126) = 0;
  BYTE4(v123) = 0;
  if ( !(unsigned int)sub_288A700(
                        v6,
                        v22,
                        v21,
                        v20,
                        *(__int64 ***)(a5 + 48),
                        *(_QWORD *)(a5 + 8),
                        v119,
                        0,
                        0,
                        0,
                        1u,
                        *(_DWORD *)a2,
                        1,
                        *(_BYTE *)(a2 + 4),
                        *(_BYTE *)(a2 + 5),
                        v123,
                        (__int64)v126,
                        0x100u,
                        0x100u,
                        0x100u,
                        0x101u,
                        0x100u,
                        (__int64)v129,
                        0) )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_112;
  }
  v126 = v128;
  v127 = 0x400000000LL;
  if ( v105 )
  {
    v26 = *(_BYTE **)(v105 + 16);
    v27 = *(_BYTE **)(v105 + 8);
  }
  else
  {
    v60 = *(_QWORD *)(a5 + 24);
    v26 = *(_BYTE **)(v60 + 40);
    v27 = *(_BYTE **)(v60 + 32);
  }
  v28 = v26 - v27;
  v29 = (v26 - v27) >> 3;
  if ( (unsigned __int64)(v26 - v27) > 0x20 )
  {
    v22 = (unsigned __int64)v128;
    sub_C8D5F0((__int64)&v126, v128, (v26 - v27) >> 3, 8u, v28, v25);
    v30 = (unsigned __int64 *)v126;
    v31 = v127;
    v28 = v26 - v27;
    v32 = (char *)v126 + 8 * (unsigned int)v127;
  }
  else
  {
    v30 = (unsigned __int64 *)v128;
    v31 = 0;
    v32 = v128;
  }
  if ( v27 != v26 )
  {
    v22 = (unsigned __int64)v27;
    memmove(v32, v27, v28);
    v31 = v127;
    v30 = (unsigned __int64 *)v126;
  }
  LODWORD(v127) = v29 + v31;
  v33 = 8LL * (unsigned int)(v29 + v31);
  src = &v30[(unsigned __int64)v33 / 8];
  v34 = v33 >> 3;
  v35 = v33 >> 5;
  if ( !v35 )
  {
LABEL_99:
    if ( v34 != 2 )
    {
      if ( v34 != 3 )
      {
        if ( v34 != 1 )
          goto LABEL_102;
        goto LABEL_170;
      }
      v22 = *v30;
      if ( v6 == *v30 )
        goto LABEL_66;
      v37 = v136;
      if ( v136 )
      {
        v104 = v133;
        v23 = (__int64)&v133[HIDWORD(v134)];
        if ( v133 != (__int64 *)v23 )
        {
          while ( v22 != *v104 )
          {
            if ( (__int64 *)v23 == ++v104 )
              goto LABEL_64;
          }
          goto LABEL_27;
        }
      }
      else if ( sub_C8CA60((__int64)&v132, v22) )
      {
        goto LABEL_28;
      }
LABEL_64:
      ++v30;
    }
    v22 = *v30;
    if ( v6 == *v30 )
      goto LABEL_66;
    v37 = v136;
    if ( v136 )
    {
      v103 = v133;
      v23 = (__int64)&v133[HIDWORD(v134)];
      if ( v133 != (__int64 *)v23 )
      {
        while ( v22 != *v103 )
        {
          if ( (__int64 *)v23 == ++v103 )
            goto LABEL_169;
        }
        goto LABEL_27;
      }
    }
    else if ( sub_C8CA60((__int64)&v132, v22) )
    {
      goto LABEL_28;
    }
LABEL_169:
    ++v30;
LABEL_170:
    v22 = *v30;
    if ( v6 != *v30 )
    {
      v37 = v136;
      if ( !v136 )
      {
        if ( sub_C8CA60((__int64)&v132, v22) )
          goto LABEL_28;
        goto LABEL_102;
      }
      v94 = v133;
      v23 = (__int64)&v133[HIDWORD(v134)];
      if ( v133 == (__int64 *)v23 )
      {
LABEL_102:
        v30 = src;
        v37 = 0;
        goto LABEL_37;
      }
      while ( v22 != *v94 )
      {
        if ( (__int64 *)v23 == ++v94 )
          goto LABEL_102;
      }
LABEL_27:
      v37 = 0;
      goto LABEL_28;
    }
LABEL_66:
    v37 = 1;
    goto LABEL_28;
  }
  v36 = &v30[4 * v35];
  while ( 1 )
  {
    v22 = *v30;
    if ( v6 == *v30 )
      goto LABEL_66;
    v37 = v136;
    if ( v136 )
      break;
    if ( sub_C8CA60((__int64)&v132, v22) )
      goto LABEL_28;
    v28 = v30[1];
    v24 = (unsigned __int64)(v30 + 1);
    if ( v6 == v28 )
      goto LABEL_76;
    v37 = v136;
    if ( v136 )
    {
      v38 = v133;
      v23 = (__int64)&v133[HIDWORD(v134)];
      if ( (__int64 *)v23 != v133 )
        goto LABEL_71;
      goto LABEL_146;
    }
    v22 = v30[1];
    v61 = sub_C8CA60((__int64)&v132, v28);
    v24 = (unsigned __int64)(v30 + 1);
    if ( v61 )
      goto LABEL_220;
    v28 = v30[2];
    v24 = (unsigned __int64)(v30 + 2);
    if ( v6 == v28 )
      goto LABEL_76;
    v37 = v136;
    if ( v136 )
    {
      v22 = (unsigned __int64)v133;
      v23 = (__int64)&v133[HIDWORD(v134)];
      if ( (__int64 *)v23 != v133 )
        goto LABEL_81;
LABEL_147:
      v24 = (unsigned __int64)(v30 + 3);
      if ( v6 == v30[3] )
        goto LABEL_76;
      v30 += 4;
      if ( v36 == v30 )
      {
LABEL_98:
        v34 = src - v30;
        goto LABEL_99;
      }
    }
    else
    {
      v22 = v30[2];
      v63 = sub_C8CA60((__int64)&v132, v28);
      v24 = (unsigned __int64)(v30 + 2);
      if ( v63 )
        goto LABEL_220;
      v22 = v30[3];
      v24 = (unsigned __int64)(v30 + 3);
      if ( v6 == v22 )
      {
LABEL_76:
        v30 = (unsigned __int64 *)v24;
        v37 = 1;
        goto LABEL_28;
      }
      v37 = v136;
      if ( v136 )
      {
        v62 = v133;
        v23 = (__int64)&v133[HIDWORD(v134)];
        goto LABEL_91;
      }
      v64 = sub_C8CA60((__int64)&v132, v22);
      v24 = (unsigned __int64)(v30 + 3);
      if ( v64 )
      {
LABEL_220:
        v30 = (unsigned __int64 *)v24;
        goto LABEL_28;
      }
LABEL_97:
      v30 += 4;
      if ( v36 == v30 )
        goto LABEL_98;
    }
  }
  v38 = v133;
  v23 = (__int64)&v133[HIDWORD(v134)];
  if ( v133 == (__int64 *)v23 )
  {
    v24 = (unsigned __int64)(v30 + 1);
    if ( v6 == v30[1] )
      goto LABEL_76;
LABEL_146:
    v24 = (unsigned __int64)(v30 + 2);
    if ( v6 == v30[2] )
      goto LABEL_76;
    goto LABEL_147;
  }
  v24 = (unsigned __int64)v133;
  do
  {
    if ( v22 == *(_QWORD *)v24 )
      goto LABEL_27;
    v24 += 8LL;
  }
  while ( v23 != v24 );
  v28 = v30[1];
  v24 = (unsigned __int64)(v30 + 1);
  if ( v6 == v28 )
    goto LABEL_76;
LABEL_71:
  v22 = (unsigned __int64)v38;
  do
  {
    if ( v28 == *v38 )
      goto LABEL_74;
    ++v38;
  }
  while ( v38 != (__int64 *)v23 );
  v28 = v30[2];
  v24 = (unsigned __int64)(v30 + 2);
  if ( v6 == v28 )
  {
    v30 += 2;
    v37 = 1;
    goto LABEL_28;
  }
LABEL_81:
  v62 = (_QWORD *)v22;
  do
  {
    if ( *(_QWORD *)v22 == v28 )
      goto LABEL_74;
    v22 += 8LL;
  }
  while ( v23 != v22 );
  v22 = v30[3];
  v24 = (unsigned __int64)(v30 + 3);
  if ( v6 == v22 )
  {
    v30 += 3;
    v37 = 1;
    goto LABEL_28;
  }
LABEL_91:
  if ( (_QWORD *)v23 == v62 )
    goto LABEL_97;
  while ( v22 != *v62 )
  {
    if ( (_QWORD *)v23 == ++v62 )
      goto LABEL_97;
  }
LABEL_74:
  v30 = (unsigned __int64 *)v24;
  v37 = 0;
LABEL_28:
  if ( src != v30 )
  {
    for ( i = v30 + 1; src != i; ++v30 )
    {
LABEL_30:
      while ( 1 )
      {
        v22 = *i;
        if ( v6 != *i )
          break;
        ++i;
        v37 = 1;
        if ( src == i )
          goto LABEL_37;
      }
      if ( v136 )
      {
        v40 = v133;
        v23 = (__int64)&v133[HIDWORD(v134)];
        if ( v133 != (__int64 *)v23 )
        {
          while ( v22 != *v40 )
          {
            if ( (__int64 *)v23 == ++v40 )
              goto LABEL_61;
          }
LABEL_36:
          if ( src == ++i )
            break;
          goto LABEL_30;
        }
      }
      else
      {
        if ( sub_C8CA60((__int64)&v132, v22) )
          goto LABEL_36;
        v22 = *i;
      }
LABEL_61:
      ++i;
      *v30 = v22;
    }
  }
LABEL_37:
  v41 = v126;
  v42 = (char *)((_BYTE *)v126 + 8 * (unsigned int)v127 - (_BYTE *)src);
  if ( src != (unsigned __int64 *)((char *)v126 + 8 * (unsigned int)v127) )
  {
    v22 = (unsigned __int64)src;
    memmove(v30, src, (_BYTE *)v126 + 8 * (unsigned int)v127 - (_BYTE *)src);
    v41 = v126;
  }
  v129 = v41;
  v43 = (&v42[(_QWORD)v30] - v41) >> 3;
  v44 = *(_BYTE *)(a6 + 25) == 0;
  v45 = *(_QWORD *)a6;
  LODWORD(v127) = v43;
  v130 = (unsigned int)v43;
  if ( v44 )
  {
    v22 = v45;
    sub_F76FB0(&v129, v45, v23, v24, v28, v25);
  }
  else
  {
    v46 = 8LL * (unsigned int)v43;
    v47 = v46;
    if ( v46 )
    {
      v48 = *(unsigned int *)(v45 + 88);
      v49 = v46 >> 3;
      v50 = v48 + (v46 >> 3);
      v51 = v48;
      if ( v50 > *(unsigned int *)(v45 + 92) )
      {
        srcb = v47 >> 3;
        v114 = v47;
        sub_C8D5F0(v45 + 80, (const void *)(v45 + 96), v50, 8u, v47, v49);
        v48 = *(unsigned int *)(v45 + 88);
        v49 = srcb;
        v47 = v114;
      }
      v22 = (unsigned __int64)v41;
      v112 = v49;
      memcpy((void *)(*(_QWORD *)(v45 + 80) + 8 * v48), v41, v47);
      v25 = v112;
      v52 = v112 + *(_DWORD *)(v45 + 88);
      *(_DWORD *)(v45 + 88) = v52;
      if ( v51 <= v52 - 1LL )
      {
        v113 = v6;
        v53 = v37;
        v54 = v52 - 1LL;
        while ( 1 )
        {
          v25 = *(_QWORD *)(v45 + 80);
          v47 = v25 + 8 * v54;
          v59 = *(_QWORD *)v47;
          v22 = *(_BYTE *)(v45 + 8) & 1;
          if ( (*(_BYTE *)(v45 + 8) & 1) != 0 )
          {
            v55 = v45 + 16;
            v24 = 3;
          }
          else
          {
            v24 = *(unsigned int *)(v45 + 24);
            v55 = *(_QWORD *)(v45 + 16);
            if ( !(_DWORD)v24 )
            {
              v67 = *(_DWORD *)(v45 + 8);
              ++*(_QWORD *)v45;
              v68 = 0;
              v69 = (v67 >> 1) + 1;
LABEL_120:
              v47 = (unsigned int)(3 * v24);
              goto LABEL_121;
            }
            v24 = (unsigned int)(v24 - 1);
          }
          v56 = v24 & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
          v57 = (__int64 *)(v55 + 16LL * v56);
          v58 = *v57;
          if ( v59 == *v57 )
          {
LABEL_47:
            v23 = v57[1];
            if ( v51 > v23 )
            {
              *(_QWORD *)(v25 + 8 * v23) = 0;
              v57[1] = v54;
            }
            else
            {
              *(_QWORD *)v47 = 0;
            }
            if ( v51 > --v54 )
              goto LABEL_126;
          }
          else
          {
            srca = 1;
            v68 = 0;
            v106 = v24;
            while ( v58 != -4096 )
            {
              if ( !v68 && v58 == -8192 )
                v68 = v57;
              v56 = v106 & (srca + v56);
              v24 = (unsigned int)(srca + 1);
              v57 = (__int64 *)(v55 + 16LL * v56);
              v58 = *v57;
              if ( v59 == *v57 )
                goto LABEL_47;
              ++srca;
            }
            if ( !v68 )
              v68 = v57;
            v67 = *(_DWORD *)(v45 + 8);
            ++*(_QWORD *)v45;
            v69 = (v67 >> 1) + 1;
            if ( !(_BYTE)v22 )
            {
              v24 = *(unsigned int *)(v45 + 24);
              goto LABEL_120;
            }
            v47 = 12;
            v24 = 4;
LABEL_121:
            if ( (unsigned int)v47 <= 4 * v69 )
            {
              v22 = (unsigned int)(2 * v24);
              sub_F76580(v45, v22);
              if ( (*(_BYTE *)(v45 + 8) & 1) != 0 )
              {
                v47 = v45 + 16;
                v96 = 3;
              }
              else
              {
                v95 = *(_DWORD *)(v45 + 24);
                v47 = *(_QWORD *)(v45 + 16);
                if ( !v95 )
                  goto LABEL_227;
                v96 = v95 - 1;
              }
              v24 = (unsigned int)v59 >> 9;
              v97 = v96 & (v24 ^ ((unsigned int)v59 >> 4));
              v68 = (__int64 *)(v47 + 16LL * v97);
              v98 = *v68;
              if ( *v68 != v59 )
              {
                v22 = 1;
                v24 = 0;
                while ( v98 != -4096 )
                {
                  if ( v98 == -8192 && !v24 )
                    v24 = (unsigned __int64)v68;
                  v25 = (unsigned int)(v22 + 1);
                  v22 = v97 + (unsigned int)v22;
                  v97 = v96 & v22;
                  v68 = (__int64 *)(v47 + 16LL * (v96 & (unsigned int)v22));
                  v98 = *v68;
                  if ( v59 == *v68 )
                    goto LABEL_181;
                  v22 = (unsigned int)v25;
                }
LABEL_188:
                if ( v24 )
                  v68 = (__int64 *)v24;
              }
LABEL_181:
              v67 = *(_DWORD *)(v45 + 8);
              goto LABEL_123;
            }
            v22 = (unsigned int)(v24 - *(_DWORD *)(v45 + 12) - v69);
            if ( (unsigned int)v22 <= (unsigned int)v24 >> 3 )
            {
              v22 = (unsigned int)v24;
              sub_F76580(v45, v24);
              if ( (*(_BYTE *)(v45 + 8) & 1) != 0 )
              {
                v47 = v45 + 16;
                v100 = 3;
              }
              else
              {
                v99 = *(_DWORD *)(v45 + 24);
                v47 = *(_QWORD *)(v45 + 16);
                if ( !v99 )
                {
LABEL_227:
                  *(_DWORD *)(v45 + 8) = (2 * (*(_DWORD *)(v45 + 8) >> 1) + 2) | *(_DWORD *)(v45 + 8) & 1;
                  BUG();
                }
                v100 = v99 - 1;
              }
              v24 = (unsigned int)v59 >> 9;
              v101 = v100 & (v24 ^ ((unsigned int)v59 >> 4));
              v68 = (__int64 *)(v47 + 16LL * v101);
              v102 = *v68;
              if ( v59 != *v68 )
              {
                v22 = 1;
                v24 = 0;
                while ( v102 != -4096 )
                {
                  if ( v102 == -8192 && !v24 )
                    v24 = (unsigned __int64)v68;
                  v25 = (unsigned int)(v22 + 1);
                  v22 = v101 + (unsigned int)v22;
                  v101 = v100 & v22;
                  v68 = (__int64 *)(v47 + 16LL * (v100 & (unsigned int)v22));
                  v102 = *v68;
                  if ( v59 == *v68 )
                    goto LABEL_181;
                  v22 = (unsigned int)v25;
                }
                goto LABEL_188;
              }
              goto LABEL_181;
            }
LABEL_123:
            v23 = 2 * (v67 >> 1) + 2;
            *(_DWORD *)(v45 + 8) = v23 | v67 & 1;
            if ( *v68 != -4096 )
              --*(_DWORD *)(v45 + 12);
            v68[1] = v54--;
            *v68 = v59;
            if ( v51 > v54 )
            {
LABEL_126:
              v70 = v53;
              v6 = v113;
              if ( !v70 )
                goto LABEL_127;
LABEL_109:
              if ( !(_BYTE)qword_50027C8 )
                goto LABEL_110;
              v80 = *(_BYTE **)(v6 + 16);
              v81 = *(_BYTE **)(v6 + 8);
              v129 = v131;
              v130 = 0x400000000LL;
              v82 = (v80 - v81) >> 3;
              if ( (unsigned __int64)(v80 - v81) > 0x20 )
              {
                sub_C8D5F0((__int64)&v129, v131, (v80 - v81) >> 3, 8u, v47, v25);
                v85 = v129;
                v84 = v130;
                v83 = &v129[8 * (unsigned int)v130];
              }
              else
              {
                v83 = v131;
                v84 = 0;
                v85 = v131;
              }
              if ( v81 != v80 )
              {
                memmove(v83, v81, v80 - v81);
                v85 = v129;
                v84 = v130;
              }
              v116[0] = v85;
              LODWORD(v130) = v82 + v84;
              v116[1] = (unsigned int)(v82 + v84);
              v86 = *(_QWORD *)a6;
              v117 = *(_QWORD *)(a6 + 16);
              v118 = *(unsigned int *)(v86 + 88);
              sub_28840F0((__int64)&v123, v86, &v117, &v118);
              if ( v125 )
              {
LABEL_157:
                v90 = *(unsigned int *)(v86 + 92);
                v91 = *(_QWORD *)(a6 + 16);
                v92 = *(unsigned int *)(v86 + 88);
                if ( v92 + 1 > v90 )
                {
                  sub_C8D5F0(v86 + 80, (const void *)(v86 + 96), v92 + 1, 8u, v87, v88);
                  v92 = *(unsigned int *)(v86 + 88);
                }
                v89 = *(_QWORD *)(v86 + 80);
                *(_QWORD *)(v89 + 8 * v92) = v91;
                ++*(_DWORD *)(v86 + 88);
              }
              else
              {
                v89 = v124;
                v90 = *(_QWORD *)(v124 + 8);
                if ( v90 != *(unsigned int *)(v86 + 88) - 1LL )
                {
                  *(_QWORD *)(*(_QWORD *)(v86 + 80) + 8 * v90) = 0;
                  *(_QWORD *)(v89 + 8) = *(unsigned int *)(v86 + 88);
                  goto LABEL_157;
                }
              }
              v22 = *(_QWORD *)a6;
              sub_F76FB0(v116, *(_QWORD *)a6, v89, v90, v87, v88);
              v93 = v129;
              *(_BYTE *)(a6 + 24) = 1;
              if ( v93 != v131 )
                _libc_free((unsigned __int64)v93);
              goto LABEL_110;
            }
          }
        }
      }
    }
  }
  if ( v37 )
    goto LABEL_109;
LABEL_127:
  v22 = v6;
  sub_22D0060(*(_QWORD *)(a6 + 8), v6, v121[0], v121[1]);
  if ( v6 == *(_QWORD *)(a6 + 16) )
    *(_BYTE *)(a6 + 24) = 1;
LABEL_110:
  sub_22D0390(a1, v22, v23, v24, v47, v25);
  if ( v126 != v128 )
    _libc_free((unsigned __int64)v126);
LABEL_112:
  if ( (_QWORD *)v121[0] != v122 )
    j_j___libc_free_0(v121[0]);
  if ( !v136 )
    _libc_free((unsigned __int64)v133);
  v65 = v120;
  if ( v120 )
  {
    sub_FDC110(v120);
    j_j___libc_free_0((unsigned __int64)v65);
  }
  return a1;
}
