// Function: sub_F1B280
// Address: 0xf1b280
//
__int64 __fastcall sub_F1B280(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r14d
  __int16 v6; // ax
  __int64 v7; // r15
  int v8; // ebx
  __int64 v9; // r12
  __int64 v10; // r13
  unsigned __int64 i; // rbx
  unsigned __int8 *v12; // r13
  unsigned __int8 *v13; // rax
  __int16 *v14; // rdx
  __int64 v15; // rcx
  __int16 *v16; // rax
  char v17; // r13
  __int64 **v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned int v21; // r12d
  char v22; // r9
  unsigned int v23; // edi
  unsigned int v24; // eax
  __int64 v25; // rdx
  unsigned int v26; // ebx
  unsigned int v27; // eax
  __int64 *v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rdx
  unsigned int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // r15
  __int64 v36; // rdx
  unsigned int v37; // r12d
  char v38; // dl
  __int64 v39; // r13
  unsigned int v40; // r14d
  unsigned __int8 *v41; // rbx
  __int64 *v42; // rdx
  int v43; // eax
  char v44; // dl
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  unsigned __int8 **v47; // rdx
  unsigned __int8 *v48; // r13
  unsigned __int8 **v49; // rax
  bool v50; // al
  __int64 v51; // rax
  __int64 v52; // rbx
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  char v55; // dl
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  bool v58; // al
  char v59; // bl
  __int64 *v60; // rdx
  int v61; // eax
  __int64 *v62; // rdi
  char *v63; // rsi
  char *v64; // rax
  int v65; // edx
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 *v68; // r14
  __int64 v69; // rax
  __int64 *v70; // r9
  __int64 v71; // rcx
  __int64 v72; // r15
  char *v73; // rax
  char *v74; // r13
  __int64 v75; // r13
  __int64 v77; // rax
  unsigned __int64 v78; // rdx
  __int64 v79; // rax
  __int64 *v80; // rbx
  __int64 *v81; // r12
  int v82; // eax
  unsigned __int8 v83; // cl
  __int64 v84; // [rsp+8h] [rbp-2D8h]
  unsigned __int8 v85; // [rsp+16h] [rbp-2CAh]
  char v86; // [rsp+17h] [rbp-2C9h]
  __int64 v87; // [rsp+18h] [rbp-2C8h]
  int dest; // [rsp+28h] [rbp-2B8h]
  char *desta; // [rsp+28h] [rbp-2B8h]
  int j; // [rsp+30h] [rbp-2B0h]
  __int64 v91; // [rsp+38h] [rbp-2A8h]
  unsigned __int64 src; // [rsp+40h] [rbp-2A0h]
  unsigned __int8 **srca; // [rsp+40h] [rbp-2A0h]
  __int64 *srcb; // [rsp+40h] [rbp-2A0h]
  unsigned int v95; // [rsp+48h] [rbp-298h]
  unsigned int v96; // [rsp+48h] [rbp-298h]
  __int64 v97; // [rsp+48h] [rbp-298h]
  __int64 *v98; // [rsp+50h] [rbp-290h] BYREF
  __int64 v99; // [rsp+58h] [rbp-288h]
  _BYTE v100[128]; // [rsp+60h] [rbp-280h] BYREF
  __int64 *v101; // [rsp+E0h] [rbp-200h] BYREF
  __int64 v102; // [rsp+E8h] [rbp-1F8h]
  _BYTE v103[128]; // [rsp+F0h] [rbp-1F0h] BYREF
  __int64 v104; // [rsp+170h] [rbp-170h] BYREF
  unsigned __int8 **v105; // [rsp+178h] [rbp-168h]
  __int64 v106; // [rsp+180h] [rbp-160h]
  int v107; // [rsp+188h] [rbp-158h]
  unsigned __int8 v108; // [rsp+18Ch] [rbp-154h]
  char v109; // [rsp+190h] [rbp-150h] BYREF
  __int64 v110; // [rsp+210h] [rbp-D0h] BYREF
  __int16 *v111; // [rsp+218h] [rbp-C8h]
  __int64 v112; // [rsp+220h] [rbp-C0h]
  int v113; // [rsp+228h] [rbp-B8h]
  char v114; // [rsp+22Ch] [rbp-B4h]
  __int16 v115; // [rsp+230h] [rbp-B0h] BYREF

  v87 = a2;
  v2 = sub_B2E500(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL));
  v108 = 1;
  v104 = 0;
  v5 = sub_B2A630(v2);
  v98 = (__int64 *)v100;
  v99 = 0x1000000000LL;
  v6 = *(_WORD *)(a2 + 2);
  v106 = 16;
  v107 = 0;
  v85 = v6 & 1;
  v105 = (unsigned __int8 **)&v109;
  dest = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( !dest )
  {
    v75 = 0;
    goto LABEL_140;
  }
  v86 = 0;
  v7 = 0;
  v95 = v5 - 12;
  while ( 1 )
  {
    v8 = v7 + 1;
    if ( (*(_BYTE *)(v87 + 7) & 0x40) == 0 )
    {
      a2 = v87;
      v9 = *(_QWORD *)(v87 + 32 * (v7 - (*(_DWORD *)(v87 + 4) & 0x7FFFFFF)));
      v91 = *(_QWORD *)(v9 + 8);
      if ( *(_BYTE *)(v91 + 8) == 16 )
        goto LABEL_5;
LABEL_78:
      v48 = sub_BD3990((unsigned __int8 *)v9, a2);
      if ( !v108 )
        goto LABEL_105;
      v49 = v105;
      a2 = HIDWORD(v106);
      v47 = &v105[HIDWORD(v106)];
      if ( v105 != v47 )
      {
        while ( v48 != *v49 )
        {
          if ( v47 == ++v49 )
            goto LABEL_109;
        }
        v86 = v108;
LABEL_84:
        if ( v5 == 11 )
          goto LABEL_30;
        if ( v5 > 11 )
        {
          if ( v95 > 2 )
            goto LABEL_162;
        }
        else if ( v5 <= 3 )
        {
          if ( v5 >= 0 )
            goto LABEL_30;
          goto LABEL_162;
        }
        v50 = sub_AC30F0((__int64)v48);
        if ( !v50 )
          goto LABEL_30;
        v21 = v99;
        if ( v8 == dest )
          goto LABEL_153;
        if ( (unsigned int)v99 > 1 )
        {
          v86 = v50;
          v85 = 0;
          goto LABEL_33;
        }
        v85 = 0;
        goto LABEL_155;
      }
LABEL_109:
      if ( HIDWORD(v106) < (unsigned int)v106 )
      {
        a2 = (unsigned int)++HIDWORD(v106);
        *v47 = v48;
        ++v104;
      }
      else
      {
LABEL_105:
        a2 = (__int64)v48;
        sub_C8CC70((__int64)&v104, (__int64)v48, (__int64)v47, v108, v3, v4);
        if ( !v55 )
        {
          v86 = 1;
          goto LABEL_84;
        }
      }
      v56 = (unsigned int)v99;
      v57 = (unsigned int)v99 + 1LL;
      if ( v57 > HIDWORD(v99) )
      {
        a2 = (__int64)v100;
        sub_C8D5F0((__int64)&v98, v100, v57, 8u, v3, v4);
        v56 = (unsigned int)v99;
      }
      v98[v56] = v9;
      LODWORD(v99) = v99 + 1;
      goto LABEL_84;
    }
    v9 = *(_QWORD *)(*(_QWORD *)(v87 - 8) + 32 * v7);
    v91 = *(_QWORD *)(v9 + 8);
    if ( *(_BYTE *)(v91 + 8) != 16 )
      goto LABEL_78;
LABEL_5:
    v10 = *(_QWORD *)(v91 + 32);
    if ( !(_DWORD)v10 )
    {
      v77 = (unsigned int)v99;
      v78 = (unsigned int)v99 + 1LL;
      if ( v78 > HIDWORD(v99) )
      {
        sub_C8D5F0((__int64)&v98, v100, v78, 8u, v3, v4);
        v77 = (unsigned int)v99;
      }
      v98[v77] = v9;
      v21 = v99 + 1;
      LODWORD(v99) = v99 + 1;
      if ( v8 == dest )
        goto LABEL_153;
      v85 = 0;
      if ( v21 > 1 )
      {
        v86 = 1;
        goto LABEL_33;
      }
      goto LABEL_155;
    }
    v101 = (__int64 *)v103;
    v102 = 0x1000000000LL;
    if ( *(_BYTE *)v9 != 14 )
      break;
    v51 = sub_AD6530(*(_QWORD *)(v91 + 24), a2);
    v52 = v51;
    if ( v5 == 11 )
      goto LABEL_98;
    if ( v5 > 11 )
    {
      if ( v95 > 2 )
        goto LABEL_162;
    }
    else if ( v5 <= 3 )
    {
      if ( v5 >= 0 )
      {
LABEL_98:
        v53 = (unsigned int)v102;
        v54 = (unsigned int)v102 + 1LL;
        if ( v54 > HIDWORD(v102) )
        {
          sub_C8D5F0((__int64)&v101, v103, v54, 8u, v3, v4);
          v53 = (unsigned int)v102;
        }
        v101[v53] = v52;
        a2 = (unsigned int)(v102 + 1);
        LODWORD(v102) = v102 + 1;
        if ( (_DWORD)v10 != 1 )
        {
LABEL_23:
          v17 = 1;
          v18 = (__int64 **)sub_BCD420(*(__int64 **)(v91 + 24), a2);
          a2 = (__int64)v101;
          v86 = 1;
          v9 = sub_AD1300(v18, v101, (unsigned int)v102);
          goto LABEL_24;
        }
        goto LABEL_101;
      }
LABEL_162:
      BUG();
    }
    if ( !sub_AC30F0(v51) )
      goto LABEL_98;
LABEL_66:
    if ( v101 != (__int64 *)v103 )
      _libc_free(v101, a2);
    v86 = 1;
LABEL_30:
    if ( dest == (_DWORD)++v7 )
    {
      v21 = v99;
      goto LABEL_32;
    }
  }
  v110 = 0;
  v111 = &v115;
  v112 = 16;
  v113 = 0;
  v114 = 1;
  src = (unsigned int)v10;
  if ( (unsigned int)v10 > 0x10 )
  {
    a2 = (__int64)v103;
    sub_C8D5F0((__int64)&v101, v103, (unsigned int)v10, 8u, v3, v4);
  }
  for ( i = 0; i != src; ++i )
  {
    while ( 1 )
    {
      v12 = *(unsigned __int8 **)(v9 + 32 * (i - (*(_DWORD *)(v9 + 4) & 0x7FFFFFF)));
      v13 = sub_BD3990(v12, a2);
      a2 = (__int64)v13;
      if ( v5 != 11 )
      {
        if ( v5 > 11 )
        {
          if ( v95 > 2 )
            goto LABEL_162;
        }
        else if ( v5 <= 3 )
        {
          if ( v5 < 0 )
            goto LABEL_162;
          goto LABEL_14;
        }
        a2 = (__int64)v13;
        if ( sub_AC30F0((__int64)v13) )
        {
          if ( !v114 )
            _libc_free(v111, a2);
          goto LABEL_66;
        }
      }
LABEL_14:
      if ( v114 )
      {
        v16 = v111;
        v15 = HIDWORD(v112);
        v14 = &v111[4 * HIDWORD(v112)];
        if ( v111 != v14 )
        {
          while ( a2 != *(_QWORD *)v16 )
          {
            v16 += 4;
            if ( v14 == v16 )
              goto LABEL_75;
          }
          goto LABEL_19;
        }
LABEL_75:
        if ( HIDWORD(v112) < (unsigned int)v112 )
          break;
      }
      sub_C8CC70((__int64)&v110, a2, (__int64)v14, v15, v3, v4);
      if ( v44 )
        goto LABEL_70;
LABEL_19:
      if ( ++i == src )
        goto LABEL_20;
    }
    ++HIDWORD(v112);
    *(_QWORD *)v14 = a2;
    ++v110;
LABEL_70:
    v45 = (unsigned int)v102;
    v46 = (unsigned int)v102 + 1LL;
    if ( v46 > HIDWORD(v102) )
    {
      a2 = (__int64)v103;
      sub_C8D5F0((__int64)&v101, v103, v46, 8u, v3, v4);
      v45 = (unsigned int)v102;
    }
    v101[v45] = (__int64)v12;
    LODWORD(v102) = v102 + 1;
  }
LABEL_20:
  a2 = (unsigned int)v102;
  v17 = v114;
  if ( (unsigned int)v102 < i )
  {
    if ( !v114 )
    {
      _libc_free(v111, (unsigned int)v102);
      a2 = (unsigned int)v102;
    }
    goto LABEL_23;
  }
  if ( v114 )
  {
LABEL_101:
    v17 = 0;
    goto LABEL_24;
  }
  _libc_free(v111, (unsigned int)v102);
LABEL_24:
  v19 = (unsigned int)v99;
  v20 = (unsigned int)v99 + 1LL;
  if ( v20 > HIDWORD(v99) )
  {
    a2 = (__int64)v100;
    sub_C8D5F0((__int64)&v98, v100, v20, 8u, v3, v4);
    v19 = (unsigned int)v99;
  }
  v98[v19] = v9;
  v21 = v99 + 1;
  LODWORD(v99) = v99 + 1;
  if ( !v17 || (a2 = (unsigned int)v102, (_DWORD)v102) )
  {
    if ( v101 != (__int64 *)v103 )
      _libc_free(v101, a2);
    goto LABEL_30;
  }
  if ( v101 != (__int64 *)v103 )
  {
    _libc_free(v101, (unsigned int)v102);
    v21 = v99;
  }
LABEL_153:
  v85 = 0;
LABEL_32:
  if ( v21 > 1 )
  {
LABEL_33:
    v22 = v86;
    v23 = 0;
LABEL_34:
    while ( 2 )
    {
      if ( v23 == v21 )
      {
        v26 = v23;
        v27 = v23;
      }
      else
      {
        v24 = v23;
        do
        {
          v25 = v24;
          v26 = v24++;
          if ( *(_BYTE *)(*(_QWORD *)(v98[v25] + 8) + 8LL) != 16 )
          {
            v27 = v23;
            goto LABEL_39;
          }
        }
        while ( v21 != v24 );
        v26 = v21;
        v27 = v23;
      }
LABEL_39:
      v28 = &v98[v23 + 1];
      do
      {
        v30 = v27++;
        if ( v27 >= v26 )
        {
          v23 = v26 + 1;
          if ( v26 + 2 < v21 )
            goto LABEL_34;
          goto LABEL_43;
        }
        v29 = *v28++;
      }
      while ( *(_QWORD *)(*(_QWORD *)(v29 + 8) + 32LL) >= *(_QWORD *)(*(_QWORD *)(v98[v30] + 8) + 32LL) );
      v66 = v26;
      v67 = v23;
      v68 = &v98[v66];
      v69 = v66 * 8 - v67 * 8;
      v70 = &v98[v67];
      v71 = v69 >> 3;
      if ( v69 <= 0 )
      {
LABEL_136:
        v72 = 0;
        sub_F18830(v70, v68, (__int64 (__fastcall *)(__int64, __int64))sub_F06500);
        v74 = 0;
      }
      else
      {
        while ( 1 )
        {
          v72 = 8 * v71;
          srcb = v70;
          v97 = v71;
          v73 = (char *)sub_2207800(8 * v71, &unk_435FF63);
          v70 = srcb;
          v74 = v73;
          if ( v73 )
            break;
          v71 = v97 >> 1;
          if ( !(v97 >> 1) )
            goto LABEL_136;
        }
        sub_F1B180(srcb, v68, v73, (void *)v97, (__int64 (__fastcall *)(__int64, __int64))sub_F06500);
      }
      j_j___libc_free_0(v74, v72);
      v23 = v26 + 1;
      v22 = 1;
      if ( v26 + 2 < v21 )
        continue;
      break;
    }
LABEL_43:
    v21 = v99;
    v86 = v22;
    v31 = v99;
    if ( (unsigned int)v99 > 1 )
    {
      for ( j = 0; ; ++j )
      {
        v32 = v98;
        v33 = *(_QWORD *)(v98[j] + 8);
        if ( *(_BYTE *)(v33 + 8) == 16 )
        {
          v84 = *(_QWORD *)(v33 + 32);
          v96 = v31 - 1;
          if ( v31 - 1 != j )
            break;
        }
LABEL_121:
        v21 = v31;
        if ( v31 <= j + 2 )
          goto LABEL_137;
      }
      srca = (unsigned __int8 **)v98[j];
      while ( 2 )
      {
        v34 = (__int64)&v32[v96];
        v35 = *(_QWORD *)v34;
        desta = (char *)v34;
        v36 = *(_QWORD *)(*(_QWORD *)v34 + 8LL);
        if ( *(_BYTE *)(v36 + 8) == 16 )
        {
          if ( !(_DWORD)v84 )
          {
            v62 = &v32[v96];
            v63 = (char *)(v34 + 8);
            v64 = (char *)&v32[(unsigned int)v99];
            v65 = v99;
            if ( v64 == v63 )
              goto LABEL_125;
            goto LABEL_124;
          }
          v37 = *(_QWORD *)(v36 + 32);
          if ( (unsigned int)v84 <= v37 )
          {
            v38 = *(_BYTE *)srca;
            if ( *(_BYTE *)v35 != 14 )
            {
              v39 = 0;
              if ( v38 == 14 )
              {
                while ( 1 )
                {
                  v58 = sub_AC30F0(*(_QWORD *)(v35
                                             + 32
                                             * ((unsigned int)v39 - (unsigned __int64)(*(_DWORD *)(v35 + 4) & 0x7FFFFFF))));
                  if ( v58 )
                    break;
                  LODWORD(v39) = v39 + 1;
                  if ( v37 == (_DWORD)v39 )
                    goto LABEL_60;
                }
                v59 = v58;
                v60 = &v98[(unsigned int)v99];
                v61 = v99;
                if ( v60 != (__int64 *)(v34 + 8) )
                {
                  memmove((void *)v34, (const void *)(v34 + 8), (size_t)v60 - v34 - 8);
                  v61 = v99;
                }
                v86 = v59;
                LODWORD(v99) = v61 - 1;
              }
              else
              {
                do
                {
                  v40 = 0;
                  v41 = sub_BD3990(srca[4 * (v39 - (*((_DWORD *)srca + 1) & 0x7FFFFFF))], v34);
                  while ( v41 != sub_BD3990(
                                   *(unsigned __int8 **)(v35
                                                       + 32
                                                       * (v40 - (unsigned __int64)(*(_DWORD *)(v35 + 4) & 0x7FFFFFF))),
                                   v34) )
                  {
                    if ( v37 == ++v40 )
                      goto LABEL_60;
                  }
                  ++v39;
                }
                while ( v39 != (unsigned int)v84 );
                v42 = &v98[(unsigned int)v99];
                v43 = v99;
                if ( v42 != (__int64 *)(v34 + 8) )
                {
                  memmove((void *)v34, (const void *)(v34 + 8), (size_t)v42 - v34 - 8);
                  v43 = v99;
                }
                v86 = 1;
                LODWORD(v99) = v43 - 1;
              }
              goto LABEL_60;
            }
            if ( v38 == 14 )
            {
              v62 = &v32[v96];
              v64 = (char *)&v32[(unsigned int)v99];
              v63 = (char *)(v34 + 8);
              v65 = v99;
              if ( v64 == desta + 8 )
              {
LABEL_125:
                v86 = 1;
                LODWORD(v99) = v65 - 1;
                goto LABEL_60;
              }
LABEL_124:
              memmove(v62, v63, v64 - v63);
              v65 = v99;
              goto LABEL_125;
            }
          }
        }
LABEL_60:
        if ( --v96 == j )
        {
          v31 = v99;
          goto LABEL_121;
        }
        v32 = v98;
        continue;
      }
    }
  }
LABEL_137:
  if ( !v86 )
  {
    a2 = v87;
    v75 = 0;
    if ( (*(_WORD *)(v87 + 2) & 1) != v85 )
    {
      v75 = v87;
      *(_WORD *)(v87 + 2) = *(_WORD *)(v87 + 2) & 0xFFFE | v85;
    }
    goto LABEL_140;
  }
LABEL_155:
  a2 = v21;
  v115 = 257;
  v79 = sub_B49060(*(_QWORD *)(v87 + 8), v21, (__int64)&v110, 0, 0);
  v80 = v98;
  v75 = v79;
  v81 = &v98[(unsigned int)v99];
  v82 = v99;
  if ( v81 != v98 )
  {
    do
    {
      a2 = *v80++;
      sub_B49100(v75, a2);
    }
    while ( v81 != v80 );
    v82 = v99;
  }
  v83 = v85;
  if ( !v82 )
    v83 = 1;
  *(_WORD *)(v75 + 2) = v83 | *(_WORD *)(v75 + 2) & 0xFFFE;
LABEL_140:
  if ( !v108 )
    _libc_free(v105, a2);
  if ( v98 != (__int64 *)v100 )
    _libc_free(v98, a2);
  return v75;
}
