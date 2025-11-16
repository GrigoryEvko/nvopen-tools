// Function: sub_2CC4A70
// Address: 0x2cc4a70
//
__int64 __fastcall sub_2CC4A70(__int64 a1)
{
  __int64 v2; // rcx
  __int64 v3; // rsi
  __int64 v4; // rdi
  unsigned int v5; // eax
  __int64 v6; // rdx
  unsigned int v7; // r12d
  unsigned __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // r13
  __int64 v16; // rbx
  __int64 v17; // r9
  __int64 v18; // r11
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // r15
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 i; // r15
  __int64 v31; // rax
  unsigned int v32; // esi
  unsigned __int64 v33; // rax
  __int64 v34; // rbx
  __int64 v35; // r15
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // r13
  __int64 j; // r15
  __int64 v40; // rax
  unsigned int v41; // esi
  __int64 v42; // rax
  __int64 *v43; // rbx
  __int64 *v44; // r13
  _BYTE *v45; // rsi
  __int64 v46; // rax
  _BYTE *v47; // r8
  _BYTE *v48; // rsi
  __int64 v49; // rax
  __int64 v50; // r15
  __int64 v51; // r12
  __int64 v52; // r13
  _BYTE *v53; // rax
  _QWORD *v54; // r8
  _QWORD *v55; // rdx
  _QWORD *v56; // rax
  __int64 v57; // rdi
  __int64 *v58; // r13
  __int64 *v59; // rbx
  __int64 v60; // rax
  unsigned __int64 v61; // rax
  __int64 v62; // rsi
  __int64 k; // r12
  __int64 v64; // r13
  _BYTE *v65; // rax
  _QWORD *v66; // r8
  _QWORD *v67; // rdx
  _QWORD *v68; // rax
  __int64 v69; // rdi
  __int64 *v70; // rax
  __int64 *v71; // rax
  unsigned __int64 v72; // rbx
  __int64 v73; // rsi
  __int64 v74; // rbx
  __int64 v75; // rax
  __int64 v76; // r13
  _QWORD *v77; // rax
  __int64 v78; // rbx
  __int64 v79; // r15
  _BYTE *v80; // r13
  __int64 v81; // rdx
  unsigned int v82; // esi
  __int64 *v83; // r13
  __int64 v84; // rbx
  __int64 m; // r13
  __int64 v86; // rax
  unsigned int v87; // esi
  __int64 v88; // rsi
  unsigned __int8 *v89; // rsi
  __int64 v90; // rdx
  unsigned __int64 v91; // rsi
  __int64 v92; // rdx
  __int64 v93; // rdx
  unsigned __int64 v94; // rsi
  __int64 v95; // rdx
  __int64 v96; // [rsp+0h] [rbp-1D0h]
  __int64 v97; // [rsp+20h] [rbp-1B0h]
  unsigned __int8 v98; // [rsp+28h] [rbp-1A8h]
  unsigned __int8 v99; // [rsp+28h] [rbp-1A8h]
  __int64 v100; // [rsp+30h] [rbp-1A0h] BYREF
  unsigned __int8 *v101; // [rsp+38h] [rbp-198h] BYREF
  __int64 v102; // [rsp+40h] [rbp-190h] BYREF
  __int64 v103; // [rsp+48h] [rbp-188h] BYREF
  __int64 v104; // [rsp+50h] [rbp-180h] BYREF
  __int64 v105; // [rsp+58h] [rbp-178h] BYREF
  __int64 v106; // [rsp+60h] [rbp-170h] BYREF
  __int64 v107; // [rsp+68h] [rbp-168h] BYREF
  __int64 v108; // [rsp+70h] [rbp-160h] BYREF
  __int64 v109; // [rsp+78h] [rbp-158h] BYREF
  __int64 v110; // [rsp+80h] [rbp-150h] BYREF
  __int64 v111; // [rsp+88h] [rbp-148h] BYREF
  __int64 v112; // [rsp+90h] [rbp-140h] BYREF
  __int64 v113; // [rsp+98h] [rbp-138h] BYREF
  __int64 v114; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v115; // [rsp+A8h] [rbp-128h] BYREF
  __int64 v116; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v117; // [rsp+B8h] [rbp-118h] BYREF
  _BYTE *v118; // [rsp+C0h] [rbp-110h] BYREF
  _BYTE *v119; // [rsp+C8h] [rbp-108h]
  _BYTE *v120; // [rsp+D0h] [rbp-100h]
  __int64 v121[4]; // [rsp+E0h] [rbp-F0h] BYREF
  __int16 v122; // [rsp+100h] [rbp-D0h]
  _BYTE *v123; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v124; // [rsp+118h] [rbp-B8h]
  _BYTE v125[32]; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v126; // [rsp+140h] [rbp-90h]
  __int64 v127; // [rsp+148h] [rbp-88h]
  __int64 v128; // [rsp+150h] [rbp-80h]
  __int64 v129; // [rsp+158h] [rbp-78h]
  void **v130; // [rsp+160h] [rbp-70h]
  void **v131; // [rsp+168h] [rbp-68h]
  __int64 v132; // [rsp+170h] [rbp-60h]
  int v133; // [rsp+178h] [rbp-58h]
  __int16 v134; // [rsp+17Ch] [rbp-54h]
  char v135; // [rsp+17Eh] [rbp-52h]
  __int64 v136; // [rsp+180h] [rbp-50h]
  __int64 v137; // [rsp+188h] [rbp-48h]
  void *v138; // [rsp+190h] [rbp-40h] BYREF
  void *v139; // [rsp+198h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 160);
  v3 = *(_QWORD *)(a1 + 24);
  v100 = 0;
  v101 = 0;
  if ( v2 )
  {
    v4 = (unsigned int)(*(_DWORD *)(v2 + 44) + 1);
    v5 = *(_DWORD *)(v2 + 44) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  if ( v5 >= *(_DWORD *)(v3 + 32) )
    BUG();
  v6 = *(_QWORD *)(a1 + 152);
  if ( v6 != **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v3 + 24) + 8 * v4) + 8LL) )
    return 0;
  v9 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 == v6 + 48 )
    goto LABEL_197;
  if ( !v9 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
LABEL_197:
    BUG();
  if ( *(_BYTE *)(v9 - 24) != 31 )
    return 0;
  if ( (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v10 = *(_QWORD *)(v9 - 120);
  if ( *(_BYTE *)v10 != 82 )
    return 0;
  v11 = *(_QWORD *)(a1 + 40);
  v12 = *(_QWORD *)a1;
  v13 = *(_WORD *)(v10 + 2) & 0x3F;
  if ( (*(_WORD *)(v10 + 2) & 0x3F) == 0x21 )
  {
    if ( v2 != *(_QWORD *)(v9 - 56) )
      return 0;
  }
  else if ( (*(_WORD *)(v10 + 2) & 0x3F) != 0x20 || v2 != *(_QWORD *)(v9 - 88) )
  {
    return 0;
  }
  if ( v11 == *(_QWORD *)(v10 - 64) && (unsigned __int8)sub_D48480(*(_QWORD *)a1, *(_QWORD *)(v10 - 32), v13, v2) )
  {
    v96 = *(_QWORD *)(v10 - 32);
  }
  else
  {
    if ( v11 != *(_QWORD *)(v10 - 32) || !(unsigned __int8)sub_D48480(v12, *(_QWORD *)(v10 - 64), v13, v2) )
      return 0;
    v96 = *(_QWORD *)(v10 - 64);
  }
  if ( !(unsigned __int8)sub_2CBFC80(
                           *(_QWORD *)a1,
                           *(_QWORD *)(a1 + 176),
                           *(_QWORD *)(a1 + 144),
                           *(_QWORD *)(a1 + 152),
                           *(_QWORD *)(a1 + 160),
                           *(_QWORD *)(a1 + 168),
                           *(_QWORD *)(a1 + 184),
                           *(_QWORD *)(a1 + 56),
                           *(_QWORD *)(a1 + 64),
                           &v100,
                           &v101) )
    return 0;
  v7 = sub_2CBF770(
         *(_QWORD *)a1,
         *(_QWORD *)(a1 + 176),
         *(_QWORD *)(a1 + 152),
         *(__int64 **)(a1 + 160),
         *(_QWORD *)(a1 + 168),
         *(_QWORD *)(a1 + 184));
  if ( !(_BYTE)v7 )
    return 0;
  v14 = *(_QWORD *)(a1 + 40);
  if ( !v14 )
    return 0;
  if ( *(_QWORD *)(a1 + 152) != *(_QWORD *)(v14 + 40) )
    return 0;
  v15 = *(_QWORD *)(a1 + 184);
  v16 = *(_QWORD *)(a1 + 168);
  v97 = sub_2CBF180(v14, *(_QWORD *)(a1 + 160), v16, v15);
  if ( !v97 )
    return 0;
  sub_2CC1B10(
    *(_QWORD *)a1,
    &v115,
    &v116,
    1,
    *(__int64 **)(a1 + 8),
    *(_QWORD *)(a1 + 200),
    *(_QWORD *)(a1 + 16),
    v97,
    &v114,
    *(_QWORD *)(a1 + 176),
    *(_QWORD *)(a1 + 144),
    v17,
    v18,
    v16,
    v15,
    *(_QWORD *)(a1 + 192),
    &v102,
    &v103,
    (__int64)&v104,
    &v105,
    &v106,
    &v107,
    &v108,
    &v109,
    (__int64)&v110,
    &v111,
    &v112,
    &v113);
  v21 = *(unsigned int *)(a1 + 216);
  v22 = v116;
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 220) )
  {
    sub_C8D5F0(a1 + 208, (const void *)(a1 + 224), v21 + 1, 8u, v19, v20);
    v21 = *(unsigned int *)(a1 + 216);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 208) + 8 * v21) = v22;
  v23 = *(_QWORD *)(a1 + 152);
  ++*(_DWORD *)(a1 + 216);
  v24 = *(_QWORD *)(v23 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v24 == v23 + 48 )
    goto LABEL_196;
  if ( !v24 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v24 - 24) - 30 > 0xA )
LABEL_196:
    BUG();
  v25 = *(_QWORD *)(v24 - 56);
  v26 = *(_QWORD *)(a1 + 160);
  if ( v25 )
  {
    if ( v25 == v26 )
    {
      v25 = *(_QWORD *)(v24 - 88);
      if ( !v25 || (v27 = *(_QWORD *)(v24 - 80), (**(_QWORD **)(v24 - 72) = v27) == 0) )
      {
        *(_QWORD *)(v24 - 88) = v26;
        goto LABEL_35;
      }
LABEL_34:
      *(_QWORD *)(v27 + 16) = *(_QWORD *)(v24 - 72);
      *(_QWORD *)(v24 - 88) = v26;
      if ( !v26 )
      {
LABEL_38:
        v26 = *(_QWORD *)(a1 + 160);
        goto LABEL_39;
      }
LABEL_35:
      v28 = *(_QWORD *)(v26 + 16);
      *(_QWORD *)(v24 - 80) = v28;
      if ( v28 )
        *(_QWORD *)(v28 + 16) = v24 - 80;
      *(_QWORD *)(v24 - 72) = v26 + 16;
      *(_QWORD *)(v26 + 16) = v24 - 88;
      goto LABEL_38;
    }
    v93 = *(_QWORD *)(v24 - 48);
    **(_QWORD **)(v24 - 40) = v93;
    if ( v93 )
      *(_QWORD *)(v93 + 16) = *(_QWORD *)(v24 - 40);
    *(_QWORD *)(v24 - 56) = v26;
    v94 = v24 - 56;
    if ( !v26 )
      goto LABEL_38;
    goto LABEL_164;
  }
  if ( v26 )
  {
    *(_QWORD *)(v24 - 56) = v26;
    v94 = v24 - 56;
LABEL_164:
    v95 = *(_QWORD *)(v26 + 16);
    *(_QWORD *)(v24 - 48) = v95;
    if ( v95 )
      *(_QWORD *)(v95 + 16) = v24 - 48;
    *(_QWORD *)(v24 - 40) = v26 + 16;
    *(_QWORD *)(v26 + 16) = v94;
    v26 = *(_QWORD *)(a1 + 160);
    goto LABEL_39;
  }
  v25 = *(_QWORD *)(v24 - 88);
  if ( v25 )
  {
    v27 = *(_QWORD *)(v24 - 80);
    **(_QWORD **)(v24 - 72) = v27;
    if ( v27 )
      goto LABEL_34;
    *(_QWORD *)(v24 - 88) = 0;
    v26 = *(_QWORD *)(a1 + 160);
  }
LABEL_39:
  v29 = *(_QWORD *)(v26 + 56);
  for ( i = v26 + 48; i != v29; v29 = *(_QWORD *)(v29 + 8) )
  {
    if ( !v29 )
      BUG();
    if ( *(_BYTE *)(v29 - 24) != 84 )
      break;
    if ( (*(_DWORD *)(v29 - 20) & 0x7FFFFFF) != 0 )
    {
      v31 = 0;
      while ( 1 )
      {
        v32 = v31;
        if ( v25 == *(_QWORD *)(*(_QWORD *)(v29 - 32) + 32LL * *(unsigned int *)(v29 + 48) + 8 * v31) )
          break;
        if ( (*(_DWORD *)(v29 - 20) & 0x7FFFFFF) == (_DWORD)++v31 )
          goto LABEL_149;
      }
    }
    else
    {
LABEL_149:
      v32 = -1;
    }
    sub_B48BF0(v29 - 24, v32, 1);
  }
  v33 = *(_QWORD *)(v110 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v33 == v110 + 48 )
    goto LABEL_199;
  if ( !v33 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v33 - 24) - 30 > 0xA )
LABEL_199:
    BUG();
  v34 = *(_QWORD *)(v33 - 56);
  v35 = v111;
  if ( v34 )
  {
    if ( v34 == v111 )
    {
      v34 = *(_QWORD *)(v33 - 88);
      if ( !v34 || (v36 = *(_QWORD *)(v33 - 80), (**(_QWORD **)(v33 - 72) = v36) == 0) )
      {
        *(_QWORD *)(v33 - 88) = v35;
        goto LABEL_55;
      }
LABEL_54:
      *(_QWORD *)(v36 + 16) = *(_QWORD *)(v33 - 72);
      *(_QWORD *)(v33 - 88) = v35;
      if ( !v35 )
      {
LABEL_58:
        v35 = v111;
        goto LABEL_59;
      }
LABEL_55:
      v37 = *(_QWORD *)(v35 + 16);
      *(_QWORD *)(v33 - 80) = v37;
      if ( v37 )
        *(_QWORD *)(v37 + 16) = v33 - 80;
      *(_QWORD *)(v33 - 72) = v35 + 16;
      *(_QWORD *)(v35 + 16) = v33 - 88;
      goto LABEL_58;
    }
    v90 = *(_QWORD *)(v33 - 48);
    **(_QWORD **)(v33 - 40) = v90;
    if ( v90 )
      *(_QWORD *)(v90 + 16) = *(_QWORD *)(v33 - 40);
    *(_QWORD *)(v33 - 56) = v35;
    v91 = v33 - 56;
    if ( !v35 )
      goto LABEL_58;
    goto LABEL_158;
  }
  if ( v111 )
  {
    *(_QWORD *)(v33 - 56) = v111;
    v91 = v33 - 56;
LABEL_158:
    v92 = *(_QWORD *)(v35 + 16);
    *(_QWORD *)(v33 - 48) = v92;
    if ( v92 )
      *(_QWORD *)(v92 + 16) = v33 - 48;
    *(_QWORD *)(v33 - 40) = v35 + 16;
    *(_QWORD *)(v35 + 16) = v91;
    v35 = v111;
    goto LABEL_59;
  }
  v34 = *(_QWORD *)(v33 - 88);
  if ( v34 )
  {
    v36 = *(_QWORD *)(v33 - 80);
    **(_QWORD **)(v33 - 72) = v36;
    if ( v36 )
      goto LABEL_54;
    *(_QWORD *)(v33 - 88) = 0;
    v35 = v111;
  }
LABEL_59:
  v38 = *(_QWORD *)(v35 + 56);
  for ( j = v35 + 48; j != v38; v38 = *(_QWORD *)(v38 + 8) )
  {
    if ( !v38 )
      BUG();
    if ( *(_BYTE *)(v38 - 24) != 84 )
      break;
    if ( (*(_DWORD *)(v38 - 20) & 0x7FFFFFF) != 0 )
    {
      v40 = 0;
      while ( 1 )
      {
        v41 = v40;
        if ( v34 == *(_QWORD *)(*(_QWORD *)(v38 - 32) + 32LL * *(unsigned int *)(v38 + 48) + 8 * v40) )
          break;
        if ( (*(_DWORD *)(v38 - 20) & 0x7FFFFFF) == (_DWORD)++v40 )
          goto LABEL_147;
      }
    }
    else
    {
LABEL_147:
      v41 = -1;
    }
    sub_B48BF0(v38 - 24, v41, 1);
  }
  v118 = 0;
  v42 = *(_QWORD *)a1;
  v119 = 0;
  v43 = *(__int64 **)(v42 + 32);
  v44 = *(__int64 **)(v42 + 40);
  v120 = 0;
  if ( v44 == v43 )
  {
    v58 = *(__int64 **)(v116 + 40);
    v59 = *(__int64 **)(v116 + 32);
    if ( v59 == v58 )
      goto LABEL_120;
    v47 = 0;
  }
  else
  {
    v45 = 0;
    do
    {
      v46 = *v43;
      if ( *v43 != *(_QWORD *)(a1 + 176)
        && v46 != *(_QWORD *)(a1 + 144)
        && v46 != *(_QWORD *)(a1 + 152)
        && v46 != *(_QWORD *)(a1 + 160)
        && v46 != *(_QWORD *)(a1 + 168)
        && v46 != *(_QWORD *)(a1 + 184) )
      {
        if ( v120 == v45 )
        {
          sub_9319A0((__int64)&v118, v45, v43);
          v45 = v119;
        }
        else
        {
          if ( v45 )
          {
            *(_QWORD *)v45 = v46;
            v45 = v119;
          }
          v45 += 8;
          v119 = v45;
        }
      }
      ++v43;
    }
    while ( v43 != v44 );
    v47 = v45;
    v48 = v118;
    v49 = (v47 - v118) >> 3;
    if ( (_DWORD)v49 )
    {
      v98 = v7;
      v50 = 8LL * (unsigned int)(v49 - 1) + 8;
      v51 = 0;
      do
      {
        v52 = *(_QWORD *)a1;
        v123 = *(_BYTE **)&v48[v51];
        v53 = sub_2CBEFC0(*(_QWORD **)(v52 + 32), *(_QWORD *)(v52 + 40), (__int64 *)&v123);
        sub_F681A0(v52 + 32, v53);
        if ( *(_BYTE *)(v52 + 84) )
        {
          v54 = *(_QWORD **)(v52 + 64);
          v55 = &v54[*(unsigned int *)(v52 + 76)];
          v56 = v54;
          if ( v54 != v55 )
          {
            while ( v123 != (_BYTE *)*v56 )
            {
              if ( v55 == ++v56 )
                goto LABEL_88;
            }
            v57 = (unsigned int)(*(_DWORD *)(v52 + 76) - 1);
            *(_DWORD *)(v52 + 76) = v57;
            *v56 = v54[v57];
            ++*(_QWORD *)(v52 + 56);
          }
        }
        else
        {
          v71 = sub_C8CA60(v52 + 56, (__int64)v123);
          if ( v71 )
          {
            *v71 = -2;
            ++*(_DWORD *)(v52 + 80);
            ++*(_QWORD *)(v52 + 56);
          }
        }
LABEL_88:
        v51 += 8;
        v48 = v118;
      }
      while ( v50 != v51 );
      v7 = v98;
      v47 = v119;
    }
    v58 = *(__int64 **)(v116 + 40);
    v59 = *(__int64 **)(v116 + 32);
    if ( v47 != v48 )
    {
      v119 = v48;
      if ( v58 == v59 )
        goto LABEL_120;
      goto LABEL_92;
    }
    if ( v58 == v59 )
    {
      v61 = (unsigned __int64)v48;
      goto LABEL_104;
    }
  }
  v48 = v47;
  do
  {
LABEL_92:
    v60 = *v59;
    if ( *v59 != v108 && v60 != v109 && v60 != v110 && v60 != v111 && v60 != v112 && v60 != v113 )
    {
      if ( v120 == v48 )
      {
        sub_9319A0((__int64)&v118, v48, v59);
        v48 = v119;
      }
      else
      {
        if ( v48 )
        {
          *(_QWORD *)v48 = v60;
          v48 = v119;
        }
        v48 += 8;
        v119 = v48;
      }
    }
    ++v59;
  }
  while ( v59 != v58 );
  v61 = (unsigned __int64)v118;
LABEL_104:
  v62 = (__int64)&v48[-v61] >> 3;
  if ( (_DWORD)v62 )
  {
    v99 = v7;
    for ( k = 0; ; k += 8 )
    {
      v64 = v116;
      v123 = *(_BYTE **)(v61 + k);
      v65 = sub_2CBEFC0(*(_QWORD **)(v116 + 32), *(_QWORD *)(v116 + 40), (__int64 *)&v123);
      sub_F681A0(v64 + 32, v65);
      if ( *(_BYTE *)(v64 + 84) )
      {
        v66 = *(_QWORD **)(v64 + 64);
        v67 = &v66[*(unsigned int *)(v64 + 76)];
        v68 = v66;
        if ( v66 != v67 )
        {
          while ( v123 != (_BYTE *)*v68 )
          {
            if ( v67 == ++v68 )
              goto LABEL_112;
          }
          v69 = (unsigned int)(*(_DWORD *)(v64 + 76) - 1);
          *(_DWORD *)(v64 + 76) = v69;
          *v68 = v66[v69];
          ++*(_QWORD *)(v64 + 56);
        }
      }
      else
      {
        v70 = sub_C8CA60(v64 + 56, (__int64)v123);
        if ( v70 )
        {
          *v70 = -2;
          ++*(_DWORD *)(v64 + 80);
          ++*(_QWORD *)(v64 + 56);
        }
      }
LABEL_112:
      if ( 8LL * (unsigned int)(v62 - 1) == k )
        break;
      v61 = (unsigned __int64)v118;
    }
    v7 = v99;
  }
LABEL_120:
  v72 = *(_QWORD *)(v105 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v72 == v105 + 48 )
    goto LABEL_204;
  if ( !v72 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v72 - 24) - 30 > 0xA )
LABEL_204:
    BUG();
  v73 = *(_QWORD *)(v72 + 24);
  v117 = v73;
  if ( v73 )
    sub_B96E90((__int64)&v117, v73, 1);
  sub_B43D60((_QWORD *)(v72 - 24));
  v74 = v105;
  v75 = sub_AA48A0(v105);
  v135 = 7;
  v129 = v75;
  v76 = v106;
  v130 = &v138;
  v131 = &v139;
  v123 = v125;
  v138 = &unk_49DA100;
  v124 = 0x200000000LL;
  v126 = v74;
  v122 = 257;
  v127 = v74 + 48;
  LOWORD(v128) = 0;
  v132 = 0;
  v133 = 0;
  v134 = 512;
  v136 = 0;
  v137 = 0;
  v139 = &unk_49DA0B0;
  v77 = sub_BD2C40(72, 1u);
  v78 = (__int64)v77;
  if ( v77 )
    sub_B4C8F0((__int64)v77, v76, 1u, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v131 + 2))(v131, v78, v121, v127, v128);
  v79 = (__int64)v123;
  v80 = &v123[16 * (unsigned int)v124];
  if ( v123 != v80 )
  {
    do
    {
      v81 = *(_QWORD *)(v79 + 8);
      v82 = *(_DWORD *)v79;
      v79 += 16;
      sub_B99FD0(v78, v82, v81);
    }
    while ( v80 != (_BYTE *)v79 );
  }
  v83 = (__int64 *)(v78 + 48);
  v121[0] = v117;
  if ( v117 )
  {
    sub_B96E90((__int64)v121, v117, 1);
    if ( v83 == v121 )
    {
      if ( v121[0] )
        sub_B91220((__int64)v121, v121[0]);
      goto LABEL_133;
    }
    v88 = *(_QWORD *)(v78 + 48);
    if ( !v88 )
    {
LABEL_153:
      v89 = (unsigned __int8 *)v121[0];
      *(_QWORD *)(v78 + 48) = v121[0];
      if ( v89 )
        sub_B976B0((__int64)v121, v89, v78 + 48);
      goto LABEL_133;
    }
LABEL_152:
    sub_B91220(v78 + 48, v88);
    goto LABEL_153;
  }
  if ( v83 != v121 )
  {
    v88 = *(_QWORD *)(v78 + 48);
    if ( v88 )
      goto LABEL_152;
  }
LABEL_133:
  v84 = *(_QWORD *)(v104 + 56);
  for ( m = v104 + 48; m != v84; v84 = *(_QWORD *)(v84 + 8) )
  {
    if ( !v84 )
      BUG();
    if ( *(_BYTE *)(v84 - 24) != 84 )
      break;
    if ( (*(_DWORD *)(v84 - 20) & 0x7FFFFFF) != 0 )
    {
      v86 = 0;
      while ( 1 )
      {
        v87 = v86;
        if ( v105 == *(_QWORD *)(*(_QWORD *)(v84 - 32) + 32LL * *(unsigned int *)(v84 + 48) + 8 * v86) )
          break;
        if ( (*(_DWORD *)(v84 - 20) & 0x7FFFFFF) == (_DWORD)++v86 )
          goto LABEL_148;
      }
    }
    else
    {
LABEL_148:
      v87 = -1;
    }
    sub_B48BF0(v84 - 24, v87, 1);
  }
  sub_2CC0800(*(_QWORD *)(a1 + 176), *(_QWORD *)(a1 + 160), v102, v108, v100, v101, v96, v97, v114);
  nullsub_61();
  v138 = &unk_49DA100;
  nullsub_63();
  if ( v123 != v125 )
    _libc_free((unsigned __int64)v123);
  if ( v117 )
    sub_B91220((__int64)&v117, v117);
  if ( v118 )
    j_j___libc_free_0((unsigned __int64)v118);
  return v7;
}
