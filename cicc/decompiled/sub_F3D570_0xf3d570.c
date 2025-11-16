// Function: sub_F3D570
// Address: 0xf3d570
//
__int64 __fastcall sub_F3D570(__int64 a1)
{
  __int64 v1; // rcx
  __int64 v2; // rdi
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  _QWORD *i; // rcx
  __m128i *v6; // rbx
  unsigned __int8 **v7; // r15
  __int64 v8; // r13
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // r12
  int v17; // r14d
  __m128i *v18; // rax
  __int64 v19; // rbx
  __int64 v20; // r14
  unsigned __int8 **v21; // r12
  __int64 v22; // r15
  __int64 v23; // r13
  __int64 v24; // rdi
  _QWORD **v25; // rbx
  _QWORD **v26; // r12
  _QWORD *v27; // rdi
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // rdi
  __int64 v31; // rsi
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  int v37; // r12d
  __m128i *v38; // r14
  __int64 v39; // r12
  int v40; // eax
  __int64 v41; // r15
  __int64 v42; // rax
  int v43; // eax
  __int64 v44; // r12
  __int64 v45; // r14
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // r8
  __int64 v50; // r9
  unsigned __int8 v51; // dl
  __int64 v52; // rcx
  __int64 v53; // r13
  __int64 v54; // rdx
  int v55; // ebx
  __int64 v56; // rbx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  unsigned __int8 **v60; // rdx
  unsigned __int8 **v61; // rdx
  _QWORD **v62; // rbx
  _QWORD **v63; // r12
  _QWORD *v64; // rdi
  __int64 v65; // rax
  int v66; // edx
  __int64 v67; // rax
  __int64 v68; // rax
  unsigned __int64 v69; // rdx
  bool v70; // al
  __int64 v71; // [rsp+28h] [rbp-1D8h]
  __int64 v72; // [rsp+30h] [rbp-1D0h]
  int v73; // [rsp+38h] [rbp-1C8h]
  unsigned __int8 v74; // [rsp+43h] [rbp-1BDh]
  int v75; // [rsp+44h] [rbp-1BCh]
  int v76; // [rsp+44h] [rbp-1BCh]
  int v77; // [rsp+44h] [rbp-1BCh]
  int v78; // [rsp+48h] [rbp-1B8h]
  bool v79; // [rsp+50h] [rbp-1B0h]
  int v80; // [rsp+50h] [rbp-1B0h]
  unsigned int v81; // [rsp+58h] [rbp-1A8h]
  __int64 v82; // [rsp+58h] [rbp-1A8h]
  bool v83; // [rsp+58h] [rbp-1A8h]
  __int64 v84; // [rsp+60h] [rbp-1A0h]
  unsigned int v85; // [rsp+60h] [rbp-1A0h]
  __int64 v86; // [rsp+68h] [rbp-198h]
  unsigned int v87; // [rsp+68h] [rbp-198h]
  __int64 v88; // [rsp+68h] [rbp-198h]
  unsigned __int8 *v89; // [rsp+70h] [rbp-190h] BYREF
  unsigned __int8 **v90; // [rsp+78h] [rbp-188h] BYREF
  __int64 v91; // [rsp+80h] [rbp-180h] BYREF
  __int64 v92; // [rsp+88h] [rbp-178h] BYREF
  __int64 v93; // [rsp+90h] [rbp-170h] BYREF
  __int64 v94; // [rsp+98h] [rbp-168h]
  __int64 v95; // [rsp+A0h] [rbp-160h]
  __int64 v96; // [rsp+A8h] [rbp-158h]
  __int64 v97; // [rsp+B0h] [rbp-150h] BYREF
  __int128 v98; // [rsp+B8h] [rbp-148h] BYREF
  char v99; // [rsp+C8h] [rbp-138h]
  __int64 v100; // [rsp+D0h] [rbp-130h]
  __m128i *v101; // [rsp+E0h] [rbp-120h] BYREF
  __int64 v102; // [rsp+E8h] [rbp-118h]
  __int64 v103; // [rsp+F0h] [rbp-110h]
  __int64 v104; // [rsp+F8h] [rbp-108h]
  __int64 v105; // [rsp+100h] [rbp-100h]
  __m128i v106; // [rsp+110h] [rbp-F0h] BYREF
  __int64 v107; // [rsp+120h] [rbp-E0h]
  __int64 v108; // [rsp+128h] [rbp-D8h]
  _QWORD *v109; // [rsp+130h] [rbp-D0h]
  __int64 v110; // [rsp+140h] [rbp-C0h] BYREF
  unsigned __int8 **v111; // [rsp+148h] [rbp-B8h]
  char v112; // [rsp+158h] [rbp-A8h]
  __int64 v113; // [rsp+160h] [rbp-A0h]
  _BYTE *v114; // [rsp+180h] [rbp-80h] BYREF
  __int64 v115; // [rsp+188h] [rbp-78h]
  _BYTE v116[112]; // [rsp+190h] [rbp-70h] BYREF

  v1 = *(_QWORD *)(a1 + 56);
  v71 = a1 + 48;
  v74 = *(_BYTE *)(a1 + 40);
  v114 = v116;
  v72 = v1;
  v115 = 0x800000000LL;
  if ( v74 )
  {
    v93 = 0;
    v94 = 0;
    v95 = 0;
    v96 = 0;
    if ( v1 == a1 + 48 )
    {
      v74 = 0;
      v30 = 0;
      v31 = 0;
      goto LABEL_35;
    }
    while ( 1 )
    {
      if ( !v72 )
        BUG();
      v2 = *(_QWORD *)(v72 + 40);
      if ( v2 )
      {
        v3 = (_QWORD *)sub_B14240(v2);
        for ( i = v4; v3 != v4; v3 = (_QWORD *)v3[1] )
        {
          if ( !*((_BYTE *)v3 + 32) )
            break;
        }
      }
      else
      {
        i = &qword_4F81430[1];
        v3 = &qword_4F81430[1];
      }
      v6 = &v106;
      v106.m128i_i64[0] = (__int64)v3;
      v106.m128i_i64[1] = (__int64)i;
      v108 = (__int64)i;
      v109 = i;
      sub_F333F0((__int64)&v110, v106.m128i_i64);
      v7 = v111;
      v8 = v110;
      v84 = v113;
      if ( v113 != v110 )
        break;
LABEL_29:
      v72 = *(_QWORD *)(v72 + 8);
      if ( v72 == v71 )
      {
        v25 = (_QWORD **)v114;
        v26 = (_QWORD **)&v114[8 * (unsigned int)v115];
        if ( v114 == (_BYTE *)v26 )
        {
          v28 = v94;
          v74 = (_DWORD)v115 != 0;
          v29 = v96;
          v31 = 40LL * (unsigned int)v96;
          v30 = v94;
        }
        else
        {
          do
          {
            v27 = *v25++;
            sub_B14290(v27);
          }
          while ( v26 != v25 );
          v28 = v94;
          v29 = v96;
          v74 = (_DWORD)v115 != 0;
          v30 = v94;
          v31 = 40LL * (unsigned int)v96;
        }
        if ( v29 )
        {
          v30 = v28;
          v31 = 40LL * (unsigned int)v96;
        }
        goto LABEL_35;
      }
    }
    while ( 1 )
    {
      v9 = *(_BYTE *)(v8 + 64);
      if ( v9 == 1 )
        break;
      if ( v9 == 2 )
      {
        v10 = sub_B13870(v8);
        v11 = sub_AE9410(v10);
        v79 = v11 == v12;
LABEL_13:
        v13 = *(_QWORD *)(v8 + 24);
        v106.m128i_i64[0] = v13;
        if ( v13 )
          sub_B96E90((__int64)v6, v13, 1);
        v14 = sub_B10D40((__int64)v6);
        v15 = sub_B12000(v8 + 72);
        v99 = 0;
        v97 = v15;
        v100 = v14;
        if ( v106.m128i_i64[0] )
          sub_B91220((__int64)v6, v106.m128i_i64[0]);
        v16 = v94;
        v17 = v96;
        v86 = v94 + 40LL * (unsigned int)v96;
        if ( !(_DWORD)v96 )
          goto LABEL_40;
        v106.m128i_i64[0] = 0;
        LOBYTE(v108) = 0;
        v109 = 0;
        LODWORD(v91) = 0;
        if ( v99 )
          LODWORD(v91) = WORD4(v98) | ((_DWORD)v98 << 16);
        v101 = (__m128i *)v100;
        v92 = v97;
        v75 = 1;
        v73 = v96 - 1;
        v81 = (v17 - 1) & sub_F11290(&v92, &v91, (__int64 *)&v101);
        v18 = v6;
        v19 = v97;
        v20 = v16;
        v21 = v7;
        v22 = v8;
        v23 = (__int64)v18;
        while ( 1 )
        {
          v24 = v20 + 40LL * v81;
          if ( *(_QWORD *)v24 == v19
            && v99 == *(_BYTE *)(v24 + 24)
            && (!v99 || v98 == *(_OWORD *)(v24 + 8))
            && v100 == *(_QWORD *)(v24 + 32) )
          {
            break;
          }
          if ( sub_F34140(v24, v23) )
          {
            v6 = (__m128i *)v23;
            v8 = v22;
            v7 = v21;
            v24 = v94 + 40LL * (unsigned int)v96;
            goto LABEL_24;
          }
          v81 = v73 & (v75 + v81);
          ++v75;
        }
        v6 = (__m128i *)v23;
        v8 = v22;
        v7 = v21;
LABEL_24:
        if ( v24 == v86 )
        {
LABEL_40:
          if ( sub_B12EE0(v8) && v79 )
          {
            if ( *(_BYTE *)(v8 + 64) == 2 )
            {
              v35 = (unsigned int)v115;
              v36 = (unsigned int)v115 + 1LL;
              if ( v36 > HIDWORD(v115) )
              {
                sub_C8D5F0((__int64)&v114, v116, v36, 8u, v33, v34);
                v35 = (unsigned int)v115;
              }
              *(_QWORD *)&v114[8 * v35] = v8;
              LODWORD(v115) = v115 + 1;
            }
          }
          else
          {
            v37 = v96;
            if ( (_DWORD)v96 )
            {
              v101 = 0;
              LOBYTE(v104) = 0;
              v82 = v94;
              v105 = 0;
              v106 = 0u;
              v107 = 0;
              v108 = 1;
              v109 = 0;
              LODWORD(v90) = 0;
              if ( v99 )
                LODWORD(v90) = WORD4(v98) | ((_DWORD)v98 << 16);
              v38 = 0;
              v92 = v100;
              v91 = v97;
              v76 = v96 - 1;
              v80 = 1;
              v87 = (v37 - 1) & sub_F11290(&v91, &v90, &v92);
              while ( 1 )
              {
                v39 = v82 + 40LL * v87;
                if ( sub_F34140((__int64)&v97, v39) )
                  break;
                if ( sub_F34140(v39, (__int64)&v101) )
                {
                  v37 = v96;
                  if ( !v38 )
                    v38 = (__m128i *)(v82 + 40LL * v87);
                  ++v93;
                  v40 = v95 + 1;
                  v101 = v38;
                  if ( 4 * ((int)v95 + 1) >= (unsigned int)(3 * v96) )
                    goto LABEL_63;
                  if ( (int)v96 - (v40 + HIDWORD(v95)) > (unsigned int)v96 >> 3 )
                    goto LABEL_55;
                  sub_F3D0B0((__int64)&v93, v96);
                  goto LABEL_64;
                }
                v70 = sub_F34140(v39, (__int64)v6);
                if ( v38 || !v70 )
                  v39 = (__int64)v38;
                v38 = (__m128i *)v39;
                v87 = v76 & (v80 + v87);
                ++v80;
              }
            }
            else
            {
              ++v93;
              v101 = 0;
LABEL_63:
              sub_F3D0B0((__int64)&v93, 2 * v37);
LABEL_64:
              sub_F38F60((__int64)&v93, (__int64)&v97, (__int64 *)&v101);
              v38 = v101;
              v40 = v95 + 1;
LABEL_55:
              LODWORD(v95) = v40;
              v106.m128i_i64[0] = 0;
              LOBYTE(v108) = 0;
              v109 = 0;
              if ( !sub_F34140((__int64)v38, (__int64)v6) )
                --HIDWORD(v95);
              *v38 = _mm_loadu_si128((const __m128i *)&v97);
              v38[1] = _mm_loadu_si128((const __m128i *)((char *)&v98 + 8));
              v38[2].m128i_i64[0] = v100;
            }
          }
        }
        goto LABEL_27;
      }
      do
LABEL_27:
        v8 = *(_QWORD *)(v8 + 8);
      while ( (unsigned __int8 **)v8 != v7 && *(_BYTE *)(v8 + 32) );
      if ( v84 == v8 )
        goto LABEL_29;
    }
    v79 = v74;
    goto LABEL_13;
  }
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  if ( v1 == v71 )
  {
    v31 = 0;
    v30 = 0;
    goto LABEL_35;
  }
  v41 = v1;
  do
  {
    while ( 1 )
    {
      if ( !v41 )
        BUG();
      if ( *(_BYTE *)(v41 - 24) != 85 )
        goto LABEL_68;
      v42 = *(_QWORD *)(v41 - 56);
      if ( !v42 )
        goto LABEL_68;
      if ( *(_BYTE *)v42 )
        goto LABEL_68;
      if ( *(_QWORD *)(v42 + 24) != *(_QWORD *)(v41 + 56) )
        goto LABEL_68;
      if ( (*(_BYTE *)(v42 + 33) & 0x20) == 0 )
        goto LABEL_68;
      v43 = *(_DWORD *)(v42 + 36);
      v83 = v43 == 68 || v43 == 71;
      if ( !v83 )
        goto LABEL_68;
      v44 = v41 - 24;
      v45 = 0;
      if ( v43 == 68 )
      {
        v45 = v41 - 24;
        v46 = sub_AE9410(*(_QWORD *)(*(_QWORD *)(v44 + 32 * (3LL - (*(_DWORD *)(v41 - 20) & 0x7FFFFFF))) + 24LL));
        v83 = v47 == v46;
      }
      v48 = sub_B10CD0(v41 + 24);
      v51 = *(_BYTE *)(v48 - 16);
      if ( (v51 & 2) == 0 )
      {
        if ( ((*(_WORD *)(v48 - 16) >> 6) & 0xF) != 2 )
          goto LABEL_80;
        v67 = v48 - 16 - 8LL * ((v51 >> 2) & 0xF);
LABEL_107:
        v52 = *(_QWORD *)(v67 + 8);
        goto LABEL_81;
      }
      if ( *(_DWORD *)(v48 - 24) == 2 )
      {
        v67 = *(_QWORD *)(v48 - 32);
        goto LABEL_107;
      }
LABEL_80:
      v52 = 0;
LABEL_81:
      v53 = v102;
      v54 = *(_QWORD *)(*(_QWORD *)(v44 + 32 * (1LL - (*(_DWORD *)(v41 - 20) & 0x7FFFFFF))) + 24LL);
      LOBYTE(v108) = 0;
      v109 = (_QWORD *)v52;
      v55 = v104;
      v106.m128i_i64[0] = v54;
      v88 = v102 + 40LL * (unsigned int)v104;
      if ( (_DWORD)v104 )
      {
        v93 = v54;
        v97 = v52;
        v110 = 0;
        v112 = 0;
        v113 = 0;
        LODWORD(v92) = 0;
        v78 = 1;
        v77 = v104 - 1;
        v85 = (v55 - 1) & sub_F11290(&v93, &v92, &v97);
        while ( 1 )
        {
          v56 = v53 + 40LL * v85;
          if ( sub_F34140((__int64)&v106, v56) )
            break;
          if ( sub_F34140(v56, (__int64)&v110) )
          {
            v44 = v41 - 24;
            v50 = v102 + 40LL * (unsigned int)v104;
            goto LABEL_86;
          }
          v85 = v77 & (v78 + v85);
          ++v78;
        }
        v44 = v41 - 24;
        v50 = v53 + 40LL * v85;
        if ( !v56 )
          v50 = v102 + 40LL * (unsigned int)v104;
LABEL_86:
        if ( v50 != v88 )
          goto LABEL_68;
      }
      v57 = *(_DWORD *)(v41 - 20) & 0x7FFFFFF;
      v58 = *(_QWORD *)(*(_QWORD *)(v44 - 32 * v57) + 24LL);
      v89 = (unsigned __int8 *)v58;
      if ( *(_BYTE *)v58 != 4 )
        break;
      if ( *(_DWORD *)(v58 + 144) || (unsigned __int8)sub_AF4500(*(_QWORD *)(*(_QWORD *)(v44 + 32 * (2 - v57)) + 24LL)) )
        goto LABEL_89;
LABEL_97:
      if ( !v83 )
        goto LABEL_98;
      if ( v45 )
      {
        v68 = (unsigned int)v115;
        v69 = (unsigned int)v115 + 1LL;
        if ( v69 > HIDWORD(v115) )
        {
          sub_C8D5F0((__int64)&v114, v116, v69, 8u, v49, v50);
          v68 = (unsigned int)v115;
        }
        *(_QWORD *)&v114[8 * v68] = v45;
        LODWORD(v115) = v115 + 1;
      }
LABEL_68:
      v41 = *(_QWORD *)(v41 + 8);
      if ( v41 == v71 )
        goto LABEL_99;
    }
    if ( (unsigned __int8)(*(_BYTE *)v58 - 5) <= 0x1Fu )
      goto LABEL_97;
LABEL_89:
    sub_B58DC0(&v110, &v89);
    v59 = v110;
    v91 = v110;
    v90 = v111;
    v92 = v110;
    v93 = v110;
    v97 = v110;
    if ( v111 == (unsigned __int8 **)v110 )
      goto LABEL_98;
    while ( 1 )
    {
      v61 = (unsigned __int8 **)(v59 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v59 & 4) == 0 )
        break;
      if ( (unsigned int)**((unsigned __int8 **)*v61 + 17) - 12 <= 1 )
        goto LABEL_96;
      v59 = (unsigned __int64)(v61 + 1) | 4;
      v60 = (unsigned __int8 **)v59;
LABEL_93:
      if ( v60 == v111 )
        goto LABEL_98;
    }
    if ( (unsigned int)*v61[17] - 12 > 1 )
    {
      v60 = v61 + 18;
      v59 = (__int64)v60;
      goto LABEL_93;
    }
LABEL_96:
    if ( v111 != (unsigned __int8 **)v59 )
      goto LABEL_97;
LABEL_98:
    sub_F3D270((__int64)&v110, (__int64)&v101, &v106);
    v41 = *(_QWORD *)(v41 + 8);
  }
  while ( v41 != v71 );
LABEL_99:
  v62 = (_QWORD **)v114;
  v63 = (_QWORD **)&v114[8 * (unsigned int)v115];
  if ( v63 == (_QWORD **)v114 )
  {
    v65 = v102;
    v74 = (_DWORD)v115 != 0;
    v66 = v104;
    v31 = 40LL * (unsigned int)v104;
    v30 = v102;
  }
  else
  {
    do
    {
      v64 = *v62++;
      sub_B43D60(v64);
    }
    while ( v63 != v62 );
    v65 = v102;
    v66 = v104;
    v30 = v102;
    v31 = 40LL * (unsigned int)v104;
    v74 = (_DWORD)v115 != 0;
  }
  if ( v66 )
  {
    v30 = v65;
    v31 = 40LL * (unsigned int)v104;
  }
LABEL_35:
  sub_C7D6A0(v30, v31, 8);
  if ( v114 != v116 )
    _libc_free(v114, v31);
  return v74;
}
