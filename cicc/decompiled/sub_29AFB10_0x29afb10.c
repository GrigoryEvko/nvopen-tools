// Function: sub_29AFB10
// Address: 0x29afb10
//
__int64 __fastcall sub_29AFB10(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        char a5,
        _QWORD *a6,
        __int64 a7,
        __int64 a8,
        char a9,
        char a10,
        __int64 a11,
        __int64 a12,
        char a13)
{
  __int64 *v14; // rbx
  __int64 *v15; // r15
  __int64 v16; // r12
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // r8
  int v20; // ecx
  _QWORD *v21; // rax
  __int64 v22; // rdx
  int v23; // r10d
  int v24; // edx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  unsigned __int64 *v27; // r10
  unsigned int v28; // ebx
  __int64 v29; // r12
  __int64 v30; // r14
  __int64 v31; // r8
  __int64 *v32; // r13
  __int64 v33; // rbx
  unsigned __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // r15
  __int64 *v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rbx
  __int64 *v40; // rax
  __int64 v41; // rax
  int v42; // r8d
  char v43; // dl
  __int64 v44; // rcx
  unsigned int v45; // edx
  __int64 v46; // rsi
  char v47; // bl
  unsigned __int64 *v48; // rax
  __int64 v49; // rax
  unsigned __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r8
  int v53; // edi
  __int64 v54; // rax
  int v55; // esi
  int v56; // ecx
  bool v57; // dl
  char v58; // dl
  __int64 v59; // rdx
  __int64 *v60; // rsi
  _QWORD *v61; // r15
  __int64 v62; // r13
  _QWORD *v63; // r12
  _BYTE *v64; // rbx
  __int64 v65; // rax
  __int64 v66; // rcx
  int v67; // edi
  unsigned int v68; // edx
  _QWORD *v69; // rcx
  __int64 v70; // rdi
  __int64 v71; // rsi
  int i; // r11d
  __int64 v74; // rdx
  __int64 v75; // r10
  __int64 v76; // rsi
  unsigned int v77; // r14d
  __int64 *v78; // r14
  __int64 *v79; // rdx
  __int64 v80; // rsi
  unsigned int v81; // ecx
  __int64 v82; // r10
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r10
  unsigned int v86; // ecx
  __int64 v87; // r14
  int v88; // esi
  int v89; // r11d
  __int64 v90; // rsi
  __int64 v91; // rcx
  __int64 v92; // rdx
  int v93; // eax
  __int64 v94; // rdi
  int v95; // esi
  _QWORD *v96; // rcx
  _QWORD *v97; // rsi
  int v98; // r14d
  __int64 v99; // rcx
  int v100; // eax
  int v101; // r11d
  int v102; // r10d
  __int64 v103; // r10
  unsigned int v104; // edx
  __int64 v105; // r14
  int v106; // esi
  int v107; // r11d
  __int64 v108; // rsi
  __int64 v109; // r10
  unsigned int v110; // edx
  __int64 v111; // rsi
  int v112; // ecx
  __int64 v113; // rdx
  int v114; // edx
  int v115; // ecx
  int v116; // r11d
  __int64 v117; // rcx
  int v118; // eax
  __int64 v119; // [rsp+8h] [rbp-1C8h]
  __int64 *v120; // [rsp+10h] [rbp-1C0h]
  __int64 *v122; // [rsp+30h] [rbp-1A0h]
  __int64 v123; // [rsp+40h] [rbp-190h] BYREF
  _QWORD *v124; // [rsp+48h] [rbp-188h]
  __int64 v125; // [rsp+50h] [rbp-180h]
  __int64 v126; // [rsp+58h] [rbp-178h]
  void *src; // [rsp+60h] [rbp-170h] BYREF
  __int64 v128; // [rsp+68h] [rbp-168h]
  __int64 *v129; // [rsp+70h] [rbp-160h] BYREF
  __int64 v130; // [rsp+78h] [rbp-158h]
  _BYTE v131[128]; // [rsp+80h] [rbp-150h] BYREF
  __int64 v132; // [rsp+100h] [rbp-D0h] BYREF
  __int64 *v133; // [rsp+108h] [rbp-C8h]
  __int64 v134; // [rsp+110h] [rbp-C0h]
  int v135; // [rsp+118h] [rbp-B8h]
  char v136; // [rsp+11Ch] [rbp-B4h]
  char v137; // [rsp+120h] [rbp-B0h] BYREF

  v14 = a2;
  v15 = &a2[a3];
  *(_QWORD *)a1 = a4;
  if ( !a5 )
    a5 = qword_5007AC8;
  *(_BYTE *)(a1 + 48) = a9;
  *(_QWORD *)(a1 + 24) = a7;
  *(_QWORD *)(a1 + 32) = a8;
  *(_BYTE *)(a1 + 8) = a5;
  *(_QWORD *)(a1 + 16) = a6;
  *(_QWORD *)(a1 + 40) = a11;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  src = &v129;
  v128 = 0;
  if ( v15 == a2 )
  {
    v27 = (unsigned __int64 *)&v129;
    v57 = 1;
    v56 = 0;
    v55 = 0;
    v53 = 0;
    v52 = 0;
    v54 = 1;
    v28 = 0;
    goto LABEL_147;
  }
  do
  {
    v16 = *v14;
    if ( a4 )
    {
      if ( v16 )
      {
        v17 = (unsigned int)(*(_DWORD *)(v16 + 44) + 1);
        v18 = *(_DWORD *)(v16 + 44) + 1;
      }
      else
      {
        v17 = 0;
        v18 = 0;
      }
      if ( v18 >= *(_DWORD *)(a4 + 32) || !*(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v17) )
        goto LABEL_23;
    }
    if ( (_DWORD)v126 )
    {
      v19 = (__int64)v124;
      v20 = (v126 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v21 = &v124[v20];
      v22 = *v21;
      if ( v16 == *v21 )
LABEL_11:
        BUG();
      v23 = 1;
      a6 = 0;
      while ( v22 != -4096 )
      {
        if ( v22 != -8192 || a6 )
          v21 = a6;
        v20 = (v126 - 1) & (v23 + v20);
        v22 = v124[v20];
        if ( v16 == v22 )
          goto LABEL_11;
        ++v23;
        a6 = v21;
        v21 = &v124[v20];
      }
      if ( !a6 )
        a6 = v21;
      ++v123;
      v24 = v125 + 1;
      if ( 4 * ((int)v125 + 1) < (unsigned int)(3 * v126) )
      {
        if ( (int)v126 - HIDWORD(v125) - v24 <= (unsigned int)v126 >> 3 )
        {
          sub_CF28B0((__int64)&v123, v126);
          if ( !(_DWORD)v126 )
          {
LABEL_209:
            LODWORD(v125) = v125 + 1;
            BUG();
          }
          v19 = (__int64)v124;
          v97 = 0;
          v98 = (v126 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          a6 = &v124[v98];
          v99 = *a6;
          v24 = v125 + 1;
          v100 = 1;
          if ( v16 != *a6 )
          {
            while ( v99 != -4096 )
            {
              if ( v99 == -8192 && !v97 )
                v97 = a6;
              v98 = (v126 - 1) & (v100 + v98);
              a6 = &v124[v98];
              v99 = *a6;
              if ( v16 == *a6 )
                goto LABEL_18;
              ++v100;
            }
            if ( v97 )
              a6 = v97;
          }
        }
        goto LABEL_18;
      }
    }
    else
    {
      ++v123;
    }
    sub_CF28B0((__int64)&v123, 2 * v126);
    if ( !(_DWORD)v126 )
      goto LABEL_209;
    v19 = (unsigned int)(v126 - 1);
    v93 = v19 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    a6 = &v124[v93];
    v24 = v125 + 1;
    v94 = *a6;
    if ( v16 != *a6 )
    {
      v95 = 1;
      v96 = 0;
      while ( v94 != -4096 )
      {
        if ( !v96 && v94 == -8192 )
          v96 = a6;
        v93 = v19 & (v95 + v93);
        a6 = &v124[v93];
        v94 = *a6;
        if ( v16 == *a6 )
          goto LABEL_18;
        ++v95;
      }
      if ( v96 )
        a6 = v96;
    }
LABEL_18:
    LODWORD(v125) = v24;
    if ( *a6 != -4096 )
      --HIDWORD(v125);
    *a6 = v16;
    v25 = (unsigned int)v128;
    v26 = (unsigned int)v128 + 1LL;
    if ( v26 > HIDWORD(v128) )
    {
      sub_C8D5F0((__int64)&src, &v129, v26, 8u, v19, (__int64)a6);
      v25 = (unsigned int)v128;
    }
    *((_QWORD *)src + v25) = v16;
    LODWORD(v128) = v128 + 1;
LABEL_23:
    ++v14;
  }
  while ( v15 != v14 );
  v27 = (unsigned __int64 *)src;
  v28 = v128;
  v120 = (__int64 *)((char *)src + 8 * (unsigned int)v128);
  if ( src == v120 )
  {
    v52 = (__int64)v124;
    v53 = v125;
    v55 = HIDWORD(v125);
    v54 = v123 + 1;
    v56 = v126;
    v57 = (_DWORD)v128 == 0;
    goto LABEL_147;
  }
  v122 = (__int64 *)src;
  do
  {
    v29 = *v122;
    if ( (*(_WORD *)(*v122 + 2) & 0x7FFF) != 0 )
    {
      v48 = (unsigned __int64 *)src;
LABEL_96:
      v27 = v48;
      *(_OWORD *)(a1 + 88) = 0;
      *(_QWORD *)(a1 + 88) = a1 + 104;
      *(_OWORD *)(a1 + 56) = 0;
      *(_OWORD *)(a1 + 72) = 0;
LABEL_97:
      if ( v27 != (unsigned __int64 *)&v129 )
        _libc_free((unsigned __int64)v27);
      v70 = (__int64)v124;
      v71 = 8LL * (unsigned int)v126;
      goto LABEL_100;
    }
    v30 = *(_QWORD *)(v29 + 56);
    v31 = 0;
    v32 = (__int64 *)v131;
    v133 = (__int64 *)&v137;
    v33 = v29 + 48;
    v34 = 16;
    v130 = 0x1000000000LL;
    v35 = 0;
    v132 = 0;
    v134 = 16;
    v135 = 0;
    v136 = 1;
    v129 = (__int64 *)v131;
    if ( v30 == v29 + 48 )
      goto LABEL_54;
    while ( 1 )
    {
      v36 = v30 - 24;
      a6 = (_QWORD *)(v35 + 1);
      if ( !v30 )
        v36 = 0;
      if ( (unsigned __int64)a6 > v34 )
      {
        sub_C8D5F0((__int64)&v129, v131, v35 + 1, 8u, 0, (__int64)a6);
        v35 = (unsigned int)v130;
        v31 = 0;
      }
      v129[v35] = v36;
      v35 = (unsigned int)(v130 + 1);
      LODWORD(v130) = v130 + 1;
      v30 = *(_QWORD *)(v30 + 8);
      if ( v33 == v30 )
        break;
      v34 = HIDWORD(v130);
    }
    if ( (_DWORD)v35 )
    {
      v119 = v29 + 48;
      while ( 1 )
      {
        v37 = v129;
        v38 = (unsigned int)v35;
        v39 = v129[(unsigned int)v35 - 1];
        LODWORD(v130) = v35 - 1;
        if ( v136 )
        {
          v40 = v133;
          v38 = HIDWORD(v134);
          v37 = &v133[HIDWORD(v134)];
          if ( v133 != v37 )
          {
            while ( v39 != *v40 )
            {
              if ( v37 == ++v40 )
                goto LABEL_82;
            }
            goto LABEL_42;
          }
LABEL_82:
          if ( HIDWORD(v134) < (unsigned int)v134 )
          {
            ++HIDWORD(v134);
            *v37 = v39;
            ++v132;
LABEL_69:
            if ( *(_BYTE *)v39 == 4 )
              goto LABEL_56;
            v35 = (unsigned int)v130;
            if ( *(_BYTE *)v39 <= 0x1Cu || v29 == *(_QWORD *)(v39 + 40) )
            {
              v59 = 4LL * (*(_DWORD *)(v39 + 4) & 0x7FFFFFF);
              a6 = (_QWORD *)(v39 - v59 * 8);
              if ( (*(_BYTE *)(v39 + 7) & 0x40) != 0 )
              {
                a6 = *(_QWORD **)(v39 - 8);
                v39 = (__int64)&a6[v59];
              }
              if ( (_QWORD *)v39 != a6 )
              {
                v60 = v32;
                v61 = a6;
                v62 = v29;
                v63 = (_QWORD *)v39;
                do
                {
                  v64 = (_BYTE *)*v61;
                  if ( (unsigned __int8)(*(_BYTE *)*v61 - 22) > 6u )
                  {
                    if ( v35 + 1 > (unsigned __int64)HIDWORD(v130) )
                    {
                      sub_C8D5F0((__int64)&v129, v60, v35 + 1, 8u, v31, (__int64)a6);
                      v35 = (unsigned int)v130;
                    }
                    v129[v35] = (__int64)v64;
                    v35 = (unsigned int)(v130 + 1);
                    LODWORD(v130) = v130 + 1;
                  }
                  v61 += 4;
                }
                while ( v63 != v61 );
                v29 = v62;
                v32 = v60;
              }
            }
            goto LABEL_43;
          }
        }
        sub_C8CC70((__int64)&v132, v39, (__int64)v37, v38, v31, (__int64)a6);
        if ( v58 )
          goto LABEL_69;
LABEL_42:
        LODWORD(v35) = v130;
LABEL_43:
        if ( !(_DWORD)v35 )
        {
          v33 = v119;
          break;
        }
      }
    }
    v41 = *(_QWORD *)(v29 + 56);
    if ( v33 == v41 )
      goto LABEL_54;
    a6 = v124;
    v42 = v126 - 1;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v41 )
          goto LABEL_208;
        v43 = *(_BYTE *)(v41 - 24);
        if ( v43 == 60 )
        {
          if ( !a10 )
            goto LABEL_56;
          goto LABEL_53;
        }
        if ( v43 == 34 )
        {
          v44 = *(_QWORD *)(v41 - 88);
          if ( v44 )
          {
            if ( !(_DWORD)v126 )
              goto LABEL_56;
            v45 = v42 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
            v46 = v124[v45];
            if ( v44 != v46 )
            {
              v102 = 1;
              while ( v46 != -4096 )
              {
                v45 = v42 & (v102 + v45);
                v46 = v124[v45];
                if ( v44 == v46 )
                  goto LABEL_53;
                ++v102;
              }
              goto LABEL_56;
            }
          }
          goto LABEL_53;
        }
        if ( v43 != 39 )
          break;
        v74 = *(_QWORD *)(v41 - 32);
        if ( (*(_BYTE *)(v41 - 22) & 1) != 0 )
        {
          v75 = *(_QWORD *)(v74 + 32);
          if ( v75 )
          {
            if ( !(_DWORD)v126 )
              goto LABEL_56;
            v76 = v124[v42 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4))];
            v77 = v42 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
            if ( v75 != v76 )
            {
              v115 = 1;
              while ( v76 != -4096 )
              {
                v116 = v115 + 1;
                v117 = v42 & (v77 + v115);
                v76 = v124[v117];
                v77 = v117;
                if ( v75 == v76 )
                  goto LABEL_109;
                v115 = v116;
              }
              goto LABEL_56;
            }
          }
LABEL_109:
          v78 = (__int64 *)(v74 + 32LL * (*(_DWORD *)(v41 - 20) & 0x7FFFFFF));
          v79 = (__int64 *)(v74 + 64);
        }
        else
        {
          v78 = (__int64 *)(v74 + 32LL * (*(_DWORD *)(v41 - 20) & 0x7FFFFFF));
          v79 = (__int64 *)(v74 + 32);
        }
        if ( v79 == v78 )
          goto LABEL_53;
        do
        {
          v80 = *v79;
          if ( !(_DWORD)v126 )
            goto LABEL_56;
          v81 = v42 & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
          v82 = v124[v81];
          if ( v82 != v80 )
          {
            v101 = 1;
            while ( v82 != -4096 )
            {
              v81 = v42 & (v101 + v81);
              v82 = v124[v81];
              if ( v80 == v82 )
                goto LABEL_113;
              ++v101;
            }
            goto LABEL_56;
          }
LABEL_113:
          v79 += 4;
        }
        while ( v78 != v79 );
        v41 = *(_QWORD *)(v41 + 8);
        if ( v33 == v41 )
        {
LABEL_54:
          v47 = 1;
          goto LABEL_57;
        }
      }
      if ( v43 != 81 )
        break;
      v83 = *(_QWORD *)(v41 - 8);
      if ( v83 )
      {
        v84 = *(_QWORD *)(v83 + 24);
        if ( *(_BYTE *)v84 == 38 )
          goto LABEL_121;
        while ( 1 )
        {
          do
          {
LABEL_119:
            v83 = *(_QWORD *)(v83 + 8);
            if ( !v83 )
              goto LABEL_53;
            v84 = *(_QWORD *)(v83 + 24);
          }
          while ( *(_BYTE *)v84 != 38 );
LABEL_121:
          v85 = *(_QWORD *)(v84 + 40);
          if ( !(_DWORD)v126 )
            goto LABEL_56;
          v86 = v42 & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
          v87 = v124[v86];
          v88 = 1;
          if ( v85 != v87 )
          {
            while ( v87 != -4096 )
            {
              v89 = v88 + 1;
              v90 = v42 & (v86 + v88);
              v87 = v124[v90];
              v86 = v90;
              if ( v85 == v87 )
                goto LABEL_119;
              v88 = v89;
            }
            goto LABEL_56;
          }
        }
      }
LABEL_53:
      v41 = *(_QWORD *)(v41 + 8);
      if ( v33 == v41 )
        goto LABEL_54;
    }
    if ( v43 == 80 )
    {
      v91 = *(_QWORD *)(v41 - 8);
      if ( v91 )
      {
        while ( 1 )
        {
          v92 = *(_QWORD *)(v91 + 24);
          if ( *(_BYTE *)v92 == 37 )
          {
            v103 = *(_QWORD *)(v92 + 40);
            if ( !(_DWORD)v126 )
              goto LABEL_56;
            v104 = v42 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
            v105 = v124[v104];
            v106 = 1;
            if ( v103 != v105 )
              break;
          }
LABEL_130:
          v91 = *(_QWORD *)(v91 + 8);
          if ( !v91 )
            goto LABEL_53;
        }
        while ( v105 != -4096 )
        {
          v107 = v106 + 1;
          v108 = v42 & (v104 + v106);
          v105 = v124[v108];
          v104 = v108;
          if ( v103 == v105 )
            goto LABEL_130;
          v106 = v107;
        }
        goto LABEL_56;
      }
      goto LABEL_53;
    }
    if ( v43 != 37 )
    {
      if ( v43 == 85 )
      {
        v113 = *(_QWORD *)(v41 - 56);
        if ( v113 )
        {
          if ( !*(_BYTE *)v113 && *(_QWORD *)(v113 + 24) == *(_QWORD *)(v41 + 56) )
          {
            v114 = *(_DWORD *)(v113 + 36);
            if ( v114 == 375 )
            {
              if ( !a9 )
                goto LABEL_56;
            }
            else if ( v114 == 86 )
            {
              goto LABEL_56;
            }
          }
        }
      }
      goto LABEL_53;
    }
    if ( (*(_BYTE *)(v41 - 22) & 1) == 0 )
      goto LABEL_53;
    v109 = *(_QWORD *)(v41 + 32 * (1LL - (*(_DWORD *)(v41 - 20) & 0x7FFFFFF)) - 24);
    if ( !v109 )
      goto LABEL_53;
    if ( !(_DWORD)v126 )
      goto LABEL_56;
    v110 = v42 & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
    v111 = v124[v110];
    if ( v109 == v111 )
      goto LABEL_53;
    v112 = 1;
    while ( v111 != -4096 )
    {
      v110 = v42 & (v112 + v110);
      v111 = v124[v110];
      if ( v109 == v111 )
        goto LABEL_53;
      ++v112;
    }
LABEL_56:
    v47 = 0;
LABEL_57:
    if ( v129 != v32 )
      _libc_free((unsigned __int64)v129);
    if ( !v136 )
      _libc_free((unsigned __int64)v133);
    v27 = (unsigned __int64 *)src;
    v48 = (unsigned __int64 *)src;
    if ( !v47 )
      goto LABEL_96;
    if ( v29 == *(_QWORD *)src )
    {
      v49 = sub_AA4FF0(v29);
      if ( !v49 )
LABEL_208:
        BUG();
      v50 = (unsigned int)*(unsigned __int8 *)(v49 - 24) - 39;
      if ( (unsigned int)v50 <= 0x38 )
      {
        v51 = 0x100060000000001LL;
        if ( _bittest64(&v51, v50) )
        {
          v27 = (unsigned __int64 *)src;
          *(_OWORD *)(a1 + 88) = 0;
          *(_OWORD *)(a1 + 56) = 0;
          *(_QWORD *)(a1 + 88) = a1 + 104;
          *(_OWORD *)(a1 + 72) = 0;
          goto LABEL_97;
        }
      }
    }
    else
    {
      v65 = *(_QWORD *)(v29 + 16);
      if ( v65 )
      {
        while ( 1 )
        {
          v66 = *(_QWORD *)(v65 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v66 - 30) <= 0xAu )
            break;
          v65 = *(_QWORD *)(v65 + 8);
          if ( !v65 )
            goto LABEL_66;
        }
        v67 = v126 - 1;
LABEL_91:
        v69 = *(_QWORD **)(v66 + 40);
        if ( !(_DWORD)v126 )
        {
LABEL_103:
          *(_OWORD *)(a1 + 88) = 0;
          *(_OWORD *)(a1 + 56) = 0;
          *(_QWORD *)(a1 + 88) = a1 + 104;
          *(_OWORD *)(a1 + 72) = 0;
          goto LABEL_97;
        }
        v68 = v67 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
        a6 = (_QWORD *)v124[v68];
        if ( v69 == a6 )
          goto LABEL_89;
        for ( i = 1; ; ++i )
        {
          if ( a6 == (_QWORD *)-4096LL )
            goto LABEL_103;
          v68 = v67 & (i + v68);
          a6 = (_QWORD *)v124[v68];
          if ( v69 == a6 )
            break;
        }
LABEL_89:
        while ( 1 )
        {
          v65 = *(_QWORD *)(v65 + 8);
          if ( !v65 )
            break;
          v66 = *(_QWORD *)(v65 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v66 - 30) <= 0xAu )
            goto LABEL_91;
        }
      }
    }
LABEL_66:
    ++v122;
  }
  while ( v120 != v122 );
  v28 = v128;
  v52 = (__int64)v124;
  v53 = v125;
  v54 = v123 + 1;
  v55 = HIDWORD(v125);
  v56 = v126;
  v27 = (unsigned __int64 *)src;
  v57 = (_DWORD)v128 == 0;
LABEL_147:
  v123 = v54;
  v124 = 0;
  *(_DWORD *)(a1 + 72) = v53;
  *(_DWORD *)(a1 + 76) = v55;
  *(_QWORD *)(a1 + 56) = 1;
  *(_QWORD *)(a1 + 64) = v52;
  *(_DWORD *)(a1 + 80) = v56;
  v125 = 0;
  LODWORD(v126) = 0;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 96) = 0;
  if ( (void **)(a1 + 88) == &src || v57 )
    goto LABEL_97;
  if ( v27 == (unsigned __int64 *)&v129 )
  {
    sub_C8D5F0(a1 + 88, (const void *)(a1 + 104), v28, 8u, v52, (__int64)a6);
    v27 = (unsigned __int64 *)src;
    if ( 8LL * (unsigned int)v128 )
    {
      memcpy(*(void **)(a1 + 88), src, 8LL * (unsigned int)v128);
      v27 = (unsigned __int64 *)src;
    }
    *(_DWORD *)(a1 + 96) = v28;
    goto LABEL_97;
  }
  v118 = HIDWORD(v128);
  *(_QWORD *)(a1 + 88) = v27;
  v70 = 0;
  v71 = 0;
  *(_DWORD *)(a1 + 96) = v28;
  *(_DWORD *)(a1 + 100) = v118;
LABEL_100:
  sub_C7D6A0(v70, v71, 8);
  *(_QWORD *)(a1 + 104) = a1 + 120;
  *(_QWORD *)(a1 + 112) = 0x600000000LL;
  *(_QWORD *)(a1 + 168) = a1 + 184;
  sub_29AA440((__int64 *)(a1 + 168), *(_BYTE **)a12, *(_QWORD *)a12 + *(_QWORD *)(a12 + 8));
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_BYTE *)(a1 + 200) = a13;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = a1 + 256;
  *(_QWORD *)(a1 + 248) = 0;
  return a1 + 256;
}
