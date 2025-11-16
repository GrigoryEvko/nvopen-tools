// Function: sub_DB9040
// Address: 0xdb9040
//
__int64 __fastcall sub_DB9040(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 *v7; // r13
  __int64 v8; // r14
  unsigned __int64 v9; // rax
  __int64 v10; // r15
  char v11; // cl
  __int64 v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  int v15; // eax
  char v16; // al
  __int64 v17; // r15
  __int64 v18; // rsi
  bool v19; // zf
  __int64 v20; // r15
  __int64 v21; // r8
  __int64 v22; // r9
  int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // rdx
  _BYTE *v26; // rdi
  __int64 v27; // r15
  unsigned __int64 v28; // rdx
  __int64 v29; // r13
  _BYTE *v30; // r15
  unsigned __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned int v34; // esi
  __int64 v35; // r9
  __int64 v36; // rdi
  int v37; // r11d
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // rdx
  __int64 v41; // r10
  __int64 v42; // rdi
  _BYTE *v43; // rbx
  _BYTE *v44; // r12
  _BYTE *v45; // rdi
  __int64 v47; // r15
  _BYTE *v48; // r15
  _BYTE *v49; // rbx
  _BYTE *v50; // rdi
  int v51; // eax
  unsigned int v52; // esi
  __int64 v53; // r9
  __int64 v54; // rdi
  int v55; // r11d
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // rdx
  __int64 v59; // r10
  __int64 v60; // rdi
  int v61; // eax
  int v62; // ecx
  __int64 v63; // rax
  _QWORD *v64; // rax
  int v65; // eax
  int v66; // ecx
  __int64 v67; // rax
  _QWORD *v68; // rax
  __int64 v69; // rcx
  __int64 v70; // r9
  __int64 v71; // r8
  __int64 v72; // rax
  __int64 v73; // rdx
  _BYTE *v74; // rbx
  _BYTE *v75; // r12
  _BYTE *v76; // r15
  char v77; // al
  __int64 v78; // rdx
  int v79; // eax
  int v80; // esi
  unsigned int v81; // eax
  __int64 v82; // rdi
  int v83; // r11d
  __int64 v84; // r10
  int v85; // eax
  int v86; // esi
  unsigned int v87; // eax
  __int64 v88; // rdi
  int v89; // r11d
  __int64 v90; // r10
  int v91; // eax
  int v92; // esi
  int v93; // r11d
  unsigned int v94; // eax
  __int64 v95; // rdi
  int v96; // eax
  int v97; // edi
  int v98; // r11d
  unsigned int v99; // eax
  __int64 v100; // rsi
  __int64 v101; // [rsp+0h] [rbp-2B0h]
  char v104; // [rsp+28h] [rbp-288h]
  int v105; // [rsp+28h] [rbp-288h]
  __int64 v106; // [rsp+28h] [rbp-288h]
  int v107; // [rsp+30h] [rbp-280h]
  _BYTE *v108; // [rsp+30h] [rbp-280h]
  __int64 v109; // [rsp+38h] [rbp-278h]
  __int64 *v110; // [rsp+40h] [rbp-270h]
  __int64 v111; // [rsp+48h] [rbp-268h]
  __int64 v112; // [rsp+50h] [rbp-260h]
  __int64 v113; // [rsp+50h] [rbp-260h]
  bool v114; // [rsp+58h] [rbp-258h]
  bool v115; // [rsp+5Dh] [rbp-253h]
  unsigned __int8 v117; // [rsp+5Fh] [rbp-251h]
  unsigned __int64 v118; // [rsp+68h] [rbp-248h] BYREF
  __int64 *v119; // [rsp+70h] [rbp-240h] BYREF
  __int64 v120; // [rsp+78h] [rbp-238h]
  _BYTE v121[64]; // [rsp+80h] [rbp-230h] BYREF
  __int64 v122; // [rsp+C0h] [rbp-1F0h] BYREF
  __int64 v123; // [rsp+C8h] [rbp-1E8h]
  __int64 v124; // [rsp+D0h] [rbp-1E0h]
  bool v125; // [rsp+D8h] [rbp-1D8h]
  char *v126; // [rsp+E0h] [rbp-1D0h] BYREF
  unsigned int v127; // [rsp+E8h] [rbp-1C8h]
  char v128; // [rsp+F0h] [rbp-1C0h] BYREF
  _BYTE *v129; // [rsp+110h] [rbp-1A0h] BYREF
  __int64 v130; // [rsp+118h] [rbp-198h]
  _BYTE v131[400]; // [rsp+120h] [rbp-190h] BYREF

  v4 = a3;
  v5 = a2;
  v119 = (__int64 *)v121;
  v120 = 0x800000000LL;
  sub_D46D90(a3, (__int64)&v119);
  v129 = v131;
  v130 = 0x400000000LL;
  v6 = sub_D47930(v4);
  v7 = v119;
  v111 = v6;
  v110 = &v119[(unsigned int)v120];
  v114 = (unsigned int)v120 == 1;
  if ( v119 == v110 )
  {
    v117 = 1;
    v109 = sub_D970F0(a2);
    v115 = 0;
    goto LABEL_36;
  }
  v115 = 0;
  v112 = 0;
  v109 = 0;
  v117 = 1;
  do
  {
    v8 = *v7;
    v9 = *(_QWORD *)(*v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v9 == *v7 + 48 )
      goto LABEL_175;
    if ( !v9 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
LABEL_175:
      BUG();
    if ( *(_BYTE *)(v9 - 24) != 31 )
      goto LABEL_17;
    v10 = *(_QWORD *)(v9 - 120);
    if ( *(_BYTE *)v10 != 17 )
      goto LABEL_17;
    v11 = *(_BYTE *)(v4 + 84);
    v12 = *(_QWORD *)(v9 - 56);
    if ( v11 )
    {
      v13 = *(_QWORD **)(v4 + 64);
      v14 = &v13[*(unsigned int *)(v4 + 76)];
      if ( v13 != v14 )
      {
        while ( v12 != *v13 )
        {
          if ( v14 == ++v13 )
            goto LABEL_14;
        }
        v11 = 0;
      }
    }
    else
    {
      v11 = sub_C8CA60(v4 + 56, v12) == 0;
    }
LABEL_14:
    if ( *(_DWORD *)(v10 + 32) <= 0x40u )
    {
      v16 = *(_QWORD *)(v10 + 24) == 0;
    }
    else
    {
      v104 = v11;
      v107 = *(_DWORD *)(v10 + 32);
      v15 = sub_C444A0(v10 + 24);
      v11 = v104;
      v16 = v107 == v15;
    }
    if ( v16 != v11 )
    {
LABEL_17:
      sub_DB8E00((__int64)&v122, v5, v4, v8, v114, a4);
      v17 = v122;
      v18 = v117;
      v19 = v17 == sub_D970F0(v5);
      v20 = v124;
      if ( v19 )
        v18 = 0;
      v117 = v18;
      if ( v20 != sub_D970F0(v5) )
      {
        v23 = v130;
        if ( HIDWORD(v130) > (unsigned int)v130 )
        {
          v24 = 11LL * (unsigned int)v130;
          v25 = (__int64)v129;
          v26 = &v129[88 * (unsigned int)v130];
          if ( v26 )
          {
            *(_QWORD *)v26 = v8;
            *((_QWORD *)v26 + 1) = v122;
            v27 = v123;
            *((_QWORD *)v26 + 2) = v123;
            *((_QWORD *)v26 + 3) = v124;
            v26[32] = v125;
            *((_QWORD *)v26 + 5) = v26 + 56;
            *((_QWORD *)v26 + 6) = 0x400000000LL;
            if ( !v127 )
            {
              v23 = v130;
LABEL_24:
              LODWORD(v130) = v23 + 1;
              goto LABEL_25;
            }
            v18 = (__int64)&v126;
            sub_D915C0((__int64)(v26 + 40), (__int64)&v126, v25, v24, v21, v22);
            v23 = v130;
          }
          v27 = v123;
          goto LABEL_24;
        }
        v18 = sub_C8D7D0((__int64)&v129, (__int64)v131, 0, 0x58u, &v118, v22);
        v108 = (_BYTE *)v18;
        v71 = 88LL * (unsigned int)v130;
        v72 = v71 + v18;
        if ( v71 + v18 )
        {
          v18 = 0x400000000LL;
          *(_QWORD *)v72 = v8;
          v73 = v122;
          *(_QWORD *)(v72 + 48) = 0x400000000LL;
          *(_QWORD *)(v72 + 8) = v73;
          *(_QWORD *)(v72 + 16) = v123;
          *(_QWORD *)(v72 + 24) = v124;
          *(_BYTE *)(v72 + 32) = v125;
          *(_QWORD *)(v72 + 40) = v72 + 56;
          v69 = v127;
          if ( v127 )
          {
            v18 = (__int64)&v126;
            sub_D915C0(v72 + 40, (__int64)&v126, v72 + 56, v127, v71, v70);
          }
          v71 = 88LL * (unsigned int)v130;
        }
        v48 = &v129[v71];
        if ( v129 != &v129[v71] )
        {
          v106 = v5;
          v74 = &v129[v71];
          v101 = v4;
          v75 = v129;
          v76 = v108;
          do
          {
            if ( v76 )
            {
              *(_QWORD *)v76 = *(_QWORD *)v75;
              *((_QWORD *)v76 + 1) = *((_QWORD *)v75 + 1);
              *((_QWORD *)v76 + 2) = *((_QWORD *)v75 + 2);
              *((_QWORD *)v76 + 3) = *((_QWORD *)v75 + 3);
              v77 = v75[32];
              *((_DWORD *)v76 + 12) = 0;
              v76[32] = v77;
              *((_QWORD *)v76 + 5) = v76 + 56;
              *((_DWORD *)v76 + 13) = 4;
              v78 = *((unsigned int *)v75 + 12);
              if ( (_DWORD)v78 )
              {
                v18 = (__int64)(v75 + 40);
                sub_D91460((__int64)(v76 + 40), (char **)v75 + 5, v78, v69, v71, v70);
              }
            }
            v75 += 88;
            v76 += 88;
          }
          while ( v74 != v75 );
          v48 = v129;
          v5 = v106;
          v4 = v101;
          if ( &v129[88 * (unsigned int)v130] != v129 )
          {
            v49 = &v129[88 * (unsigned int)v130];
            do
            {
              v49 -= 88;
              v50 = (_BYTE *)*((_QWORD *)v49 + 5);
              if ( v50 != v49 + 56 )
                _libc_free(v50, v18);
            }
            while ( v49 != v48 );
            v5 = v106;
            v48 = v129;
          }
        }
        v51 = v118;
        if ( v48 != v131 )
        {
          v105 = v118;
          _libc_free(v48, v18);
          v51 = v105;
        }
        LODWORD(v130) = v130 + 1;
        HIDWORD(v130) = v51;
        v129 = v108;
      }
      v27 = v123;
LABEL_25:
      if ( v27 != sub_D970F0(v5) && v111 && (v18 = v8, (unsigned __int8)sub_B19720(*(_QWORD *)(v5 + 40), v8, v111)) )
      {
        if ( v109 )
        {
          v18 = v109;
          v109 = sub_DCF070(v5, v109, v123, 0);
        }
        else
        {
          v109 = v123;
          v115 = v125;
        }
      }
      else if ( v112 != sub_D970F0(v5) )
      {
        if ( !v112 || (v47 = v123, v47 == sub_D970F0(v5)) )
        {
          v112 = v123;
        }
        else
        {
          v18 = v112;
          v112 = sub_DCE0B0(v5, v112, v123);
        }
      }
      if ( v126 != &v128 )
        _libc_free(v126, v18);
    }
    ++v7;
  }
  while ( v110 != v7 );
  if ( !v109 )
  {
    if ( v112 )
      v109 = v112;
    else
      v109 = sub_D970F0(v5);
  }
  if ( v115 )
    v115 = (_DWORD)v120 == 1;
LABEL_36:
  v28 = (unsigned int)v130;
  v29 = (__int64)v129;
  v30 = &v129[88 * (unsigned int)v130];
  if ( v30 == v129 )
    goto LABEL_46;
  v31 = (4LL * a4) | v4 & 0xFFFFFFFFFFFFFFFBLL;
  v113 = v5 + 712;
  while ( 2 )
  {
    while ( 2 )
    {
      v32 = *(_QWORD *)(v29 + 8);
      if ( !*(_WORD *)(v32 + 24) )
        goto LABEL_40;
      v52 = *(_DWORD *)(v5 + 736);
      if ( !v52 )
      {
        ++*(_QWORD *)(v5 + 712);
        goto LABEL_121;
      }
      v53 = v52 - 1;
      v54 = *(_QWORD *)(v5 + 720);
      v55 = 1;
      v56 = (unsigned int)v53 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
      v57 = v54 + 72 * v56;
      v58 = 0;
      v59 = *(_QWORD *)v57;
      if ( v32 == *(_QWORD *)v57 )
        goto LABEL_73;
      while ( 2 )
      {
        if ( v59 == -4096 )
        {
          v65 = *(_DWORD *)(v5 + 728);
          if ( !v58 )
            v58 = v57;
          ++*(_QWORD *)(v5 + 712);
          v66 = v65 + 1;
          if ( 4 * (v65 + 1) < 3 * v52 )
          {
            if ( v52 - *(_DWORD *)(v5 + 732) - v66 > v52 >> 3 )
            {
LABEL_97:
              *(_DWORD *)(v5 + 728) = v66;
              if ( *(_QWORD *)v58 != -4096 )
                --*(_DWORD *)(v5 + 732);
              v67 = *(_QWORD *)(v29 + 8);
              *(_QWORD *)(v58 + 8) = 0;
              v60 = v58 + 8;
              *(_QWORD *)(v58 + 24) = 4;
              *(_QWORD *)v58 = v67;
              *(_QWORD *)(v58 + 16) = v58 + 40;
              *(_DWORD *)(v58 + 32) = 0;
              *(_BYTE *)(v58 + 36) = 1;
              goto LABEL_100;
            }
            sub_DA6B40(v113, v52);
            v91 = *(_DWORD *)(v5 + 736);
            if ( v91 )
            {
              v57 = *(_QWORD *)(v29 + 8);
              v92 = v91 - 1;
              v84 = 0;
              v53 = *(_QWORD *)(v5 + 720);
              v93 = 1;
              v66 = *(_DWORD *)(v5 + 728) + 1;
              v94 = (v91 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
              v58 = v53 + 72LL * v94;
              v95 = *(_QWORD *)v58;
              if ( v57 == *(_QWORD *)v58 )
                goto LABEL_97;
              while ( v95 != -4096 )
              {
                if ( !v84 && v95 == -8192 )
                  v84 = v58;
                v94 = v92 & (v93 + v94);
                v58 = v53 + 72LL * v94;
                v95 = *(_QWORD *)v58;
                if ( v57 == *(_QWORD *)v58 )
                  goto LABEL_97;
                ++v93;
              }
              goto LABEL_143;
            }
            goto LABEL_177;
          }
LABEL_121:
          sub_DA6B40(v113, 2 * v52);
          v79 = *(_DWORD *)(v5 + 736);
          if ( v79 )
          {
            v57 = *(_QWORD *)(v29 + 8);
            v80 = v79 - 1;
            v53 = *(_QWORD *)(v5 + 720);
            v66 = *(_DWORD *)(v5 + 728) + 1;
            v81 = (v79 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
            v58 = v53 + 72LL * v81;
            v82 = *(_QWORD *)v58;
            if ( v57 == *(_QWORD *)v58 )
              goto LABEL_97;
            v83 = 1;
            v84 = 0;
            while ( v82 != -4096 )
            {
              if ( v82 == -8192 && !v84 )
                v84 = v58;
              v81 = v80 & (v83 + v81);
              v58 = v53 + 72LL * v81;
              v82 = *(_QWORD *)v58;
              if ( v57 == *(_QWORD *)v58 )
                goto LABEL_97;
              ++v83;
            }
LABEL_143:
            if ( v84 )
              v58 = v84;
            goto LABEL_97;
          }
LABEL_177:
          ++*(_DWORD *)(v5 + 728);
          BUG();
        }
        if ( v58 || v59 != -8192 )
          v57 = v58;
        v58 = (unsigned int)(v55 + 1);
        v56 = (unsigned int)v53 & (v55 + (_DWORD)v56);
        v59 = *(_QWORD *)(v54 + 72LL * (unsigned int)v56);
        if ( v32 != v59 )
        {
          ++v55;
          v58 = v57;
          v57 = v54 + 72LL * (unsigned int)v56;
          continue;
        }
        break;
      }
      v57 = v54 + 72LL * (unsigned int)v56;
LABEL_73:
      v60 = v57 + 8;
      if ( !*(_BYTE *)(v57 + 36) )
      {
LABEL_74:
        sub_C8CC70(v60, v31, v58, v56, v57, v53);
        goto LABEL_40;
      }
LABEL_100:
      v68 = *(_QWORD **)(v60 + 8);
      v56 = *(unsigned int *)(v60 + 20);
      v58 = (__int64)&v68[v56];
      if ( v68 == (_QWORD *)v58 )
      {
LABEL_103:
        if ( (unsigned int)v56 >= *(_DWORD *)(v60 + 16) )
          goto LABEL_74;
        *(_DWORD *)(v60 + 20) = v56 + 1;
        *(_QWORD *)v58 = v31;
        ++*(_QWORD *)v60;
      }
      else
      {
        while ( v31 != *v68 )
        {
          if ( (_QWORD *)v58 == ++v68 )
            goto LABEL_103;
        }
      }
LABEL_40:
      v33 = *(_QWORD *)(v29 + 24);
      if ( !*(_WORD *)(v33 + 24) )
        goto LABEL_38;
      v34 = *(_DWORD *)(v5 + 736);
      if ( !v34 )
      {
        ++*(_QWORD *)(v5 + 712);
        goto LABEL_131;
      }
      v35 = v34 - 1;
      v36 = *(_QWORD *)(v5 + 720);
      v37 = 1;
      v38 = (unsigned int)v35 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v39 = v36 + 72 * v38;
      v40 = 0;
      v41 = *(_QWORD *)v39;
      if ( v33 == *(_QWORD *)v39 )
        goto LABEL_43;
LABEL_78:
      if ( v41 == -4096 )
      {
        v61 = *(_DWORD *)(v5 + 728);
        if ( !v40 )
          v40 = v39;
        ++*(_QWORD *)(v5 + 712);
        v62 = v61 + 1;
        if ( 4 * (v61 + 1) < 3 * v34 )
        {
          if ( v34 - *(_DWORD *)(v5 + 732) - v62 > v34 >> 3 )
            goto LABEL_83;
          sub_DA6B40(v113, v34);
          v96 = *(_DWORD *)(v5 + 736);
          if ( v96 )
          {
            v39 = *(_QWORD *)(v29 + 24);
            v97 = v96 - 1;
            v90 = 0;
            v35 = *(_QWORD *)(v5 + 720);
            v98 = 1;
            v62 = *(_DWORD *)(v5 + 728) + 1;
            v99 = (v96 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
            v40 = v35 + 72LL * v99;
            v100 = *(_QWORD *)v40;
            if ( *(_QWORD *)v40 != v39 )
            {
              while ( v100 != -4096 )
              {
                if ( v100 == -8192 && !v90 )
                  v90 = v40;
                v99 = v97 & (v98 + v99);
                v40 = v35 + 72LL * v99;
                v100 = *(_QWORD *)v40;
                if ( v39 == *(_QWORD *)v40 )
                  goto LABEL_83;
                ++v98;
              }
              goto LABEL_149;
            }
            goto LABEL_83;
          }
LABEL_176:
          ++*(_DWORD *)(v5 + 728);
          BUG();
        }
LABEL_131:
        sub_DA6B40(v113, 2 * v34);
        v85 = *(_DWORD *)(v5 + 736);
        if ( !v85 )
          goto LABEL_176;
        v39 = *(_QWORD *)(v29 + 24);
        v86 = v85 - 1;
        v35 = *(_QWORD *)(v5 + 720);
        v62 = *(_DWORD *)(v5 + 728) + 1;
        v87 = (v85 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        v40 = v35 + 72LL * v87;
        v88 = *(_QWORD *)v40;
        if ( v39 != *(_QWORD *)v40 )
        {
          v89 = 1;
          v90 = 0;
          while ( v88 != -4096 )
          {
            if ( !v90 && v88 == -8192 )
              v90 = v40;
            v87 = v86 & (v89 + v87);
            v40 = v35 + 72LL * v87;
            v88 = *(_QWORD *)v40;
            if ( v39 == *(_QWORD *)v40 )
              goto LABEL_83;
            ++v89;
          }
LABEL_149:
          if ( v90 )
            v40 = v90;
        }
LABEL_83:
        *(_DWORD *)(v5 + 728) = v62;
        if ( *(_QWORD *)v40 != -4096 )
          --*(_DWORD *)(v5 + 732);
        v63 = *(_QWORD *)(v29 + 24);
        *(_QWORD *)(v40 + 8) = 0;
        v42 = v40 + 8;
        *(_QWORD *)(v40 + 24) = 4;
        *(_QWORD *)v40 = v63;
        *(_QWORD *)(v40 + 16) = v40 + 40;
        *(_DWORD *)(v40 + 32) = 0;
        *(_BYTE *)(v40 + 36) = 1;
LABEL_86:
        v64 = *(_QWORD **)(v42 + 8);
        v38 = *(unsigned int *)(v42 + 20);
        v40 = (__int64)&v64[v38];
        if ( v64 == (_QWORD *)v40 )
        {
LABEL_89:
          if ( (unsigned int)v38 >= *(_DWORD *)(v42 + 16) )
            goto LABEL_44;
          *(_DWORD *)(v42 + 20) = v38 + 1;
          *(_QWORD *)v40 = v31;
          ++*(_QWORD *)v42;
        }
        else
        {
          while ( v31 != *v64 )
          {
            if ( (_QWORD *)v40 == ++v64 )
              goto LABEL_89;
          }
        }
LABEL_38:
        v29 += 88;
        if ( (_BYTE *)v29 == v30 )
          goto LABEL_45;
        continue;
      }
      break;
    }
    if ( v40 || v41 != -8192 )
      v39 = v40;
    v40 = (unsigned int)(v37 + 1);
    v38 = (unsigned int)v35 & (v37 + (_DWORD)v38);
    v41 = *(_QWORD *)(v36 + 72LL * (unsigned int)v38);
    if ( v33 != v41 )
    {
      ++v37;
      v40 = v39;
      v39 = v36 + 72LL * (unsigned int)v38;
      goto LABEL_78;
    }
    v39 = v36 + 72LL * (unsigned int)v38;
LABEL_43:
    v42 = v39 + 8;
    if ( *(_BYTE *)(v39 + 36) )
      goto LABEL_86;
LABEL_44:
    v29 += 88;
    sub_C8CC70(v42, v31, v40, v38, v39, v35);
    if ( (_BYTE *)v29 != v30 )
      continue;
    break;
  }
LABEL_45:
  v29 = (__int64)v129;
  v28 = (unsigned int)v130;
LABEL_46:
  sub_D9E910(a1, v29, v28, v117, v109, v115);
  v43 = v129;
  v44 = &v129[88 * (unsigned int)v130];
  if ( v129 != v44 )
  {
    do
    {
      v44 -= 88;
      v45 = (_BYTE *)*((_QWORD *)v44 + 5);
      if ( v45 != v44 + 56 )
        _libc_free(v45, v29);
    }
    while ( v43 != v44 );
    v44 = v129;
  }
  if ( v44 != v131 )
    _libc_free(v44, v29);
  if ( v119 != (__int64 *)v121 )
    _libc_free(v119, v29);
  return a1;
}
