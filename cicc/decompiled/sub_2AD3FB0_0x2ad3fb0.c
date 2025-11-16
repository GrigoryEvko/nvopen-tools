// Function: sub_2AD3FB0
// Address: 0x2ad3fb0
//
__int64 __fastcall sub_2AD3FB0(__int64 a1, __int64 a2)
{
  int v2; // eax
  int v3; // edi
  int v4; // r15d
  __int64 v5; // rax
  unsigned __int64 v6; // rsi
  unsigned int v7; // edx
  __int64 v8; // r13
  __int64 v9; // r9
  int v10; // r15d
  __int64 *v11; // r11
  unsigned int v12; // edx
  __int64 *v13; // rbx
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 **v16; // r15
  _BYTE *v17; // rsi
  int v18; // ecx
  __int64 v19; // rdi
  int v20; // ecx
  unsigned int v21; // edx
  _QWORD *v22; // rax
  _BYTE *v23; // r9
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD **v26; // r8
  __int64 *v27; // rbx
  _QWORD *v28; // rax
  int v29; // r8d
  __int64 v30; // rdx
  unsigned __int64 *v31; // rcx
  __int64 v32; // rdx
  unsigned __int64 v33; // rdi
  _QWORD *v34; // rbx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 *v39; // rcx
  int v40; // esi
  __int64 v41; // r8
  __int64 *v42; // rdi
  int v43; // eax
  int v44; // eax
  __int64 *v45; // rax
  __int64 v46; // r14
  __int64 *v47; // r12
  __int64 v48; // rax
  __int64 *v49; // rdi
  __int64 v50; // rbx
  __int64 v51; // r8
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 **v54; // r15
  __int64 v55; // rdx
  __int64 *v56; // rsi
  __int64 *v57; // rsi
  __int64 *v58; // rsi
  __int64 v59; // rax
  __int64 *v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rsi
  __int64 *v63; // rdi
  __int64 v64; // rbx
  __int64 v65; // r9
  __int64 v66; // r8
  __int64 v67; // rcx
  __int64 v68; // rdx
  __int64 *v69; // r13
  __int64 **v70; // r14
  __int64 v71; // rdx
  __int64 *v72; // rax
  __int64 v73; // rax
  __int64 *v74; // rdx
  __int64 v75; // rax
  __int64 v76; // rsi
  bool v77; // zf
  __int64 *v78; // rax
  int v79; // esi
  int v80; // edx
  unsigned int v81; // esi
  unsigned __int64 v82; // rdx
  __int64 *v83; // rdi
  __int64 v84; // r12
  __int64 *v85; // rdi
  __int64 *v86; // rax
  __int64 v87; // rbx
  unsigned __int64 v88; // r12
  unsigned __int64 v89; // rdi
  int v91; // esi
  int v92; // edx
  unsigned int v93; // esi
  unsigned __int64 v94; // rdx
  __int64 v95; // rax
  unsigned __int64 v96; // rsi
  __int64 v97; // rsi
  void **v98; // rcx
  _QWORD *v99; // rax
  void *v100; // rdx
  int v101; // r8d
  __int64 **v102; // [rsp+10h] [rbp-1B0h]
  unsigned __int64 v103; // [rsp+18h] [rbp-1A8h]
  __int64 *v104; // [rsp+20h] [rbp-1A0h]
  __int64 v105; // [rsp+28h] [rbp-198h]
  __int64 *v106; // [rsp+30h] [rbp-190h]
  __int64 v107; // [rsp+38h] [rbp-188h]
  __int64 v108; // [rsp+40h] [rbp-180h]
  __int64 *v109; // [rsp+48h] [rbp-178h]
  int v111; // [rsp+70h] [rbp-150h]
  __int64 v112; // [rsp+78h] [rbp-148h]
  int v114; // [rsp+88h] [rbp-138h]
  _QWORD *v115; // [rsp+88h] [rbp-138h]
  __int64 *v116; // [rsp+88h] [rbp-138h]
  __int64 v117; // [rsp+88h] [rbp-138h]
  __int64 v118; // [rsp+98h] [rbp-128h] BYREF
  __int64 *v119; // [rsp+A0h] [rbp-120h] BYREF
  __int64 *v120; // [rsp+A8h] [rbp-118h] BYREF
  __int64 *v121; // [rsp+B0h] [rbp-110h] BYREF
  __int64 *v122; // [rsp+B8h] [rbp-108h] BYREF
  __int64 *v123; // [rsp+C0h] [rbp-100h] BYREF
  int v124; // [rsp+C8h] [rbp-F8h]
  __int64 v125; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v126; // [rsp+D8h] [rbp-E8h]
  __int64 v127; // [rsp+E0h] [rbp-E0h]
  unsigned int v128; // [rsp+E8h] [rbp-D8h]
  __int64 **v129; // [rsp+F0h] [rbp-D0h] BYREF
  __int64 v130; // [rsp+F8h] [rbp-C8h]
  __int64 *v131; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v132; // [rsp+108h] [rbp-B8h]
  _BYTE v133[48]; // [rsp+110h] [rbp-B0h] BYREF
  __int64 *v134; // [rsp+140h] [rbp-80h] BYREF
  char *v135; // [rsp+148h] [rbp-78h]
  __int64 v136; // [rsp+150h] [rbp-70h]
  char v137; // [rsp+158h] [rbp-68h] BYREF
  __int16 v138; // [rsp+160h] [rbp-60h]

  v104 = *(__int64 **)(a2 + 40);
  v2 = sub_2AC59D0(a1, **(_BYTE ***)(a2 - 8));
  v3 = *(_DWORD *)(a2 + 4);
  v125 = 0;
  v126 = 0;
  v4 = v2;
  v5 = *(_QWORD *)(a2 - 8);
  v6 = *(_QWORD *)(v5 + 32);
  v127 = 0;
  v128 = 0;
  v7 = (v3 & 0x7FFFFFFu) >> 1;
  v103 = v6;
  v129 = &v131;
  v130 = 0;
  v112 = v7 - 1;
  if ( v7 != 1 )
  {
    v111 = v4;
    v8 = 0;
    while ( 1 )
    {
      v38 = 32;
      if ( (_DWORD)v8 != -2 )
        v38 = 32LL * (unsigned int)(2 * v8 + 3);
      v39 = *(__int64 **)(v5 + v38);
      ++v8;
      if ( (__int64 *)v103 != v39 )
        break;
LABEL_18:
      if ( v112 == v8 )
        goto LABEL_30;
      v5 = *(_QWORD *)(a2 - 8);
    }
    v40 = v128;
    v123 = *(__int64 **)(v5 + v38);
    v124 = 0;
    if ( v128 )
    {
      v9 = v128 - 1;
      v10 = 1;
      v11 = 0;
      v12 = v9 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v13 = (__int64 *)(v126 + 16LL * v12);
      v14 = *v13;
      if ( v39 == (__int64 *)*v13 )
      {
LABEL_4:
        v15 = *((unsigned int *)v13 + 2);
        goto LABEL_5;
      }
      while ( v14 != -4096 )
      {
        if ( v14 == -8192 && !v11 )
          v11 = v13;
        v12 = v9 & (v10 + v12);
        v13 = (__int64 *)(v126 + 16LL * v12);
        v14 = *v13;
        if ( v39 == (__int64 *)*v13 )
          goto LABEL_4;
        ++v10;
      }
      if ( v11 )
        v13 = v11;
      ++v125;
      v43 = v127 + 1;
      v134 = v13;
      if ( 4 * ((int)v127 + 1) < 3 * v128 )
      {
        v42 = v39;
        v41 = v128 >> 3;
        if ( v128 - HIDWORD(v127) - v43 > (unsigned int)v41 )
          goto LABEL_125;
        v116 = v39;
LABEL_26:
        sub_B23080((__int64)&v125, v40);
        sub_B1C700((__int64)&v125, (__int64 *)&v123, &v134);
        v13 = v134;
        v42 = v123;
        v39 = v116;
        v43 = v127 + 1;
LABEL_125:
        LODWORD(v127) = v43;
        if ( *v13 != -4096 )
          --HIDWORD(v127);
        *v13 = (__int64)v42;
        *((_DWORD *)v13 + 2) = v124;
        v131 = (__int64 *)v133;
        v132 = 0x600000000LL;
        v136 = 0x600000000LL;
        v95 = (unsigned int)v130;
        v134 = v39;
        v96 = (unsigned int)v130 + 1LL;
        v135 = &v137;
        v15 = (unsigned int)v130;
        if ( v96 > HIDWORD(v130) )
        {
          if ( v129 > &v134 || (v117 = (__int64)v129, &v134 >= &v129[9 * (unsigned int)v130]) )
          {
            sub_2AD3B10((__int64)&v129, v96, (__int64)v129, HIDWORD(v130), v41, v9);
            v95 = (unsigned int)v130;
            v97 = (__int64)v129;
            v98 = (void **)&v134;
            v15 = (unsigned int)v130;
          }
          else
          {
            sub_2AD3B10((__int64)&v129, v96, (__int64)v129, HIDWORD(v130), v41, v9);
            v97 = (__int64)v129;
            v95 = (unsigned int)v130;
            v15 = (unsigned int)v130;
            v98 = (void **)((char *)&v134 + (_QWORD)v129 - v117);
          }
        }
        else
        {
          v97 = (__int64)v129;
          v98 = (void **)&v134;
        }
        v99 = (_QWORD *)(v97 + 72 * v95);
        if ( v99 )
        {
          v100 = *v98;
          v99[2] = 0x600000000LL;
          *v99 = v100;
          v99[1] = v99 + 3;
          if ( *((_DWORD *)v98 + 4) )
            sub_2AA8690((__int64)(v99 + 1), (char **)v98 + 1, (__int64)(v99 + 3), (__int64)v98, v41, v9);
          v15 = (unsigned int)v130;
        }
        LODWORD(v130) = v15 + 1;
        if ( v135 != &v137 )
        {
          _libc_free((unsigned __int64)v135);
          v15 = (unsigned int)(v130 - 1);
        }
        *((_DWORD *)v13 + 2) = v15;
        v5 = *(_QWORD *)(a2 - 8);
LABEL_5:
        v16 = &v129[9 * v15];
        v17 = *(_BYTE **)(v5 + 32LL * (unsigned int)(2 * v8));
        if ( *v17 > 0x1Cu )
        {
          v18 = *(_DWORD *)(a1 + 152);
          v19 = *(_QWORD *)(a1 + 136);
          if ( v18 )
          {
            v20 = v18 - 1;
            v21 = v20 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v22 = (_QWORD *)(v19 + 16LL * v21);
            v23 = (_BYTE *)*v22;
            if ( v17 == (_BYTE *)*v22 )
            {
LABEL_8:
              v24 = v22[1];
              if ( v24 )
              {
                v25 = *(_QWORD *)(v24 + 16);
                v26 = (_QWORD **)(v25 & 0xFFFFFFFFFFFFFFF8LL);
                if ( (v25 & 4) != 0 )
                  v26 = (_QWORD **)**v26;
LABEL_11:
                v114 = (int)v26;
                v27 = *(__int64 **)(a1 + 56);
                v138 = 257;
                v123 = 0;
                v131 = 0;
                v28 = (_QWORD *)sub_22077B0(0xC8u);
                if ( v28 )
                {
                  v29 = v114;
                  v115 = v28;
                  sub_2C1A5F0((_DWORD)v28, 53, 32, v111, v29, (unsigned int)&v131, (__int64)&v134);
                  v30 = *v27;
                  v28 = v115;
                  if ( !*v27 )
                  {
LABEL_14:
                    v34 = v28 + 12;
LABEL_15:
                    sub_9C6650(&v131);
                    v37 = *((unsigned int *)v16 + 4);
                    if ( v37 + 1 > (unsigned __int64)*((unsigned int *)v16 + 5) )
                    {
                      sub_C8D5F0((__int64)(v16 + 1), v16 + 3, v37 + 1, 8u, v35, v36);
                      v37 = *((unsigned int *)v16 + 4);
                    }
                    v16[1][v37] = (__int64)v34;
                    ++*((_DWORD *)v16 + 4);
                    sub_9C6650(&v123);
                    goto LABEL_18;
                  }
                }
                else
                {
                  v30 = *v27;
                  if ( !*v27 )
                  {
                    v34 = 0;
                    goto LABEL_15;
                  }
                }
                v31 = (unsigned __int64 *)v27[1];
                v28[10] = v30;
                v32 = v28[3];
                v33 = *v31;
                v28[4] = v31;
                v33 &= 0xFFFFFFFFFFFFFFF8LL;
                v28[3] = v33 | v32 & 7;
                *(_QWORD *)(v33 + 8) = v28 + 3;
                *v31 = *v31 & 7 | (unsigned __int64)(v28 + 3);
                goto LABEL_14;
              }
            }
            else
            {
              v44 = 1;
              while ( v23 != (_BYTE *)-4096LL )
              {
                v101 = v44 + 1;
                v21 = v20 & (v44 + v21);
                v22 = (_QWORD *)(v19 + 16LL * v21);
                v23 = (_BYTE *)*v22;
                if ( v17 == (_BYTE *)*v22 )
                  goto LABEL_8;
                v44 = v101;
              }
            }
          }
        }
        LODWORD(v26) = sub_2AC42A0(*(_QWORD *)a1, (__int64)v17);
        goto LABEL_11;
      }
    }
    else
    {
      ++v125;
      v134 = 0;
    }
    v116 = v39;
    v40 = 2 * v128;
    goto LABEL_26;
  }
LABEL_30:
  v107 = 0;
  v106 = (__int64 *)sub_2AB6F10(a1, (__int64)v104);
  v105 = a1 + 64;
  v108 = (__int64)v129;
  v102 = &v129[9 * (unsigned int)v130];
  if ( v102 == v129 )
    goto LABEL_114;
  do
  {
    v45 = *(__int64 **)(v108 + 8);
    v46 = *v45;
    v47 = v45 + 1;
    v109 = &v45[*(unsigned int *)(v108 + 16)];
    if ( v109 != v45 + 1 )
    {
      while ( 1 )
      {
        v48 = *v47;
        v119 = 0;
        v131 = (__int64 *)v46;
        v49 = *(__int64 **)(a1 + 56);
        v132 = v48;
        v120 = 0;
        v138 = 257;
        v50 = sub_22077B0(0xC8u);
        if ( v50 )
          break;
        v46 = *v49;
        if ( !*v49 )
          goto LABEL_53;
        v59 = *v49;
        v46 = 96;
LABEL_52:
        v60 = (__int64 *)v49[1];
        *(_QWORD *)(v50 + 80) = v59;
        v61 = *(_QWORD *)(v50 + 24);
        v62 = *v60;
        *(_QWORD *)(v50 + 32) = v60;
        v62 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v50 + 24) = v62 | v61 & 7;
        *(_QWORD *)(v62 + 8) = v50 + 24;
        *v60 = *v60 & 7 | (v50 + 24);
LABEL_53:
        if ( v120 )
          sub_B91220((__int64)&v120, (__int64)v120);
        if ( v119 )
          sub_B91220((__int64)&v119, (__int64)v119);
        if ( v109 == ++v47 )
          goto LABEL_58;
      }
      v121 = v120;
      if ( v120 )
      {
        sub_B96E90((__int64)&v121, (__int64)v120, 1);
        v122 = v121;
        if ( v121 )
        {
          sub_B96E90((__int64)&v122, (__int64)v121, 1);
          v123 = v122;
          if ( v122 )
            sub_B96E90((__int64)&v123, (__int64)v122, 1);
LABEL_37:
          *(_BYTE *)(v50 + 8) = 4;
          v52 = v50 + 64;
          *(_QWORD *)(v50 + 48) = v50 + 64;
          v53 = 0;
          *(_QWORD *)v50 = &unk_4A231A8;
          *(_QWORD *)(v50 + 24) = 0;
          *(_QWORD *)(v50 + 32) = 0;
          *(_QWORD *)(v50 + 40) = &unk_4A23170;
          *(_QWORD *)(v50 + 56) = 0x200000000LL;
          v54 = &v131;
          *(_QWORD *)(v50 + 16) = 0;
          while ( 1 )
          {
            *(_QWORD *)(v52 + 8 * v53) = v46;
            ++*(_DWORD *)(v50 + 56);
            v55 = *(unsigned int *)(v46 + 24);
            if ( v55 + 1 > (unsigned __int64)*(unsigned int *)(v46 + 28) )
            {
              sub_C8D5F0(v46 + 16, (const void *)(v46 + 32), v55 + 1, 8u, v51, v55 + 1);
              v55 = *(unsigned int *)(v46 + 24);
            }
            ++v54;
            *(_QWORD *)(*(_QWORD *)(v46 + 16) + 8 * v55) = v50 + 40;
            ++*(_DWORD *)(v46 + 24);
            if ( v54 == (__int64 **)v133 )
              break;
            v53 = *(unsigned int *)(v50 + 56);
            v46 = (__int64)*v54;
            if ( v53 + 1 > (unsigned __int64)*(unsigned int *)(v50 + 60) )
            {
              sub_C8D5F0(v50 + 48, (const void *)(v50 + 64), v53 + 1, 8u, v51, v53 + 1);
              v53 = *(unsigned int *)(v50 + 56);
            }
            v52 = *(_QWORD *)(v50 + 48);
          }
          v56 = v123;
          *(_QWORD *)(v50 + 80) = 0;
          *(_QWORD *)(v50 + 88) = v56;
          *(_QWORD *)v50 = &unk_4A23A70;
          *(_QWORD *)(v50 + 40) = &unk_4A23AA8;
          if ( v56 )
          {
            sub_B96E90(v50 + 88, (__int64)v56, 1);
            if ( v123 )
              sub_B91220((__int64)&v123, (__int64)v123);
          }
          v46 = v50 + 96;
          sub_2BF0340(v50 + 96, 1, 0, v50);
          v57 = v122;
          *(_QWORD *)v50 = &unk_4A231C8;
          *(_QWORD *)(v50 + 40) = &unk_4A23200;
          *(_QWORD *)(v50 + 96) = &unk_4A23238;
          if ( v57 )
            sub_B91220((__int64)&v122, (__int64)v57);
          v58 = v121;
          *(_BYTE *)(v50 + 152) = 2;
          *(_BYTE *)(v50 + 156) &= ~1u;
          *(_QWORD *)v50 = &unk_4A23258;
          *(_QWORD *)(v50 + 40) = &unk_4A23290;
          *(_QWORD *)(v50 + 96) = &unk_4A232C8;
          if ( v58 )
            sub_B91220((__int64)&v121, (__int64)v58);
          *(_BYTE *)(v50 + 160) = 29;
          *(_QWORD *)v50 = &unk_4A23B70;
          *(_QWORD *)(v50 + 96) = &unk_4A23BF0;
          *(_QWORD *)(v50 + 40) = &unk_4A23BB8;
          sub_CA0F50((__int64 *)(v50 + 168), (void **)&v134);
          v59 = *v49;
          if ( !*v49 )
            goto LABEL_53;
          goto LABEL_52;
        }
      }
      else
      {
        v122 = 0;
      }
      v123 = 0;
      goto LABEL_37;
    }
LABEL_58:
    if ( v106 )
    {
      v118 = 0;
      v138 = 257;
      v63 = *(__int64 **)(a1 + 56);
      v131 = v106;
      v132 = v46;
      v119 = 0;
      v64 = sub_22077B0(0xC8u);
      if ( !v64 )
      {
        v46 = *v63;
        if ( !*v63 )
          goto LABEL_77;
        v73 = *v63;
        v46 = 96;
LABEL_76:
        v74 = (__int64 *)v63[1];
        *(_QWORD *)(v64 + 80) = v73;
        v75 = *(_QWORD *)(v64 + 24);
        v76 = *v74;
        *(_QWORD *)(v64 + 32) = v74;
        v76 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v64 + 24) = v76 | v75 & 7;
        *(_QWORD *)(v76 + 8) = v64 + 24;
        *v74 = *v74 & 7 | (v64 + 24);
LABEL_77:
        sub_9C6650(&v119);
        sub_9C6650(&v118);
        goto LABEL_78;
      }
      v120 = v119;
      if ( v119 )
      {
        sub_2AAAFA0((__int64 *)&v120);
        v121 = v120;
        if ( v120 )
        {
          sub_2AAAFA0((__int64 *)&v121);
          v122 = v121;
          if ( v121 )
          {
            sub_2AAAFA0((__int64 *)&v122);
            v123 = v122;
            if ( v122 )
              sub_2AAAFA0((__int64 *)&v123);
LABEL_65:
            v65 = v64 + 64;
            *(_BYTE *)(v64 + 8) = 4;
            v66 = v64 + 40;
            v67 = v64 + 64;
            v68 = 0;
            v69 = v106;
            *(_QWORD *)v64 = &unk_4A231A8;
            *(_QWORD *)(v64 + 24) = 0;
            v70 = &v131;
            *(_QWORD *)(v64 + 32) = 0;
            *(_QWORD *)(v64 + 40) = &unk_4A23170;
            *(_QWORD *)(v64 + 56) = 0x200000000LL;
            *(_QWORD *)(v64 + 16) = 0;
            for ( *(_QWORD *)(v64 + 48) = v64 + 64; ; v67 = *(_QWORD *)(v64 + 48) )
            {
              *(_QWORD *)(v67 + 8 * v68) = v69;
              ++*(_DWORD *)(v64 + 56);
              v71 = *((unsigned int *)v69 + 6);
              if ( v71 + 1 > (unsigned __int64)*((unsigned int *)v69 + 7) )
              {
                sub_C8D5F0((__int64)(v69 + 2), v69 + 4, v71 + 1, 8u, v66, v65);
                v71 = *((unsigned int *)v69 + 6);
              }
              ++v70;
              *(_QWORD *)(v69[2] + 8 * v71) = v64 + 40;
              ++*((_DWORD *)v69 + 6);
              if ( v70 == (__int64 **)v133 )
                break;
              v68 = *(unsigned int *)(v64 + 56);
              v69 = *v70;
              if ( v68 + 1 > (unsigned __int64)*(unsigned int *)(v64 + 60) )
              {
                sub_C8D5F0(v64 + 48, (const void *)(v64 + 64), v68 + 1, 8u, v66, v65);
                v68 = *(unsigned int *)(v64 + 56);
              }
            }
            *(_QWORD *)(v64 + 80) = 0;
            *(_QWORD *)(v64 + 40) = &unk_4A23AA8;
            v72 = v123;
            *(_QWORD *)v64 = &unk_4A23A70;
            *(_QWORD *)(v64 + 88) = v72;
            if ( v72 )
            {
              sub_2AAAFA0((__int64 *)(v64 + 88));
              if ( v123 )
                sub_B91220((__int64)&v123, (__int64)v123);
            }
            v46 = v64 + 96;
            sub_2BF0340(v64 + 96, 1, 0, v64);
            *(_QWORD *)v64 = &unk_4A231C8;
            *(_QWORD *)(v64 + 40) = &unk_4A23200;
            *(_QWORD *)(v64 + 96) = &unk_4A23238;
            sub_9C6650(&v122);
            *(_BYTE *)(v64 + 152) = 7;
            *(_DWORD *)(v64 + 156) = 0;
            *(_QWORD *)v64 = &unk_4A23258;
            *(_QWORD *)(v64 + 40) = &unk_4A23290;
            *(_QWORD *)(v64 + 96) = &unk_4A232C8;
            sub_9C6650(&v121);
            *(_BYTE *)(v64 + 160) = 83;
            *(_QWORD *)v64 = &unk_4A23B70;
            *(_QWORD *)(v64 + 96) = &unk_4A23BF0;
            *(_QWORD *)(v64 + 40) = &unk_4A23BB8;
            sub_CA0F50((__int64 *)(v64 + 168), (void **)&v134);
            sub_9C6650(&v120);
            v73 = *v63;
            if ( !*v63 )
              goto LABEL_77;
            goto LABEL_76;
          }
LABEL_104:
          v123 = 0;
          goto LABEL_65;
        }
      }
      else
      {
        v121 = 0;
      }
      v122 = 0;
      goto LABEL_104;
    }
LABEL_78:
    v134 = v104;
    v135 = *(char **)v108;
    v77 = (unsigned __int8)sub_2AC19D0(v105, (__int64 *)&v134, &v123) == 0;
    v78 = v123;
    if ( !v77 )
      goto LABEL_84;
    v131 = v123;
    v79 = *(_DWORD *)(a1 + 80);
    ++*(_QWORD *)(a1 + 64);
    v80 = v79 + 1;
    v81 = *(_DWORD *)(a1 + 88);
    if ( 4 * v80 >= 3 * v81 )
    {
      v81 *= 2;
    }
    else if ( v81 - *(_DWORD *)(a1 + 84) - v80 > v81 >> 3 )
    {
      goto LABEL_81;
    }
    sub_2AD3C30(v105, v81);
    sub_2AC19D0(v105, (__int64 *)&v134, &v131);
    v80 = *(_DWORD *)(a1 + 80) + 1;
    v78 = v131;
LABEL_81:
    *(_DWORD *)(a1 + 80) = v80;
    if ( *v78 != -4096 || v78[1] != -4096 )
      --*(_DWORD *)(a1 + 84);
    *v78 = (__int64)v134;
    v82 = (unsigned __int64)v135;
    v78[2] = 0;
    v78[1] = v82;
LABEL_84:
    v78[2] = v46;
    if ( v107 )
    {
      v138 = 257;
      v83 = *(__int64 **)(a1 + 56);
      v131 = 0;
      v107 = sub_2AB0F70(v83, v107, v46, (__int64 *)&v131, (void **)&v134);
      sub_9C6650(&v131);
    }
    else
    {
      v107 = v46;
    }
    v108 += 72;
  }
  while ( v102 != (__int64 **)v108 );
  if ( !v107 )
  {
LABEL_114:
    v84 = 0;
    goto LABEL_90;
  }
  v131 = 0;
  v138 = 257;
  v84 = sub_2AB0C10(*(_QWORD **)(a1 + 56), v107, (__int64 *)&v131, (void **)&v134);
  sub_9C6650(&v131);
  if ( v106 )
  {
    v85 = *(__int64 **)(a1 + 56);
    v138 = 257;
    v131 = 0;
    v84 = sub_2AB1320(v85, (__int64)v106, v84, (__int64 *)&v131, (void **)&v134);
    sub_9C6650(&v131);
  }
LABEL_90:
  v134 = v104;
  v135 = (char *)v103;
  v77 = (unsigned __int8)sub_2AC19D0(v105, (__int64 *)&v134, &v123) == 0;
  v86 = v123;
  if ( v77 )
  {
    v131 = v123;
    v91 = *(_DWORD *)(a1 + 80);
    ++*(_QWORD *)(a1 + 64);
    v92 = v91 + 1;
    v93 = *(_DWORD *)(a1 + 88);
    if ( 4 * v92 >= 3 * v93 )
    {
      v93 *= 2;
    }
    else if ( v93 - *(_DWORD *)(a1 + 84) - v92 > v93 >> 3 )
    {
      goto LABEL_111;
    }
    sub_2AD3C30(v105, v93);
    sub_2AC19D0(v105, (__int64 *)&v134, &v131);
    v92 = *(_DWORD *)(a1 + 80) + 1;
    v86 = v131;
LABEL_111:
    *(_DWORD *)(a1 + 80) = v92;
    if ( *v86 != -4096 || v86[1] != -4096 )
      --*(_DWORD *)(a1 + 84);
    *v86 = (__int64)v134;
    v94 = (unsigned __int64)v135;
    v86[2] = 0;
    v86[1] = v94;
  }
  v86[2] = v84;
  v87 = (__int64)v129;
  v88 = (unsigned __int64)&v129[9 * (unsigned int)v130];
  if ( v129 != (__int64 **)v88 )
  {
    do
    {
      v88 -= 72LL;
      v89 = *(_QWORD *)(v88 + 8);
      if ( v89 != v88 + 24 )
        _libc_free(v89);
    }
    while ( v87 != v88 );
    v88 = (unsigned __int64)v129;
  }
  if ( (__int64 **)v88 != &v131 )
    _libc_free(v88);
  return sub_C7D6A0(v126, 16LL * v128, 8);
}
