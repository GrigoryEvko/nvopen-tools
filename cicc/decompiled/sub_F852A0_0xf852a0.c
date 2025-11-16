// Function: sub_F852A0
// Address: 0xf852a0
//
__int64 __fastcall sub_F852A0(__int64 *a1, __int64 a2, __int64 a3, unsigned __int64 **a4, __int64 a5)
{
  __int64 *v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rax
  char *v15; // r12
  char *v16; // r8
  __int64 v17; // rbx
  char *v18; // r14
  __int64 v19; // r15
  char *v20; // rax
  char *v21; // r13
  char *v22; // rsi
  __int64 *v23; // rbx
  __int64 v24; // rax
  __int64 v25; // r13
  _QWORD *v26; // rsi
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 *v31; // rax
  __int64 v32; // rax
  __int64 *v33; // rax
  int v34; // esi
  __int64 v35; // rdx
  unsigned int v36; // r10d
  unsigned int v37; // ecx
  _QWORD *v38; // rax
  __int64 v39; // rdi
  unsigned __int8 **v40; // r11
  __int64 v41; // rax
  unsigned __int8 *v42; // rdi
  __int64 v43; // r9
  char v44; // dl
  char v45; // dh
  __int64 v46; // r8
  char v47; // al
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 *v53; // r11
  __int64 v54; // rsi
  __int64 v55; // rsi
  __int64 *v56; // r11
  _BYTE *v57; // rax
  __int64 v58; // r12
  int v59; // r13d
  bool v60; // zf
  unsigned int v61; // eax
  __int64 v62; // r9
  __int64 v63; // rdx
  int v64; // eax
  unsigned __int64 *v65; // rdi
  unsigned __int8 *v66; // rax
  __int64 v67; // rdi
  __int64 v68; // rsi
  __int64 v70; // rdx
  int v71; // eax
  unsigned __int64 *v72; // rdi
  unsigned __int8 *v73; // rax
  __int64 *v74; // rsi
  unsigned __int8 *v75; // r13
  unsigned __int8 **v76; // rax
  unsigned __int64 *v77; // r12
  unsigned __int64 *v78; // rdi
  unsigned __int8 *v79; // rax
  int v80; // r13d
  int v81; // ecx
  int v82; // r11d
  unsigned __int64 *v83; // r12
  unsigned __int64 *v84; // rdi
  unsigned __int8 *v85; // rax
  int v86; // eax
  char v87; // al
  __int64 v88; // r9
  int v89; // r8d
  __int64 v90; // rdx
  unsigned int *v91; // r12
  __int64 v92; // rdx
  _QWORD *v93; // rax
  __int64 v94; // rdx
  __int64 v95; // [rsp+0h] [rbp-210h]
  __int64 *v96; // [rsp+8h] [rbp-208h]
  __int64 v97; // [rsp+8h] [rbp-208h]
  int v98; // [rsp+8h] [rbp-208h]
  __int64 v99; // [rsp+8h] [rbp-208h]
  __int64 *v100; // [rsp+28h] [rbp-1E8h]
  __int64 v101; // [rsp+28h] [rbp-1E8h]
  __int64 *v102; // [rsp+28h] [rbp-1E8h]
  __int64 v103; // [rsp+28h] [rbp-1E8h]
  int v104; // [rsp+28h] [rbp-1E8h]
  __int64 v105; // [rsp+28h] [rbp-1E8h]
  int v106; // [rsp+28h] [rbp-1E8h]
  __int64 v107; // [rsp+28h] [rbp-1E8h]
  __int64 v108; // [rsp+28h] [rbp-1E8h]
  unsigned int *v109; // [rsp+28h] [rbp-1E8h]
  __int64 v110; // [rsp+30h] [rbp-1E0h]
  unsigned int v114; // [rsp+5Ch] [rbp-1B4h]
  __int64 *v116; // [rsp+68h] [rbp-1A8h]
  unsigned __int8 *v117; // [rsp+78h] [rbp-198h] BYREF
  __int64 v118; // [rsp+80h] [rbp-190h] BYREF
  __int64 v119; // [rsp+88h] [rbp-188h]
  __int64 v120; // [rsp+90h] [rbp-180h]
  unsigned int v121; // [rsp+98h] [rbp-178h]
  _BYTE *v122; // [rsp+A0h] [rbp-170h] BYREF
  __int16 v123; // [rsp+C0h] [rbp-150h]
  __int64 v124[4]; // [rsp+D0h] [rbp-140h] BYREF
  __int16 v125; // [rsp+F0h] [rbp-120h]
  void *src; // [rsp+100h] [rbp-110h] BYREF
  __int64 v127; // [rsp+108h] [rbp-108h]
  _BYTE v128[64]; // [rsp+110h] [rbp-100h] BYREF
  _QWORD *v129; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v130; // [rsp+158h] [rbp-B8h]
  _QWORD v131[4]; // [rsp+160h] [rbp-B0h] BYREF
  __int64 v132; // [rsp+180h] [rbp-90h]
  __int64 v133; // [rsp+188h] [rbp-88h]
  __int64 v134; // [rsp+190h] [rbp-80h]
  __int64 v135; // [rsp+198h] [rbp-78h]
  void **v136; // [rsp+1A0h] [rbp-70h]
  void **v137; // [rsp+1A8h] [rbp-68h]
  __int64 v138; // [rsp+1B0h] [rbp-60h]
  int v139; // [rsp+1B8h] [rbp-58h]
  __int16 v140; // [rsp+1BCh] [rbp-54h]
  char v141; // [rsp+1BEh] [rbp-52h]
  __int64 v142; // [rsp+1C0h] [rbp-50h]
  __int64 v143; // [rsp+1C8h] [rbp-48h]
  void *v144; // [rsp+1D0h] [rbp-40h] BYREF
  void *v145; // [rsp+1D8h] [rbp-38h] BYREF

  v5 = a1;
  src = v128;
  v127 = 0x800000000LL;
  v6 = sub_AA5930(**(_QWORD **)(a2 + 32));
  if ( v6 == v7 )
  {
    v11 = (unsigned int)v127;
  }
  else
  {
    v10 = v6;
    v11 = (unsigned int)v127;
    v12 = v7;
    do
    {
      if ( v11 + 1 > (unsigned __int64)HIDWORD(v127) )
      {
        sub_C8D5F0((__int64)&src, v128, v11 + 1, 8u, v8, v9);
        v11 = (unsigned int)v127;
      }
      *((_QWORD *)src + v11) = v10;
      v11 = (unsigned int)(v127 + 1);
      LODWORD(v127) = v127 + 1;
      if ( !v10 )
        BUG();
      v13 = *(_QWORD *)(v10 + 32);
      if ( !v13 )
        BUG();
      v10 = 0;
      if ( *(_BYTE *)(v13 - 24) == 84 )
        v10 = v13 - 24;
    }
    while ( v12 != v10 );
  }
  if ( a5 )
  {
    v14 = 8 * v11;
    v15 = (char *)src;
    v16 = (char *)src + v14;
    v17 = v14 >> 3;
    if ( v14 )
    {
      v18 = (char *)src + v14;
      do
      {
        v19 = 8 * v17;
        v20 = (char *)sub_2207800(8 * v17, &unk_435FF63);
        v21 = v20;
        if ( v20 )
        {
          v22 = v18;
          v5 = a1;
          sub_F7C940(v15, v22, v20, v17);
          goto LABEL_15;
        }
        v17 >>= 1;
      }
      while ( v17 );
      v16 = v18;
      v5 = a1;
    }
    v19 = 0;
    v21 = 0;
    sub_F7A870(v15, v16);
LABEL_15:
    j_j___libc_free_0(v21, v19);
    v11 = (unsigned int)v127;
  }
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  v116 = (__int64 *)((char *)src + 8 * v11);
  if ( src != v116 )
  {
    v114 = 0;
    v23 = (__int64 *)src;
    while ( 1 )
    {
      while ( 1 )
      {
        v24 = *v5;
        v25 = *v23;
        v26 = (_QWORD *)v5[1];
        v27 = *(_QWORD *)(*v5 + 24);
        v28 = *(_QWORD *)(*v5 + 40);
        v117 = (unsigned __int8 *)v25;
        v29 = *(_QWORD *)(v24 + 32);
        v129 = v26;
        LOWORD(v134) = 257;
        v130 = v27;
        v131[0] = 0;
        v131[1] = v28;
        v131[2] = v29;
        v131[3] = 0;
        v132 = 0;
        v133 = 0;
        v30 = sub_1020E10(v25, &v129, v28, v27, v8, v9);
        if ( v30 )
          goto LABEL_18;
        if ( sub_D97040(*v5, *(_QWORD *)(v25 + 8)) )
        {
          v31 = sub_DD8400(*v5, v25);
          if ( !*((_WORD *)v31 + 12) )
          {
            v32 = v31[4];
            if ( v32 )
            {
              v30 = v32;
LABEL_18:
              if ( *(_QWORD *)(v30 + 8) == *((_QWORD *)v117 + 1) )
              {
                sub_DAC8D0(*v5, v117);
                sub_BD84D0((__int64)v117, v30);
                v70 = *((unsigned int *)a4 + 2);
                v71 = v70;
                if ( *((_DWORD *)a4 + 3) <= (unsigned int)v70 )
                {
                  v77 = (unsigned __int64 *)sub_C8D7D0(
                                              (__int64)a4,
                                              (__int64)(a4 + 2),
                                              0,
                                              0x18u,
                                              (unsigned __int64 *)&v129,
                                              v9);
                  v78 = &v77[3 * *((unsigned int *)a4 + 2)];
                  if ( v78 )
                  {
                    v79 = v117;
                    v78[1] = 0;
                    *v78 = 6;
                    v78[2] = (unsigned __int64)v79;
                    if ( v79 != 0 && v79 + 4096 != 0 && v79 != (unsigned __int8 *)-8192LL )
                      sub_BD73F0((__int64)v78);
                  }
                  sub_F17F80((__int64)a4, v77);
                  v80 = (int)v129;
                  if ( a4 + 2 != (unsigned __int64 **)*a4 )
                    _libc_free(*a4, v77);
                  ++*((_DWORD *)a4 + 2);
                  *a4 = v77;
                  *((_DWORD *)a4 + 3) = v80;
                }
                else
                {
                  v72 = &(*a4)[3 * v70];
                  if ( v72 )
                  {
                    v73 = v117;
                    v72[1] = 0;
                    *v72 = 6;
                    v72[2] = (unsigned __int64)v73;
                    if ( v73 != 0 && v73 + 4096 != 0 && v73 != (unsigned __int8 *)-8192LL )
                      sub_BD73F0((__int64)v72);
                    v71 = *((_DWORD *)a4 + 2);
                  }
                  *((_DWORD *)a4 + 2) = v71 + 1;
                }
                ++v114;
              }
              goto LABEL_19;
            }
          }
        }
        if ( sub_D97040(*v5, *((_QWORD *)v117 + 1)) )
          break;
LABEL_19:
        if ( v116 == ++v23 )
          goto LABEL_53;
      }
      v33 = sub_DD8400(*v5, (__int64)v117);
      v34 = v121;
      v124[0] = (__int64)v33;
      v35 = (__int64)v33;
      if ( v121 )
      {
        v36 = v121 - 1;
        v9 = v119;
        v8 = 1;
        v37 = (v121 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        v38 = (_QWORD *)(v119 + 16LL * v37);
        v39 = *v38;
        if ( v35 == *v38 )
          goto LABEL_28;
        while ( v39 != -4096 )
        {
          if ( !v30 && v39 == -8192 )
            v30 = (__int64)v38;
          v82 = v8 + 1;
          v8 = v37 + (unsigned int)v8;
          v37 = v36 & v8;
          v38 = (_QWORD *)(v119 + 16LL * (v36 & (unsigned int)v8));
          v39 = *v38;
          if ( v35 == *v38 )
            goto LABEL_28;
          LODWORD(v8) = v82;
        }
        if ( v30 )
          v38 = (_QWORD *)v30;
        ++v118;
        v81 = v120 + 1;
        v129 = v38;
        if ( 4 * ((int)v120 + 1) < 3 * v121 )
        {
          v8 = v121 >> 3;
          if ( v121 - HIDWORD(v120) - v81 > (unsigned int)v8 )
            goto LABEL_87;
          goto LABEL_86;
        }
      }
      else
      {
        ++v118;
        v129 = 0;
      }
      v34 = 2 * v121;
LABEL_86:
      sub_F84F20((__int64)&v118, v34);
      sub_F81DD0((__int64)&v118, v124, &v129);
      v35 = v124[0];
      v81 = v120 + 1;
      v38 = v129;
LABEL_87:
      LODWORD(v120) = v81;
      if ( *v38 != -4096 )
        --HIDWORD(v120);
      *v38 = v35;
      v38[1] = 0;
LABEL_28:
      v40 = (unsigned __int8 **)(v38 + 1);
      v41 = v38[1];
      if ( !v41 )
      {
        *v40 = v117;
        if ( *(_BYTE *)(*((_QWORD *)v117 + 1) + 8LL) == 12 )
        {
          if ( a5 )
          {
            if ( (unsigned __int8)sub_DFA860(a5) )
            {
              v74 = sub_DD8400(*v5, (__int64)v117);
              if ( *((_WORD *)v74 + 12) == 8 )
              {
                v124[0] = (__int64)sub_DC5200(
                                     *v5,
                                     (__int64)v74,
                                     *(_QWORD *)(*((_QWORD *)src + (unsigned int)v127 - 1) + 8LL),
                                     0);
                v75 = v117;
                if ( (unsigned __int8)sub_F81DD0((__int64)&v118, v124, &v129) )
                {
                  v76 = (unsigned __int8 **)(v129 + 1);
                }
                else
                {
                  v93 = sub_F85100((__int64)&v118, v124, v129);
                  v94 = v124[0];
                  v76 = (unsigned __int8 **)(v93 + 1);
                  *v76 = 0;
                  *(v76 - 1) = (unsigned __int8 *)v94;
                }
                *v76 = v75;
              }
            }
          }
        }
        goto LABEL_19;
      }
      if ( (*(_BYTE *)(*((_QWORD *)v117 + 1) + 8LL) == 14) != (*(_BYTE *)(*(_QWORD *)(v41 + 8) + 8LL) == 14) )
        goto LABEL_19;
      v100 = (__int64 *)v40;
      sub_F84490((__int64)v5, &v117, v40, a2, a3, (__int64)a4);
      v42 = v117;
      ++v114;
      v43 = *v100;
      if ( *((_QWORD *)v117 + 1) != *(_QWORD *)(*v100 + 8) )
      {
        v46 = sub_AA5190(**(_QWORD **)(a2 + 32));
        if ( v46 )
        {
          v47 = v45;
        }
        else
        {
          v47 = 0;
          v44 = 0;
        }
        v96 = v100;
        v95 = v46;
        v48 = v110;
        LOBYTE(v48) = v44;
        v101 = **(_QWORD **)(a2 + 32);
        BYTE1(v48) = v47;
        v110 = v48;
        v49 = sub_AA48A0(v101);
        v141 = 7;
        v135 = v49;
        v136 = &v144;
        v137 = &v145;
        v129 = v131;
        v144 = &unk_49DA100;
        v140 = 512;
        LOWORD(v134) = 0;
        v130 = 0x200000000LL;
        v145 = &unk_49DA0B0;
        v138 = 0;
        v139 = 0;
        v142 = 0;
        v143 = 0;
        v132 = 0;
        v133 = 0;
        sub_A88F30((__int64)&v129, v101, v95, v110);
        v53 = v96;
        v124[0] = *((_QWORD *)v117 + 6);
        v54 = v124[0];
        if ( v124[0] )
        {
          sub_B96E90((__int64)v124, v124[0], 1);
          v54 = v124[0];
          v53 = v96;
        }
        v102 = v53;
        sub_F80810((__int64)&v129, 0, v54, v50, v51, v52);
        v55 = v124[0];
        v56 = v102;
        if ( v124[0] )
        {
          sub_B91220((__int64)v124, v124[0]);
          v56 = v102;
        }
        v57 = (_BYTE *)v5[2];
        v123 = 257;
        if ( *v57 )
        {
          v122 = v57;
          LOBYTE(v123) = 3;
        }
        v58 = *((_QWORD *)v117 + 1);
        v97 = *v56;
        v103 = *(_QWORD *)(*v56 + 8);
        v59 = sub_BCB060(v103);
        v60 = v59 == (unsigned int)sub_BCB060(v58);
        v61 = 38;
        if ( v60 )
          v61 = 49;
        if ( v58 == v103 )
        {
          v62 = v97;
        }
        else
        {
          v55 = v61;
          v104 = v61;
          v62 = (*((__int64 (__fastcall **)(void **, _QWORD, __int64, __int64))*v136 + 15))(v136, v61, v97, v58);
          if ( !v62 )
          {
            v125 = 257;
            v107 = sub_B51D30(v104, v97, v58, (__int64)v124, 0, 0);
            v87 = sub_920620(v107);
            v88 = v107;
            if ( v87 )
            {
              v89 = v139;
              if ( v138 )
              {
                v98 = v139;
                sub_B99FD0(v107, 3u, v138);
                v89 = v98;
                v88 = v107;
              }
              v108 = v88;
              sub_B45150(v88, v89);
              v88 = v108;
            }
            v55 = v88;
            v99 = v88;
            (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64))*v137 + 2))(
              v137,
              v88,
              &v122,
              v133,
              v134);
            v62 = v99;
            v90 = 2LL * (unsigned int)v130;
            v109 = (unsigned int *)&v129[v90];
            if ( v129 != &v129[v90] )
            {
              v91 = (unsigned int *)v129;
              do
              {
                v92 = *((_QWORD *)v91 + 1);
                v55 = *v91;
                v91 += 4;
                sub_B99FD0(v99, v55, v92);
              }
              while ( v109 != v91 );
              v62 = v99;
            }
          }
        }
        v105 = v62;
        nullsub_61();
        v144 = &unk_49DA100;
        nullsub_63();
        v43 = v105;
        if ( v129 != v131 )
        {
          _libc_free(v129, v55);
          v43 = v105;
        }
        v42 = v117;
      }
      sub_BD84D0((__int64)v42, v43);
      v63 = *((unsigned int *)a4 + 2);
      v64 = v63;
      if ( *((_DWORD *)a4 + 3) <= (unsigned int)v63 )
      {
        v83 = (unsigned __int64 *)sub_C8D7D0((__int64)a4, (__int64)(a4 + 2), 0, 0x18u, (unsigned __int64 *)&v129, v9);
        v84 = &v83[3 * *((unsigned int *)a4 + 2)];
        if ( v84 )
        {
          v85 = v117;
          v84[1] = 0;
          *v84 = 6;
          v84[2] = (unsigned __int64)v85;
          if ( v85 + 4096 != 0 && v85 != 0 && v85 != (unsigned __int8 *)-8192LL )
            sub_BD73F0((__int64)v84);
        }
        sub_F17F80((__int64)a4, v83);
        v86 = (int)v129;
        if ( a4 + 2 != (unsigned __int64 **)*a4 )
        {
          v106 = (int)v129;
          _libc_free(*a4, v83);
          v86 = v106;
        }
        ++*((_DWORD *)a4 + 2);
        *a4 = v83;
        *((_DWORD *)a4 + 3) = v86;
        goto LABEL_19;
      }
      v65 = &(*a4)[3 * v63];
      if ( v65 )
      {
        v66 = v117;
        v65[1] = 0;
        *v65 = 6;
        v65[2] = (unsigned __int64)v66;
        if ( v66 + 4096 != 0 && v66 != 0 && v66 != (unsigned __int8 *)-8192LL )
          sub_BD73F0((__int64)v65);
        v64 = *((_DWORD *)a4 + 2);
      }
      ++v23;
      *((_DWORD *)a4 + 2) = v64 + 1;
      if ( v116 == v23 )
      {
LABEL_53:
        v67 = v119;
        v68 = 16LL * v121;
        goto LABEL_54;
      }
    }
  }
  v114 = 0;
  v67 = 0;
  v68 = 0;
LABEL_54:
  sub_C7D6A0(v67, v68, 8);
  if ( src != v128 )
    _libc_free(src, v68);
  return v114;
}
