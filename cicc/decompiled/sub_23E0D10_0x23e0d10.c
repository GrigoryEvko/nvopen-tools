// Function: sub_23E0D10
// Address: 0x23e0d10
//
void __fastcall sub_23E0D10(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  _QWORD *v4; // rax
  __int64 v5; // r15
  __int64 v6; // rsi
  __int64 v7; // rax
  int v8; // eax
  __int64 v9; // r10
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r15
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // r9
  __int64 v18; // r15
  unsigned int v19; // ebx
  unsigned int v20; // ebx
  __int64 v21; // r9
  __int64 v22; // rax
  unsigned __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 *v28; // rdi
  _QWORD *v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r15
  unsigned int *v33; // rax
  int v34; // ecx
  unsigned int *v35; // rdx
  __int64 v36; // rbx
  int v37; // eax
  __int64 v38; // rax
  __int64 v39; // r15
  __int64 v40; // r9
  __int64 v41; // r15
  unsigned int v42; // ebx
  unsigned int v43; // eax
  __int64 v44; // r15
  unsigned int v45; // ebx
  unsigned int v46; // eax
  __int64 v47; // rdx
  char v48; // al
  __int64 v49; // r9
  int v50; // r14d
  unsigned int *v51; // r14
  unsigned int *v52; // rbx
  __int64 v53; // rdx
  unsigned int v54; // esi
  char v55; // al
  __int64 v56; // r9
  int v57; // r15d
  unsigned int *v58; // r12
  unsigned int *v59; // r15
  __int64 v60; // rbx
  __int64 v61; // rdx
  unsigned int v62; // esi
  char v63; // al
  __int64 v64; // r9
  int v65; // r15d
  unsigned int *v66; // r12
  unsigned int *v67; // r15
  __int64 v68; // rbx
  __int64 v69; // rdx
  unsigned int v70; // esi
  char v71; // al
  __int64 v72; // r9
  int v73; // r15d
  unsigned int *v74; // r12
  unsigned int *v75; // r15
  __int64 v76; // rbx
  __int64 v77; // rdx
  unsigned int v78; // esi
  unsigned __int64 v79; // rsi
  __int64 v80; // [rsp+18h] [rbp-208h]
  __int64 v81; // [rsp+20h] [rbp-200h]
  __int64 v82; // [rsp+20h] [rbp-200h]
  __int64 v83; // [rsp+20h] [rbp-200h]
  __int64 v84; // [rsp+20h] [rbp-200h]
  __int64 v85; // [rsp+20h] [rbp-200h]
  __int64 v86; // [rsp+20h] [rbp-200h]
  __int64 v87; // [rsp+20h] [rbp-200h]
  __int64 v88; // [rsp+20h] [rbp-200h]
  __int64 v89; // [rsp+20h] [rbp-200h]
  __int64 v90; // [rsp+20h] [rbp-200h]
  __int64 v91; // [rsp+20h] [rbp-200h]
  __int64 v92; // [rsp+20h] [rbp-200h]
  __int64 v93; // [rsp+20h] [rbp-200h]
  __int64 v94; // [rsp+20h] [rbp-200h]
  __int64 v95; // [rsp+20h] [rbp-200h]
  __int64 v96; // [rsp+20h] [rbp-200h]
  __int64 v97; // [rsp+20h] [rbp-200h]
  __int64 v98; // [rsp+20h] [rbp-200h]
  __int64 v99; // [rsp+20h] [rbp-200h]
  __int64 v102; // [rsp+48h] [rbp-1D8h]
  _QWORD v103[4]; // [rsp+50h] [rbp-1D0h] BYREF
  _QWORD v104[4]; // [rsp+70h] [rbp-1B0h] BYREF
  __int16 v105; // [rsp+90h] [rbp-190h]
  _BYTE v106[32]; // [rsp+A0h] [rbp-180h] BYREF
  __int16 v107; // [rsp+C0h] [rbp-160h]
  _BYTE v108[32]; // [rsp+D0h] [rbp-150h] BYREF
  __int16 v109; // [rsp+F0h] [rbp-130h]
  _BYTE v110[32]; // [rsp+100h] [rbp-120h] BYREF
  __int16 v111; // [rsp+120h] [rbp-100h]
  int v112[8]; // [rsp+130h] [rbp-F0h] BYREF
  __int16 v113; // [rsp+150h] [rbp-D0h]
  unsigned int *v114; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v115; // [rsp+168h] [rbp-B8h]
  _BYTE v116[32]; // [rsp+170h] [rbp-B0h] BYREF
  __int64 v117; // [rsp+190h] [rbp-90h]
  __int64 v118; // [rsp+198h] [rbp-88h]
  __int64 v119; // [rsp+1A0h] [rbp-80h]
  _QWORD *v120; // [rsp+1A8h] [rbp-78h]
  void **v121; // [rsp+1B0h] [rbp-70h]
  void **v122; // [rsp+1B8h] [rbp-68h]
  __int64 v123; // [rsp+1C0h] [rbp-60h]
  int v124; // [rsp+1C8h] [rbp-58h]
  __int16 v125; // [rsp+1CCh] [rbp-54h]
  char v126; // [rsp+1CEh] [rbp-52h]
  __int64 v127; // [rsp+1D0h] [rbp-50h]
  __int64 v128; // [rsp+1D8h] [rbp-48h]
  void *v129; // [rsp+1E0h] [rbp-40h] BYREF
  void *v130; // [rsp+1E8h] [rbp-38h] BYREF

  v3 = a2;
  v4 = (_QWORD *)sub_BD5C60(a2);
  v125 = 512;
  v120 = v4;
  v121 = &v129;
  v122 = &v130;
  v114 = (unsigned int *)v116;
  v129 = &unk_49DA100;
  LOWORD(v119) = 0;
  v115 = 0x200000000LL;
  v130 = &unk_49DA0B0;
  v123 = 0;
  v124 = 0;
  v126 = 7;
  v127 = 0;
  v128 = 0;
  v117 = 0;
  v118 = 0;
  sub_D5F1F0((__int64)&v114, a2);
  v5 = sub_B43CB0(a2);
  sub_B33910(v112, (__int64 *)&v114);
  v6 = *(_QWORD *)v112;
  if ( *(_QWORD *)v112 )
    goto LABEL_2;
  v27 = sub_B92180(v5);
  if ( !v27 )
    goto LABEL_3;
  v28 = (__int64 *)(*(_QWORD *)(v27 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(v27 + 8) & 4) != 0 )
    v28 = (__int64 *)*v28;
  v29 = sub_B01860(v28, 0, 0, v27, 0, 0, 0, 1);
  sub_B10CB0(v112, (__int64)v29);
  v32 = *(_QWORD *)v112;
  if ( !*(_QWORD *)v112 )
  {
    sub_93FB40((__int64)&v114, 0);
    v32 = *(_QWORD *)v112;
    goto LABEL_72;
  }
  v33 = v114;
  v34 = v115;
  v35 = &v114[4 * (unsigned int)v115];
  if ( v114 == v35 )
  {
LABEL_68:
    if ( (unsigned int)v115 >= (unsigned __int64)HIDWORD(v115) )
    {
      v79 = (unsigned int)v115 + 1LL;
      if ( HIDWORD(v115) < v79 )
      {
        sub_C8D5F0((__int64)&v114, v116, v79, 0x10u, v30, v31);
        v35 = &v114[4 * (unsigned int)v115];
      }
      *(_QWORD *)v35 = 0;
      *((_QWORD *)v35 + 1) = v32;
      v32 = *(_QWORD *)v112;
      LODWORD(v115) = v115 + 1;
    }
    else
    {
      if ( v35 )
      {
        *v35 = 0;
        *((_QWORD *)v35 + 1) = v32;
        v34 = v115;
        v32 = *(_QWORD *)v112;
      }
      LODWORD(v115) = v34 + 1;
    }
LABEL_72:
    if ( !v32 )
      goto LABEL_3;
    goto LABEL_35;
  }
  while ( *v33 )
  {
    v33 += 4;
    if ( v35 == v33 )
      goto LABEL_68;
  }
  *((_QWORD *)v33 + 1) = *(_QWORD *)v112;
LABEL_35:
  v6 = v32;
LABEL_2:
  sub_B91220((__int64)v112, v6);
LABEL_3:
  v7 = *(_QWORD *)(v3 - 32);
  if ( !v7 || *(_BYTE *)v7 )
    BUG();
  if ( *(_QWORD *)(v7 + 24) != *(_QWORD *)(v3 + 80) )
    goto LABEL_89;
  v8 = *(_DWORD *)(v7 + 36);
  if ( v8 == 238 || (unsigned int)(v8 - 240) <= 1 )
  {
    v111 = 257;
    v9 = a1[14];
    v10 = *(_DWORD *)(v3 + 4);
    v105 = 257;
    v11 = v10 & 0x7FFFFFF;
    v12 = *(_QWORD *)(v3 - 32 * v11);
    v13 = *(_QWORD *)(v12 + 8);
    if ( v9 == v13 )
    {
      v14 = *(_QWORD *)(v3 - 32 * v11);
    }
    else
    {
      v81 = v9;
      v14 = (*((__int64 (__fastcall **)(void **, __int64, _QWORD, __int64))*v121 + 15))(
              v121,
              50,
              *(_QWORD *)(v3 - 32 * v11),
              v9);
      if ( !v14 )
      {
        v113 = 257;
        v92 = sub_B51D30(50, v12, v81, (__int64)v112, 0, 0);
        v63 = sub_920620(v92);
        v64 = v92;
        if ( v63 )
        {
          v65 = v124;
          if ( v123 )
          {
            sub_B99FD0(v92, 3u, v123);
            v64 = v92;
          }
          v93 = v64;
          sub_B45150(v64, v65);
          v64 = v93;
        }
        v94 = v64;
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v122 + 2))(v122, v64, v104, v118, v119);
        v14 = v94;
        if ( v114 != &v114[4 * (unsigned int)v115] )
        {
          v95 = v3;
          v66 = v114;
          v67 = &v114[4 * (unsigned int)v115];
          v68 = v14;
          do
          {
            v69 = *((_QWORD *)v66 + 1);
            v70 = *v66;
            v66 += 4;
            sub_B99FD0(v68, v70, v69);
          }
          while ( v67 != v66 );
          v3 = v95;
          v14 = v68;
        }
      }
      v13 = a1[14];
      v11 = *(_DWORD *)(v3 + 4) & 0x7FFFFFF;
    }
    v103[0] = v14;
    v107 = 257;
    v15 = 32 * (1 - v11);
    v16 = *(_QWORD *)(v3 + v15);
    if ( v13 == *(_QWORD *)(v16 + 8) )
    {
      v17 = *(_QWORD *)(v3 + v15);
    }
    else
    {
      v17 = (*((__int64 (__fastcall **)(void **, __int64, _QWORD, __int64))*v121 + 15))(
              v121,
              50,
              *(_QWORD *)(v3 + v15),
              v13);
      if ( !v17 )
      {
        v113 = 257;
        v88 = sub_B51D30(50, v16, v13, (__int64)v112, 0, 0);
        v55 = sub_920620(v88);
        v56 = v88;
        if ( v55 )
        {
          v57 = v124;
          if ( v123 )
          {
            sub_B99FD0(v88, 3u, v123);
            v56 = v88;
          }
          v89 = v56;
          sub_B45150(v56, v57);
          v56 = v89;
        }
        v90 = v56;
        (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v122 + 2))(v122, v56, v106, v118, v119);
        v17 = v90;
        if ( v114 != &v114[4 * (unsigned int)v115] )
        {
          v91 = v3;
          v58 = v114;
          v59 = &v114[4 * (unsigned int)v115];
          v60 = v17;
          do
          {
            v61 = *((_QWORD *)v58 + 1);
            v62 = *v58;
            v58 += 4;
            sub_B99FD0(v60, v62, v61);
          }
          while ( v59 != v58 );
          v3 = v91;
          v17 = v60;
        }
      }
      v11 = *(_DWORD *)(v3 + 4) & 0x7FFFFFF;
    }
    v103[1] = v17;
    v109 = 257;
    v18 = a1[12];
    v80 = *(_QWORD *)(v3 + 32 * (2 - v11));
    v82 = *(_QWORD *)(v80 + 8);
    v19 = sub_BCB060(v82);
    v20 = (v19 <= (unsigned int)sub_BCB060(v18)) + 38;
    if ( v18 == v82 )
    {
      v21 = v80;
    }
    else
    {
      v21 = (*((__int64 (__fastcall **)(void **, _QWORD, __int64, __int64))*v121 + 15))(v121, v20, v80, v18);
      if ( !v21 )
      {
        v113 = 257;
        v85 = sub_B51D30(v20, v80, v18, (__int64)v112, 0, 0);
        v48 = sub_920620(v85);
        v49 = v85;
        if ( v48 )
        {
          v50 = v124;
          if ( v123 )
          {
            sub_B99FD0(v85, 3u, v123);
            v49 = v85;
          }
          v86 = v49;
          sub_B45150(v49, v50);
          v49 = v86;
        }
        v87 = v49;
        (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v122 + 2))(v122, v49, v108, v118, v119);
        v51 = v114;
        v21 = v87;
        v52 = &v114[4 * (unsigned int)v115];
        if ( v114 != v52 )
        {
          do
          {
            v53 = *((_QWORD *)v51 + 1);
            v54 = *v51;
            v51 += 4;
            sub_B99FD0(v87, v54, v53);
          }
          while ( v52 != v51 );
          v21 = v87;
        }
      }
    }
    v22 = *(_QWORD *)(v3 - 32);
    v103[2] = v21;
    if ( v22 && !*(_BYTE *)v22 && *(_QWORD *)(v22 + 24) == *(_QWORD *)(v3 + 80) )
    {
      if ( *(_DWORD *)(v22 + 36) == 241 )
      {
        v23 = a1[121];
        v24 = a1[122];
      }
      else
      {
        v23 = a1[123];
        v24 = a1[124];
      }
      v25 = sub_921880(&v114, v23, v24, (int)v103, 3, (__int64)v110, 0);
      if ( *(_BYTE *)(a3 + 8) )
      {
        v47 = *(unsigned int *)(a3 + 24);
        if ( v47 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 28) )
        {
          v102 = v25;
          sub_C8D5F0(a3 + 16, (const void *)(a3 + 32), v47 + 1, 8u, v47 + 1, v26);
          v47 = *(unsigned int *)(a3 + 24);
          v25 = v102;
        }
        *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v47) = v25;
        ++*(_DWORD *)(a3 + 24);
      }
      goto LABEL_22;
    }
LABEL_89:
    BUG();
  }
  if ( ((v8 - 243) & 0xFFFFFFFD) == 0 )
  {
    v111 = 257;
    v36 = a1[14];
    v37 = *(_DWORD *)(v3 + 4);
    v107 = 257;
    v38 = -32LL * (v37 & 0x7FFFFFF);
    v39 = *(_QWORD *)(v3 + v38);
    if ( v36 == *(_QWORD *)(v39 + 8) )
    {
      v40 = *(_QWORD *)(v3 + v38);
    }
    else
    {
      v40 = (*((__int64 (__fastcall **)(void **, __int64, _QWORD, __int64))*v121 + 15))(
              v121,
              50,
              *(_QWORD *)(v3 + v38),
              v36);
      if ( !v40 )
      {
        v113 = 257;
        v96 = sub_B51D30(50, v39, v36, (__int64)v112, 0, 0);
        v71 = sub_920620(v96);
        v72 = v96;
        if ( v71 )
        {
          v73 = v124;
          if ( v123 )
          {
            sub_B99FD0(v96, 3u, v123);
            v72 = v96;
          }
          v97 = v72;
          sub_B45150(v72, v73);
          v72 = v97;
        }
        v98 = v72;
        (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v122 + 2))(v122, v72, v106, v118, v119);
        v40 = v98;
        if ( v114 != &v114[4 * (unsigned int)v115] )
        {
          v99 = v3;
          v74 = v114;
          v75 = &v114[4 * (unsigned int)v115];
          v76 = v40;
          do
          {
            v77 = *((_QWORD *)v74 + 1);
            v78 = *v74;
            v74 += 4;
            sub_B99FD0(v76, v78, v77);
          }
          while ( v75 != v74 );
          v3 = v99;
          v40 = v76;
        }
      }
    }
    v104[0] = v40;
    v109 = 257;
    v41 = sub_BCB2D0(v120);
    v83 = *(_QWORD *)(v3 + 32 * (1LL - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
    v42 = sub_BCB060(*(_QWORD *)(v83 + 8));
    v43 = sub_BCB060(v41);
    v104[1] = sub_23DDFD0((__int64 *)&v114, (unsigned int)(v42 <= v43) + 38, v83, v41, (__int64)v108, 0, v112[0], 0);
    v113 = 257;
    v44 = a1[12];
    v84 = *(_QWORD *)(v3 + 32 * (2LL - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
    v45 = sub_BCB060(*(_QWORD *)(v84 + 8));
    v46 = sub_BCB060(v44);
    v104[2] = sub_23DDFD0((__int64 *)&v114, (unsigned int)(v45 <= v46) + 38, v84, v44, (__int64)v112, 0, v103[0], 0);
    sub_23DEC70(a3, &v114, a1[125], a1[126], (int)v104, 3, (__int64)v110);
  }
LABEL_22:
  sub_B43D60((_QWORD *)v3);
  nullsub_61();
  v129 = &unk_49DA100;
  nullsub_63();
  if ( v114 != (unsigned int *)v116 )
    _libc_free((unsigned __int64)v114);
}
