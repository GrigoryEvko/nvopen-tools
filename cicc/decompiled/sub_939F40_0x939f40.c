// Function: sub_939F40
// Address: 0x939f40
//
__int64 __fastcall sub_939F40(__int64 a1, __int64 a2, __int64 **a3, _DWORD *a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  _QWORD *v8; // r15
  _BYTE *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rax
  int v13; // edx
  __int64 v14; // rdx
  bool v15; // cc
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // r8
  __int64 v18; // rcx
  __int64 i; // rdx
  char v20; // al
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // r13
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r15
  __int64 v30; // r14
  int v31; // eax
  __int64 v32; // rdi
  __int64 (__fastcall *v33)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v34; // r12
  int v35; // r15d
  __int64 v36; // rax
  char v37; // al
  __int16 v38; // cx
  __int64 v39; // rax
  int v40; // r9d
  __int64 v41; // r14
  unsigned int *v42; // r15
  unsigned int *v43; // r13
  __int64 v44; // rdx
  __int64 v45; // rsi
  __int64 v46; // r13
  unsigned int v47; // r12d
  int v48; // r8d
  __int64 v49; // r12
  int v51; // r15d
  __int64 v52; // r12
  __int64 v53; // rax
  char v54; // al
  __int16 v55; // cx
  __int64 v56; // rax
  unsigned __int64 v57; // r14
  unsigned int *v58; // r15
  unsigned int *v59; // r13
  __int64 v60; // rdx
  __int64 v61; // rsi
  __int64 v62; // rdi
  __int64 v63; // rax
  _BYTE *v64; // rax
  __int64 v65; // rdi
  __int64 (__fastcall *v66)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v67; // rax
  int v68; // r10d
  _BYTE **v69; // rcx
  __int64 v70; // r11
  __int64 v71; // rcx
  __int64 v72; // rax
  __int64 v73; // rdx
  unsigned int *v74; // rbx
  unsigned int *v75; // r12
  __int64 v76; // rdx
  __int64 v77; // rsi
  __int64 v78; // rdx
  int v79; // eax
  char v80; // si
  int v81; // eax
  __int64 v82; // rax
  __int64 v83; // rdx
  unsigned int v84; // r14d
  unsigned int *v85; // r15
  unsigned int *v86; // r14
  __int64 v87; // rdx
  __int64 v88; // rsi
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rdx
  __int64 v94; // rdx
  __int64 v95; // r15
  __int64 v96; // rax
  unsigned __int64 v97; // rdx
  __int64 v98; // rdx
  __int64 v99; // rax
  _BYTE *v100; // rax
  __int64 v101; // rax
  __int64 v102; // [rsp+0h] [rbp-1D0h]
  unsigned int v103; // [rsp+0h] [rbp-1D0h]
  int v104; // [rsp+8h] [rbp-1C8h]
  unsigned __int64 v105; // [rsp+10h] [rbp-1C0h]
  __int64 v107; // [rsp+28h] [rbp-1A8h]
  __int64 v108; // [rsp+30h] [rbp-1A0h]
  __int64 v109; // [rsp+38h] [rbp-198h]
  __int64 v110; // [rsp+40h] [rbp-190h]
  __int64 v111; // [rsp+48h] [rbp-188h]
  int v112; // [rsp+50h] [rbp-180h]
  __int64 v113; // [rsp+58h] [rbp-178h]
  unsigned int v114; // [rsp+68h] [rbp-168h]
  __int16 v115; // [rsp+6Ch] [rbp-164h]
  __int16 v116; // [rsp+6Eh] [rbp-162h]
  __int64 v117; // [rsp+70h] [rbp-160h]
  unsigned __int64 *v118; // [rsp+70h] [rbp-160h]
  __int64 v119; // [rsp+78h] [rbp-158h]
  int v120; // [rsp+78h] [rbp-158h]
  unsigned int v121; // [rsp+78h] [rbp-158h]
  _BYTE *v122; // [rsp+80h] [rbp-150h] BYREF
  __int64 v123; // [rsp+88h] [rbp-148h] BYREF
  __int64 v124; // [rsp+90h] [rbp-140h] BYREF
  _BYTE *v125; // [rsp+98h] [rbp-138h]
  _BYTE *v126; // [rsp+A0h] [rbp-130h]
  _QWORD v127[4]; // [rsp+B0h] [rbp-120h] BYREF
  __int16 v128; // [rsp+D0h] [rbp-100h]
  char *v129; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v130; // [rsp+E8h] [rbp-E8h]
  __int64 v131; // [rsp+F0h] [rbp-E0h]
  unsigned int v132; // [rsp+F8h] [rbp-D8h]
  __int16 v133; // [rsp+100h] [rbp-D0h]
  _QWORD *v134; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v135; // [rsp+118h] [rbp-B8h]
  _QWORD v136[22]; // [rsp+120h] [rbp-B0h] BYREF

  v6 = a1;
  v7 = sub_BCB2B0(*(_QWORD *)(a1 + 40));
  v124 = 0;
  v107 = v7;
  v125 = 0;
  v126 = 0;
  v134 = (_QWORD *)sub_BCE760(v7, 0);
  v8 = v134;
  sub_9183A0((__int64)&v124, 0, &v134);
  v100 = v125;
  v134 = v8;
  if ( v126 == v125 )
  {
    sub_9183A0((__int64)&v124, v125, &v134);
    v9 = v125;
  }
  else
  {
    if ( v125 )
    {
      *(_QWORD *)v125 = v8;
      v100 = v125;
    }
    v9 = v100 + 8;
    v125 = v9;
  }
  v117 = v124;
  v119 = (__int64)&v9[-v124] >> 3;
  v10 = sub_BCB2D0(*(_QWORD *)(a1 + 40));
  v11 = sub_BCF480(v10, v117, v119, 0);
  v105 = sub_BA8CA0(**(_QWORD **)(a1 + 32), "vprintf", 7, v11);
  v134 = v136;
  v135 = 0x1000000000LL;
  v12 = *a3;
  v104 = v13;
  v14 = **a3;
  v15 = *((_DWORD *)a3 + 2) <= 1u;
  LODWORD(v135) = 1;
  v118 = (unsigned __int64 *)(v12 + 1);
  v136[0] = v14;
  if ( !v15 )
  {
    v111 = *(_QWORD *)(a1 + 224);
    if ( v111 )
    {
      v16 = 16;
      v17 = 2;
      v18 = 1;
    }
    else
    {
      v129 = "tmp";
      v133 = 259;
      v101 = sub_921B80(a1, v107, (__int64)&v129, 0x103u, 0);
      v18 = (unsigned int)v135;
      v16 = HIDWORD(v135);
      v111 = v101;
      *(_QWORD *)(a1 + 224) = v101;
      v17 = v18 + 1;
    }
    for ( i = v111; ; i = *(_QWORD *)(i - 32) )
    {
      v20 = *(_BYTE *)i;
      if ( *(_BYTE *)i <= 0x1Cu )
        goto LABEL_16;
      if ( v20 != 78 )
        break;
    }
    if ( v20 != 85 )
    {
      if ( v20 == 60 )
      {
        v108 = i;
        v21 = i;
        goto LABEL_17;
      }
LABEL_16:
      v108 = 0;
      v21 = 0;
      goto LABEL_17;
    }
    while ( 1 )
    {
      v22 = *(_QWORD *)(i - 32);
      if ( !v22 )
        break;
      if ( *(_BYTE *)v22 || *(_QWORD *)(v22 + 24) != *(_QWORD *)(i + 80) || (*(_BYTE *)(v22 + 33) & 0x20) == 0 )
        goto LABEL_16;
    }
    v108 = 0;
    v21 = 0;
LABEL_17:
    if ( v16 < v17 )
    {
      sub_C8D5F0(&v134, v136, v17, 8);
      v18 = (unsigned int)v135;
    }
    v134[v18] = v21;
    v23 = *(_QWORD *)(a2 + 16);
    v24 = *((unsigned int *)a3 + 2);
    LODWORD(v135) = v135 + 1;
    v113 = v23 + 80;
    v109 = (__int64)&(*a3)[v24];
    if ( v118 != (unsigned __int64 *)v109 )
      goto LABEL_20;
    v120 = 0;
    v110 = a1 + 48;
LABEL_41:
    if ( !v108 )
      goto LABEL_44;
    v46 = *(_QWORD *)(v108 - 32);
    v47 = *(_DWORD *)(v46 + 32);
    if ( v47 > 0x40 )
    {
      if ( v47 - (unsigned int)sub_C444A0(v46 + 24) > 0x40 || (unsigned __int64)v120 <= **(_QWORD **)(v46 + 24) )
        goto LABEL_44;
    }
    else if ( (unsigned __int64)v120 <= *(_QWORD *)(v46 + 24) )
    {
LABEL_44:
      v48 = v135;
      goto LABEL_45;
    }
    v91 = sub_BCB2D0(*(_QWORD *)(v6 + 40));
    v92 = sub_ACD640(v91, v120, 0);
    if ( *(_QWORD *)(v108 - 32) )
    {
      v93 = *(_QWORD *)(v108 - 24);
      **(_QWORD **)(v108 - 16) = v93;
      if ( v93 )
        *(_QWORD *)(v93 + 16) = *(_QWORD *)(v108 - 16);
    }
    *(_QWORD *)(v108 - 32) = v92;
    if ( v92 )
    {
      v94 = *(_QWORD *)(v92 + 16);
      *(_QWORD *)(v108 - 24) = v94;
      if ( v94 )
        *(_QWORD *)(v94 + 16) = v108 - 24;
      *(_QWORD *)(v108 - 16) = v92 + 16;
      *(_QWORD *)(v92 + 16) = v108 - 32;
    }
    goto LABEL_44;
  }
  v95 = sub_AD6530(v8);
  v96 = (unsigned int)v135;
  v97 = (unsigned int)v135 + 1LL;
  if ( v97 > HIDWORD(v135) )
  {
    sub_C8D5F0(&v134, v136, v97, 8);
    v96 = (unsigned int)v135;
  }
  v111 = 0;
  v108 = 0;
  v134[v96] = v95;
  v98 = *((unsigned int *)a3 + 2);
  v48 = v135 + 1;
  v99 = *(_QWORD *)(a2 + 16);
  LODWORD(v135) = v135 + 1;
  v113 = v99 + 80;
  v109 = (__int64)&(*a3)[v98];
  if ( v118 != (unsigned __int64 *)v109 )
  {
LABEL_20:
    v120 = 0;
    v110 = a1 + 48;
    while ( 1 )
    {
      v25 = *v118;
      if ( *(_DWORD *)(v113 + 12) == 2 && *(_BYTE *)(v113 + 16) )
      {
        v128 = 257;
        v51 = unk_4D0463C;
        if ( unk_4D0463C )
          v51 = sub_90AA40(*(_QWORD *)(v6 + 32), v25);
        v52 = *(_QWORD *)(v25 + 8);
        v53 = sub_AA4E30(*(_QWORD *)(v6 + 96));
        v54 = sub_AE5020(v53, v52);
        HIBYTE(v55) = HIBYTE(v115);
        v133 = 257;
        LOBYTE(v55) = v54;
        v115 = v55;
        v56 = sub_BD2C40(80, unk_3F10A14);
        v57 = v56;
        if ( v56 )
          sub_B4D190(v56, v52, v25, (unsigned int)&v129, v51, (unsigned __int8)v115, 0, 0);
        (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v6 + 136) + 16LL))(
          *(_QWORD *)(v6 + 136),
          v57,
          v127,
          *(_QWORD *)(v110 + 56),
          *(_QWORD *)(v110 + 64));
        v58 = *(unsigned int **)(v6 + 48);
        v59 = &v58[4 * *(unsigned int *)(v6 + 56)];
        while ( v59 != v58 )
        {
          v60 = *((_QWORD *)v58 + 1);
          v61 = *v58;
          v58 += 4;
          sub_B99FD0(v57, v61, v60);
        }
        v25 = v57;
      }
      v26 = *(_QWORD *)(v25 + 8);
      if ( *(_BYTE *)(v26 + 8) == 14 && *(_DWORD *)(v26 + 8) >> 8 )
      {
        v89 = sub_BCE3C0(*(_QWORD *)v26, 0);
        v90 = sub_92C9E0(v6, v25, 0, v89, 0, 0, a4);
        v26 = *(_QWORD *)(v90 + 8);
        v25 = v90;
      }
      v27 = sub_9208B0(*(_QWORD *)(*(_QWORD *)(v6 + 32) + 352LL), v26);
      v130 = v28;
      v129 = (char *)((unsigned __int64)(v27 + 7) >> 3);
      v112 = sub_CA1930(&v129);
      v29 = v111;
      v30 = sub_BCE760(v26, 0);
      v31 = v112 + v120 - v120 % v112;
      if ( !(v120 % v112) )
        v31 = v120;
      v121 = v31;
      if ( v31 )
        break;
LABEL_28:
      v127[0] = "casted";
      v128 = 259;
      if ( v30 == *(_QWORD *)(v29 + 8) )
      {
        v34 = v29;
      }
      else
      {
        v32 = *(_QWORD *)(v6 + 128);
        v33 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v32 + 120LL);
        if ( v33 == sub_920130 )
        {
          if ( *(_BYTE *)v29 > 0x15u )
            goto LABEL_74;
          if ( (unsigned __int8)sub_AC4810(49) )
            v34 = sub_ADAB70(49, v29, v30, 0);
          else
            v34 = sub_AA93C0(49, v29, v30);
        }
        else
        {
          v34 = v33(v32, 49u, (_BYTE *)v29, v30);
        }
        if ( !v34 )
        {
LABEL_74:
          v133 = 257;
          v34 = sub_B51D30(49, v29, v30, &v129, 0, 0);
          if ( (unsigned __int8)sub_920620(v34) )
          {
            v83 = *(_QWORD *)(v6 + 144);
            v84 = *(_DWORD *)(v6 + 152);
            if ( v83 )
              sub_B99FD0(v34, 3, v83);
            sub_B45150(v34, v84);
          }
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v6 + 136) + 16LL))(
            *(_QWORD *)(v6 + 136),
            v34,
            v127,
            *(_QWORD *)(v110 + 56),
            *(_QWORD *)(v110 + 64));
          v85 = *(unsigned int **)(v6 + 48);
          v86 = &v85[4 * *(unsigned int *)(v6 + 56)];
          while ( v86 != v85 )
          {
            v87 = *((_QWORD *)v85 + 1);
            v88 = *v85;
            v85 += 4;
            sub_B99FD0(v34, v88, v87);
          }
        }
      }
      v35 = unk_4D0463C;
      if ( unk_4D0463C )
        v35 = sub_90AA40(*(_QWORD *)(v6 + 32), v34);
      v36 = sub_AA4E30(*(_QWORD *)(v6 + 96));
      v37 = sub_AE5020(v36, *(_QWORD *)(v25 + 8));
      HIBYTE(v38) = HIBYTE(v116);
      v133 = 257;
      LOBYTE(v38) = v37;
      v116 = v38;
      v39 = sub_BD2C40(80, unk_3F10A10);
      v41 = v39;
      if ( v39 )
        sub_B4D3C0(v39, v25, v34, v35, (unsigned __int8)v116, v40, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v6 + 136) + 16LL))(
        *(_QWORD *)(v6 + 136),
        v41,
        &v129,
        *(_QWORD *)(v110 + 56),
        *(_QWORD *)(v110 + 64));
      v42 = *(unsigned int **)(v6 + 48);
      v43 = &v42[4 * *(unsigned int *)(v6 + 56)];
      while ( v43 != v42 )
      {
        v44 = *((_QWORD *)v42 + 1);
        v45 = *v42;
        v42 += 4;
        sub_B99FD0(v41, v45, v44);
      }
      ++v118;
      v113 += 40;
      v120 = v112 + v121;
      if ( v118 == (unsigned __int64 *)v109 )
        goto LABEL_41;
    }
    v62 = *(_QWORD *)(v6 + 120);
    v127[0] = "buf.indexed";
    v128 = 259;
    v63 = sub_BCB2D0(v62);
    v64 = (_BYTE *)sub_ACD640(v63, v121, 0);
    v65 = *(_QWORD *)(v6 + 128);
    v122 = v64;
    v66 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v65 + 64LL);
    if ( v66 == sub_920540 )
    {
      if ( (unsigned __int8)sub_BCEA30(v107) )
        goto LABEL_64;
      if ( *(_BYTE *)v111 > 0x15u )
        goto LABEL_64;
      v67 = sub_937330(&v122, (__int64)&v123);
      if ( v69 != v67 )
        goto LABEL_64;
      LOBYTE(v133) = 0;
      v29 = sub_AD9FD0(v107, v68, (unsigned int)&v122, 1, 3, (unsigned int)&v129, 0);
      if ( (_BYTE)v133 )
      {
        LOBYTE(v133) = 0;
        if ( v132 > 0x40 && v131 )
          j_j___libc_free_0_0(v131);
        if ( (unsigned int)v130 > 0x40 && v129 )
          j_j___libc_free_0_0(v129);
      }
    }
    else
    {
      v29 = v66(v65, v107, (_BYTE *)v111, &v122, 1, 3);
    }
    if ( v29 )
      goto LABEL_28;
LABEL_64:
    v133 = 257;
    v29 = sub_BD2C40(88, 2);
    if ( !v29 )
      goto LABEL_67;
    v70 = *(_QWORD *)(v111 + 8);
    v71 = v114 & 0xE0000000 | 2;
    v114 = v114 & 0xE0000000 | 2;
    if ( (unsigned int)*(unsigned __int8 *)(v70 + 8) - 17 <= 1 )
    {
LABEL_66:
      sub_B44260(v29, v70, 34, v71, 0, 0);
      *(_QWORD *)(v29 + 72) = v107;
      *(_QWORD *)(v29 + 80) = sub_B4DC50(v107, &v122, 1);
      sub_B4D9A0(v29, v111, &v122, 1, &v129);
LABEL_67:
      sub_B4DDE0(v29, 3);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v6 + 136) + 16LL))(
        *(_QWORD *)(v6 + 136),
        v29,
        v127,
        *(_QWORD *)(v110 + 56),
        *(_QWORD *)(v110 + 64));
      v72 = *(_QWORD *)(v6 + 48);
      v73 = 16LL * *(unsigned int *)(v6 + 56);
      if ( v72 != v72 + v73 )
      {
        v102 = v6;
        v74 = *(unsigned int **)(v6 + 48);
        v75 = (unsigned int *)(v72 + v73);
        do
        {
          v76 = *((_QWORD *)v74 + 1);
          v77 = *v74;
          v74 += 4;
          sub_B99FD0(v29, v77, v76);
        }
        while ( v75 != v74 );
        v6 = v102;
      }
      goto LABEL_28;
    }
    v78 = *((_QWORD *)v122 + 1);
    v79 = *(unsigned __int8 *)(v78 + 8);
    if ( v79 == 17 )
    {
      v80 = 0;
    }
    else
    {
      v80 = 1;
      if ( v79 != 18 )
        goto LABEL_66;
    }
    v81 = *(_DWORD *)(v78 + 32);
    BYTE4(v123) = v80;
    v103 = v71;
    LODWORD(v123) = v81;
    v82 = sub_BCE1B0(v70, v123);
    v71 = v103;
    v70 = v82;
    goto LABEL_66;
  }
  v110 = a1 + 48;
LABEL_45:
  v133 = 257;
  v49 = sub_921880((unsigned int **)v110, v105, v104, (int)v134, v48, (__int64)&v129, 0);
  if ( v134 != v136 )
    _libc_free(v134, v105);
  if ( v124 )
    j_j___libc_free_0(v124, &v126[-v124]);
  return v49;
}
