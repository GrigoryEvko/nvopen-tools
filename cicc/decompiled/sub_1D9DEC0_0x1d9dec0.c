// Function: sub_1D9DEC0
// Address: 0x1d9dec0
//
__int64 __fastcall sub_1D9DEC0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  __int64 v15; // r12
  __int64 (*v16)(); // rdx
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rdi
  unsigned __int64 v20; // rax
  int v21; // r8d
  int v22; // r9d
  _QWORD *v23; // r13
  __int64 v24; // rax
  unsigned int v25; // r15d
  int v26; // ebx
  __int64 v27; // rsi
  __int64 *v28; // rax
  __int64 *v29; // rdi
  __int64 *v30; // rcx
  __int64 v31; // rax
  __int64 v32; // r13
  __int64 *v33; // rax
  __int64 v34; // rbx
  __int64 *v35; // r15
  __int64 v36; // r15
  __int64 *v37; // r15
  int v38; // r8d
  int v39; // r9d
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // rax
  __int64 v43; // rax
  double v44; // xmm4_8
  double v45; // xmm5_8
  unsigned __int64 v46; // r13
  __int64 v47; // rbx
  _BYTE *v48; // r12
  __int64 **v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v53; // r15
  _QWORD *v54; // rdi
  __int64 v55; // r12
  _QWORD *v56; // r13
  int v57; // r12d
  __int64 v58; // rax
  __int64 v59; // r15
  unsigned __int64 v60; // r12
  __int64 *v61; // rax
  char v62; // al
  __int64 v63; // rdx
  _QWORD *v64; // rdx
  __int64 v65; // rsi
  __int64 *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  int v71; // eax
  __int64 v72; // rax
  int v73; // edx
  __int64 v74; // rdx
  __int64 *v75; // rax
  __int64 v76; // rsi
  unsigned __int64 v77; // rdx
  __int64 v78; // rsi
  __int64 v79; // rdx
  __int64 v80; // rcx
  _QWORD *v81; // rdi
  __int64 v82; // r13
  __int64 v83; // r14
  __int64 v84; // r12
  __int64 *v85; // rdx
  char v86; // al
  __int64 v87; // rdx
  _QWORD *v88; // rdx
  __int64 *v89; // rdi
  __int64 v90; // r14
  __int64 v91; // rax
  __int64 v92; // r12
  __int64 v93; // r14
  __int64 v94; // r13
  __int64 v95; // rsi
  __int64 v96; // r15
  __int64 v97; // rax
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // r9
  unsigned __int64 v101; // rdi
  __int64 *v102; // rax
  _QWORD **v103; // rbx
  _QWORD **v104; // r12
  _QWORD *v105; // r15
  __int64 v106; // r14
  _QWORD *v107; // rdi
  __int64 *v108; // rdx
  __int64 v109; // [rsp+8h] [rbp-338h]
  unsigned __int8 v110; // [rsp+37h] [rbp-309h]
  __int64 v111; // [rsp+38h] [rbp-308h]
  _BYTE *v113; // [rsp+40h] [rbp-300h]
  __int64 v114; // [rsp+48h] [rbp-2F8h]
  int v115; // [rsp+48h] [rbp-2F8h]
  __int64 v116; // [rsp+48h] [rbp-2F8h]
  int v117; // [rsp+48h] [rbp-2F8h]
  const char *v118; // [rsp+50h] [rbp-2F0h] BYREF
  __int64 v119; // [rsp+58h] [rbp-2E8h]
  _BYTE *v120; // [rsp+60h] [rbp-2E0h] BYREF
  __int64 v121; // [rsp+68h] [rbp-2D8h]
  _BYTE v122[16]; // [rsp+70h] [rbp-2D0h] BYREF
  _QWORD v123[2]; // [rsp+80h] [rbp-2C0h] BYREF
  char v124; // [rsp+90h] [rbp-2B0h]
  char v125; // [rsp+91h] [rbp-2AFh]
  const char *v126; // [rsp+A0h] [rbp-2A0h] BYREF
  _QWORD *v127; // [rsp+A8h] [rbp-298h]
  __int16 v128; // [rsp+B0h] [rbp-290h]
  _BYTE *v129; // [rsp+C0h] [rbp-280h] BYREF
  __int64 v130; // [rsp+C8h] [rbp-278h]
  _BYTE v131[32]; // [rsp+D0h] [rbp-270h] BYREF
  __int64 v132; // [rsp+F0h] [rbp-250h] BYREF
  __int64 *v133; // [rsp+F8h] [rbp-248h]
  __int64 *v134; // [rsp+100h] [rbp-240h]
  __int64 v135; // [rsp+108h] [rbp-238h]
  int v136; // [rsp+110h] [rbp-230h]
  _BYTE v137[40]; // [rsp+118h] [rbp-228h] BYREF
  _QWORD v138[57]; // [rsp+140h] [rbp-200h] BYREF
  __int64 v139; // [rsp+308h] [rbp-38h]

  v111 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v11 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FCBA30, 1u);
  if ( !v11 )
    return 0;
  v12 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v11 + 104LL))(v11, &unk_4FCBA30);
  if ( !v12 )
    return 0;
  v13 = *(_QWORD *)(v12 + 208);
  v14 = *(__int64 (**)())(*(_QWORD *)v13 + 16LL);
  if ( v14 == sub_16FF750 )
    BUG();
  v15 = ((__int64 (__fastcall *)(__int64, __int64))v14)(v13, a2);
  v110 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v15 + 192LL))(v15);
  if ( !v110 )
    return 0;
  v16 = *(__int64 (**)())(*(_QWORD *)v15 + 56LL);
  v17 = 0;
  if ( v16 != sub_1D12D20 )
    v17 = ((__int64 (__fastcall *)(__int64))v16)(v15);
  *(_QWORD *)(a1 + 160) = v17;
  v120 = v122;
  v121 = 0x100000000LL;
  v133 = (__int64 *)v137;
  v134 = (__int64 *)v137;
  v132 = 0;
  v18 = *(_QWORD *)(a2 + 80);
  v135 = 4;
  v136 = 0;
  v114 = a2 + 72;
  if ( v18 == a2 + 72 )
  {
    v110 = 0;
    goto LABEL_132;
  }
  do
  {
    while ( 1 )
    {
      v19 = v18 - 24;
      if ( !v18 )
        v19 = 0;
      v20 = sub_157EBA0(v19);
      v23 = (_QWORD *)v20;
      if ( *(_BYTE *)(v20 + 16) != 28 )
        goto LABEL_27;
      if ( (*(_DWORD *)(v20 + 20) & 0xFFFFFFF) == 1 )
        break;
      v24 = (unsigned int)v121;
      if ( (unsigned int)v121 >= HIDWORD(v121) )
      {
        sub_16CD150((__int64)&v120, v122, 0, 8, v21, v22);
        v24 = (unsigned int)v121;
      }
      v25 = 0;
      *(_QWORD *)&v120[8 * v24] = v23;
      LODWORD(v121) = v121 + 1;
      v26 = sub_15F4D60((__int64)v23);
      if ( v26 )
      {
        do
        {
LABEL_18:
          v27 = sub_15F4DF0((__int64)v23, v25);
          v28 = v133;
          if ( v134 != v133 )
            goto LABEL_16;
          v29 = &v133[HIDWORD(v135)];
          if ( v133 != v29 )
          {
            v30 = 0;
            while ( v27 != *v28 )
            {
              if ( *v28 == -2 )
                v30 = v28;
              if ( v29 == ++v28 )
              {
                if ( !v30 )
                  goto LABEL_73;
                ++v25;
                *v30 = v27;
                --v136;
                ++v132;
                if ( v26 != v25 )
                  goto LABEL_18;
                goto LABEL_27;
              }
            }
            goto LABEL_17;
          }
LABEL_73:
          if ( HIDWORD(v135) < (unsigned int)v135 )
          {
            ++HIDWORD(v135);
            *v29 = v27;
            ++v132;
          }
          else
          {
LABEL_16:
            sub_16CCBA0((__int64)&v132, v27);
          }
LABEL_17:
          ++v25;
        }
        while ( v26 != v25 );
      }
LABEL_27:
      v18 = *(_QWORD *)(v18 + 8);
      if ( v114 == v18 )
        goto LABEL_28;
    }
    v53 = sub_15E0530(a2);
    v54 = sub_1648A60(56, 0);
    if ( v54 )
      sub_15F82A0((__int64)v54, v53, (__int64)v23);
    sub_15F20C0(v23);
    v18 = *(_QWORD *)(v18 + 8);
  }
  while ( v114 != v18 );
LABEL_28:
  v31 = (unsigned int)v121;
  if ( !(_DWORD)v121 )
  {
    v110 = 0;
    v101 = (unsigned __int64)v134;
    v102 = v133;
    goto LABEL_129;
  }
  v130 = 0x400000000LL;
  v32 = *(_QWORD *)(a2 + 80);
  v129 = v131;
  if ( v18 == v32 )
  {
LABEL_139:
    v103 = (_QWORD **)v120;
    v104 = (_QWORD **)&v120[8 * v31];
    if ( v120 != (_BYTE *)v104 )
    {
      do
      {
        v105 = *v103;
        v106 = sub_15E0530(a2);
        v107 = sub_1648A60(56, 0);
        if ( v107 )
          sub_15F82A0((__int64)v107, v106, (__int64)v105);
        ++v103;
        sub_15F20C0(v105);
      }
      while ( v104 != v103 );
    }
    goto LABEL_126;
  }
  do
  {
    v33 = v133;
    v34 = v32 - 24;
    if ( !v32 )
      v34 = 0;
    if ( v134 == v133 )
    {
      v35 = &v133[HIDWORD(v135)];
      if ( v133 == v35 )
      {
        v108 = v133;
      }
      else
      {
        do
        {
          if ( v34 == *v33 )
            break;
          ++v33;
        }
        while ( v35 != v33 );
        v108 = &v133[HIDWORD(v135)];
      }
    }
    else
    {
      v35 = &v134[(unsigned int)v135];
      v33 = sub_16CC9F0((__int64)&v132, v34);
      if ( v34 == *v33 )
      {
        if ( v134 == v133 )
          v108 = &v134[HIDWORD(v135)];
        else
          v108 = &v134[(unsigned int)v135];
      }
      else
      {
        if ( v134 != v133 )
        {
          v33 = &v134[(unsigned int)v135];
          goto LABEL_36;
        }
        v33 = &v134[HIDWORD(v135)];
        v108 = v33;
      }
    }
    for ( ; v108 != v33; ++v33 )
    {
      if ( (unsigned __int64)*v33 < 0xFFFFFFFFFFFFFFFELL )
        break;
    }
LABEL_36:
    if ( v35 != v33 )
    {
      v36 = *(_QWORD *)(v34 + 8);
      if ( v36 )
      {
        while ( *((_BYTE *)sub_1648700(v36) + 16) != 4 )
        {
          v36 = *(_QWORD *)(v36 + 8);
          if ( !v36 )
            goto LABEL_45;
        }
        v37 = sub_1648700(v36);
        if ( (unsigned __int8)sub_1593E70((__int64)v37) )
        {
          v40 = (unsigned int)v130;
          v115 = v130 + 1;
          if ( (unsigned int)v130 >= HIDWORD(v130) )
          {
            sub_16CD150((__int64)&v129, v131, 0, 8, v38, v39);
            v40 = (unsigned int)v130;
          }
          *(_QWORD *)&v129[8 * v40] = v34;
          LODWORD(v130) = v130 + 1;
          v41 = sub_15A9650(v111, *v37);
          v42 = sub_159C470(v41, v115, 0);
          v43 = sub_15A3BA0(v42, (__int64 **)*v37, 0);
          sub_164D160((__int64)v37, v43, a3, a4, a5, a6, v44, v45, a9, a10);
        }
      }
    }
LABEL_45:
    v32 = *(_QWORD *)(v32 + 8);
  }
  while ( v18 != v32 );
  v31 = (unsigned int)v121;
  if ( !(_DWORD)v130 )
    goto LABEL_139;
  v46 = (unsigned __int64)v120;
  v47 = 0;
  v48 = &v120[8 * (unsigned int)v121];
  if ( v120 != v48 )
  {
    do
    {
      while ( 1 )
      {
        v51 = *(_QWORD *)v46;
        v49 = (*(_BYTE *)(*(_QWORD *)v46 + 23LL) & 0x40) != 0
            ? *(__int64 ***)(v51 - 8)
            : (__int64 **)(v51 - 24LL * (*(_DWORD *)(v51 + 20) & 0xFFFFFFF));
        v50 = sub_15A9650(v111, **v49);
        if ( v47 )
          break;
        v46 += 8LL;
        v47 = v50;
        if ( v48 == (_BYTE *)v46 )
          goto LABEL_76;
      }
      if ( *(_DWORD *)(v47 + 8) >> 8 < *(_DWORD *)(v50 + 8) >> 8 )
        v47 = v50;
      v46 += 8LL;
    }
    while ( v48 != (_BYTE *)v46 );
  }
LABEL_76:
  sub_12BDBC0((__int64)v138, v111);
  v139 = v47;
  if ( (_DWORD)v121 == 1 )
  {
    v84 = *(_QWORD *)v120;
    v56 = *(_QWORD **)(*(_QWORD *)v120 + 40LL);
    v125 = 1;
    v123[0] = ".switch_cast";
    v124 = 3;
    if ( (*(_BYTE *)(v84 + 23) & 0x40) != 0 )
      v85 = *(__int64 **)(v84 - 8);
    else
      v85 = (__int64 *)(v84 - 24LL * (*(_DWORD *)(v84 + 20) & 0xFFFFFFF));
    v118 = sub_1649960(*v85);
    v86 = v124;
    v119 = v87;
    if ( v124 )
    {
      if ( v124 == 1 )
      {
        v126 = (const char *)&v118;
        v128 = 261;
      }
      else
      {
        v88 = (_QWORD *)v123[0];
        if ( v125 != 1 )
        {
          v88 = v123;
          v86 = 2;
        }
        v127 = v88;
        v126 = (const char *)&v118;
        LOBYTE(v128) = 5;
        HIBYTE(v128) = v86;
      }
    }
    else
    {
      v128 = 256;
    }
    if ( (*(_BYTE *)(v84 + 23) & 0x40) != 0 )
      v89 = *(__int64 **)(v84 - 8);
    else
      v89 = (__int64 *)(v84 - 24LL * (*(_DWORD *)(v84 + 20) & 0xFFFFFFF));
    v59 = sub_15FDFF0(*v89, v139, (__int64)&v126, v84);
    sub_15F20C0(*(_QWORD **)v120);
  }
  else
  {
    v126 = "switch_bb";
    v128 = 259;
    v55 = sub_15E0530(a2);
    v56 = (_QWORD *)sub_22077B0(64);
    if ( v56 )
      sub_157FB60(v56, v55, (__int64)&v126, a2, 0);
    v57 = v121;
    v126 = "switch_value_phi";
    v128 = 259;
    v58 = sub_1648B60(64);
    v59 = v58;
    if ( v58 )
    {
      sub_15F1F50(v58, v47, 53, 0, 0, (__int64)v56);
      *(_DWORD *)(v59 + 56) = v57;
      sub_164B780(v59, (__int64 *)&v126);
      sub_1648880(v59, *(_DWORD *)(v59 + 56), 1);
    }
    v60 = (unsigned __int64)v120;
    v113 = &v120[8 * (unsigned int)v121];
    if ( v120 != v113 )
    {
      v116 = (__int64)v56;
      while ( 1 )
      {
        v82 = *(_QWORD *)v60;
        v83 = *(_QWORD *)(*(_QWORD *)v60 + 40LL);
        v125 = 1;
        v123[0] = ".switch_cast";
        v124 = 3;
        if ( (*(_BYTE *)(v82 + 23) & 0x40) != 0 )
          v61 = *(__int64 **)(v82 - 8);
        else
          v61 = (__int64 *)(v82 - 24LL * (*(_DWORD *)(v82 + 20) & 0xFFFFFFF));
        v118 = sub_1649960(*v61);
        v62 = v124;
        v119 = v63;
        if ( v124 )
        {
          if ( v124 == 1 )
          {
            v126 = (const char *)&v118;
            v128 = 261;
          }
          else
          {
            v64 = (_QWORD *)v123[0];
            if ( v125 != 1 )
            {
              v64 = v123;
              v62 = 2;
            }
            v127 = v64;
            LOBYTE(v128) = 5;
            v126 = (const char *)&v118;
            HIBYTE(v128) = v62;
          }
          v65 = v139;
          if ( (*(_BYTE *)(v82 + 23) & 0x40) == 0 )
          {
LABEL_135:
            v66 = (__int64 *)(v82 - 24LL * (*(_DWORD *)(v82 + 20) & 0xFFFFFFF));
            goto LABEL_91;
          }
        }
        else
        {
          v65 = v139;
          v128 = 256;
          if ( (*(_BYTE *)(v82 + 23) & 0x40) == 0 )
            goto LABEL_135;
        }
        v66 = *(__int64 **)(v82 - 8);
LABEL_91:
        v68 = sub_15FDFF0(*v66, v65, (__int64)&v126, v82);
        v71 = *(_DWORD *)(v59 + 20) & 0xFFFFFFF;
        if ( v71 == *(_DWORD *)(v59 + 56) )
        {
          v109 = v68;
          sub_15F55D0(v59, v65, v67, v68, v69, v70);
          v68 = v109;
          v71 = *(_DWORD *)(v59 + 20) & 0xFFFFFFF;
        }
        v72 = (v71 + 1) & 0xFFFFFFF;
        v73 = v72 | *(_DWORD *)(v59 + 20) & 0xF0000000;
        *(_DWORD *)(v59 + 20) = v73;
        if ( (v73 & 0x40000000) != 0 )
          v74 = *(_QWORD *)(v59 - 8);
        else
          v74 = v59 - 24 * v72;
        v75 = (__int64 *)(v74 + 24LL * (unsigned int)(v72 - 1));
        if ( *v75 )
        {
          v76 = v75[1];
          v77 = v75[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v77 = v76;
          if ( v76 )
            *(_QWORD *)(v76 + 16) = *(_QWORD *)(v76 + 16) & 3LL | v77;
        }
        *v75 = v68;
        if ( v68 )
        {
          v78 = *(_QWORD *)(v68 + 8);
          v75[1] = v78;
          if ( v78 )
            *(_QWORD *)(v78 + 16) = (unsigned __int64)(v75 + 1) | *(_QWORD *)(v78 + 16) & 3LL;
          v75[2] = v75[2] & 3 | (v68 + 8);
          *(_QWORD *)(v68 + 8) = v75;
        }
        v79 = *(_DWORD *)(v59 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v59 + 23) & 0x40) != 0 )
          v80 = *(_QWORD *)(v59 - 8);
        else
          v80 = v59 - 24 * v79;
        *(_QWORD *)(v80 + 8LL * (unsigned int)(v79 - 1) + 24LL * *(unsigned int *)(v59 + 56) + 8) = v83;
        v81 = sub_1648A60(56, 1u);
        if ( v81 )
          sub_15F8320((__int64)v81, v116, v82);
        v60 += 8LL;
        sub_15F20C0((_QWORD *)v82);
        if ( v113 == (_BYTE *)v60 )
        {
          v56 = (_QWORD *)v116;
          break;
        }
      }
    }
  }
  v117 = v130;
  v90 = *(_QWORD *)v129;
  v91 = sub_1648B60(64);
  v92 = v91;
  if ( v91 )
    sub_15FFB20(v91, v59, v90, v117, (__int64)v56);
  if ( (_DWORD)v130 != 1 )
  {
    v93 = 2;
    v94 = (unsigned int)(v130 - 2) + 3LL;
    do
    {
      v95 = v93;
      v96 = *(_QWORD *)&v129[8 * v93++ - 8];
      v97 = sub_159C470(v47, v95, 0);
      sub_15FFFB0(v92, v97, v96, v98, v99, v100);
    }
    while ( v94 != v93 );
  }
  sub_15A93E0(v138);
LABEL_126:
  if ( v129 != v131 )
    _libc_free((unsigned __int64)v129);
  v101 = (unsigned __int64)v134;
  v102 = v133;
LABEL_129:
  if ( v102 != (__int64 *)v101 )
    _libc_free(v101);
LABEL_132:
  if ( v120 != v122 )
    _libc_free((unsigned __int64)v120);
  return v110;
}
