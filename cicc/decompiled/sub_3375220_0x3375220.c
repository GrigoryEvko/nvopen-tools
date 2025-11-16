// Function: sub_3375220
// Address: 0x3375220
//
void __fastcall sub_3375220(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned int v7; // r15d
  __int64 v8; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int16 v12; // r14
  __int64 v13; // r13
  __int64 v14; // rax
  int v15; // edx
  int v16; // r14d
  int v17; // edx
  __int64 v18; // rdx
  int v19; // r9d
  __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // r14
  __int64 v24; // r15
  __int64 v25; // r13
  __int64 v28; // r12
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // r15d
  int v34; // edx
  __int128 v35; // rax
  int v36; // r9d
  __int64 v37; // rax
  unsigned int v38; // edx
  __int64 v39; // r8
  __int64 v40; // r9
  int v41; // eax
  __int128 v42; // rax
  int v43; // r9d
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r13
  __int64 v47; // r12
  __int64 v48; // r14
  __int128 v49; // rax
  int v50; // r9d
  __int128 v51; // rax
  __int64 v52; // r12
  __int64 v53; // r13
  __int64 v54; // rdx
  __int64 v55; // r14
  __int64 v56; // rax
  __int64 v57; // rax
  int v58; // r15d
  int v59; // edx
  __int128 v60; // rax
  int v61; // r9d
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // r12
  __int64 v65; // rsi
  __int64 v66; // r14
  __int64 v67; // rdx
  __int64 v68; // r15
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int128 v72; // rax
  int v73; // r9d
  __int64 v74; // r14
  __int64 v75; // rdx
  __int64 v76; // r14
  __int128 v77; // rax
  int v78; // r9d
  int v79; // edx
  __int64 v80; // r12
  __int64 v82; // rdi
  __int64 v83; // rdx
  __int64 (__fastcall *v84)(__int64, __int64, __int64, __int64, _QWORD); // r12
  __int64 v85; // rax
  __int64 v86; // rax
  int v87; // eax
  int v88; // edx
  int v89; // r15d
  __int128 v90; // rax
  int v91; // r9d
  unsigned int v92; // edx
  __int128 v93; // [rsp-30h] [rbp-120h]
  __int128 v94; // [rsp-30h] [rbp-120h]
  __int128 v95; // [rsp-30h] [rbp-120h]
  __int128 v96; // [rsp-10h] [rbp-100h]
  __int128 v97; // [rsp-10h] [rbp-100h]
  __int64 v98; // [rsp+0h] [rbp-F0h]
  __int64 (__fastcall *v99)(__int64, __int64, __int64, __int64, _QWORD); // [rsp+0h] [rbp-F0h]
  __int64 v100; // [rsp+0h] [rbp-F0h]
  __int128 v101; // [rsp+0h] [rbp-F0h]
  __int128 v102; // [rsp+0h] [rbp-F0h]
  int v103; // [rsp+10h] [rbp-E0h]
  __int128 v104; // [rsp+10h] [rbp-E0h]
  __int64 (__fastcall *v105)(__int64, __int64, __int64, __int64, _QWORD); // [rsp+10h] [rbp-E0h]
  unsigned int v106; // [rsp+20h] [rbp-D0h]
  __int64 v107; // [rsp+20h] [rbp-D0h]
  unsigned int v108; // [rsp+20h] [rbp-D0h]
  __int64 v109; // [rsp+20h] [rbp-D0h]
  __int64 v110; // [rsp+20h] [rbp-D0h]
  __int64 v111; // [rsp+20h] [rbp-D0h]
  unsigned int v112; // [rsp+2Ch] [rbp-C4h]
  int v115; // [rsp+40h] [rbp-B0h]
  __int128 v116; // [rsp+40h] [rbp-B0h]
  __int128 v117; // [rsp+40h] [rbp-B0h]
  unsigned __int16 v118; // [rsp+50h] [rbp-A0h]
  __int64 v119; // [rsp+50h] [rbp-A0h]
  __int64 v120; // [rsp+50h] [rbp-A0h]
  __int64 v121; // [rsp+50h] [rbp-A0h]
  __int64 v122; // [rsp+90h] [rbp-60h] BYREF
  int v123; // [rsp+98h] [rbp-58h]
  __int64 v124; // [rsp+A0h] [rbp-50h] BYREF
  int v125; // [rsp+A8h] [rbp-48h]
  __int64 v126; // [rsp+B0h] [rbp-40h]
  __int64 v127; // [rsp+B8h] [rbp-38h]

  v7 = a5;
  v8 = a2;
  v10 = *(_QWORD *)a1;
  v11 = *(unsigned int *)(a1 + 848);
  v112 = a4;
  v122 = 0;
  v123 = v11;
  if ( v10 )
  {
    v11 = v10 + 48;
    if ( &v122 != (__int64 *)(v10 + 48) )
    {
      a2 = *(_QWORD *)(v10 + 48);
      v122 = a2;
      if ( a2 )
        sub_B96E90((__int64)&v122, a2, 1);
    }
  }
  v12 = *(_WORD *)(v8 + 44);
  v13 = *(_QWORD *)(a1 + 864);
  v118 = v12;
  v14 = sub_3373A60(a1, a2, v11, a4, a5, a6);
  v103 = v15;
  v106 = v12;
  v98 = v14;
  v16 = sub_33E5110(v13, v12, 0, 1, 0);
  v115 = v17;
  v124 = v98;
  v125 = v103;
  v126 = sub_33F0B60(v13, v7, v106, 0);
  v127 = v18;
  *((_QWORD *)&v96 + 1) = 2;
  *(_QWORD *)&v96 = &v124;
  *(_QWORD *)&v104 = sub_3411630(v13, 50, (unsigned int)&v122, v16, v115, v19, v96);
  *((_QWORD *)&v104 + 1) = v20;
  v107 = *(_QWORD *)a6;
  v21 = sub_39FAC40(*(_QWORD *)a6);
  v22 = *(_QWORD *)(a1 + 864);
  _RDX = v107;
  v24 = *(_QWORD *)(v22 + 16);
  if ( v21 != 1 )
  {
    v25 = v21;
    v108 = *(_DWORD *)(v8 + 24);
    if ( v108 > 0x40 )
    {
      v100 = _RDX;
      v41 = sub_C444A0(v8 + 16);
      _RDX = v100;
      if ( v108 - v41 <= 0x40 && v25 == **(_QWORD **)(v8 + 16) )
        goto LABEL_8;
    }
    else if ( v21 == *(_QWORD *)(v8 + 16) )
    {
LABEL_8:
      _RDX = ~_RDX;
      __asm { tzcnt   rsi, rdx }
      if ( !_RDX )
        LODWORD(_RSI) = 64;
      v28 = sub_3400BD0(v22, _RSI, (unsigned int)&v122, v118, 0, 0, 0);
      v30 = v29;
      v109 = v118;
      v31 = *(_QWORD *)(a1 + 864);
      v99 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v24 + 528LL);
      v119 = *(_QWORD *)(v31 + 64);
      v32 = sub_2E79000(*(__int64 **)(v31 + 40));
      v33 = v99(v24, v32, v119, v109, 0);
      LODWORD(v119) = v34;
      *(_QWORD *)&v35 = sub_33ED040(v22, 22);
      *((_QWORD *)&v93 + 1) = v30;
      *(_QWORD *)&v93 = v28;
      v37 = sub_340F900(v22, 208, (unsigned int)&v122, v33, v119, v36, v104, v93, v35);
      goto LABEL_14;
    }
    *(_QWORD *)&v42 = sub_3400BD0(v22, 1, (unsigned int)&v122, v118, 0, 0, 0);
    v44 = sub_3406EB0(v22, 190, (unsigned int)&v122, v118, 0, v43, v42, v104);
    v46 = v45;
    v47 = v44;
    v48 = *(_QWORD *)(a1 + 864);
    *(_QWORD *)&v49 = sub_3400BD0(v48, *(_QWORD *)a6, (unsigned int)&v122, v118, 0, 0, 0);
    *((_QWORD *)&v94 + 1) = v46;
    *(_QWORD *)&v94 = v47;
    *(_QWORD *)&v51 = sub_3406EB0(v48, 186, (unsigned int)&v122, v118, 0, v50, v94, v49);
    v52 = *(_QWORD *)(a1 + 864);
    v101 = v51;
    v53 = sub_3400BD0(v52, 0, (unsigned int)&v122, v118, 0, 0, 0);
    v55 = v54;
    v110 = v118;
    v56 = *(_QWORD *)(a1 + 864);
    v105 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v24 + 528LL);
    v120 = *(_QWORD *)(v56 + 64);
    v57 = sub_2E79000(*(__int64 **)(v56 + 40));
    v58 = v105(v24, v57, v120, v110, 0);
    LODWORD(v120) = v59;
    *(_QWORD *)&v60 = sub_33ED040(v52, 22);
    *((_QWORD *)&v95 + 1) = v55;
    *(_QWORD *)&v95 = v53;
    v37 = sub_340F900(v52, 208, (unsigned int)&v122, v58, v120, v61, v101, v95, v60);
LABEL_14:
    *(_QWORD *)&v116 = v37;
    *((_QWORD *)&v116 + 1) = v38;
    goto LABEL_15;
  }
  __asm { tzcnt   rsi, rdx }
  v82 = *(_QWORD *)(a1 + 864);
  if ( !v107 )
    LODWORD(_RSI) = 64;
  *(_QWORD *)&v102 = sub_3400BD0(v82, _RSI, (unsigned int)&v122, v118, 0, 0, 0);
  *((_QWORD *)&v102 + 1) = v83;
  v84 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v24 + 528LL);
  v85 = *(_QWORD *)(a1 + 864);
  v111 = v118;
  v121 = *(_QWORD *)(v85 + 64);
  v86 = sub_2E79000(*(__int64 **)(v85 + 40));
  v87 = v84(v24, v86, v121, v111, 0);
  v89 = v88;
  LODWORD(v84) = v87;
  *(_QWORD *)&v90 = sub_33ED040(v22, 17);
  *(_QWORD *)&v116 = sub_340F900(v22, 208, (unsigned int)&v122, (_DWORD)v84, v89, v91, v104, v102, v90);
  *((_QWORD *)&v116 + 1) = v92;
LABEL_15:
  sub_3373E10(a1, a7, *(_QWORD *)(a6 + 16), *(unsigned int *)(a6 + 24), v39, v40);
  sub_3373E10(a1, a7, a3, v112, v62, v63);
  sub_2E33470(*(unsigned int **)(a7 + 144), *(unsigned int **)(a7 + 152));
  v64 = *(_QWORD *)(a1 + 864);
  v65 = *(_QWORD *)(a6 + 16);
  v66 = sub_33EEAD0(v64, v65);
  v68 = v67;
  *(_QWORD *)&v72 = sub_3373A60(a1, v65, v67, v69, v70, v71);
  *((_QWORD *)&v97 + 1) = v68;
  *(_QWORD *)&v97 = v66;
  *(_QWORD *)&v117 = sub_340F900(v64, 305, (unsigned int)&v122, 1, 0, v73, v72, v116, v97);
  v74 = v117;
  *((_QWORD *)&v117 + 1) = v75;
  if ( a3 != sub_3374B60(a1, a7) )
  {
    v76 = *(_QWORD *)(a1 + 864);
    *(_QWORD *)&v77 = sub_33EEAD0(v76, a3);
    v74 = sub_3406EB0(v76, 301, (unsigned int)&v122, 1, 0, v78, v117, v77);
    DWORD2(v117) = v79;
  }
  v80 = *(_QWORD *)(a1 + 864);
  if ( v74 )
  {
    nullsub_1875(v74, v80, 0);
    *(_QWORD *)(v80 + 384) = v74;
    *(_DWORD *)(v80 + 392) = DWORD2(v117);
    sub_33E2B60(v80, 0);
  }
  else
  {
    *(_QWORD *)(v80 + 384) = 0;
    *(_DWORD *)(v80 + 392) = DWORD2(v117);
  }
  if ( v122 )
    sub_B91220((__int64)&v122, v122);
}
