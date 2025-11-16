// Function: sub_34696C0
// Address: 0x34696c0
//
unsigned __int8 *__fastcall sub_34696C0(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __m128i a7,
        __int128 a8,
        __int64 a9)
{
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v15; // rdx
  unsigned int v16; // r15d
  __int64 v17; // r9
  int v18; // eax
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // r10
  unsigned __int64 v24; // rdi
  __int128 v25; // rax
  __int64 v26; // r9
  unsigned int v27; // edx
  __int64 v28; // r14
  unsigned int *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r9
  unsigned __int8 *v32; // r15
  unsigned __int8 *v33; // rax
  unsigned int v34; // edx
  __int64 v35; // r9
  unsigned int v36; // edx
  __int64 v37; // rdx
  __int128 v38; // rax
  __int64 v39; // rdx
  __int128 v40; // rax
  __int64 v41; // r9
  __int64 v42; // r14
  __int64 v43; // rdx
  __int64 v44; // r15
  __int128 v45; // rax
  __int64 v46; // r9
  __int128 v47; // rax
  __int64 v48; // r9
  __int128 v49; // rax
  __int128 v50; // rax
  __int64 v51; // r9
  unsigned __int8 *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r15
  unsigned __int8 *v55; // r14
  __int64 v56; // r9
  unsigned __int8 *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rbx
  __int64 v60; // rcx
  __int64 v61; // rax
  unsigned int v62; // edx
  __int16 v63; // si
  __int64 v64; // rax
  __int64 v65; // r9
  unsigned __int8 *v66; // r8
  bool v67; // al
  unsigned int v68; // esi
  __int128 v69; // rax
  __int64 v70; // r9
  unsigned __int8 *v71; // rax
  unsigned int v72; // edx
  __int128 v73; // [rsp-40h] [rbp-120h]
  __int128 v74; // [rsp-30h] [rbp-110h]
  __int128 v75; // [rsp-30h] [rbp-110h]
  __int128 v76; // [rsp-20h] [rbp-100h]
  __int128 v77; // [rsp-20h] [rbp-100h]
  __int128 v78; // [rsp-20h] [rbp-100h]
  __int128 v79; // [rsp-20h] [rbp-100h]
  __int128 v80; // [rsp-20h] [rbp-100h]
  __int128 v81; // [rsp-20h] [rbp-100h]
  __int128 v82; // [rsp-10h] [rbp-F0h]
  int v83; // [rsp+Ch] [rbp-D4h]
  int v84; // [rsp+Ch] [rbp-D4h]
  __int128 v85; // [rsp+10h] [rbp-D0h]
  __int64 v86; // [rsp+20h] [rbp-C0h]
  unsigned __int8 *v87; // [rsp+28h] [rbp-B8h]
  unsigned int v88; // [rsp+30h] [rbp-B0h]
  unsigned __int8 *v89; // [rsp+30h] [rbp-B0h]
  unsigned int v90; // [rsp+38h] [rbp-A8h]
  unsigned __int16 v91; // [rsp+3Eh] [rbp-A2h]
  unsigned int v92; // [rsp+40h] [rbp-A0h]
  __int64 (__fastcall *v93)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+48h] [rbp-98h]
  __int64 v94; // [rsp+48h] [rbp-98h]
  __int128 v96; // [rsp+50h] [rbp-90h]
  __int64 v97; // [rsp+50h] [rbp-90h]
  unsigned int v98; // [rsp+50h] [rbp-90h]
  __int64 v99; // [rsp+60h] [rbp-80h]
  __int64 v100; // [rsp+68h] [rbp-78h]
  __int128 v101; // [rsp+70h] [rbp-70h]
  __int128 v102; // [rsp+70h] [rbp-70h]
  unsigned __int64 v103; // [rsp+78h] [rbp-68h]
  unsigned __int64 v104; // [rsp+90h] [rbp-50h] BYREF
  __int64 v105; // [rsp+98h] [rbp-48h]
  unsigned __int64 v106; // [rsp+A0h] [rbp-40h]
  unsigned int v107; // [rsp+A8h] [rbp-38h]

  *(_QWORD *)&v101 = a4;
  *((_QWORD *)&v101 + 1) = a5;
  v87 = (unsigned __int8 *)a4;
  v86 = (unsigned int)a5;
  v12 = 16LL * (unsigned int)a5 + *(_QWORD *)(a4 + 48);
  v99 = *(_QWORD *)(a9 + 64);
  v100 = *(_QWORD *)(v12 + 8);
  v91 = *(_WORD *)v12;
  v88 = a2 & 0xFFFFFFFD;
  v93 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
  v13 = sub_2E79000(*(__int64 **)(a9 + 40));
  v14 = v93(a1, v13, v99, v91, v100);
  v94 = v15;
  v92 = v14;
  if ( (a2 & 0xFFFFFFFD) == 0x5C )
  {
    v16 = sub_33D4D80(a9, v101, a5, 0) - 1;
    goto LABEL_3;
  }
  sub_33DD090((__int64)&v104, a9, v101, *((__int64 *)&v101 + 1), 0);
  v16 = v105;
  if ( (unsigned int)v105 > 0x40 )
  {
    v16 = sub_C44500((__int64)&v104);
    if ( v107 <= 0x40 || (v24 = v106) == 0 )
    {
LABEL_39:
      if ( v104 )
        j_j___libc_free_0_0(v104);
      goto LABEL_3;
    }
LABEL_38:
    j_j___libc_free_0_0(v24);
    if ( (unsigned int)v105 <= 0x40 )
      goto LABEL_3;
    goto LABEL_39;
  }
  if ( (_DWORD)v105 )
  {
    v16 = 64;
    if ( v104 << (64 - (unsigned __int8)v105) != -1 )
    {
      _BitScanReverse64(&v23, ~(v104 << (64 - (unsigned __int8)v105)));
      v16 = v23 ^ 0x3F;
    }
  }
  if ( v107 > 0x40 )
  {
    v24 = v106;
    if ( v106 )
      goto LABEL_38;
  }
LABEL_3:
  sub_33DD090((__int64)&v104, a9, a8, *((__int64 *)&a8 + 1), 0);
  if ( (unsigned int)v105 > 0x40 )
  {
    v18 = sub_C445E0((__int64)&v104);
    if ( v107 <= 0x40 || (v22 = v106) == 0 )
    {
LABEL_11:
      if ( v104 )
      {
        v84 = v18;
        j_j___libc_free_0_0(v104);
        v18 = v84;
      }
      goto LABEL_7;
    }
LABEL_10:
    v83 = v18;
    j_j___libc_free_0_0(v22);
    v18 = v83;
    if ( (unsigned int)v105 <= 0x40 )
      goto LABEL_7;
    goto LABEL_11;
  }
  v18 = 64;
  _RDX = ~v104;
  __asm { tzcnt   rcx, rdx }
  if ( v104 != -1 )
    v18 = _RCX;
  if ( v107 > 0x40 )
  {
    v22 = v106;
    if ( v106 )
      goto LABEL_10;
  }
LABEL_7:
  if ( v16 + v18 < a6 + ((a2 & 0xFFFFFFFD) == 92 && (unsigned int)(a2 - 94) <= 1) )
    return 0;
  if ( a6 >= v16 )
  {
    v17 = a6 - v16;
    if ( !v16 )
      goto LABEL_25;
    goto LABEL_44;
  }
  if ( a6 )
  {
    v16 = a6;
    LODWORD(v17) = 0;
LABEL_44:
    v98 = v17;
    *(_QWORD *)&v69 = sub_3400E40(a9, v16, v91, v100, a3, a7);
    v71 = sub_3406EB0((_QWORD *)a9, 0xBEu, a3, v91, v100, v70, v101, v69);
    v17 = v98;
    v87 = v71;
    v86 = v72;
LABEL_25:
    if ( (_DWORD)v17 )
    {
      *(_QWORD *)&v25 = sub_3400E40(a9, (unsigned int)v17, v91, v100, a3, a7);
      *(_QWORD *)&a8 = sub_3406EB0((_QWORD *)a9, (unsigned int)(v88 != 92) + 191, a3, v91, v100, v26, a8, v25);
      *((_QWORD *)&a8 + 1) = v27 | *((_QWORD *)&a8 + 1) & 0xFFFFFFFF00000000LL;
    }
  }
  if ( v88 == 92 )
  {
    if ( v91 && *(_QWORD *)(a1 + 8LL * v91 + 112) && (*(_BYTE *)(a1 + 500LL * v91 + 6479) & 0xFB) == 0 )
    {
      v28 = 1;
      v29 = (unsigned int *)sub_33E5110((__int64 *)a9, v91, v100, v91, v100);
      v103 = v86 | *((_QWORD *)&v101 + 1) & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v76 + 1) = v103;
      *(_QWORD *)&v76 = v87;
      v90 = 0;
      v89 = sub_3411F20((_QWORD *)a9, 65, a3, v29, v30, v31, v76, a8);
      v32 = v89;
    }
    else
    {
      v103 = v86 | *((_QWORD *)&v101 + 1) & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v77 + 1) = v103;
      *(_QWORD *)&v77 = v87;
      v33 = sub_3406EB0((_QWORD *)a9, 0x3Bu, a3, v91, v100, v17, v77, a8);
      *((_QWORD *)&v78 + 1) = v103;
      *(_QWORD *)&v78 = v87;
      v90 = v34;
      v89 = v33;
      v32 = sub_3406EB0((_QWORD *)a9, 0x3Du, a3, v91, v100, v35, v78, a8);
      v28 = v36;
    }
    *(_QWORD *)&v96 = sub_3400BD0(a9, 0, a3, v91, v100, 0, a7, 0);
    *((_QWORD *)&v96 + 1) = v37;
    *(_QWORD *)&v38 = sub_33ED040((_QWORD *)a9, 0x16u);
    *((_QWORD *)&v73 + 1) = v28;
    *(_QWORD *)&v73 = v32;
    *(_QWORD *)&v85 = sub_340F900((_QWORD *)a9, 0xD0u, a3, v92, v94, v28, v73, v96, v38);
    *((_QWORD *)&v85 + 1) = v39;
    *(_QWORD *)&v40 = sub_33ED040((_QWORD *)a9, 0x14u);
    *((_QWORD *)&v74 + 1) = v86 | v103 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v74 = v87;
    v42 = sub_340F900((_QWORD *)a9, 0xD0u, a3, v92, v94, v41, v74, v96, v40);
    v44 = v43;
    *(_QWORD *)&v45 = sub_33ED040((_QWORD *)a9, 0x14u);
    *(_QWORD *)&v47 = sub_340F900((_QWORD *)a9, 0xD0u, a3, v92, v94, v46, a8, v96, v45);
    *((_QWORD *)&v79 + 1) = v44;
    *(_QWORD *)&v79 = v42;
    *(_QWORD *)&v49 = sub_3406EB0((_QWORD *)a9, 0xBCu, a3, v92, v94, v48, v79, v47);
    v102 = v49;
    *(_QWORD *)&v50 = sub_3400BD0(a9, 1, a3, v91, v100, 0, a7, 0);
    *((_QWORD *)&v75 + 1) = v90;
    *(_QWORD *)&v75 = v89;
    v52 = sub_3406EB0((_QWORD *)a9, 0x39u, a3, v91, v100, v51, v75, v50);
    v54 = v53;
    v55 = v52;
    v57 = sub_3406EB0((_QWORD *)a9, 0xBAu, a3, v92, v94, v56, v85, v102);
    v59 = v58;
    v60 = (__int64)v57;
    v61 = *((_QWORD *)v57 + 6) + 16LL * (unsigned int)v58;
    v62 = v91;
    v63 = *(_WORD *)v61;
    v64 = *(_QWORD *)(v61 + 8);
    v65 = v90;
    v66 = v89;
    LOWORD(v104) = v63;
    v105 = v64;
    if ( v63 )
    {
      v68 = ((unsigned __int16)(v63 - 17) < 0xD4u) + 205;
    }
    else
    {
      v97 = v60;
      v67 = sub_30070B0((__int64)&v104);
      v66 = v89;
      v62 = v91;
      v60 = v97;
      v65 = v90;
      v68 = 205 - (!v67 - 1);
    }
    *((_QWORD *)&v82 + 1) = v65;
    *(_QWORD *)&v82 = v66;
    *((_QWORD *)&v80 + 1) = v54;
    *(_QWORD *)&v80 = v55;
    return (unsigned __int8 *)sub_340EC60((_QWORD *)a9, v68, a3, v62, v100, 0, v60, v59, v80, v82);
  }
  else
  {
    *((_QWORD *)&v81 + 1) = v86 | *((_QWORD *)&v101 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v81 = v87;
    return sub_3406EB0((_QWORD *)a9, 0x3Cu, a3, v91, v100, v17, v81, a8);
  }
}
