// Function: sub_3050DB0
// Address: 0x3050db0
//
__int64 __fastcall sub_3050DB0(__int64 a1, __int64 a2, char a3)
{
  int v5; // eax
  __int64 result; // rax
  __int16 v7; // ax
  _QWORD *v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  unsigned __int16 v11; // dx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // r10d
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int16 v21; // dx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rdx
  char v26; // al
  __int64 v27; // rax
  int v28; // r10d
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int16 v38; // dx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // r8
  __int64 v49; // r14
  int v50; // r10d
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rcx
  __int64 v57; // r14
  int v58; // r10d
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  int v64; // r9d
  __int64 v65; // r12
  unsigned __int64 v66; // r14
  __int64 v67; // r15
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // r14
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // r14
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  int v84; // r9d
  __int64 v85; // r12
  unsigned __int64 v86; // r14
  __int64 v87; // r15
  __int128 v88; // [rsp-10h] [rbp-110h]
  __int128 v89; // [rsp-10h] [rbp-110h]
  __int64 v90; // [rsp+8h] [rbp-F8h]
  __int64 v91; // [rsp+10h] [rbp-F0h]
  __int64 v92; // [rsp+20h] [rbp-E0h]
  int v93; // [rsp+20h] [rbp-E0h]
  __int64 v94; // [rsp+20h] [rbp-E0h]
  __int64 v95; // [rsp+20h] [rbp-E0h]
  __int64 v96; // [rsp+28h] [rbp-D8h]
  int v97; // [rsp+28h] [rbp-D8h]
  int v98; // [rsp+28h] [rbp-D8h]
  int v99; // [rsp+28h] [rbp-D8h]
  __int64 v100; // [rsp+28h] [rbp-D8h]
  unsigned __int16 v101; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v102; // [rsp+38h] [rbp-C8h]
  __int64 v103; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v104; // [rsp+48h] [rbp-B8h]
  __int64 v105; // [rsp+50h] [rbp-B0h]
  __int64 v106; // [rsp+58h] [rbp-A8h]
  __int64 v107; // [rsp+60h] [rbp-A0h]
  __int64 v108; // [rsp+68h] [rbp-98h]
  __int64 v109; // [rsp+70h] [rbp-90h] BYREF
  __int64 v110; // [rsp+78h] [rbp-88h]
  _BYTE *v111; // [rsp+80h] [rbp-80h] BYREF
  __int64 v112; // [rsp+88h] [rbp-78h]
  _BYTE v113[112]; // [rsp+90h] [rbp-70h] BYREF

  v5 = *(_DWORD *)(a1 + 24);
  if ( v5 < 0 )
  {
    if ( (unsigned int)(-v5 - 5591) > 1 )
      return 0;
  }
  else
  {
    if ( v5 != 156 )
      return 0;
    v7 = **(_WORD **)(a1 + 48);
    if ( v7 != 58 && v7 != 47 )
      return 0;
  }
  v8 = *(_QWORD **)(a1 + 40);
  v9 = *v8;
  v96 = v8[1];
  v10 = *(_QWORD *)(*v8 + 48LL) + 16LL * *((unsigned int *)v8 + 2);
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v101 = v11;
  v102 = v12;
  if ( v11 )
  {
    if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
      goto LABEL_49;
    v14 = 16LL * (v11 - 1);
    v13 = *(_QWORD *)&byte_444C4A0[v14];
    LOBYTE(v14) = byte_444C4A0[v14 + 8];
  }
  else
  {
    v13 = sub_3007260((__int64)&v101);
    v105 = v13;
    v106 = v14;
  }
  v111 = (_BYTE *)v13;
  LOBYTE(v112) = v14;
  v15 = sub_CA1930(&v111);
  if ( *(_DWORD *)(v9 + 24) != 216 )
    return 0;
  v16 = *(_QWORD *)(v9 + 40);
  v17 = *(_QWORD *)v16;
  v18 = *(unsigned int *)(v16 + 8);
  v19 = v18 | v96 & 0xFFFFFFFF00000000LL;
  v20 = *(_QWORD *)(v17 + 48) + 16 * v18;
  v92 = v19;
  v21 = *(_WORD *)v20;
  v22 = *(_QWORD *)(v20 + 8);
  LOWORD(v109) = v21;
  v110 = v22;
  if ( v21 )
  {
    if ( v21 == 1 || (unsigned __int16)(v21 - 504) <= 7u )
      goto LABEL_49;
    v68 = 16LL * (v21 - 1);
    v25 = *(_QWORD *)&byte_444C4A0[v68];
    v26 = byte_444C4A0[v68 + 8];
  }
  else
  {
    v97 = v15;
    v23 = sub_3007260((__int64)&v109);
    v15 = v97;
    v107 = v23;
    v108 = v24;
    v25 = v23;
    v26 = v108;
  }
  v98 = v15;
  v111 = (_BYTE *)v25;
  LOBYTE(v112) = v26;
  v27 = sub_CA1930(&v111);
  v28 = v98;
  v29 = (unsigned int)(2 * v98);
  if ( v27 != v29 )
    return 0;
  v30 = *(_QWORD *)(a1 + 40);
  v31 = *(_QWORD *)(v30 + 48);
  v32 = *(_QWORD *)(v30 + 40);
  if ( *(_DWORD *)(v32 + 24) != 216 )
    return 0;
  v33 = *(_QWORD *)(v32 + 40);
  v34 = *(_QWORD *)v33;
  v35 = *(unsigned int *)(v33 + 8);
  v36 = v35 | v31 & 0xFFFFFFFF00000000LL;
  v37 = *(_QWORD *)(v34 + 48) + 16 * v35;
  v91 = v34;
  v90 = v36;
  v38 = *(_WORD *)v37;
  v39 = *(_QWORD *)(v37 + 8);
  LOWORD(v103) = v38;
  v104 = v39;
  if ( !v38 )
  {
    v40 = sub_3007260((__int64)&v103);
    v29 = (unsigned int)(2 * v98);
    v28 = v98;
    v109 = v40;
    v110 = v41;
    goto LABEL_16;
  }
  if ( v38 == 1 || (unsigned __int16)(v38 - 504) <= 7u )
LABEL_49:
    BUG();
  v41 = 16LL * (v38 - 1);
  v40 = *(_QWORD *)&byte_444C4A0[v41];
  LOBYTE(v41) = byte_444C4A0[v41 + 8];
LABEL_16:
  v99 = v28;
  v111 = (_BYTE *)v40;
  LOBYTE(v112) = v41;
  if ( v29 != sub_CA1930(&v111) || *(_DWORD *)(v17 + 24) == 298 || *(_DWORD *)(v91 + 24) == 298 )
    return 0;
  if ( v99 == 16 )
  {
    v111 = v113;
    v112 = 0x400000000LL;
    sub_3050D50((__int64)&v111, v17, v92, v42, v43, v44);
    sub_3050D50((__int64)&v111, v91, v90, v69, v70, v71);
    v72 = *(_QWORD *)(a2 + 16);
    v103 = *(_QWORD *)(a1 + 80);
    if ( v103 )
      sub_2AAAFA0(&v103);
    LODWORD(v104) = *(_DWORD *)(a1 + 72);
    v73 = sub_3400BD0(v72, 21520, (unsigned int)&v103, 7, 0, 0, 0, v91);
    sub_3050D50((__int64)&v111, v73, v74, v75, v76, v77);
    sub_9C6650(&v103);
    v78 = *(_QWORD *)(a2 + 16);
    v103 = *(_QWORD *)(a1 + 80);
    if ( v103 )
      sub_2AAAFA0(&v103);
    LODWORD(v104) = *(_DWORD *)(a1 + 72);
    v79 = sub_3400BD0(v78, 0, (unsigned int)&v103, 7, 0, 0, 0, (unsigned int)v104);
    sub_3050D50((__int64)&v111, v79, v80, v81, v82, v83);
    sub_9C6650(&v103);
    v85 = *(_QWORD *)(a2 + 16);
    v86 = (unsigned __int64)v111;
    v87 = (unsigned int)v112;
    v103 = *(_QWORD *)(a1 + 80);
    if ( v103 )
      sub_2AAAFA0(&v103);
    *((_QWORD *)&v89 + 1) = v87;
    *(_QWORD *)&v89 = v86;
    LODWORD(v104) = *(_DWORD *)(a1 + 72);
    v95 = sub_33FC220(v85, 537, (unsigned int)&v103, 7, 0, v84, v89);
    sub_9C6650(&v103);
    result = v95;
    if ( v111 != v113 )
      goto LABEL_29;
    return result;
  }
  if ( v99 != 32 || !a3 )
    return 0;
  v111 = v113;
  v112 = 0x400000000LL;
  sub_3050D50((__int64)&v111, v91, v90, v42, v43, v44);
  sub_3050D50((__int64)&v111, v17, v92, v45, v46, v47);
  v49 = *(_QWORD *)(a2 + 16);
  v50 = v99;
  v103 = *(_QWORD *)(a1 + 80);
  if ( v103 )
  {
    sub_2AAAFA0(&v103);
    v50 = v99;
  }
  v93 = v50;
  LODWORD(v104) = *(_DWORD *)(a1 + 72);
  v51 = sub_3400BD0(v49, v50, (unsigned int)&v103, 7, 0, 0, 0, v48);
  sub_3050D50((__int64)&v111, v51, v52, v53, v54, v55);
  sub_9C6650(&v103);
  v57 = *(_QWORD *)(a2 + 16);
  v103 = *(_QWORD *)(a1 + 80);
  v58 = v93;
  if ( v103 )
  {
    sub_2AAAFA0(&v103);
    v58 = v93;
  }
  LODWORD(v104) = *(_DWORD *)(a1 + 72);
  v59 = sub_3400BD0(v57, v58, (unsigned int)&v103, 7, 0, 0, 0, v56);
  sub_3050D50((__int64)&v111, v59, v60, v61, v62, v63);
  sub_9C6650(&v103);
  v65 = *(_QWORD *)(a2 + 16);
  v66 = (unsigned __int64)v111;
  v67 = (unsigned int)v112;
  v103 = *(_QWORD *)(a1 + 80);
  if ( v103 )
    sub_2AAAFA0(&v103);
  *((_QWORD *)&v88 + 1) = v67;
  *(_QWORD *)&v88 = v66;
  LODWORD(v104) = *(_DWORD *)(a1 + 72);
  v94 = sub_33FC220(v65, 536, (unsigned int)&v103, 8, 0, v64, v88);
  sub_9C6650(&v103);
  result = v94;
  if ( v111 != v113 )
  {
LABEL_29:
    v100 = result;
    _libc_free((unsigned __int64)v111);
    return v100;
  }
  return result;
}
