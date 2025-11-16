// Function: sub_3283260
// Address: 0x3283260
//
__int64 __fastcall sub_3283260(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  unsigned __int16 *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdi
  int v14; // eax
  bool v15; // r12
  bool v16; // cl
  __int64 v17; // rt0
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rdi
  int v22; // eax
  _QWORD *v23; // rsi
  __int64 v24; // r9
  int v25; // eax
  __int64 v26; // rdi
  _QWORD *v27; // rax
  __int64 v28; // rdi
  _QWORD *v29; // rax
  unsigned __int64 v30; // r11
  __int64 v31; // r13
  __int64 v32; // r9
  __int64 v33; // rdx
  unsigned int v34; // eax
  int v35; // r9d
  unsigned __int64 v36; // r11
  int v37; // r9d
  __int64 v38; // r12
  unsigned int v39; // ecx
  char v40; // r12
  __int64 v41; // r12
  __int64 v42; // r12
  __int64 v43; // rdx
  __int64 v44; // r13
  int v45; // eax
  __int64 v46; // r14
  __int128 v47; // rax
  int v48; // r9d
  __int64 v49; // r12
  __int64 v50; // rax
  __int64 *v51; // rsi
  __int64 v52; // rcx
  __int64 v53; // rcx
  _QWORD *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 *v57; // rdi
  __int64 v58; // rsi
  __int64 v59; // rsi
  _QWORD *v60; // rax
  bool v61; // zf
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // r12
  __int64 v65; // rax
  __int64 *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rdi
  _QWORD *v69; // rdx
  __int64 v70; // rax
  __int64 *v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rsi
  _QWORD *v74; // rax
  __int128 v75; // [rsp-D8h] [rbp-D8h]
  __int128 v76; // [rsp-C8h] [rbp-C8h]
  __int128 v77; // [rsp-C8h] [rbp-C8h]
  unsigned __int64 v78; // [rsp-B0h] [rbp-B0h]
  __int64 v79; // [rsp-A8h] [rbp-A8h]
  bool v80; // [rsp-98h] [rbp-98h]
  unsigned int v81; // [rsp-90h] [rbp-90h]
  unsigned int v82; // [rsp-88h] [rbp-88h]
  unsigned __int64 v83; // [rsp-80h] [rbp-80h]
  unsigned __int64 v84; // [rsp-78h] [rbp-78h]
  unsigned int v86; // [rsp-6Ch] [rbp-6Ch]
  unsigned int v87; // [rsp-58h] [rbp-58h] BYREF
  __int64 v88; // [rsp-50h] [rbp-50h]
  __int64 v89; // [rsp-48h] [rbp-48h] BYREF
  __int64 v90; // [rsp-40h] [rbp-40h]
  __int64 v91; // [rsp-8h] [rbp-8h] BYREF

  if ( !*((_BYTE *)a1 + 33) )
    return 0;
  v10 = *(unsigned __int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOWORD(v87) = v11;
  v88 = v12;
  if ( (_WORD)v11 != 8 && (unsigned __int16)(v11 - 6) > 1u )
    return 0;
  v13 = a1[1];
  if ( !*(_QWORD *)(v13 + 8 * v11 + 112) || (*(_BYTE *)(v13 + 500LL * (unsigned __int16)v11 + 6611) & 0xFB) != 0 )
    return 0;
  v14 = *(_DWORD *)(a3 + 24);
  if ( v14 != 186 )
  {
    if ( *(_DWORD *)(a4 + 24) != 186 )
    {
      v15 = 0;
      v16 = 0;
      goto LABEL_11;
    }
LABEL_65:
    v61 = *(_DWORD *)(**(_QWORD **)(a4 + 40) + 24LL) == 190;
    v62 = a3;
    a3 = a4;
    a4 = v62;
    if ( v61 )
      goto LABEL_50;
    goto LABEL_66;
  }
  if ( *(_DWORD *)(**(_QWORD **)(a3 + 40) + 24LL) != 192 )
  {
    if ( *(_DWORD *)(a4 + 24) != 186 )
      goto LABEL_50;
    goto LABEL_65;
  }
LABEL_66:
  if ( *(_DWORD *)(a4 + 24) != 186 )
  {
    v15 = 0;
    goto LABEL_58;
  }
  v63 = a3;
  a3 = a4;
  a4 = v63;
LABEL_50:
  v50 = *(_QWORD *)(a3 + 56);
  if ( !v50 )
    return 0;
  if ( *(_QWORD *)(v50 + 32) )
    return 0;
  v51 = *(__int64 **)(a3 + 40);
  v52 = v51[5];
  v15 = *(_DWORD *)(v52 + 24) == 11 || *(_DWORD *)(v52 + 24) == 35;
  if ( !v15 )
    return 0;
  v53 = *(_QWORD *)(v52 + 96);
  v54 = *(_QWORD **)(v53 + 24);
  if ( *(_DWORD *)(v53 + 32) > 0x40u )
    v54 = (_QWORD *)*v54;
  v16 = v54 != (_QWORD *)65280 && v54 != (_QWORD *)0xFFFF;
  if ( v16 )
    return 0;
  a3 = *v51;
  if ( *(_DWORD *)(a4 + 24) != 186 )
  {
    v14 = *(_DWORD *)(a3 + 24);
    goto LABEL_11;
  }
  v55 = a3;
  a3 = a4;
  a4 = v55;
LABEL_58:
  v56 = *(_QWORD *)(a3 + 56);
  if ( !v56 )
    return 0;
  if ( *(_QWORD *)(v56 + 32) )
    return 0;
  v57 = *(__int64 **)(a3 + 40);
  v58 = v57[5];
  v16 = *(_DWORD *)(v58 + 24) == 11 || *(_DWORD *)(v58 + 24) == 35;
  if ( !v16 )
    return 0;
  v59 = *(_QWORD *)(v58 + 96);
  v60 = *(_QWORD **)(v59 + 24);
  if ( *(_DWORD *)(v59 + 32) > 0x40u )
    v60 = (_QWORD *)*v60;
  if ( v60 != (_QWORD *)255 )
    return 0;
  v14 = *(_DWORD *)(a4 + 24);
  a3 = a4;
  a4 = *v57;
LABEL_11:
  if ( v14 == 192 )
  {
    if ( *(_DWORD *)(a4 + 24) != 190 )
      return 0;
  }
  else
  {
    if ( v14 != 190 || *(_DWORD *)(a4 + 24) != 192 )
      return 0;
    v17 = a3;
    a3 = a4;
    a4 = v17;
  }
  v18 = *(_QWORD *)(a4 + 56);
  if ( !v18 )
    return 0;
  if ( *(_QWORD *)(v18 + 32) )
    return 0;
  v19 = *(_QWORD *)(a3 + 56);
  if ( !v19 )
    return 0;
  if ( *(_QWORD *)(v19 + 32) )
    return 0;
  v20 = *(_QWORD **)(a4 + 40);
  v21 = v20[5];
  v22 = *(_DWORD *)(v21 + 24);
  if ( v22 != 35 && v22 != 11 )
    return 0;
  v23 = *(_QWORD **)(a3 + 40);
  v24 = v23[5];
  v25 = *(_DWORD *)(v24 + 24);
  if ( v25 != 11 && v25 != 35 )
    return 0;
  v26 = *(_QWORD *)(v21 + 96);
  v27 = *(_QWORD **)(v26 + 24);
  if ( *(_DWORD *)(v26 + 32) > 0x40u )
    v27 = (_QWORD *)*v27;
  if ( v27 != (_QWORD *)8 )
    return 0;
  v28 = *(_QWORD *)(v24 + 96);
  v29 = *(_QWORD **)(v28 + 24);
  if ( *(_DWORD *)(v28 + 32) > 0x40u )
    v29 = (_QWORD *)*v29;
  if ( v29 != (_QWORD *)8 )
    return 0;
  v30 = v20[1];
  v31 = *v20;
  v86 = *((_DWORD *)v20 + 2);
  if ( !v15 && *(_DWORD *)(v31 + 24) == 186 )
  {
    v65 = *(_QWORD *)(v31 + 56);
    if ( !v65 )
      return 0;
    if ( *(_QWORD *)(v65 + 32) )
      return 0;
    v66 = *(__int64 **)(v31 + 40);
    v67 = v66[5];
    v15 = *(_DWORD *)(v67 + 24) == 35 || *(_DWORD *)(v67 + 24) == 11;
    if ( !v15 )
      return 0;
    v68 = *(_QWORD *)(v67 + 96);
    v69 = *(_QWORD **)(v68 + 24);
    if ( *(_DWORD *)(v68 + 32) > 0x40u )
      v69 = (_QWORD *)*v69;
    if ( v69 != (_QWORD *)255 )
      return 0;
    v31 = *v66;
    v86 = *((_DWORD *)v66 + 2);
    v30 = v86 | v30 & 0xFFFFFFFF00000000LL;
  }
  v32 = *v23;
  v84 = v23[1];
  v82 = *((_DWORD *)v23 + 2);
  if ( !v16 && *(_DWORD *)(v32 + 24) == 186 )
  {
    v70 = *(_QWORD *)(v32 + 56);
    if ( !v70 )
      return 0;
    if ( *(_QWORD *)(v70 + 32) )
      return 0;
    v71 = *(__int64 **)(v32 + 40);
    v72 = v71[5];
    v16 = *(_DWORD *)(v72 + 24) == 35 || *(_DWORD *)(v72 + 24) == 11;
    if ( !v16 )
      return 0;
    v73 = *(_QWORD *)(v72 + 96);
    v74 = *(_QWORD **)(v73 + 24);
    if ( *(_DWORD *)(v73 + 32) > 0x40u )
      v74 = (_QWORD *)*v74;
    if ( v74 != (_QWORD *)65280 && v74 != (_QWORD *)0xFFFF )
      return 0;
    v32 = *v71;
    v84 = *((unsigned int *)v71 + 2) | v84 & 0xFFFFFFFF00000000LL;
    v82 = *((_DWORD *)v71 + 2);
  }
  if ( v82 != v86 || v32 != v31 )
    return 0;
  v78 = v30;
  v80 = v16;
  v79 = v32;
  v89 = sub_2D5B750((unsigned __int16 *)&v87);
  v90 = v33;
  v34 = sub_CA1930(&v89);
  v36 = v78;
  v81 = v34;
  if ( v34 <= 0x10 )
  {
    v64 = *a1;
    v89 = *(_QWORD *)(a2 + 80);
    if ( v89 )
      sub_325F5D0(&v89);
    LODWORD(v90) = *(_DWORD *)(a2 + 72);
    *((_QWORD *)&v77 + 1) = v86 | v78 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v77 = v31;
    v49 = sub_33FAF80(v64, 197, (unsigned int)&v91 - 64, v87, v88, v35, v77);
    sub_9C6650(&v89);
    return v49;
  }
  v37 = v79;
  if ( !v15 )
  {
    if ( a5 )
      return 0;
  }
  if ( !v80 )
  {
    v38 = *a1;
    v39 = 24;
    if ( a5 )
      v39 = v34;
    sub_327D420((__int64)&v89, v34, 0x10u, v39);
    v40 = sub_33DD210(v38, v79, v82 | v84 & 0xFFFFFFFF00000000LL, &v89, 0);
    sub_969240(&v89);
    v36 = v78;
    if ( !v40 )
      return 0;
  }
  v41 = *a1;
  v89 = *(_QWORD *)(a2 + 80);
  if ( v89 )
  {
    v83 = v36;
    sub_325F5D0(&v89);
    v36 = v83;
  }
  LODWORD(v90) = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v76 + 1) = v86 | v36 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v76 = v31;
  v42 = sub_33FAF80(v41, 197, (unsigned int)&v91 - 64, v87, v88, v37, v76);
  v44 = v43;
  sub_9C6650(&v89);
  v89 = *(_QWORD *)(a2 + 80);
  if ( v89 )
    sub_325F5D0(&v89);
  v45 = *(_DWORD *)(a2 + 72);
  v46 = *a1;
  LODWORD(v90) = v45;
  *(_QWORD *)&v47 = sub_3400E40(v46, v81 - 16, v87, v88, &v89);
  *((_QWORD *)&v75 + 1) = v44;
  *(_QWORD *)&v75 = v42;
  v49 = sub_3406EB0(v46, 192, (unsigned int)&v91 - 64, v87, v88, v48, v75, v47);
  sub_9C6650(&v89);
  return v49;
}
