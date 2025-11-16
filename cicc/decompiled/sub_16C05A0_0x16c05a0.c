// Function: sub_16C05A0
// Address: 0x16c05a0
//
__int64 *__fastcall sub_16C05A0(__int64 *a1, char a2, char a3, __int64 a4, __int64 a5)
{
  unsigned int v8; // eax
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rsi
  unsigned __int64 v13; // rdx
  unsigned int v14; // eax
  unsigned int v15; // eax
  unsigned int v16; // eax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rdx
  unsigned int v20; // edx
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rcx
  unsigned int v24; // edx
  unsigned __int64 v25; // r8
  unsigned int v26; // ecx
  unsigned __int64 v27; // r8
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // r8
  unsigned int v32; // eax
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // r15
  unsigned int v35; // r15d
  unsigned __int64 v36; // r14
  bool v37; // cc
  __int64 v38; // rdi
  unsigned int v39; // edx
  unsigned __int64 v40; // rdi
  __int64 v41; // rax
  unsigned int v43; // edx
  unsigned int v44; // edi
  __int64 v45; // rsi
  unsigned __int64 v46; // rax
  unsigned int v47; // edi
  unsigned __int64 v48; // rsi
  unsigned int v49; // edi
  __int64 v50; // rsi
  __int64 v51; // rax
  unsigned int v52; // edi
  __int64 v53; // rax
  unsigned __int64 v54; // rcx
  unsigned int v55; // edx
  unsigned int v56; // edi
  __int64 v57; // rsi
  unsigned int v58; // ecx
  __int64 v59; // rax
  unsigned int v60; // edi
  unsigned int v61; // [rsp+4h] [rbp-DCh]
  unsigned int v62; // [rsp+10h] [rbp-D0h]
  unsigned int v63; // [rsp+10h] [rbp-D0h]
  unsigned int v64; // [rsp+10h] [rbp-D0h]
  unsigned int v65; // [rsp+10h] [rbp-D0h]
  unsigned int v66; // [rsp+10h] [rbp-D0h]
  const void **v68; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v69; // [rsp+18h] [rbp-C8h]
  __int64 *v70; // [rsp+20h] [rbp-C0h]
  unsigned int v71; // [rsp+20h] [rbp-C0h]
  __int64 v72; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v73; // [rsp+28h] [rbp-B8h]
  unsigned int v74; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v75; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v76; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v77; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v78; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v79; // [rsp+48h] [rbp-98h]
  unsigned __int64 v80; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v81; // [rsp+58h] [rbp-88h]
  unsigned __int64 v82; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v83; // [rsp+68h] [rbp-78h]
  __int64 v84; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v85; // [rsp+78h] [rbp-68h]
  unsigned __int64 v86; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v87; // [rsp+88h] [rbp-58h]
  unsigned __int64 v88; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v89; // [rsp+98h] [rbp-48h]
  unsigned __int64 v90; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v91; // [rsp+A8h] [rbp-38h]

  v8 = *(_DWORD *)(a5 + 8);
  v72 = 0;
  if ( !a2 )
  {
    v54 = *(_QWORD *)a5;
    v72 = 1;
    *(_QWORD *)a5 = *(_QWORD *)(a5 + 16);
    v55 = *(_DWORD *)(a5 + 24);
    *(_QWORD *)(a5 + 16) = v54;
    *(_DWORD *)(a5 + 24) = v8;
    v8 = v55;
    *(_DWORD *)(a5 + 8) = v55;
  }
  v87 = v8;
  if ( v8 <= 0x40 )
  {
    v9 = *(_QWORD *)a5;
LABEL_5:
    v10 = ~v9 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v8);
    v86 = v10;
    goto LABEL_6;
  }
  sub_16A4FD0((__int64)&v86, (const void **)a5);
  v8 = v87;
  if ( v87 <= 0x40 )
  {
    v9 = v86;
    goto LABEL_5;
  }
  sub_16A8F40((__int64 *)&v86);
  v8 = v87;
  v10 = v86;
LABEL_6:
  v89 = v8;
  v11 = *(_DWORD *)(a4 + 8);
  v88 = v10;
  v87 = 0;
  v83 = v11;
  if ( v11 > 0x40 )
  {
    sub_16A4FD0((__int64)&v82, (const void **)a4);
    v11 = v83;
    if ( v83 > 0x40 )
    {
      sub_16A8F40((__int64 *)&v82);
      v11 = v83;
      v13 = v82;
      goto LABEL_9;
    }
    v12 = v82;
  }
  else
  {
    v12 = *(_QWORD *)a4;
  }
  v13 = ~v12 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v11);
  v82 = v13;
LABEL_9:
  v85 = v11;
  v84 = v13;
  v83 = 0;
  sub_16A7200((__int64)&v88, &v84);
  v14 = v89;
  v89 = 0;
  v91 = v14;
  v90 = v88;
  sub_16A7490((__int64)&v90, v72);
  v77 = v91;
  v76 = v90;
  if ( v85 > 0x40 && v84 )
    j_j___libc_free_0_0(v84);
  if ( v83 > 0x40 && v82 )
    j_j___libc_free_0_0(v82);
  if ( v89 > 0x40 && v88 )
    j_j___libc_free_0_0(v88);
  if ( v87 > 0x40 && v86 )
    j_j___libc_free_0_0(v86);
  v70 = (__int64 *)(a5 + 16);
  v68 = (const void **)(a4 + 16);
  v89 = *(_DWORD *)(a4 + 24);
  if ( v89 > 0x40 )
    sub_16A4FD0((__int64)&v88, v68);
  else
    v88 = *(_QWORD *)(a4 + 16);
  sub_16A7200((__int64)&v88, v70);
  v15 = v89;
  v89 = 0;
  v91 = v15;
  v90 = v88;
  sub_16A7490((__int64)&v90, v72);
  v79 = v91;
  v78 = v90;
  if ( v89 > 0x40 && v88 )
    j_j___libc_free_0_0(v88);
  v16 = v77;
  v87 = v77;
  if ( v77 <= 0x40 )
  {
    v17 = v76;
LABEL_28:
    v18 = *(_QWORD *)a4 ^ v17;
    v87 = 0;
    v86 = v18;
LABEL_29:
    v19 = *(_QWORD *)a5 ^ v18;
LABEL_30:
    v73 = ~v19 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v16);
    goto LABEL_31;
  }
  sub_16A4FD0((__int64)&v86, (const void **)&v76);
  v16 = v87;
  if ( v87 <= 0x40 )
  {
    v17 = v86;
    goto LABEL_28;
  }
  sub_16A8F00((__int64 *)&v86, (__int64 *)a4);
  v16 = v87;
  v18 = v86;
  v87 = 0;
  v89 = v16;
  v88 = v86;
  if ( v16 <= 0x40 )
    goto LABEL_29;
  sub_16A8F00((__int64 *)&v88, (__int64 *)a5);
  v16 = v89;
  v19 = v88;
  v89 = 0;
  v91 = v16;
  v90 = v88;
  if ( v16 <= 0x40 )
    goto LABEL_30;
  sub_16A8F40((__int64 *)&v90);
  v16 = v91;
  v73 = v90;
  if ( v89 > 0x40 && v88 )
  {
    v66 = v91;
    j_j___libc_free_0_0(v88);
    v16 = v66;
  }
LABEL_31:
  if ( v87 > 0x40 && v86 )
  {
    v62 = v16;
    j_j___libc_free_0_0(v86);
    v16 = v62;
  }
  v20 = v79;
  v89 = v79;
  if ( v79 <= 0x40 )
  {
    v21 = v78;
LABEL_36:
    v22 = *(_QWORD *)(a4 + 16) ^ v21;
LABEL_37:
    v23 = *(_QWORD *)(a5 + 16) ^ v22;
    v81 = v20;
    v80 = v23;
    goto LABEL_38;
  }
  v63 = v16;
  sub_16A4FD0((__int64)&v88, (const void **)&v78);
  v20 = v89;
  v16 = v63;
  if ( v89 <= 0x40 )
  {
    v21 = v88;
    goto LABEL_36;
  }
  sub_16A8F00((__int64 *)&v88, (__int64 *)v68);
  v20 = v89;
  v22 = v88;
  v89 = 0;
  v16 = v63;
  v91 = v20;
  v90 = v88;
  if ( v20 <= 0x40 )
    goto LABEL_37;
  sub_16A8F00((__int64 *)&v90, v70);
  v16 = v63;
  v81 = v91;
  v80 = v90;
  if ( v89 > 0x40 && v88 )
  {
    j_j___libc_free_0_0(v88);
    v24 = *(_DWORD *)(a4 + 8);
    v16 = v63;
    v91 = v24;
    if ( v24 <= 0x40 )
      goto LABEL_39;
    goto LABEL_115;
  }
LABEL_38:
  v24 = *(_DWORD *)(a4 + 8);
  v91 = v24;
  if ( v24 <= 0x40 )
  {
LABEL_39:
    v25 = *(_QWORD *)a4;
    goto LABEL_40;
  }
LABEL_115:
  v64 = v16;
  sub_16A4FD0((__int64)&v90, (const void **)a4);
  v24 = v91;
  v16 = v64;
  if ( v91 > 0x40 )
  {
    sub_16A89F0((__int64 *)&v90, (__int64 *)v68);
    v26 = *(_DWORD *)(a5 + 8);
    v24 = v91;
    v27 = v90;
    v16 = v64;
    v91 = v26;
    if ( v26 <= 0x40 )
      goto LABEL_41;
    goto LABEL_117;
  }
  v25 = v90;
LABEL_40:
  v26 = *(_DWORD *)(a5 + 8);
  v27 = *(_QWORD *)(a4 + 16) | v25;
  v91 = v26;
  if ( v26 <= 0x40 )
  {
LABEL_41:
    v28 = *(_QWORD *)a5;
LABEL_42:
    v29 = *(_QWORD *)(a5 + 16) | v28;
    goto LABEL_43;
  }
LABEL_117:
  v61 = v16;
  v65 = v24;
  v69 = v27;
  sub_16A4FD0((__int64)&v90, (const void **)a5);
  v26 = v91;
  v27 = v69;
  v24 = v65;
  v16 = v61;
  if ( v91 <= 0x40 )
  {
    v28 = v90;
    goto LABEL_42;
  }
  sub_16A89F0((__int64 *)&v90, v70);
  v26 = v91;
  v29 = v90;
  v16 = v61;
  v24 = v65;
  v27 = v69;
LABEL_43:
  v82 = v29;
  v83 = v26;
  v91 = v16;
  v90 = v73;
  if ( v16 > 0x40 )
  {
    v71 = v24;
    v75 = v27;
    sub_16A89F0((__int64 *)&v90, (__int64 *)&v80);
    v16 = v91;
    v30 = v90;
    v24 = v71;
    v27 = v75;
  }
  else
  {
    v30 = v73 | v80;
  }
  v85 = v16;
  v84 = v30;
  v89 = v24;
  v88 = v27;
  if ( v24 <= 0x40 )
  {
    v31 = v82 & v27;
LABEL_47:
    v87 = v24;
    v86 = v31 & v30;
    goto LABEL_48;
  }
  sub_16A8890((__int64 *)&v88, (__int64 *)&v82);
  v24 = v89;
  v31 = v88;
  v89 = 0;
  v91 = v24;
  v90 = v88;
  if ( v24 <= 0x40 )
  {
    v30 = v84;
    goto LABEL_47;
  }
  sub_16A8890((__int64 *)&v90, &v84);
  v87 = v91;
  v86 = v90;
  if ( v89 > 0x40 && v88 )
    j_j___libc_free_0_0(v88);
LABEL_48:
  v32 = v77;
  v33 = v76;
  *((_DWORD *)a1 + 2) = 1;
  *a1 = 0;
  *((_DWORD *)a1 + 6) = 1;
  a1[2] = 0;
  v89 = v32;
  v88 = v33;
  v77 = 0;
  if ( v32 <= 0x40 )
  {
    v89 = 0;
    v88 = ~v33 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v32);
    v34 = v86 & v88;
LABEL_50:
    *a1 = v34;
    *((_DWORD *)a1 + 2) = v32;
    goto LABEL_51;
  }
  sub_16A8F40((__int64 *)&v88);
  v32 = v89;
  v89 = 0;
  v91 = v32;
  v90 = v88;
  if ( v32 <= 0x40 )
  {
    v34 = v86 & v88;
    v43 = *((_DWORD *)a1 + 2);
    v90 = v86 & v88;
  }
  else
  {
    sub_16A8890((__int64 *)&v90, (__int64 *)&v86);
    v32 = v91;
    v34 = v90;
    v43 = *((_DWORD *)a1 + 2);
  }
  v91 = 0;
  if ( v43 <= 0x40 || !*a1 )
    goto LABEL_50;
  v74 = v32;
  j_j___libc_free_0_0(*a1);
  v37 = v91 <= 0x40;
  *a1 = v34;
  *((_DWORD *)a1 + 2) = v74;
  if ( !v37 && v90 )
    j_j___libc_free_0_0(v90);
LABEL_51:
  if ( v89 > 0x40 && v88 )
    j_j___libc_free_0_0(v88);
  v35 = v79;
  v79 = 0;
  v91 = v35;
  v90 = v78;
  if ( v35 > 0x40 )
  {
    sub_16A8890((__int64 *)&v90, (__int64 *)&v86);
    v36 = v90;
    v35 = v91;
  }
  else
  {
    v90 = v86 & v78;
    v36 = v86 & v78;
  }
  v37 = *((_DWORD *)a1 + 6) <= 0x40u;
  v91 = 0;
  if ( v37 || (v38 = a1[2]) == 0 )
  {
    a1[2] = v36;
    *((_DWORD *)a1 + 6) = v35;
  }
  else
  {
    j_j___libc_free_0_0(v38);
    v37 = v91 <= 0x40;
    a1[2] = v36;
    *((_DWORD *)a1 + 6) = v35;
    if ( !v37 && v90 )
      j_j___libc_free_0_0(v90);
  }
  v39 = v87;
  v40 = v86;
  v41 = 1LL << ((unsigned __int8)v87 - 1);
  if ( v87 > 0x40 )
  {
    if ( (*(_QWORD *)(v86 + 8LL * ((v87 - 1) >> 6)) & v41) != 0 || !a3 )
    {
      if ( !v86 )
        goto LABEL_65;
      goto LABEL_84;
    }
LABEL_96:
    v44 = *(_DWORD *)(a4 + 8);
    v45 = 1LL << ((unsigned __int8)v44 - 1);
    v46 = *(_QWORD *)a4;
    if ( v44 > 0x40 )
    {
      if ( (*(_QWORD *)(v46 + 8LL * ((v44 - 1) >> 6)) & v45) == 0 )
        goto LABEL_101;
    }
    else if ( (v46 & v45) == 0 )
    {
      goto LABEL_101;
    }
    v47 = *(_DWORD *)(a5 + 8);
    v48 = *(_QWORD *)a5;
    if ( v47 > 0x40 )
      v48 = *(_QWORD *)(v48 + 8LL * ((v47 - 1) >> 6));
    if ( (v48 & (1LL << ((unsigned __int8)v47 - 1))) != 0 )
    {
      v56 = *((_DWORD *)a1 + 2);
      v57 = *a1;
      v58 = v56 - 1;
      v59 = 1LL << ((unsigned __int8)v56 - 1);
      if ( v56 <= 0x40 )
      {
        *a1 = v57 | v59;
        goto LABEL_106;
      }
      goto LABEL_140;
    }
LABEL_101:
    v49 = *(_DWORD *)(a4 + 24);
    v50 = 1LL << ((unsigned __int8)v49 - 1);
    v51 = *(_QWORD *)(a4 + 16);
    if ( v49 > 0x40 )
    {
      if ( (*(_QWORD *)(v51 + 8LL * ((v49 - 1) >> 6)) & v50) == 0 )
        goto LABEL_106;
    }
    else if ( (v51 & v50) == 0 )
    {
      goto LABEL_106;
    }
    v52 = *(_DWORD *)(a5 + 24);
    v53 = *(_QWORD *)(a5 + 16);
    if ( v52 > 0x40 )
      v53 = *(_QWORD *)(v53 + 8LL * ((v52 - 1) >> 6));
    if ( (v53 & (1LL << ((unsigned __int8)v52 - 1))) == 0 )
    {
LABEL_106:
      if ( v39 <= 0x40 )
        goto LABEL_65;
      v40 = v86;
      if ( !v86 )
        goto LABEL_65;
LABEL_84:
      j_j___libc_free_0_0(v40);
      goto LABEL_65;
    }
    v60 = *((_DWORD *)a1 + 6);
    v57 = a1[2];
    v58 = v60 - 1;
    v59 = 1LL << ((unsigned __int8)v60 - 1);
    if ( v60 <= 0x40 )
    {
      a1[2] = v57 | v59;
      goto LABEL_106;
    }
LABEL_140:
    *(_QWORD *)(v57 + 8LL * (v58 >> 6)) |= v59;
    v39 = v87;
    goto LABEL_106;
  }
  if ( (v41 & v86) == 0 && a3 )
    goto LABEL_96;
LABEL_65:
  if ( v85 > 0x40 && v84 )
    j_j___libc_free_0_0(v84);
  if ( v83 > 0x40 && v82 )
    j_j___libc_free_0_0(v82);
  if ( v81 > 0x40 && v80 )
    j_j___libc_free_0_0(v80);
  if ( v79 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
  if ( v77 > 0x40 && v76 )
    j_j___libc_free_0_0(v76);
  return a1;
}
