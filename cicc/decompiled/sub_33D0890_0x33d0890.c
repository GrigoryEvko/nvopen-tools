// Function: sub_33D0890
// Address: 0x33d0890
//
__int64 __fastcall sub_33D0890(
        __int64 a1,
        __int64 a2,
        unsigned __int64 *a3,
        unsigned int *a4,
        bool *a5,
        unsigned int a6,
        char a7)
{
  unsigned __int16 *v9; // rdx
  unsigned __int16 v10; // ax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  char v14; // al
  __int64 v15; // rsi
  unsigned int v16; // ebx
  bool v18; // cc
  unsigned __int16 v19; // r15
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // ebx
  int v24; // r13d
  unsigned int v25; // r12d
  unsigned int v26; // r14d
  __int64 *v27; // rsi
  unsigned int v28; // eax
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rdx
  int v32; // eax
  __int64 v33; // rax
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  unsigned int v37; // r14d
  unsigned __int64 v38; // rax
  bool v39; // al
  unsigned int v40; // r14d
  unsigned __int64 v41; // r15
  _QWORD *v42; // r15
  unsigned int v43; // r15d
  unsigned __int64 v44; // r14
  unsigned __int64 v45; // r14
  const void *v46; // r14
  unsigned int v47; // eax
  unsigned __int64 v48; // r8
  unsigned __int64 v49; // r8
  unsigned __int64 v50; // r8
  bool v51; // si
  unsigned int v52; // r14d
  unsigned __int64 v53; // r15
  unsigned __int64 v54; // r15
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  unsigned int v58; // eax
  bool v59; // al
  unsigned __int64 v60; // r8
  unsigned int v61; // r14d
  unsigned __int64 v62; // [rsp+0h] [rbp-150h]
  unsigned __int64 *v64; // [rsp+18h] [rbp-138h]
  unsigned int v65; // [rsp+18h] [rbp-138h]
  int v67; // [rsp+28h] [rbp-128h]
  __int64 v68; // [rsp+28h] [rbp-128h]
  unsigned int v69; // [rsp+30h] [rbp-120h]
  bool v72; // [rsp+48h] [rbp-108h]
  unsigned __int64 v73; // [rsp+48h] [rbp-108h]
  unsigned __int16 v74; // [rsp+50h] [rbp-100h] BYREF
  __int64 v75; // [rsp+58h] [rbp-F8h]
  unsigned __int64 v76; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v77; // [rsp+68h] [rbp-E8h]
  unsigned __int64 v78; // [rsp+70h] [rbp-E0h] BYREF
  unsigned int v79; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v80; // [rsp+80h] [rbp-D0h] BYREF
  unsigned int v81; // [rsp+88h] [rbp-C8h]
  unsigned __int64 v82; // [rsp+90h] [rbp-C0h] BYREF
  unsigned int v83; // [rsp+98h] [rbp-B8h]
  unsigned __int64 v84; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned int v85; // [rsp+A8h] [rbp-A8h]
  unsigned __int64 v86; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned int v87; // [rsp+B8h] [rbp-98h]
  unsigned __int64 v88; // [rsp+C0h] [rbp-90h] BYREF
  unsigned int v89; // [rsp+C8h] [rbp-88h]
  const void *v90; // [rsp+D0h] [rbp-80h] BYREF
  unsigned int v91; // [rsp+D8h] [rbp-78h]
  const void *v92; // [rsp+E0h] [rbp-70h] BYREF
  unsigned int v93; // [rsp+E8h] [rbp-68h]
  __int64 v94; // [rsp+F0h] [rbp-60h]
  __int64 v95; // [rsp+F8h] [rbp-58h]
  unsigned __int64 v96; // [rsp+100h] [rbp-50h] BYREF
  __int64 v97; // [rsp+108h] [rbp-48h]
  _QWORD *v98; // [rsp+110h] [rbp-40h] BYREF
  __int64 v99; // [rsp+118h] [rbp-38h]

  v9 = *(unsigned __int16 **)(a1 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v74 = v10;
  v75 = v11;
  if ( v10 )
  {
    if ( v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
      goto LABEL_163;
    v33 = 16LL * (v10 - 1);
    v13 = *(_QWORD *)&byte_444C4A0[v33];
    v14 = byte_444C4A0[v33 + 8];
  }
  else
  {
    v94 = sub_3007260((__int64)&v74);
    v95 = v12;
    v13 = v94;
    v14 = v95;
  }
  v98 = (_QWORD *)v13;
  LOBYTE(v99) = v14;
  v15 = a6;
  v69 = sub_CA1930(&v98);
  v16 = v69;
  if ( v69 < a6 )
    return 0;
  LODWORD(v99) = v69;
  if ( v69 > 0x40 )
  {
    v15 = 0;
    sub_C43690((__int64)&v98, 0, 0);
    if ( *(_DWORD *)(a2 + 8) <= 0x40u || (v34 = *(_QWORD *)a2) == 0 )
    {
      v15 = 0;
      *(_QWORD *)a2 = v98;
      *(_DWORD *)(a2 + 8) = v99;
      LODWORD(v99) = v69;
      sub_C43690((__int64)&v98, 0, 0);
      goto LABEL_9;
    }
  }
  else
  {
    v18 = *(_DWORD *)(a2 + 8) <= 0x40u;
    v98 = 0;
    if ( v18 )
    {
      *(_QWORD *)a2 = 0;
      *(_DWORD *)(a2 + 8) = v69;
      goto LABEL_8;
    }
    v34 = *(_QWORD *)a2;
    if ( !*(_QWORD *)a2 )
    {
      *(_DWORD *)(a2 + 8) = v69;
      goto LABEL_8;
    }
  }
  j_j___libc_free_0_0(v34);
  *(_QWORD *)a2 = v98;
  *(_DWORD *)(a2 + 8) = v99;
  LODWORD(v99) = v69;
  if ( v69 > 0x40 )
  {
    v15 = 0;
    sub_C43690((__int64)&v98, 0, 0);
    goto LABEL_9;
  }
LABEL_8:
  v98 = 0;
LABEL_9:
  if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
    j_j___libc_free_0_0(*a3);
  v19 = v74;
  *a3 = (unsigned __int64)v98;
  *((_DWORD *)a3 + 2) = v99;
  v67 = *(_DWORD *)(a1 + 64);
  if ( v19 )
  {
    if ( (unsigned __int16)(v19 - 17) > 0xD3u )
    {
LABEL_14:
      v20 = v75;
      goto LABEL_15;
    }
    v19 = word_4456580[v19 - 1];
    v20 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v74) )
      goto LABEL_14;
    v19 = sub_3009970((__int64)&v74, v15, v55, v56, v57);
  }
LABEL_15:
  LOWORD(v96) = v19;
  v97 = v20;
  if ( v19 )
  {
    if ( v19 != 1 && (unsigned __int16)(v19 - 504) > 7u )
    {
      v21 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
      goto LABEL_17;
    }
LABEL_163:
    BUG();
  }
  v21 = sub_3007260((__int64)&v96);
  v98 = (_QWORD *)v21;
  v99 = v22;
LABEL_17:
  if ( !v67 )
    goto LABEL_46;
  v64 = a3;
  v23 = v67;
  v68 = a2;
  v24 = 1;
  v25 = 0;
  v62 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v21);
  v26 = v21;
  while ( 1 )
  {
    v30 = (unsigned int)(v24 - 1);
    if ( a7 )
      v30 = (unsigned int)(v23 - v24);
    v31 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40 * v30);
    v32 = *(_DWORD *)(v31 + 24);
    if ( v32 != 51 )
    {
      if ( v32 == 35 || v32 == 11 )
      {
        sub_C44AB0((__int64)&v96, *(_QWORD *)(v31 + 96) + 24LL, v26);
      }
      else
      {
        if ( v32 != 12 && v32 != 36 )
          return 0;
        v27 = (__int64 *)(*(_QWORD *)(v31 + 96) + 24LL);
        if ( (void *)*v27 == sub_C33340() )
          sub_C3E660((__int64)&v96, (__int64)v27);
        else
          sub_C3A850((__int64)&v96, v27);
      }
      sub_C43D80(v68, (__int64)&v96, v25);
      v28 = v26 + v25;
      if ( (unsigned int)v97 > 0x40 && v96 )
      {
        j_j___libc_free_0_0(v96);
        v28 = v26 + v25;
      }
      goto LABEL_27;
    }
    v28 = v26 + v25;
    if ( v26 + v25 != v25 )
      break;
LABEL_27:
    v29 = v24 + 1;
    v25 = v28;
    if ( v23 == v24 )
      goto LABEL_45;
LABEL_28:
    v24 = v29;
  }
  if ( v25 > 0x3F || v28 > 0x40 )
  {
    sub_C43C90(v64, v25, v28);
    v28 = v26 + v25;
    goto LABEL_27;
  }
  v35 = v62 << v25;
  v36 = *v64;
  if ( *((_DWORD *)v64 + 2) > 0x40u )
  {
    *(_QWORD *)v36 |= v35;
    goto LABEL_27;
  }
  v25 += v26;
  *v64 = v35 | v36;
  v29 = v24 + 1;
  if ( v23 != v24 )
    goto LABEL_28;
LABEL_45:
  v16 = v69;
  a2 = v68;
  a3 = v64;
LABEL_46:
  v37 = *((_DWORD *)a3 + 2);
  if ( v37 <= 0x40 )
  {
    v38 = *a3;
    goto LABEL_48;
  }
  v61 = v37 - sub_C444A0((__int64)a3);
  v39 = 1;
  if ( v61 <= 0x40 )
  {
    v38 = *(_QWORD *)*a3;
LABEL_48:
    v39 = v38 != 0;
  }
  *a5 = v39;
  if ( v69 <= 8 || (v69 & 1) != 0 )
    goto LABEL_152;
  while ( 2 )
  {
    v65 = v16;
    v16 >>= 1;
    sub_C440A0((__int64)&v76, (__int64 *)a2, v16, v16);
    sub_C440A0((__int64)&v78, (__int64 *)a2, v16, 0);
    sub_C440A0((__int64)&v80, (__int64 *)a3, v16, v16);
    sub_C440A0((__int64)&v82, (__int64 *)a3, v16, 0);
    v43 = v81;
    v91 = v81;
    if ( v81 <= 0x40 )
    {
      v44 = v80;
      goto LABEL_80;
    }
    sub_C43780((__int64)&v90, (const void **)&v80);
    v43 = v91;
    if ( v91 <= 0x40 )
    {
      v44 = (unsigned __int64)v90;
LABEL_80:
      v91 = 0;
      v45 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v43) & ~v44;
      if ( !v43 )
        v45 = 0;
      v90 = (const void *)v45;
      goto LABEL_83;
    }
    sub_C43D10((__int64)&v90);
    v43 = v91;
    v45 = (unsigned __int64)v90;
    v91 = 0;
    v93 = v43;
    v92 = v90;
    if ( v43 <= 0x40 )
    {
LABEL_83:
      v46 = (const void *)(v78 & v45);
      v92 = v46;
      goto LABEL_84;
    }
    sub_C43B90(&v92, (__int64 *)&v78);
    v43 = v93;
    v46 = v92;
LABEL_84:
    v47 = v83;
    LODWORD(v97) = v43;
    v96 = (unsigned __int64)v46;
    v93 = 0;
    v85 = v83;
    if ( v83 <= 0x40 )
    {
      v48 = v82;
      goto LABEL_86;
    }
    sub_C43780((__int64)&v84, (const void **)&v82);
    v47 = v85;
    if ( v85 <= 0x40 )
    {
      v48 = v84;
LABEL_86:
      v85 = 0;
      v49 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v47) & ~v48;
      if ( !v47 )
        v49 = 0;
      v84 = v49;
      goto LABEL_89;
    }
    sub_C43D10((__int64)&v84);
    v47 = v85;
    v49 = v84;
    v85 = 0;
    v87 = v47;
    v86 = v84;
    if ( v47 <= 0x40 )
    {
LABEL_89:
      v50 = v76 & v49;
      v89 = v47;
      v86 = v50;
      v88 = v50;
      v87 = 0;
LABEL_90:
      v51 = a6 > v16;
      if ( (const void *)v50 != v46 )
        v51 = 1;
      v72 = v51;
      goto LABEL_93;
    }
    sub_C43B90(&v86, (__int64 *)&v76);
    v58 = v87;
    v50 = v86;
    v87 = 0;
    v89 = v58;
    v88 = v86;
    if ( v58 <= 0x40 )
      goto LABEL_90;
    v73 = v86;
    v59 = sub_C43C50((__int64)&v88, (const void **)&v96);
    v60 = v73;
    v72 = !v59 || a6 > v16;
    if ( v60 )
      j_j___libc_free_0_0(v60);
LABEL_93:
    if ( v87 > 0x40 && v86 )
      j_j___libc_free_0_0(v86);
    if ( v85 > 0x40 && v84 )
      j_j___libc_free_0_0(v84);
    if ( v43 > 0x40 && v46 )
      j_j___libc_free_0_0((unsigned __int64)v46);
    if ( v93 > 0x40 && v92 )
      j_j___libc_free_0_0((unsigned __int64)v92);
    if ( v91 > 0x40 && v90 )
      j_j___libc_free_0_0((unsigned __int64)v90);
    if ( !v72 )
    {
      v52 = v77;
      LODWORD(v97) = v77;
      if ( v77 <= 0x40 )
      {
        v53 = v76;
        goto LABEL_111;
      }
      sub_C43780((__int64)&v96, (const void **)&v76);
      v52 = v97;
      if ( (unsigned int)v97 <= 0x40 )
      {
        v53 = v96;
LABEL_111:
        v54 = v78 | v53;
        v96 = v54;
      }
      else
      {
        sub_C43BD0(&v96, (__int64 *)&v78);
        v52 = v97;
        v54 = v96;
      }
      v18 = *(_DWORD *)(a2 + 8) <= 0x40u;
      LODWORD(v97) = 0;
      if ( v18 || !*(_QWORD *)a2 )
      {
        *(_QWORD *)a2 = v54;
        *(_DWORD *)(a2 + 8) = v52;
      }
      else
      {
        j_j___libc_free_0_0(*(_QWORD *)a2);
        v18 = (unsigned int)v97 <= 0x40;
        *(_QWORD *)a2 = v54;
        *(_DWORD *)(a2 + 8) = v52;
        if ( !v18 && v96 )
          j_j___libc_free_0_0(v96);
      }
      v40 = v81;
      LODWORD(v97) = v81;
      if ( v81 <= 0x40 )
      {
        v41 = v80;
        goto LABEL_58;
      }
      sub_C43780((__int64)&v96, (const void **)&v80);
      v40 = v97;
      if ( (unsigned int)v97 <= 0x40 )
      {
        v41 = v96;
LABEL_58:
        v42 = (_QWORD *)(v82 & v41);
        v96 = (unsigned __int64)v42;
      }
      else
      {
        sub_C43B90(&v96, (__int64 *)&v82);
        v40 = v97;
        v42 = (_QWORD *)v96;
      }
      v18 = *((_DWORD *)a3 + 2) <= 0x40u;
      LODWORD(v97) = 0;
      if ( v18 || !*a3 )
      {
        *a3 = (unsigned __int64)v42;
        *((_DWORD *)a3 + 2) = v40;
      }
      else
      {
        j_j___libc_free_0_0(*a3);
        v18 = (unsigned int)v97 <= 0x40;
        *a3 = (unsigned __int64)v42;
        *((_DWORD *)a3 + 2) = v40;
        if ( !v18 && v96 )
          j_j___libc_free_0_0(v96);
      }
      if ( v83 > 0x40 && v82 )
        j_j___libc_free_0_0(v82);
      if ( v81 > 0x40 && v80 )
        j_j___libc_free_0_0(v80);
      if ( v79 > 0x40 && v78 )
        j_j___libc_free_0_0(v78);
      if ( v77 > 0x40 && v76 )
        j_j___libc_free_0_0(v76);
      if ( v16 <= 8 || (v16 & 1) != 0 )
        goto LABEL_152;
      continue;
    }
    break;
  }
  if ( v83 > 0x40 && v82 )
    j_j___libc_free_0_0(v82);
  if ( v81 > 0x40 && v80 )
    j_j___libc_free_0_0(v80);
  if ( v79 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
  if ( v77 > 0x40 && v76 )
    j_j___libc_free_0_0(v76);
  v16 = v65;
LABEL_152:
  *a4 = v16;
  return 1;
}
