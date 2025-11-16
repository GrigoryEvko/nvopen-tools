// Function: sub_14981D0
// Address: 0x14981d0
//
char __fastcall sub_14981D0(__int64 a1, __int64 *a2, __int64 *a3)
{
  char result; // al
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 *v26; // rax
  __int64 *v27; // rax
  unsigned __int64 v28; // rdx
  char v29; // cl
  unsigned __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // rdi
  __int64 *v33; // rax
  char v34; // cl
  unsigned __int64 v35; // rdx
  unsigned int v36; // eax
  __int64 v37; // rdi
  __int64 *v38; // rax
  __int64 *v39; // [rsp-130h] [rbp-130h]
  unsigned __int64 v40; // [rsp-128h] [rbp-128h]
  __int64 *v41; // [rsp-128h] [rbp-128h]
  __int64 v42; // [rsp-120h] [rbp-120h]
  unsigned int v43; // [rsp-120h] [rbp-120h]
  unsigned __int64 v44; // [rsp-118h] [rbp-118h]
  unsigned int v45; // [rsp-110h] [rbp-110h]
  __int64 v46; // [rsp-108h] [rbp-108h]
  __int64 v47; // [rsp-108h] [rbp-108h]
  bool v48; // [rsp-108h] [rbp-108h]
  __int64 *v49; // [rsp-108h] [rbp-108h]
  __int64 v50; // [rsp-100h] [rbp-100h]
  __int64 v51; // [rsp-100h] [rbp-100h]
  __int64 v52; // [rsp-100h] [rbp-100h]
  unsigned int v53; // [rsp-100h] [rbp-100h]
  __int64 *v54; // [rsp-100h] [rbp-100h]
  unsigned __int64 v55; // [rsp-F8h] [rbp-F8h] BYREF
  unsigned int v56; // [rsp-F0h] [rbp-F0h]
  unsigned __int64 v57; // [rsp-E8h] [rbp-E8h] BYREF
  unsigned int v58; // [rsp-E0h] [rbp-E0h]
  __int64 v59; // [rsp-D8h] [rbp-D8h] BYREF
  unsigned int v60; // [rsp-D0h] [rbp-D0h]
  unsigned __int64 v61; // [rsp-C8h] [rbp-C8h] BYREF
  unsigned int v62; // [rsp-C0h] [rbp-C0h]
  unsigned __int64 v63; // [rsp-B8h] [rbp-B8h] BYREF
  unsigned int v64; // [rsp-B0h] [rbp-B0h]
  __int64 v65; // [rsp-A8h] [rbp-A8h] BYREF
  unsigned int v66; // [rsp-A0h] [rbp-A0h]
  __int64 v67; // [rsp-98h] [rbp-98h] BYREF
  __int64 v68; // [rsp-90h] [rbp-90h]
  __int64 v69; // [rsp-88h] [rbp-88h] BYREF
  __int64 v70; // [rsp-80h] [rbp-80h]
  __int64 v71; // [rsp-78h] [rbp-78h]
  __int64 v72; // [rsp-68h] [rbp-68h] BYREF
  __int64 v73; // [rsp-60h] [rbp-60h]
  __int64 v74; // [rsp-58h] [rbp-58h] BYREF
  __int64 v75; // [rsp-50h] [rbp-50h]
  __int64 v76; // [rsp-48h] [rbp-48h]

  result = 0;
  if ( !a2[1] || !a3[1] )
    return result;
  v7 = sub_146F1B0(*(_QWORD *)(a1 + 8), *a2);
  v8 = sub_146F1B0(*(_QWORD *)(a1 + 8), *a3);
  if ( v7 == v8 )
    return 3;
  v50 = *(_QWORD *)(a1 + 8);
  v9 = sub_1456040(v7);
  v51 = sub_1456E10(v50, v9);
  v46 = *(_QWORD *)(a1 + 8);
  v10 = sub_1456040(v8);
  if ( v51 != sub_1456E10(v46, v10) )
  {
LABEL_6:
    v11 = sub_1498160(a1, v7);
    v12 = sub_1498160(a1, v8);
    v13 = v12;
    if ( !v11 || *a2 == v11 )
    {
      result = 1;
      if ( !v13 || *a3 == v13 )
        return result;
      v72 = v13;
      v73 = -1;
      v74 = 0;
      v75 = 0;
      v76 = 0;
      if ( v11 )
      {
        v19 = 0;
        v20 = 0;
        v18 = 0;
        v21 = -1;
      }
      else
      {
        v18 = a2[2];
        v19 = a2[3];
        v20 = a2[4];
        v21 = a2[1];
        v11 = *a2;
      }
    }
    else
    {
      if ( v12 )
      {
        v14 = 0;
        v15 = 0;
        v16 = 0;
        v17 = -1;
      }
      else
      {
        v15 = a3[2];
        v14 = a3[3];
        v16 = a3[4];
        v17 = a3[1];
        v13 = *a3;
      }
      v73 = v17;
      v72 = v13;
      v18 = 0;
      v74 = v15;
      v19 = 0;
      v75 = v14;
      v20 = 0;
      v76 = v16;
      v21 = -1;
    }
    v69 = v18;
    v70 = v19;
    v67 = v11;
    v68 = v21;
    v71 = v20;
    return (unsigned __int8)sub_14981D0(a1, &v67, &v72) != 0;
  }
  v52 = *(_QWORD *)(a1 + 8);
  v22 = sub_1456040(v7);
  v23 = sub_1456C90(v52, v22);
  v24 = a2[1];
  v56 = v23;
  if ( v23 <= 0x40 )
  {
    v58 = v23;
    v28 = a3[1] & (0xFFFFFFFFFFFFFFFFLL >> -(char)v23);
    v55 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v23) & v24;
    v57 = v28;
  }
  else
  {
    v53 = v23;
    sub_16A4EF0(&v55, v24, 0);
    v25 = a3[1];
    v58 = v53;
    sub_16A4EF0(&v57, v25, 0);
  }
  v47 = sub_14806B0(*(_QWORD *)(a1 + 8), v8, v7, 0, 0);
  v26 = sub_1477920(*(_QWORD *)(a1 + 8), v47, 0);
  LODWORD(v68) = *((_DWORD *)v26 + 2);
  if ( (unsigned int)v68 > 0x40 )
  {
    v54 = v26;
    sub_16A4FD0(&v67, v26);
    v26 = v54;
  }
  else
  {
    v67 = *v26;
  }
  LODWORD(v70) = *((_DWORD *)v26 + 6);
  if ( (unsigned int)v70 > 0x40 )
    sub_16A4FD0(&v69, v26 + 2);
  else
    v69 = v26[2];
  sub_158AAD0(&v59, &v67);
  if ( (int)sub_16A9900(&v55, &v59) > 0 )
  {
    v48 = 0;
    goto LABEL_21;
  }
  v29 = v58;
  v62 = v58;
  if ( v58 > 0x40 )
  {
    sub_16A4FD0(&v61, &v57);
    v29 = v62;
    if ( v62 > 0x40 )
    {
      sub_16A8F40(&v61);
      goto LABEL_72;
    }
    v30 = v61;
  }
  else
  {
    v30 = v57;
  }
  v61 = ~v30 & (0xFFFFFFFFFFFFFFFFLL >> -v29);
LABEL_72:
  sub_16A7400(&v61);
  v31 = v62;
  v32 = *(_QWORD *)(a1 + 8);
  v62 = 0;
  v43 = v31;
  v64 = v31;
  v40 = v61;
  v63 = v61;
  v33 = sub_1477920(v32, v47, 0);
  LODWORD(v73) = *((_DWORD *)v33 + 2);
  if ( (unsigned int)v73 > 0x40 )
  {
    v39 = v33;
    sub_16A4FD0(&v72, v33);
    v33 = v39;
  }
  else
  {
    v72 = *v33;
  }
  LODWORD(v75) = *((_DWORD *)v33 + 6);
  if ( (unsigned int)v75 > 0x40 )
    sub_16A4FD0(&v74, v33 + 2);
  else
    v74 = v33[2];
  sub_158A9F0(&v65, &v72);
  v48 = (int)sub_16A9900(&v63, &v65) >= 0;
  if ( v66 > 0x40 && v65 )
    j_j___libc_free_0_0(v65);
  if ( (unsigned int)v75 > 0x40 && v74 )
    j_j___libc_free_0_0(v74);
  if ( (unsigned int)v73 > 0x40 && v72 )
    j_j___libc_free_0_0(v72);
  if ( v43 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v62 > 0x40 && v61 )
    j_j___libc_free_0_0(v61);
LABEL_21:
  if ( v60 > 0x40 && v59 )
    j_j___libc_free_0_0(v59);
  if ( (unsigned int)v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  if ( (unsigned int)v68 > 0x40 && v67 )
    j_j___libc_free_0_0(v67);
  if ( v48 )
    goto LABEL_59;
  v42 = sub_14806B0(*(_QWORD *)(a1 + 8), v7, v8, 0, 0);
  v27 = sub_1477920(*(_QWORD *)(a1 + 8), v42, 0);
  LODWORD(v68) = *((_DWORD *)v27 + 2);
  if ( (unsigned int)v68 > 0x40 )
  {
    v41 = v27;
    sub_16A4FD0(&v67, v27);
    v27 = v41;
  }
  else
  {
    v67 = *v27;
  }
  LODWORD(v70) = *((_DWORD *)v27 + 6);
  if ( (unsigned int)v70 > 0x40 )
    sub_16A4FD0(&v69, v27 + 2);
  else
    v69 = v27[2];
  sub_158AAD0(&v59, &v67);
  if ( (int)sub_16A9900(&v57, &v59) <= 0 )
  {
    v34 = v56;
    v62 = v56;
    if ( v56 > 0x40 )
    {
      sub_16A4FD0(&v61, &v55);
      v34 = v62;
      if ( v62 > 0x40 )
      {
        sub_16A8F40(&v61);
        goto LABEL_98;
      }
      v35 = v61;
    }
    else
    {
      v35 = v55;
    }
    v61 = ~v35 & (0xFFFFFFFFFFFFFFFFLL >> -v34);
LABEL_98:
    sub_16A7400(&v61);
    v36 = v62;
    v37 = *(_QWORD *)(a1 + 8);
    v62 = 0;
    v45 = v36;
    v64 = v36;
    v44 = v61;
    v63 = v61;
    v38 = sub_1477920(v37, v42, 0);
    LODWORD(v73) = *((_DWORD *)v38 + 2);
    if ( (unsigned int)v73 > 0x40 )
    {
      v49 = v38;
      sub_16A4FD0(&v72, v38);
      v38 = v49;
    }
    else
    {
      v72 = *v38;
    }
    LODWORD(v75) = *((_DWORD *)v38 + 6);
    if ( (unsigned int)v75 > 0x40 )
      sub_16A4FD0(&v74, v38 + 2);
    else
      v74 = v38[2];
    sub_158A9F0(&v65, &v72);
    v48 = (int)sub_16A9900(&v63, &v65) >= 0;
    if ( v66 > 0x40 && v65 )
      j_j___libc_free_0_0(v65);
    if ( (unsigned int)v75 > 0x40 && v74 )
      j_j___libc_free_0_0(v74);
    if ( (unsigned int)v73 > 0x40 && v72 )
      j_j___libc_free_0_0(v72);
    if ( v45 > 0x40 && v44 )
      j_j___libc_free_0_0(v44);
    if ( v62 > 0x40 && v61 )
      j_j___libc_free_0_0(v61);
  }
  if ( v60 > 0x40 && v59 )
    j_j___libc_free_0_0(v59);
  if ( (unsigned int)v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  if ( (unsigned int)v68 > 0x40 && v67 )
    j_j___libc_free_0_0(v67);
  if ( !v48 )
  {
    if ( v58 > 0x40 && v57 )
      j_j___libc_free_0_0(v57);
    if ( v56 > 0x40 && v55 )
      j_j___libc_free_0_0(v55);
    goto LABEL_6;
  }
LABEL_59:
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  if ( v56 > 0x40 && v55 )
    j_j___libc_free_0_0(v55);
  return 0;
}
