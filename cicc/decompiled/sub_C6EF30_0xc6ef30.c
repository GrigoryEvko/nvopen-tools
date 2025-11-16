// Function: sub_C6EF30
// Address: 0xc6ef30
//
__int64 __fastcall sub_C6EF30(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, unsigned __int8 a5)
{
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  unsigned int v10; // eax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rdx
  unsigned int v13; // eax
  unsigned int v14; // eax
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  unsigned int v21; // edx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rcx
  __int64 v24; // rcx
  unsigned int v25; // edx
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // r13
  unsigned int v28; // ecx
  unsigned __int64 v29; // rsi
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // r13
  unsigned int v33; // r13d
  __int64 v34; // rax
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rbx
  __int64 v37; // rbx
  unsigned int v38; // r13d
  unsigned __int64 v39; // rbx
  bool v40; // cc
  __int64 v41; // rdi
  unsigned int v43; // eax
  unsigned int v44; // [rsp+Ch] [rbp-D4h]
  unsigned int v45; // [rsp+Ch] [rbp-D4h]
  unsigned int v46; // [rsp+Ch] [rbp-D4h]
  unsigned int v47; // [rsp+Ch] [rbp-D4h]
  unsigned int v48; // [rsp+Ch] [rbp-D4h]
  const void **v49; // [rsp+10h] [rbp-D0h]
  const void **v51; // [rsp+18h] [rbp-C8h]
  unsigned int v52; // [rsp+18h] [rbp-C8h]
  __int64 v53; // [rsp+20h] [rbp-C0h]
  __int64 v54; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v55; // [rsp+20h] [rbp-C0h]
  unsigned int v56; // [rsp+20h] [rbp-C0h]
  __int64 v57; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v58; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v59; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v60; // [rsp+48h] [rbp-98h]
  __int64 v61; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v62; // [rsp+58h] [rbp-88h]
  __int64 v63; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v64; // [rsp+68h] [rbp-78h]
  __int64 v65; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v66; // [rsp+78h] [rbp-68h]
  __int64 v67; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v68; // [rsp+88h] [rbp-58h]
  __int64 v69; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v70; // [rsp+98h] [rbp-48h]
  __int64 v71; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v72; // [rsp+A8h] [rbp-38h]

  v53 = a4 ^ 1u;
  v7 = *(_DWORD *)(a3 + 8);
  v72 = v7;
  if ( v7 > 0x40 )
  {
    sub_C43780((__int64)&v71, (const void **)a3);
    v7 = v72;
    if ( v72 > 0x40 )
    {
      sub_C43D10((__int64)&v71);
      v7 = v72;
      v9 = v71;
      goto LABEL_5;
    }
    v8 = v71;
  }
  else
  {
    v8 = *(_QWORD *)a3;
  }
  v9 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & ~v8;
  if ( !v7 )
    v9 = 0;
LABEL_5:
  v70 = v7;
  v10 = *(_DWORD *)(a2 + 8);
  v69 = v9;
  v72 = v10;
  if ( v10 > 0x40 )
  {
    sub_C43780((__int64)&v71, (const void **)a2);
    v10 = v72;
    if ( v72 > 0x40 )
    {
      sub_C43D10((__int64)&v71);
      v10 = v72;
      v12 = v71;
      goto LABEL_9;
    }
    v11 = v71;
  }
  else
  {
    v11 = *(_QWORD *)a2;
  }
  v12 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v10) & ~v11;
  if ( !v10 )
    v12 = 0;
LABEL_9:
  v68 = v10;
  v67 = v12;
  sub_C45EE0((__int64)&v69, &v67);
  v13 = v70;
  v70 = 0;
  v72 = v13;
  v71 = v69;
  sub_C46A40((__int64)&v71, v53);
  v58 = v72;
  v57 = v71;
  if ( v68 > 0x40 && v67 )
    j_j___libc_free_0_0(v67);
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  v54 = a5;
  v49 = (const void **)(a3 + 16);
  v70 = *(_DWORD *)(a3 + 24);
  if ( v70 > 0x40 )
    sub_C43780((__int64)&v69, v49);
  else
    v69 = *(_QWORD *)(a3 + 16);
  v51 = (const void **)(a2 + 16);
  v68 = *(_DWORD *)(a2 + 24);
  if ( v68 > 0x40 )
    sub_C43780((__int64)&v67, v51);
  else
    v67 = *(_QWORD *)(a2 + 16);
  sub_C45EE0((__int64)&v69, &v67);
  v14 = v70;
  v70 = 0;
  v72 = v14;
  v71 = v69;
  sub_C46A40((__int64)&v71, v54);
  v60 = v72;
  v59 = v71;
  if ( v68 > 0x40 && v67 )
    j_j___libc_free_0_0(v67);
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  v15 = v58;
  v68 = v58;
  if ( v58 <= 0x40 )
  {
    v16 = v57;
LABEL_27:
    v17 = *(_QWORD *)a2 ^ v16;
    v68 = 0;
    v67 = v17;
LABEL_28:
    v18 = *(_QWORD *)a3 ^ v17;
LABEL_29:
    v19 = 0;
    v20 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & ~v18;
    if ( v15 )
      v19 = v20;
    v55 = v19;
    goto LABEL_32;
  }
  sub_C43780((__int64)&v67, (const void **)&v57);
  v15 = v68;
  if ( v68 <= 0x40 )
  {
    v16 = v67;
    goto LABEL_27;
  }
  sub_C43C10(&v67, (__int64 *)a2);
  v15 = v68;
  v17 = v67;
  v68 = 0;
  v70 = v15;
  v69 = v67;
  if ( v15 <= 0x40 )
    goto LABEL_28;
  sub_C43C10(&v69, (__int64 *)a3);
  v15 = v70;
  v18 = v69;
  v70 = 0;
  v72 = v15;
  v71 = v69;
  if ( v15 <= 0x40 )
    goto LABEL_29;
  sub_C43D10((__int64)&v71);
  v15 = v72;
  v55 = v71;
  if ( v70 > 0x40 && v69 )
  {
    v46 = v72;
    j_j___libc_free_0_0(v69);
    v15 = v46;
  }
LABEL_32:
  if ( v68 > 0x40 && v67 )
  {
    v44 = v15;
    j_j___libc_free_0_0(v67);
    v15 = v44;
  }
  v21 = v60;
  v70 = v60;
  if ( v60 <= 0x40 )
  {
    v22 = v59;
LABEL_37:
    v23 = *(_QWORD *)(a2 + 16) ^ v22;
LABEL_38:
    v24 = *(_QWORD *)(a3 + 16) ^ v23;
    v62 = v21;
    v61 = v24;
    goto LABEL_39;
  }
  v48 = v15;
  sub_C43780((__int64)&v69, (const void **)&v59);
  v21 = v70;
  v15 = v48;
  if ( v70 <= 0x40 )
  {
    v22 = v69;
    goto LABEL_37;
  }
  sub_C43C10(&v69, (__int64 *)v51);
  v21 = v70;
  v23 = v69;
  v70 = 0;
  v15 = v48;
  v72 = v21;
  v71 = v69;
  if ( v21 <= 0x40 )
    goto LABEL_38;
  sub_C43C10(&v71, (__int64 *)v49);
  v15 = v48;
  v62 = v72;
  v61 = v71;
  if ( v70 > 0x40 && v69 )
  {
    j_j___libc_free_0_0(v69);
    v15 = v48;
  }
LABEL_39:
  v25 = *(_DWORD *)(a2 + 8);
  v72 = v25;
  if ( v25 <= 0x40 )
  {
    v26 = *(_QWORD *)a2;
LABEL_41:
    v27 = *(_QWORD *)(a2 + 16) | v26;
    goto LABEL_42;
  }
  v47 = v15;
  sub_C43780((__int64)&v71, (const void **)a2);
  v25 = v72;
  v15 = v47;
  if ( v72 <= 0x40 )
  {
    v26 = v71;
    goto LABEL_41;
  }
  sub_C43BD0(&v71, (__int64 *)v51);
  v25 = v72;
  v27 = v71;
  v15 = v47;
LABEL_42:
  v28 = *(_DWORD *)(a3 + 8);
  v72 = v28;
  if ( v28 <= 0x40 )
  {
    v29 = *(_QWORD *)a3;
LABEL_44:
    v30 = *(_QWORD *)(a3 + 16) | v29;
    goto LABEL_45;
  }
  v45 = v15;
  v52 = v25;
  sub_C43780((__int64)&v71, (const void **)a3);
  v28 = v72;
  v25 = v52;
  v15 = v45;
  if ( v72 <= 0x40 )
  {
    v29 = v71;
    goto LABEL_44;
  }
  sub_C43BD0(&v71, (__int64 *)v49);
  v28 = v72;
  v30 = v71;
  v15 = v45;
  v25 = v52;
LABEL_45:
  v64 = v28;
  v63 = v30;
  v72 = v15;
  v71 = v55;
  if ( v15 > 0x40 )
  {
    v56 = v25;
    sub_C43BD0(&v71, &v61);
    v15 = v72;
    v31 = v71;
    v25 = v56;
  }
  else
  {
    v31 = v61 | v55;
  }
  v66 = v15;
  v65 = v31;
  v70 = v25;
  v69 = v27;
  if ( v25 <= 0x40 )
  {
    v32 = v63 & v27;
LABEL_49:
    v68 = v25;
    v67 = v32 & v31;
    goto LABEL_50;
  }
  sub_C43B90(&v69, &v63);
  v25 = v70;
  v32 = v69;
  v70 = 0;
  v72 = v25;
  v71 = v69;
  if ( v25 <= 0x40 )
  {
    v31 = v65;
    goto LABEL_49;
  }
  sub_C43B90(&v71, &v65);
  v68 = v72;
  v67 = v71;
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
LABEL_50:
  v33 = v58;
  v34 = v57;
  *(_DWORD *)(a1 + 8) = 1;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  v70 = v33;
  v69 = v34;
  v58 = 0;
  if ( v33 <= 0x40 )
  {
    v70 = 0;
    v35 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v33) & ~v34;
    v36 = 0;
    if ( v33 )
      v36 = v35;
    v69 = v36;
    v37 = v67 & v36;
    goto LABEL_54;
  }
  sub_C43D10((__int64)&v69);
  v33 = v70;
  v70 = 0;
  v72 = v33;
  v71 = v69;
  if ( v33 <= 0x40 )
  {
    v37 = v67 & v69;
    v43 = *(_DWORD *)(a1 + 8);
    v71 = v67 & v69;
  }
  else
  {
    sub_C43B90(&v71, &v67);
    v33 = v72;
    v37 = v71;
    v43 = *(_DWORD *)(a1 + 8);
  }
  v72 = 0;
  if ( v43 <= 0x40 || !*(_QWORD *)a1 )
  {
LABEL_54:
    *(_QWORD *)a1 = v37;
    *(_DWORD *)(a1 + 8) = v33;
    goto LABEL_55;
  }
  j_j___libc_free_0_0(*(_QWORD *)a1);
  v40 = v72 <= 0x40;
  *(_QWORD *)a1 = v37;
  *(_DWORD *)(a1 + 8) = v33;
  if ( !v40 && v71 )
    j_j___libc_free_0_0(v71);
LABEL_55:
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  v38 = v60;
  v60 = 0;
  v72 = v38;
  v71 = v59;
  if ( v38 > 0x40 )
  {
    sub_C43B90(&v71, &v67);
    v39 = v71;
    v38 = v72;
  }
  else
  {
    v71 = v67 & v59;
    v39 = v67 & v59;
  }
  v40 = *(_DWORD *)(a1 + 24) <= 0x40u;
  v72 = 0;
  if ( v40 || (v41 = *(_QWORD *)(a1 + 16)) == 0 )
  {
    *(_QWORD *)(a1 + 16) = v39;
    *(_DWORD *)(a1 + 24) = v38;
  }
  else
  {
    j_j___libc_free_0_0(v41);
    v40 = v72 <= 0x40;
    *(_QWORD *)(a1 + 16) = v39;
    *(_DWORD *)(a1 + 24) = v38;
    if ( !v40 && v71 )
    {
      j_j___libc_free_0_0(v71);
      if ( v68 <= 0x40 )
        goto LABEL_65;
      goto LABEL_83;
    }
  }
  if ( v68 <= 0x40 )
    goto LABEL_65;
LABEL_83:
  if ( v67 )
    j_j___libc_free_0_0(v67);
LABEL_65:
  if ( v66 > 0x40 && v65 )
    j_j___libc_free_0_0(v65);
  if ( v64 > 0x40 && v63 )
    j_j___libc_free_0_0(v63);
  if ( v62 > 0x40 && v61 )
    j_j___libc_free_0_0(v61);
  if ( v60 > 0x40 && v59 )
    j_j___libc_free_0_0(v59);
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  return a1;
}
