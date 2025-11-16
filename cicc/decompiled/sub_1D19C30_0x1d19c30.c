// Function: sub_1D19C30
// Address: 0x1d19c30
//
__int64 __fastcall sub_1D19C30(
        __int64 a1,
        __int64 a2,
        unsigned __int64 *a3,
        unsigned int *a4,
        bool *a5,
        __int64 a6,
        char a7)
{
  unsigned __int64 *v7; // r13
  unsigned __int64 *v8; // rbx
  __int64 v9; // rax
  char v10; // di
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned int v15; // r12d
  bool v17; // cc
  unsigned __int64 v18; // rdi
  int v19; // eax
  int v20; // eax
  char v21; // r15
  __int64 v22; // rdx
  unsigned int v23; // esi
  unsigned int v24; // r12d
  unsigned int v25; // r14d
  int v26; // r13d
  __int64 *v27; // rsi
  unsigned int v28; // eax
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int16 v32; // ax
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rdx
  unsigned int v35; // r14d
  unsigned __int64 v36; // rax
  bool v37; // al
  unsigned int v38; // eax
  unsigned int v39; // eax
  unsigned int v40; // edx
  __int64 v41; // rax
  unsigned __int64 v42; // r14
  const void *v43; // r14
  unsigned int v44; // eax
  __int64 v45; // rsi
  unsigned __int64 v46; // r8
  unsigned __int64 v47; // r8
  bool v48; // al
  unsigned int v49; // r14d
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rax
  unsigned int v52; // r14d
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // rax
  unsigned int v55; // eax
  bool v56; // r9
  __int64 v57; // rdi
  unsigned int v58; // r14d
  unsigned __int64 v60; // [rsp+10h] [rbp-120h]
  unsigned int v61; // [rsp+10h] [rbp-120h]
  bool v62; // [rsp+18h] [rbp-118h]
  bool v63; // [rsp+18h] [rbp-118h]
  unsigned int v64; // [rsp+18h] [rbp-118h]
  __int64 v65; // [rsp+18h] [rbp-118h]
  bool v66; // [rsp+18h] [rbp-118h]
  unsigned __int64 *v67; // [rsp+20h] [rbp-110h]
  unsigned __int64 v68; // [rsp+20h] [rbp-110h]
  unsigned int v69; // [rsp+20h] [rbp-110h]
  unsigned int v70; // [rsp+20h] [rbp-110h]
  bool v71; // [rsp+20h] [rbp-110h]
  bool v72; // [rsp+20h] [rbp-110h]
  bool v73; // [rsp+20h] [rbp-110h]
  unsigned __int64 v74; // [rsp+20h] [rbp-110h]
  unsigned int v75; // [rsp+20h] [rbp-110h]
  unsigned int v77; // [rsp+30h] [rbp-100h]
  unsigned int v78; // [rsp+34h] [rbp-FCh]
  int v80; // [rsp+48h] [rbp-E8h]
  char v81[8]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v82; // [rsp+58h] [rbp-D8h]
  unsigned __int64 v83; // [rsp+60h] [rbp-D0h] BYREF
  unsigned int v84; // [rsp+68h] [rbp-C8h]
  __int64 v85; // [rsp+70h] [rbp-C0h] BYREF
  unsigned int v86; // [rsp+78h] [rbp-B8h]
  unsigned __int64 v87; // [rsp+80h] [rbp-B0h] BYREF
  unsigned int v88; // [rsp+88h] [rbp-A8h]
  unsigned __int64 v89; // [rsp+90h] [rbp-A0h] BYREF
  unsigned int v90; // [rsp+98h] [rbp-98h]
  __int64 v91; // [rsp+A0h] [rbp-90h] BYREF
  unsigned int v92; // [rsp+A8h] [rbp-88h]
  __int64 v93; // [rsp+B0h] [rbp-80h] BYREF
  unsigned int v94; // [rsp+B8h] [rbp-78h]
  __int64 v95; // [rsp+C0h] [rbp-70h] BYREF
  unsigned int v96; // [rsp+C8h] [rbp-68h]
  const void *v97; // [rsp+D0h] [rbp-60h] BYREF
  unsigned int v98; // [rsp+D8h] [rbp-58h]
  const void *v99; // [rsp+E0h] [rbp-50h] BYREF
  unsigned int v100; // [rsp+E8h] [rbp-48h]
  unsigned __int64 v101; // [rsp+F0h] [rbp-40h] BYREF
  __int64 v102; // [rsp+F8h] [rbp-38h]

  v7 = a3;
  v8 = (unsigned __int64 *)a2;
  v9 = *(_QWORD *)(a1 + 40);
  v10 = *(_BYTE *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v78 = a6;
  v81[0] = v10;
  v82 = v11;
  if ( v10 )
  {
    v15 = sub_1D13440(v10);
    if ( v78 > v15 )
      return 0;
  }
  else
  {
    v15 = sub_1F58D40(v81, a2, a3, a4, a5, a6);
    if ( v78 > v15 )
      return 0;
  }
  LODWORD(v102) = v15;
  if ( v15 > 0x40 )
  {
    a2 = 0;
    sub_16A4EF0((__int64)&v101, 0, 0);
    if ( *((_DWORD *)v8 + 2) <= 0x40u || (v18 = *v8) == 0 )
    {
      *v8 = v101;
      v20 = v102;
      LODWORD(v102) = v15;
      *((_DWORD *)v8 + 2) = v20;
      goto LABEL_13;
    }
LABEL_10:
    j_j___libc_free_0_0(v18);
    *v8 = v101;
    v19 = v102;
    LODWORD(v102) = v15;
    *((_DWORD *)v8 + 2) = v19;
    if ( v15 <= 0x40 )
      goto LABEL_11;
LABEL_13:
    a2 = 0;
    sub_16A4EF0((__int64)&v101, 0, 0);
    goto LABEL_14;
  }
  v17 = *(_DWORD *)(a2 + 8) <= 0x40u;
  v101 = 0;
  if ( !v17 )
  {
    v18 = *(_QWORD *)a2;
    if ( !*(_QWORD *)a2 )
    {
      *(_DWORD *)(a2 + 8) = v15;
      goto LABEL_11;
    }
    goto LABEL_10;
  }
  *(_QWORD *)a2 = 0;
  *(_DWORD *)(a2 + 8) = v15;
LABEL_11:
  v101 = 0;
LABEL_14:
  if ( *((_DWORD *)v7 + 2) > 0x40u && *v7 )
    j_j___libc_free_0_0(*v7);
  v21 = v81[0];
  *v7 = v101;
  *((_DWORD *)v7 + 2) = v102;
  v80 = *(_DWORD *)(a1 + 56);
  if ( v21 )
  {
    if ( (unsigned __int8)(v21 - 14) <= 0x5Fu )
    {
      switch ( v21 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          v21 = 3;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          v21 = 4;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          v21 = 5;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          v21 = 6;
          break;
        case 55:
          v21 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v21 = 8;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          v21 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v21 = 10;
          break;
        default:
          v21 = 2;
          break;
      }
      goto LABEL_125;
    }
    goto LABEL_19;
  }
  if ( !(unsigned __int8)sub_1F58D20(v81) )
  {
LABEL_19:
    v22 = v82;
    goto LABEL_20;
  }
  v21 = sub_1F596B0(v81);
LABEL_20:
  LOBYTE(v101) = v21;
  v102 = v22;
  if ( v21 )
  {
LABEL_125:
    v23 = sub_1D13440(v21);
    goto LABEL_22;
  }
  v23 = sub_1F58D40(&v101, a2, v22, v12, v13, v14);
LABEL_22:
  if ( !v80 )
    goto LABEL_47;
  v77 = v15;
  v24 = 0;
  v25 = v23;
  v67 = v7;
  v26 = 1;
  v60 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v23);
  while ( 1 )
  {
    v30 = (unsigned int)(v26 - 1);
    if ( a7 )
      v30 = (unsigned int)(v80 - v26);
    v31 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40 * v30);
    v32 = *(_WORD *)(v31 + 24);
    if ( v32 != 48 )
    {
      if ( v32 == 32 || v32 == 10 )
      {
        sub_16A5D10((__int64)&v101, *(_QWORD *)(v31 + 88) + 24LL, v25);
      }
      else
      {
        if ( v32 != 33 && v32 != 11 )
          return 0;
        v27 = (__int64 *)(*(_QWORD *)(v31 + 88) + 32LL);
        if ( (void *)*v27 == sub_16982C0() )
          sub_169D930((__int64)&v101, (__int64)v27);
        else
          sub_169D7E0((__int64)&v101, v27);
      }
      sub_16A52E0((__int64)v8, (__int64)&v101, v24);
      v28 = v25 + v24;
      if ( (unsigned int)v102 > 0x40 && v101 )
      {
        j_j___libc_free_0_0(v101);
        v28 = v25 + v24;
      }
      goto LABEL_32;
    }
    v28 = v25 + v24;
    if ( v25 + v24 != v24 )
      break;
LABEL_32:
    v29 = v26 + 1;
    v24 = v28;
    if ( v80 == v26 )
      goto LABEL_46;
LABEL_33:
    v26 = v29;
  }
  if ( v24 > 0x3F || v28 > 0x40 )
  {
    sub_16A5260(v67, v24, v28);
    v28 = v25 + v24;
    goto LABEL_32;
  }
  v33 = v60 << v24;
  v34 = *v67;
  if ( *((_DWORD *)v67 + 2) > 0x40u )
  {
    *(_QWORD *)v34 |= v33;
    goto LABEL_32;
  }
  v24 += v25;
  *v67 = v33 | v34;
  v29 = v26 + 1;
  if ( v80 != v26 )
    goto LABEL_33;
LABEL_46:
  v15 = v77;
  v7 = v67;
LABEL_47:
  v35 = *((_DWORD *)v7 + 2);
  if ( v35 <= 0x40 )
  {
    v36 = *v7;
    goto LABEL_49;
  }
  v58 = v35 - sub_16A57B0((__int64)v7);
  v37 = 1;
  if ( v58 <= 0x40 )
  {
    v36 = *(_QWORD *)*v7;
LABEL_49:
    v37 = v36 != 0;
  }
  *a5 = v37;
  if ( v15 <= 8 )
    goto LABEL_153;
  while ( 2 )
  {
    v38 = *((_DWORD *)v8 + 2);
    v61 = v15;
    v15 >>= 1;
    LODWORD(v102) = v38;
    if ( v38 > 0x40 )
    {
      sub_16A4FD0((__int64)&v101, (const void **)v8);
      v38 = v102;
      if ( (unsigned int)v102 <= 0x40 )
        goto LABEL_71;
      sub_16A8110((__int64)&v101, v15);
    }
    else
    {
      v101 = *v8;
LABEL_71:
      if ( v15 == v38 )
        v101 = 0;
      else
        v101 >>= v15;
    }
    sub_16A5A50((__int64)&v83, (__int64 *)&v101, v15);
    if ( (unsigned int)v102 > 0x40 && v101 )
      j_j___libc_free_0_0(v101);
    sub_16A5A50((__int64)&v85, (__int64 *)v8, v15);
    v39 = *((_DWORD *)v7 + 2);
    LODWORD(v102) = v39;
    if ( v39 > 0x40 )
    {
      sub_16A4FD0((__int64)&v101, (const void **)v7);
      v39 = v102;
      if ( (unsigned int)v102 <= 0x40 )
        goto LABEL_78;
      sub_16A8110((__int64)&v101, v15);
    }
    else
    {
      v101 = *v7;
LABEL_78:
      if ( v15 == v39 )
        v101 = 0;
      else
        v101 >>= v15;
    }
    sub_16A5A50((__int64)&v87, (__int64 *)&v101, v15);
    if ( (unsigned int)v102 > 0x40 && v101 )
      j_j___libc_free_0_0(v101);
    sub_16A5A50((__int64)&v89, (__int64 *)v7, v15);
    v40 = v88;
    v98 = v88;
    if ( v88 <= 0x40 )
    {
      v41 = v87;
      goto LABEL_85;
    }
    sub_16A4FD0((__int64)&v97, (const void **)&v87);
    v40 = v98;
    if ( v98 <= 0x40 )
    {
      v41 = (__int64)v97;
LABEL_85:
      v98 = 0;
      v42 = ~v41 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v40);
      v97 = (const void *)v42;
      goto LABEL_86;
    }
    sub_16A8F40((__int64 *)&v97);
    v40 = v98;
    v42 = (unsigned __int64)v97;
    v98 = 0;
    v100 = v40;
    v99 = v97;
    if ( v40 <= 0x40 )
    {
LABEL_86:
      v43 = (const void *)(v85 & v42);
      v99 = v43;
      goto LABEL_87;
    }
    sub_16A8890((__int64 *)&v99, &v85);
    v40 = v100;
    v43 = v99;
LABEL_87:
    v44 = v90;
    LODWORD(v102) = v40;
    v101 = (unsigned __int64)v43;
    v100 = 0;
    v92 = v90;
    if ( v90 <= 0x40 )
    {
      v45 = v89;
      goto LABEL_89;
    }
    v64 = v40;
    sub_16A4FD0((__int64)&v91, (const void **)&v89);
    v44 = v92;
    v40 = v64;
    if ( v92 <= 0x40 )
    {
      v45 = v91;
LABEL_89:
      v92 = 0;
      v46 = ~v45 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v44);
      v91 = v46;
      goto LABEL_90;
    }
    v75 = v64;
    sub_16A8F40(&v91);
    v44 = v92;
    v46 = v91;
    v92 = 0;
    v40 = v64;
    v94 = v44;
    v93 = v91;
    if ( v44 <= 0x40 )
    {
LABEL_90:
      v47 = v83 & v46;
      v96 = v44;
      v93 = v47;
      v95 = v47;
      v94 = 0;
      goto LABEL_91;
    }
    sub_16A8890(&v93, (__int64 *)&v83);
    v55 = v94;
    v47 = v93;
    v94 = 0;
    v40 = v64;
    v96 = v55;
    v95 = v93;
    if ( v55 <= 0x40 )
    {
LABEL_91:
      v48 = v78 > v15;
      if ( v43 != (const void *)v47 )
        v48 = 1;
      goto LABEL_93;
    }
    v65 = v93;
    v56 = sub_16A5220((__int64)&v95, (const void **)&v101);
    v48 = v78 > v15;
    v40 = v75;
    if ( !v56 )
      v48 = 1;
    if ( v65 )
    {
      v57 = v65;
      v66 = v48;
      j_j___libc_free_0_0(v57);
      v48 = v66;
      v40 = v75;
    }
LABEL_93:
    if ( v94 > 0x40 && v93 )
    {
      v62 = v48;
      v69 = v40;
      j_j___libc_free_0_0(v93);
      v48 = v62;
      v40 = v69;
    }
    if ( v92 > 0x40 && v91 )
    {
      v63 = v48;
      v70 = v40;
      j_j___libc_free_0_0(v91);
      v48 = v63;
      v40 = v70;
    }
    if ( v40 > 0x40 && v43 )
    {
      v71 = v48;
      j_j___libc_free_0_0(v43);
      v48 = v71;
    }
    if ( v100 > 0x40 && v99 )
    {
      v72 = v48;
      j_j___libc_free_0_0(v99);
      v48 = v72;
    }
    if ( v98 > 0x40 && v97 )
    {
      v73 = v48;
      j_j___libc_free_0_0(v97);
      v48 = v73;
    }
    if ( !v48 )
    {
      v49 = v84;
      LODWORD(v102) = v84;
      if ( v84 <= 0x40 )
      {
        v50 = v83;
        goto LABEL_111;
      }
      sub_16A4FD0((__int64)&v101, (const void **)&v83);
      v49 = v102;
      if ( (unsigned int)v102 <= 0x40 )
      {
        v50 = v101;
LABEL_111:
        v51 = v85 | v50;
        v101 = v51;
      }
      else
      {
        sub_16A89F0((__int64 *)&v101, &v85);
        v49 = v102;
        v51 = v101;
      }
      v17 = *((_DWORD *)v8 + 2) <= 0x40u;
      LODWORD(v102) = 0;
      if ( v17 || !*v8 )
      {
        *v8 = v51;
        *((_DWORD *)v8 + 2) = v49;
      }
      else
      {
        v74 = v51;
        j_j___libc_free_0_0(*v8);
        v17 = (unsigned int)v102 <= 0x40;
        *((_DWORD *)v8 + 2) = v49;
        *v8 = v74;
        if ( !v17 && v101 )
          j_j___libc_free_0_0(v101);
      }
      v52 = v88;
      LODWORD(v102) = v88;
      if ( v88 <= 0x40 )
      {
        v53 = v87;
        goto LABEL_119;
      }
      sub_16A4FD0((__int64)&v101, (const void **)&v87);
      v52 = v102;
      if ( (unsigned int)v102 <= 0x40 )
      {
        v53 = v101;
LABEL_119:
        v54 = v89 & v53;
        v101 = v54;
      }
      else
      {
        sub_16A8890((__int64 *)&v101, (__int64 *)&v89);
        v52 = v102;
        v54 = v101;
      }
      v17 = *((_DWORD *)v7 + 2) <= 0x40u;
      LODWORD(v102) = 0;
      if ( v17 || !*v7 )
      {
        *v7 = v54;
        *((_DWORD *)v7 + 2) = v52;
      }
      else
      {
        v68 = v54;
        j_j___libc_free_0_0(*v7);
        v17 = (unsigned int)v102 <= 0x40;
        *((_DWORD *)v7 + 2) = v52;
        *v7 = v68;
        if ( !v17 && v101 )
          j_j___libc_free_0_0(v101);
      }
      if ( v90 > 0x40 && v89 )
        j_j___libc_free_0_0(v89);
      if ( v88 > 0x40 && v87 )
        j_j___libc_free_0_0(v87);
      if ( v86 > 0x40 && v85 )
        j_j___libc_free_0_0(v85);
      if ( v84 > 0x40 )
      {
        if ( v83 )
          j_j___libc_free_0_0(v83);
      }
      if ( v15 <= 8 )
        goto LABEL_153;
      continue;
    }
    break;
  }
  if ( v90 > 0x40 && v89 )
    j_j___libc_free_0_0(v89);
  if ( v88 > 0x40 && v87 )
    j_j___libc_free_0_0(v87);
  if ( v86 > 0x40 && v85 )
    j_j___libc_free_0_0(v85);
  if ( v84 > 0x40 && v83 )
    j_j___libc_free_0_0(v83);
  v15 = v61;
LABEL_153:
  *a4 = v15;
  return 1;
}
