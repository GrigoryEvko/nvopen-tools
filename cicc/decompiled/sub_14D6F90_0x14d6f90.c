// Function: sub_14D6F90
// Address: 0x14d6f90
//
__int64 __fastcall sub_14D6F90(unsigned int a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // r8
  bool v12; // al
  __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  unsigned int v15; // ebx
  unsigned __int64 v16; // r8
  unsigned __int64 v17; // r8
  bool v18; // al
  unsigned int v19; // r15d
  unsigned int v20; // ebx
  int v21; // eax
  __int64 v22; // rsi
  int v23; // ecx
  __int64 v24; // rax
  unsigned int v25; // eax
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  int v33; // [rsp+10h] [rbp-D0h]
  __int64 v34; // [rsp+18h] [rbp-C8h]
  unsigned int v35; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v36; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v37; // [rsp+20h] [rbp-C0h]
  bool v38; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v39; // [rsp+20h] [rbp-C0h]
  bool v40; // [rsp+20h] [rbp-C0h]
  int v41; // [rsp+28h] [rbp-B8h]
  int v42; // [rsp+28h] [rbp-B8h]
  __int64 v43; // [rsp+28h] [rbp-B8h]
  int v44; // [rsp+28h] [rbp-B8h]
  __int64 v45; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+38h] [rbp-A8h] BYREF
  unsigned __int64 v47; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v48; // [rsp+48h] [rbp-98h]
  unsigned __int64 v49; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v50; // [rsp+58h] [rbp-88h]
  unsigned __int64 v51; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v52; // [rsp+68h] [rbp-78h]
  unsigned __int64 v53; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v54; // [rsp+78h] [rbp-68h]
  unsigned __int64 v55; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v56; // [rsp+88h] [rbp-58h]
  unsigned __int64 v57; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v58; // [rsp+98h] [rbp-48h]
  unsigned __int64 v59; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v60; // [rsp+A8h] [rbp-38h]

  v7 = a4;
  if ( *((_BYTE *)a2 + 16) != 5 && *(_BYTE *)(a3 + 16) != 5 )
    return sub_15A2A30(a1, a2, a3, 0, 0);
  if ( a1 != 26 )
  {
    if ( a1 != 13 )
      return sub_15A2A30(a1, a2, a3, 0, 0);
    v48 = 1;
    v47 = 0;
    v50 = 1;
    v49 = 0;
    if ( (unsigned __int8)sub_14D5D40((__int64)a2, &v45, (__int64)&v47, a4)
      && (unsigned __int8)sub_14D5D40(a3, &v46, (__int64)&v49, v7)
      && v45 == v46 )
    {
      v22 = *a2;
      v23 = 1;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v22 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v30 = *(_QWORD *)(v22 + 32);
            v22 = *(_QWORD *)(v22 + 24);
            v23 *= (_DWORD)v30;
            continue;
          case 1:
            LODWORD(v24) = 16;
            goto LABEL_52;
          case 2:
            LODWORD(v24) = 32;
            goto LABEL_52;
          case 3:
          case 9:
            LODWORD(v24) = 64;
            goto LABEL_52;
          case 4:
            LODWORD(v24) = 80;
            goto LABEL_52;
          case 5:
          case 6:
            LODWORD(v24) = 128;
            goto LABEL_52;
          case 7:
            v27 = 0;
            v42 = v23;
            goto LABEL_73;
          case 0xB:
            LODWORD(v24) = *(_DWORD *)(v22 + 8) >> 8;
            goto LABEL_52;
          case 0xD:
            v44 = v23;
            v29 = (_QWORD *)sub_15A9930(v7, v22);
            v23 = v44;
            v24 = 8LL * *v29;
            goto LABEL_52;
          case 0xE:
            v33 = v23;
            v34 = *(_QWORD *)(v22 + 24);
            v43 = *(_QWORD *)(v22 + 32);
            v36 = (unsigned int)sub_15A9FE0(v7, v34);
            v28 = sub_127FA20(v7, v34);
            v23 = v33;
            v24 = 8 * v43 * v36 * ((v36 + ((unsigned __int64)(v28 + 7) >> 3) - 1) / v36);
            goto LABEL_52;
          case 0xF:
            v42 = v23;
            v27 = *(_DWORD *)(v22 + 8) >> 8;
LABEL_73:
            LODWORD(v24) = sub_15A9520(v7, v27);
            v23 = v42;
            LODWORD(v24) = 8 * v24;
LABEL_52:
            v35 = v23 * v24;
            sub_16A5D10(&v53, &v49, (unsigned int)(v23 * v24));
            sub_16A5D10(&v51, &v47, v35);
            if ( v54 > 0x40 )
              sub_16A8F40(&v53);
            else
              v53 = ~v53 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v54);
            sub_16A7400(&v53);
            sub_16A7200(&v53, &v51);
            v25 = v54;
            v26 = *a2;
            v54 = 0;
            v58 = v25;
            v57 = v53;
            v13 = sub_15A1070(v26, &v57);
            if ( v58 > 0x40 && v57 )
              j_j___libc_free_0_0(v57);
            if ( v52 > 0x40 && v51 )
              j_j___libc_free_0_0(v51);
            if ( v54 > 0x40 && v53 )
              j_j___libc_free_0_0(v53);
            if ( v50 > 0x40 && v49 )
              j_j___libc_free_0_0(v49);
            if ( v48 <= 0x40 )
              goto LABEL_32;
            v14 = v47;
            if ( !v47 )
              goto LABEL_32;
            goto LABEL_31;
        }
      }
    }
    if ( v50 > 0x40 && v49 )
      j_j___libc_free_0_0(v49);
    if ( v48 <= 0x40 )
      return sub_15A2A30(a1, a2, a3, 0, 0);
    v9 = v47;
    if ( !v47 )
      return sub_15A2A30(a1, a2, a3, 0, 0);
    goto LABEL_14;
  }
  sub_14C2530((__int64)&v53, a2, a4, 0, 0, 0, 0, 0);
  sub_14C2530((__int64)&v57, (__int64 *)a3, v7, 0, 0, 0, 0, 0);
  LOBYTE(v7) = v60;
  v50 = v60;
  if ( v60 <= 0x40 )
  {
    v10 = v59;
LABEL_17:
    v11 = v53 | v10;
LABEL_18:
    v12 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) == v11;
    goto LABEL_19;
  }
  sub_16A4FD0(&v49, &v59);
  LOBYTE(v7) = v50;
  if ( v50 <= 0x40 )
  {
    v10 = v49;
    goto LABEL_17;
  }
  sub_16A89F0(&v49, &v53);
  LODWORD(v7) = v50;
  v11 = v49;
  v50 = 0;
  v52 = v7;
  v51 = v49;
  if ( (unsigned int)v7 <= 0x40 )
    goto LABEL_18;
  v37 = v49;
  v12 = (_DWORD)v7 == (unsigned int)sub_16A58F0(&v51);
  if ( v37 )
  {
    v31 = v37;
    v38 = v12;
    j_j___libc_free_0_0(v31);
    v12 = v38;
    if ( v50 > 0x40 )
    {
      if ( v49 )
      {
        j_j___libc_free_0_0(v49);
        v12 = v38;
      }
    }
  }
LABEL_19:
  v13 = (__int64)a2;
  if ( v12 )
    goto LABEL_20;
  LOBYTE(v15) = v56;
  v50 = v56;
  if ( v56 <= 0x40 )
  {
    v16 = v55;
LABEL_36:
    v17 = v57 | v16;
LABEL_37:
    v18 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15) == v17;
    goto LABEL_38;
  }
  sub_16A4FD0(&v49, &v55);
  LOBYTE(v15) = v50;
  if ( v50 <= 0x40 )
  {
    v16 = v49;
    goto LABEL_36;
  }
  sub_16A89F0(&v49, &v57);
  v15 = v50;
  v17 = v49;
  v50 = 0;
  v52 = v15;
  v51 = v49;
  if ( v15 <= 0x40 )
    goto LABEL_37;
  v39 = v49;
  v18 = v15 == (unsigned int)sub_16A58F0(&v51);
  if ( v39 )
  {
    v32 = v39;
    v40 = v18;
    j_j___libc_free_0_0(v32);
    v18 = v40;
    if ( v50 > 0x40 )
    {
      if ( v49 )
      {
        j_j___libc_free_0_0(v49);
        v18 = v40;
      }
    }
  }
LABEL_38:
  v13 = a3;
  if ( !v18 )
  {
    if ( v54 > 0x40 )
      sub_16A89F0(&v53, &v57);
    else
      v53 |= v57;
    v19 = v56;
    if ( v56 > 0x40 )
    {
      sub_16A8890(&v55, &v59);
      v19 = v56;
    }
    else
    {
      v55 &= v59;
    }
    v20 = v54;
    if ( v54 > 0x40 )
      v41 = sub_16A5940(&v53);
    else
      v41 = sub_39FAC40(v53);
    if ( v19 > 0x40 )
      v21 = sub_16A5940(&v55);
    else
      v21 = sub_39FAC40(v55);
    if ( v41 + v21 != v20 )
    {
      if ( v60 > 0x40 && v59 )
        j_j___libc_free_0_0(v59);
      if ( v58 > 0x40 && v57 )
        j_j___libc_free_0_0(v57);
      if ( v56 > 0x40 && v55 )
        j_j___libc_free_0_0(v55);
      if ( v54 <= 0x40 )
        return sub_15A2A30(a1, a2, a3, 0, 0);
      v9 = v53;
      if ( !v53 )
        return sub_15A2A30(a1, a2, a3, 0, 0);
LABEL_14:
      j_j___libc_free_0_0(v9);
      return sub_15A2A30(a1, a2, a3, 0, 0);
    }
    v13 = sub_15A1070(*a2, &v55);
  }
LABEL_20:
  if ( v60 > 0x40 && v59 )
    j_j___libc_free_0_0(v59);
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  if ( v56 > 0x40 && v55 )
    j_j___libc_free_0_0(v55);
  if ( v54 > 0x40 )
  {
    v14 = v53;
    if ( v53 )
LABEL_31:
      j_j___libc_free_0_0(v14);
  }
LABEL_32:
  if ( !v13 )
    return sub_15A2A30(a1, a2, a3, 0, 0);
  return v13;
}
