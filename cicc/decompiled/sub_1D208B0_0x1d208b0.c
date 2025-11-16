// Function: sub_1D208B0
// Address: 0x1d208b0
//
__int64 __fastcall sub_1D208B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v3; // r14
  __int64 v4; // rax
  unsigned __int64 *v5; // rbx
  char v6; // di
  __int64 v7; // rdx
  int v8; // r8d
  int v9; // r9d
  unsigned int v10; // r13d
  int v11; // ecx
  __int16 v12; // ax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  int v17; // eax
  unsigned __int64 v18; // rdi
  unsigned int v20; // ebx
  int v21; // eax
  unsigned int v22; // r12d
  int v23; // eax
  int v24; // eax
  _QWORD *v25; // r15
  signed __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rdx
  int v29; // eax
  __int64 v30; // rdx
  int v31; // eax
  __int64 v32; // rdx
  int v33; // eax
  int v34; // eax
  int v35; // eax
  int v36; // eax
  int v37; // eax
  __int64 v38; // rax
  __int64 v39; // r15
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned int v43; // edx
  __int64 v44; // rdi
  int v45; // edx
  int v46; // eax
  int v47; // r13d
  int v48; // ebx
  _QWORD *v49; // [rsp+0h] [rbp-90h]
  int v50; // [rsp+Ch] [rbp-84h]
  int v51; // [rsp+Ch] [rbp-84h]
  int v52; // [rsp+Ch] [rbp-84h]
  _QWORD *v53; // [rsp+10h] [rbp-80h]
  _QWORD *v56; // [rsp+28h] [rbp-68h]
  int v57; // [rsp+28h] [rbp-68h]
  char v58[8]; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 *v59; // [rsp+38h] [rbp-58h]
  unsigned __int64 v60; // [rsp+40h] [rbp-50h] BYREF
  __int64 v61; // [rsp+48h] [rbp-48h]
  __int64 v62; // [rsp+50h] [rbp-40h] BYREF
  __int64 v63; // [rsp+58h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v5 = *(unsigned __int64 **)(v4 + 8);
  v6 = *(_BYTE *)v4;
  v58[0] = v6;
  v59 = v5;
  if ( v6 )
  {
    if ( (unsigned __int8)(v6 - 14) <= 0x5Fu )
    {
      switch ( v6 )
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
          v6 = 3;
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
          v6 = 4;
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
          v6 = 5;
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
          v6 = 6;
          break;
        case 55:
          v6 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v6 = 8;
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
          v6 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v6 = 10;
          break;
        default:
          v6 = 2;
          break;
      }
    }
    goto LABEL_3;
  }
  if ( (unsigned __int8)sub_1F58D20(v58) )
  {
    LOBYTE(v60) = sub_1F596B0(v58);
    v6 = v60;
    v61 = v13;
    if ( (_BYTE)v60 )
    {
LABEL_3:
      v10 = sub_1D13440(v6);
      goto LABEL_4;
    }
  }
  else
  {
    LOBYTE(v60) = 0;
    v61 = (__int64)v5;
  }
  v10 = sub_1F58D40(&v60, a2, v13, v14, v15, v16);
LABEL_4:
  v11 = *(unsigned __int16 *)(a2 + 24);
  v12 = *(_WORD *)(a2 + 24);
  LOBYTE(v7) = (_WORD)v11 == 32;
  LOBYTE(v3) = (_WORD)v11 == 32 || (_WORD)v11 == 10;
  if ( (_BYTE)v3 )
  {
    v3 = &v60;
    sub_16A5D10((__int64)&v60, *(_QWORD *)(a2 + 88) + 24LL, v10);
    if ( (unsigned int)v61 <= 0x40 )
    {
      LODWORD(v3) = 0;
      if ( v60 )
        LOBYTE(v3) = (v60 & (v60 - 1)) == 0;
      return (unsigned int)v3;
    }
    v17 = sub_16A5940((__int64)&v60);
    v18 = v60;
    LOBYTE(v3) = v17 == 1;
    if ( !v60 )
      return (unsigned int)v3;
LABEL_12:
    j_j___libc_free_0_0(v18);
    return (unsigned int)v3;
  }
  if ( v11 == 122 )
  {
    v38 = sub_1D1ADA0(**(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), v7, 122, v8, v9);
    if ( v38 )
    {
      v39 = *(_QWORD *)(v38 + 88);
      LODWORD(v5) = *(_DWORD *)(v39 + 32);
      if ( (unsigned int)v5 <= 0x40 )
      {
        v40 = *(_QWORD *)(v39 + 24);
LABEL_87:
        if ( v40 == 1 )
          goto LABEL_37;
        goto LABEL_88;
      }
      LODWORD(v5) = (_DWORD)v5 - sub_16A57B0(v39 + 24);
      if ( (unsigned int)v5 <= 0x40 )
      {
        v40 = **(_QWORD **)(v39 + 24);
        goto LABEL_87;
      }
    }
LABEL_88:
    v12 = *(_WORD *)(a2 + 24);
  }
  if ( v12 == 124 )
  {
    v41 = sub_1D1ADA0(**(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), v7, v11, v8, v9);
    if ( v41 )
    {
      v42 = *(_QWORD *)(v41 + 88);
      v43 = *(_DWORD *)(v42 + 32);
      v44 = *(_QWORD *)(v42 + 24);
      LODWORD(v5) = v43 - 1;
      if ( v43 <= 0x40 )
      {
        if ( v44 == 1LL << (char)v5 )
          goto LABEL_37;
      }
      else if ( (*(_QWORD *)(v44 + 8LL * ((unsigned int)v5 >> 6)) & (1LL << (char)v5)) != 0
             && (unsigned int)sub_16A58A0(v42 + 24) == (_DWORD)v5 )
      {
        goto LABEL_37;
      }
    }
    v12 = *(_WORD *)(a2 + 24);
  }
  if ( v12 != 104 )
    goto LABEL_17;
  v25 = *(_QWORD **)(a2 + 32);
  v26 = *(unsigned int *)(a2 + 56);
  v49 = &v25[5 * v26];
  if ( !(v26 >> 2) )
  {
LABEL_59:
    if ( v26 != 2 )
    {
      if ( v26 != 3 )
      {
        if ( v26 != 1 )
          goto LABEL_37;
        goto LABEL_62;
      }
      v45 = *(unsigned __int16 *)(*v25 + 24LL);
      if ( v45 != 32 && v45 != 10 )
        goto LABEL_36;
      sub_16A5D10((__int64)&v60, *(_QWORD *)(*v25 + 88LL) + 24LL, v10);
      if ( (unsigned int)v61 > 0x40 )
      {
        v48 = sub_16A5940((__int64)&v60);
        if ( v60 )
          j_j___libc_free_0_0(v60);
        if ( v48 != 1 )
          goto LABEL_36;
      }
      else if ( !v60 || (v60 & (v60 - 1)) != 0 )
      {
        goto LABEL_36;
      }
      v25 += 5;
    }
    v46 = *(unsigned __int16 *)(*v25 + 24LL);
    if ( v46 != 32 && v46 != 10 )
      goto LABEL_36;
    v5 = &v60;
    sub_16A5D10((__int64)&v60, *(_QWORD *)(*v25 + 88LL) + 24LL, v10);
    if ( (unsigned int)v61 > 0x40 )
    {
      LODWORD(v5) = sub_16A5940((__int64)&v60);
      if ( v60 )
        j_j___libc_free_0_0(v60);
      if ( (_DWORD)v5 != 1 )
        goto LABEL_36;
    }
    else if ( !v60 || (v60 & (v60 - 1)) != 0 )
    {
      goto LABEL_36;
    }
    v25 += 5;
LABEL_62:
    LOBYTE(v5) = *(_WORD *)(*v25 + 24LL) == 32 || *(_WORD *)(*v25 + 24LL) == 10;
    if ( !(_BYTE)v5 )
      goto LABEL_36;
    sub_16A5D10((__int64)&v60, *(_QWORD *)(*v25 + 88LL) + 24LL, v10);
    if ( (unsigned int)v61 > 0x40 )
    {
      v47 = sub_16A5940((__int64)&v60);
      if ( v60 )
        j_j___libc_free_0_0(v60);
      if ( v47 != 1 )
        goto LABEL_36;
    }
    else if ( !v60 || (v60 & (v60 - 1)) != 0 )
    {
      goto LABEL_36;
    }
    LODWORD(v3) = (_DWORD)v5;
    return (unsigned int)v3;
  }
  v5 = &v60;
  v53 = &v25[20 * (v26 >> 2)];
  while ( 1 )
  {
    v27 = *(unsigned __int16 *)(*v25 + 24LL);
    if ( v27 != 10 && v27 != 32 )
      break;
    sub_16A5D10((__int64)&v60, *(_QWORD *)(*v25 + 88LL) + 24LL, v10);
    if ( (unsigned int)v61 > 0x40 )
    {
      v34 = sub_16A5940((__int64)&v60);
      if ( v60 )
      {
        v57 = v34;
        j_j___libc_free_0_0(v60);
        v34 = v57;
      }
      if ( v34 != 1 )
        break;
    }
    else if ( !v60 || (v60 & (v60 - 1)) != 0 )
    {
      break;
    }
    v28 = v25[5];
    v56 = v25 + 5;
    v29 = *(unsigned __int16 *)(v28 + 24);
    if ( v29 != 10 && v29 != 32 )
    {
LABEL_43:
      v25 = v56;
      break;
    }
    sub_16A5D10((__int64)&v60, *(_QWORD *)(v28 + 88) + 24LL, v10);
    if ( (unsigned int)v61 > 0x40 )
    {
      v35 = sub_16A5940((__int64)&v60);
      if ( v60 )
      {
        v50 = v35;
        j_j___libc_free_0_0(v60);
        v35 = v50;
      }
      if ( v35 != 1 )
        goto LABEL_43;
    }
    else if ( !v60 || (v60 & (v60 - 1)) != 0 )
    {
      goto LABEL_43;
    }
    v30 = v25[10];
    v56 = v25 + 10;
    v31 = *(unsigned __int16 *)(v30 + 24);
    if ( v31 != 32 && v31 != 10 )
      goto LABEL_43;
    sub_16A5D10((__int64)&v60, *(_QWORD *)(v30 + 88) + 24LL, v10);
    if ( (unsigned int)v61 > 0x40 )
    {
      v36 = sub_16A5940((__int64)&v60);
      if ( v60 )
      {
        v51 = v36;
        j_j___libc_free_0_0(v60);
        v36 = v51;
      }
      if ( v36 != 1 )
        goto LABEL_43;
    }
    else if ( !v60 || (v60 & (v60 - 1)) != 0 )
    {
      goto LABEL_43;
    }
    v32 = v25[15];
    v56 = v25 + 15;
    v33 = *(unsigned __int16 *)(v32 + 24);
    if ( v33 != 32 && v33 != 10 )
      goto LABEL_43;
    sub_16A5D10((__int64)&v60, *(_QWORD *)(v32 + 88) + 24LL, v10);
    if ( (unsigned int)v61 > 0x40 )
    {
      v37 = sub_16A5940((__int64)&v60);
      if ( v60 )
      {
        v52 = v37;
        j_j___libc_free_0_0(v60);
        v37 = v52;
      }
      if ( v37 != 1 )
        goto LABEL_43;
    }
    else if ( !v60 || (v60 & (v60 - 1)) != 0 )
    {
      goto LABEL_43;
    }
    v25 += 20;
    if ( v53 == v25 )
    {
      v26 = 0xCCCCCCCCCCCCCCCDLL * (v49 - v25);
      goto LABEL_59;
    }
  }
LABEL_36:
  if ( v49 == v25 )
  {
LABEL_37:
    LODWORD(v3) = 1;
    return (unsigned int)v3;
  }
LABEL_17:
  v60 = 0;
  v61 = 1;
  v62 = 0;
  v63 = 1;
  sub_1D1F820(a1, a2, a3, &v60, 0);
  v20 = v61;
  if ( (unsigned int)v61 <= 0x40 )
  {
    v23 = sub_39FAC40(v60);
    v22 = v63;
    if ( v20 - v23 != 1 )
      goto LABEL_19;
  }
  else
  {
    v21 = sub_16A5940((__int64)&v60);
    v22 = v63;
    if ( v20 - v21 != 1 )
      goto LABEL_19;
  }
  if ( v22 > 0x40 )
    v24 = sub_16A5940((__int64)&v62);
  else
    v24 = sub_39FAC40(v62);
  LOBYTE(v3) = v24 == 1;
LABEL_19:
  if ( v22 > 0x40 && v62 )
  {
    j_j___libc_free_0_0(v62);
    v20 = v61;
  }
  if ( v20 > 0x40 )
  {
    v18 = v60;
    if ( v60 )
      goto LABEL_12;
  }
  return (unsigned int)v3;
}
