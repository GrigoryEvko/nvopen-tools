// Function: sub_13F7530
// Address: 0x13f7530
//
__int64 __fastcall sub_13F7530(
        unsigned __int64 a1,
        unsigned int a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int64 *v10; // rax
  unsigned __int64 *v11; // rcx
  unsigned int v12; // edx
  unsigned __int8 v13; // dl
  __int64 v14; // rax
  unsigned int v15; // ecx
  unsigned __int64 v16; // rsi
  unsigned __int8 v17; // al
  __int64 v18; // rax
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rsi
  int v23; // ebx
  unsigned __int64 *v24; // r10
  unsigned __int64 *v25; // rsi
  unsigned int v26; // edi
  unsigned __int64 *v27; // rcx
  __int64 result; // rax
  __int64 *v29; // rax
  char v30; // al
  unsigned int v31; // edx
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rdx
  unsigned __int64 v36; // rsi
  unsigned __int64 v37; // rdi
  __int64 v38; // rdx
  _QWORD *v39; // rax
  __int64 v40; // rax
  unsigned int v41; // eax
  unsigned __int64 v42; // rsi
  unsigned int v43; // ebx
  __int64 v44; // rcx
  __int64 v45; // r8
  int v46; // eax
  __int64 v47; // rax
  __int64 v48; // r13
  __int64 v49; // rsi
  unsigned __int64 v50; // r15
  unsigned __int64 v51; // rax
  __int64 v52; // rax
  unsigned int v53; // ebx
  __int64 v54; // r13
  unsigned __int64 v55; // rax
  __int64 v56; // rdx
  unsigned int v57; // eax
  unsigned int v58; // eax
  unsigned __int64 v59; // r12
  unsigned int v60; // r13d
  int v61; // eax
  int v62; // ebx
  unsigned __int64 v63; // rcx
  __int64 v64; // rax
  __int64 v65; // [rsp+8h] [rbp-D8h]
  unsigned int v66; // [rsp+10h] [rbp-D0h]
  __int64 v67; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v68; // [rsp+10h] [rbp-D0h]
  __int64 v70; // [rsp+18h] [rbp-C8h]
  __int64 v71; // [rsp+20h] [rbp-C0h]
  __int64 v72; // [rsp+20h] [rbp-C0h]
  unsigned int v73; // [rsp+28h] [rbp-B8h]
  bool v74; // [rsp+28h] [rbp-B8h]
  unsigned __int8 v76; // [rsp+2Ch] [rbp-B4h]
  unsigned __int8 v77; // [rsp+2Ch] [rbp-B4h]
  unsigned __int8 v78; // [rsp+2Ch] [rbp-B4h]
  unsigned __int8 v79; // [rsp+2Ch] [rbp-B4h]
  unsigned __int8 v80; // [rsp+2Ch] [rbp-B4h]
  unsigned __int8 v81; // [rsp+2Ch] [rbp-B4h]
  unsigned __int8 v82; // [rsp+2Ch] [rbp-B4h]
  char v83; // [rsp+3Fh] [rbp-A1h] BYREF
  unsigned __int64 v84; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v85; // [rsp+48h] [rbp-98h]
  unsigned __int64 v86; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v87; // [rsp+58h] [rbp-88h]
  _QWORD *v88; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v89; // [rsp+68h] [rbp-78h]
  unsigned __int64 v90; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v91; // [rsp+78h] [rbp-68h]
  unsigned __int64 v92; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v93; // [rsp+88h] [rbp-58h]
  unsigned __int64 v94; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v95; // [rsp+98h] [rbp-48h]
  unsigned __int64 v96; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v97; // [rsp+A8h] [rbp-38h]

  v71 = a5;
  v10 = *(unsigned __int64 **)(a7 + 8);
  if ( v10 == *(unsigned __int64 **)(a7 + 16) )
    goto LABEL_29;
LABEL_2:
  sub_16CCBA0(a7, a1);
  v11 = *(unsigned __int64 **)(a7 + 16);
  v10 = *(unsigned __int64 **)(a7 + 8);
  a5 = v12;
  if ( !(_BYTE)v12 )
    return (unsigned int)a5;
  while ( 1 )
  {
    v13 = *(_BYTE *)(a1 + 16);
    if ( v13 <= 0x17u )
      break;
    if ( v13 != 71 )
      goto LABEL_5;
LABEL_26:
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v24 = *(unsigned __int64 **)(a1 - 8);
    else
      v24 = (unsigned __int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    a1 = *v24;
    if ( v10 != v11 )
      goto LABEL_2;
LABEL_29:
    v25 = &v10[*(unsigned int *)(a7 + 28)];
    v26 = *(_DWORD *)(a7 + 28);
    if ( v25 == v10 )
      goto LABEL_40;
    v27 = 0;
    do
    {
      if ( a1 == *v10 )
      {
        LODWORD(a5) = 0;
        return (unsigned int)a5;
      }
      if ( *v10 == -2 )
        v27 = v10;
      ++v10;
    }
    while ( v25 != v10 );
    if ( !v27 )
    {
LABEL_40:
      if ( v26 >= *(_DWORD *)(a7 + 24) )
        goto LABEL_2;
      *(_DWORD *)(a7 + 28) = v26 + 1;
      *v25 = a1;
      v10 = *(unsigned __int64 **)(a7 + 8);
      ++*(_QWORD *)a7;
      v11 = *(unsigned __int64 **)(a7 + 16);
    }
    else
    {
      *v27 = a1;
      v11 = *(unsigned __int64 **)(a7 + 16);
      --*(_DWORD *)(a7 + 32);
      v10 = *(unsigned __int64 **)(a7 + 8);
      ++*(_QWORD *)a7;
    }
  }
  if ( v13 == 5 && *(_WORD *)(a1 + 18) == 47 )
    goto LABEL_26;
LABEL_5:
  v83 = 0;
  v14 = sub_1649000(a1, a4, &v83, v11, a5);
  v15 = *(_DWORD *)(a3 + 8);
  v16 = v14;
  v85 = v15;
  if ( v15 <= 0x40 )
  {
    v16 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & v14;
    v84 = v16;
    goto LABEL_7;
  }
  sub_16A4EF0(&v84, v14, 0);
  if ( v85 <= 0x40 )
  {
    v16 = v84;
LABEL_7:
    if ( !v16 )
      goto LABEL_8;
LABEL_20:
    v16 = a3;
    if ( (int)sub_16A9900(&v84, a3) < 0 )
      goto LABEL_8;
    if ( v83 )
    {
      v16 = a4;
      if ( !(unsigned __int8)sub_14BFF20(a1, a4, 0, 0, v71, a6) )
        goto LABEL_8;
    }
    v22 = *(_QWORD *)a1;
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
          v47 = *(_QWORD *)(v22 + 32);
          v22 = *(_QWORD *)(v22 + 24);
          v23 *= (_DWORD)v47;
          continue;
        case 1:
          LODWORD(v40) = 16;
          break;
        case 2:
          LODWORD(v40) = 32;
          break;
        case 3:
        case 9:
          LODWORD(v40) = 64;
          break;
        case 4:
          LODWORD(v40) = 80;
          break;
        case 5:
        case 6:
          LODWORD(v40) = 128;
          break;
        case 7:
          LODWORD(v40) = 8 * sub_15A9520(a4, 0);
          break;
        case 0xB:
          LODWORD(v40) = *(_DWORD *)(v22 + 8) >> 8;
          break;
        case 0xD:
          v40 = 8LL * *(_QWORD *)sub_15A9930(a4, v22);
          break;
        case 0xE:
          v48 = 1;
          v72 = *(_QWORD *)(v22 + 32);
          v49 = *(_QWORD *)(v22 + 24);
          v50 = (unsigned int)sub_15A9FE0(a4, v49);
          while ( 2 )
          {
            switch ( *(_BYTE *)(v49 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v52 = *(_QWORD *)(v49 + 32);
                v49 = *(_QWORD *)(v49 + 24);
                v48 *= v52;
                continue;
              case 1:
                v51 = 16;
                goto LABEL_123;
              case 2:
                v51 = 32;
                goto LABEL_123;
              case 3:
              case 9:
                v51 = 64;
                goto LABEL_123;
              case 4:
                v51 = 80;
                goto LABEL_123;
              case 5:
              case 6:
                v51 = 128;
                goto LABEL_123;
              case 7:
                v51 = 8 * (unsigned int)sub_15A9520(a4, 0);
                goto LABEL_123;
              case 0xB:
                v51 = *(_DWORD *)(v49 + 8) >> 8;
                goto LABEL_123;
              case 0xD:
                JUMPOUT(0x13F7DE1);
              case 0xE:
                v65 = *(_QWORD *)(v49 + 24);
                v70 = *(_QWORD *)(v49 + 32);
                v68 = (unsigned int)sub_15A9FE0(a4, v65);
                v51 = 8 * v70 * v68 * ((v68 + ((unsigned __int64)(sub_127FA20(a4, v65) + 7) >> 3) - 1) / v68);
LABEL_123:
                v40 = 8 * v72 * v50 * ((v50 + ((v48 * v51 + 7) >> 3) - 1) / v50);
                break;
              case 0xF:
                JUMPOUT(0x13F7DEF);
            }
            return result;
          }
        case 0xF:
          LODWORD(v40) = 8 * sub_15A9520(a4, *(_DWORD *)(v22 + 8) >> 8);
          break;
      }
      break;
    }
    v87 = (v40 * v23 + 7) & 0xFFFFFFF8;
    if ( v87 > 0x40 )
      sub_16A4EF0(&v86, 0, 0);
    else
      v86 = 0;
    v41 = sub_1649510(a1, a4);
    v89 = v87;
    if ( v87 > 0x40 )
    {
      sub_16A4EF0(&v88, v41, 0);
      v53 = v89;
      if ( v89 > 0x40 )
      {
        if ( v53 != (unsigned int)sub_16A57B0(&v88) )
          goto LABEL_84;
        goto LABEL_135;
      }
      v42 = (unsigned __int64)v88;
    }
    else
    {
      v42 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v87) & v41;
      v88 = (_QWORD *)v42;
    }
    if ( v42 )
      goto LABEL_84;
LABEL_135:
    v54 = **(_QWORD **)(*(_QWORD *)a1 + 16LL);
    v55 = *(unsigned __int8 *)(v54 + 8);
    if ( (unsigned __int8)v55 > 0xFu || (v56 = 35454, !_bittest64(&v56, v55)) )
    {
      if ( (unsigned int)(v55 - 13) > 1 && (_DWORD)v55 != 16
        || !(unsigned __int8)sub_16435F0(**(_QWORD **)(*(_QWORD *)a1 + 16LL), 0) )
      {
        LODWORD(a5) = 0;
        goto LABEL_90;
      }
    }
    v57 = sub_15A9FE0(a4, v54);
    if ( v89 > 0x40 )
    {
      *v88 = v57;
      memset(v88 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v89 + 63) >> 6) - 8);
    }
    else
    {
      v88 = (_QWORD *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v89) & v57);
    }
LABEL_84:
    v43 = v87;
    v91 = v87;
    if ( v87 > 0x40 )
    {
      sub_16A4EF0(&v90, a2, 0);
      v43 = v91;
      if ( (int)sub_16A9900(&v88, &v90) < 0 )
        goto LABEL_86;
      v93 = v43;
      if ( v43 > 0x40 )
      {
        sub_16A4FD0(&v92, &v90);
        goto LABEL_147;
      }
    }
    else
    {
      v90 = a2 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v87);
      if ( (int)sub_16A9900(&v88, &v90) < 0 )
      {
LABEL_86:
        LODWORD(a5) = 0;
LABEL_87:
        if ( v43 > 0x40 && v90 )
        {
          v78 = a5;
          j_j___libc_free_0_0(v90);
          LODWORD(a5) = v78;
        }
LABEL_90:
        if ( v89 > 0x40 && v88 )
        {
          v79 = a5;
          j_j___libc_free_0_0(v88);
          LODWORD(a5) = v79;
        }
        if ( v87 <= 0x40 )
          goto LABEL_58;
        v32 = v86;
        if ( !v86 )
          goto LABEL_58;
LABEL_54:
        v76 = a5;
        j_j___libc_free_0_0(v32);
        LODWORD(a5) = v76;
        goto LABEL_58;
      }
      v93 = v43;
    }
    v92 = v90;
LABEL_147:
    sub_16A7800(&v92, 1);
    v58 = v93;
    v93 = 0;
    v95 = v58;
    v94 = v92;
    if ( v58 > 0x40 )
    {
      sub_16A8890(&v94, &v86);
      v60 = v95;
      v59 = v94;
      v95 = 0;
      v97 = v60;
      v96 = v94;
      if ( v60 > 0x40 )
      {
        v61 = sub_16A57B0(&v96);
        v62 = v61;
        if ( v59 )
        {
          j_j___libc_free_0_0(v59);
          LOBYTE(a5) = v60 == v62;
          if ( v95 > 0x40 && v94 )
          {
            j_j___libc_free_0_0(v94);
            LODWORD(a5) = v60 == v62;
          }
        }
        else
        {
          LOBYTE(a5) = v60 == v61;
        }
        goto LABEL_150;
      }
    }
    else
    {
      v59 = v86 & v92;
    }
    LOBYTE(a5) = v59 == 0;
LABEL_150:
    if ( v93 > 0x40 && v92 )
    {
      v82 = a5;
      j_j___libc_free_0_0(v92);
      LODWORD(a5) = v82;
    }
    v43 = v91;
    goto LABEL_87;
  }
  v66 = v85;
  if ( v66 != (unsigned int)sub_16A57B0(&v84) )
    goto LABEL_20;
LABEL_8:
  v17 = *(_BYTE *)(a1 + 16);
  if ( v17 <= 0x17u )
  {
    if ( v17 != 5 || *(_WORD *)(a1 + 18) != 32 )
      goto LABEL_57;
LABEL_44:
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v29 = *(__int64 **)(a1 - 8);
    else
      v29 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v67 = *v29;
    v93 = sub_15A95F0(a4, *(_QWORD *)a1);
    if ( v93 > 0x40 )
      sub_16A4EF0(&v92, 0, 0);
    else
      v92 = 0;
    v30 = sub_1634900(a1, a4, &v92);
    v31 = v93;
    if ( !v30 )
      goto LABEL_51;
    if ( v93 > 0x40 )
    {
      if ( (*(_QWORD *)(v92 + 8LL * ((v93 - 1) >> 6)) & (1LL << ((unsigned __int8)v93 - 1))) != 0 )
        goto LABEL_51;
      v95 = v93;
      sub_16A4EF0(&v94, a2, 0);
    }
    else
    {
      if ( ((1LL << ((unsigned __int8)v93 - 1)) & v92) != 0 )
        goto LABEL_51;
      v95 = v93;
      v94 = a2 & (unsigned int)(0xFFFFFFFFFFFFFFFFLL >> (63 - ((v93 - 1) & 0x3F)));
    }
    sub_16AB0A0(&v96, &v92, &v94);
    if ( v97 <= 0x40 )
    {
      v74 = v96 != 0;
    }
    else
    {
      v73 = v97;
      v74 = v73 != (unsigned int)sub_16A57B0(&v96);
      if ( v96 )
        j_j___libc_free_0_0(v96);
    }
    if ( v95 > 0x40 && v94 )
      j_j___libc_free_0_0(v94);
    v31 = v93;
    if ( !v74 )
    {
      sub_16A5D70(&v94, a3, v93, v44, v45);
      sub_16A7200(&v94, &v92);
      v97 = v95;
      v95 = 0;
      v96 = v94;
      v46 = sub_13F7530(v67, a2, (unsigned int)&v96, a4, v71, a6, a7);
      LODWORD(a5) = v46;
      if ( v97 > 0x40 && v96 )
      {
        v80 = v46;
        j_j___libc_free_0_0(v96);
        LODWORD(a5) = v80;
      }
      if ( v95 > 0x40 && v94 )
      {
        v81 = a5;
        j_j___libc_free_0_0(v94);
        LODWORD(a5) = v81;
      }
      v31 = v93;
LABEL_52:
      if ( v31 <= 0x40 )
        goto LABEL_58;
      v32 = v92;
      if ( !v92 )
        goto LABEL_58;
      goto LABEL_54;
    }
LABEL_51:
    LODWORD(a5) = 0;
    goto LABEL_52;
  }
  if ( v17 == 56 )
    goto LABEL_44;
  if ( v17 != 78 )
  {
    if ( v17 == 72 )
    {
      v21 = *(_QWORD *)(a1 - 24);
      goto LABEL_17;
    }
    if ( v17 == 29 )
    {
      v19 = a1 & 0xFFFFFFFFFFFFFFFBLL;
      goto LABEL_15;
    }
    goto LABEL_57;
  }
  v18 = *(_QWORD *)(a1 - 24);
  if ( !*(_BYTE *)(v18 + 16) && (*(_BYTE *)(v18 + 33) & 0x20) != 0 && *(_DWORD *)(v18 + 36) == 76 )
  {
    v33 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    v34 = *(_QWORD *)(a1 - 24 * v33);
    v35 = *(unsigned __int8 *)(v34 + 16);
    if ( (_BYTE)v35 == 88 )
    {
      v64 = sub_157F120(*(_QWORD *)(v34 + 40), v16, v35);
      v34 = sub_157EBA0(v64);
      LOBYTE(v35) = *(_BYTE *)(v34 + 16);
      v33 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    }
    if ( (unsigned __int8)v35 > 0x17u )
    {
      if ( (_BYTE)v35 == 78 )
      {
        v63 = v34 | 4;
      }
      else
      {
        v36 = 0;
        if ( (_BYTE)v35 != 29 )
        {
LABEL_69:
          v37 = v36 - 24LL * (*(_DWORD *)(v36 + 20) & 0xFFFFFFF);
LABEL_70:
          v38 = *(_QWORD *)(a1 + 24 * (2 - v33));
          v39 = *(_QWORD **)(v38 + 24);
          if ( *(_DWORD *)(v38 + 32) > 0x40u )
            v39 = (_QWORD *)*v39;
          LODWORD(a5) = sub_13F7530(*(_QWORD *)(v37 + 24LL * (unsigned int)v39), a2, a3, a4, v71, a6, a7);
          goto LABEL_58;
        }
        v63 = v34 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v36 = v63 & 0xFFFFFFFFFFFFFFF8LL;
      v37 = (v63 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v63 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
      if ( (v63 & 4) != 0 )
        goto LABEL_70;
      goto LABEL_69;
    }
    v36 = 0;
    goto LABEL_69;
  }
  v19 = a1 | 4;
LABEL_15:
  if ( (v19 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v20 = sub_14AD130();
    LODWORD(v21) = v20;
    if ( v20 )
    {
LABEL_17:
      LODWORD(a5) = sub_13F7530(v21, a2, a3, a4, v71, a6, a7);
      goto LABEL_58;
    }
  }
LABEL_57:
  LODWORD(a5) = 0;
LABEL_58:
  if ( v85 > 0x40 && v84 )
  {
    v77 = a5;
    j_j___libc_free_0_0(v84);
    LODWORD(a5) = v77;
  }
  return (unsigned int)a5;
}
