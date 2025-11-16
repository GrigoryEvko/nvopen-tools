// Function: sub_2171180
// Address: 0x2171180
//
__int64 __fastcall sub_2171180(__int64 a1, unsigned int a2, unsigned __int64 *a3)
{
  __int64 v3; // r14
  int v6; // eax
  __int64 v7; // r15
  char v8; // r8
  __int64 v9; // rdx
  _QWORD *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // r14d
  unsigned __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // r14
  unsigned int v20; // eax
  char v21; // al
  __int64 v22; // rdx
  int v23; // eax
  __int64 *v24; // rdx
  __int64 v25; // r12
  int v26; // eax
  unsigned __int64 v27; // rcx
  __int64 v28; // rax
  char v29; // di
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned int v34; // r13d
  __int64 v35; // r14
  char v36; // di
  __int64 v37; // rax
  unsigned int v38; // eax
  bool v39; // zf
  unsigned __int64 v40; // r14
  _QWORD *v41; // rax
  __int64 v42; // rax
  char v43; // di
  __int64 v44; // rax
  unsigned int v45; // eax
  unsigned __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdi
  unsigned int v49; // r13d
  __int64 v51; // rax
  __int64 *v52; // rax
  __int64 v53; // r12
  unsigned int v54; // r13d
  char v55; // r8
  unsigned int v56; // r14d
  __int64 v57; // rax
  char v58; // r13
  __int64 v59; // r15
  unsigned int v60; // eax
  char v61; // r8
  int v62; // r14d
  int v63; // eax
  char v64; // r8
  unsigned int v65; // r14d
  char v66; // r15
  unsigned int v67; // eax
  char v68; // r8
  int v69; // r14d
  int v70; // eax
  char v71; // al
  __int64 v72; // rdx
  char v73; // r9
  char v74; // r9
  unsigned int v75; // r15d
  unsigned int v76; // eax
  unsigned int v77; // eax
  unsigned int v78; // eax
  unsigned int v79; // eax
  __int64 v80; // rax
  unsigned __int64 v81; // r14
  char v82; // [rsp+7h] [rbp-79h]
  char v83; // [rsp+7h] [rbp-79h]
  char v84; // [rsp+8h] [rbp-78h]
  __int64 v85; // [rsp+8h] [rbp-78h]
  char v86; // [rsp+8h] [rbp-78h]
  __int64 v87; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v88; // [rsp+18h] [rbp-68h] BYREF
  char v89[8]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v90; // [rsp+28h] [rbp-58h]
  _QWORD v91[2]; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v92; // [rsp+40h] [rbp-40h] BYREF
  __int64 v93; // [rsp+48h] [rbp-38h]

  v3 = 16LL * a2;
  v6 = *(__int16 *)(a1 + 24);
  v7 = v3 + *(_QWORD *)(a1 + 40);
  v8 = *(_BYTE *)v7;
  v9 = *(_QWORD *)(v7 + 8);
  v89[0] = *(_BYTE *)v7;
  v90 = v9;
  if ( (_WORD)v6 == 143 )
  {
    v52 = *(__int64 **)(a1 + 32);
    v53 = *v52;
    v54 = *((_DWORD *)v52 + 2);
    if ( v8 )
    {
      v56 = sub_216FFF0(v8);
    }
    else
    {
      v77 = sub_1F58D40((__int64)v89);
      v55 = 0;
      v56 = v77;
    }
    v57 = *(_QWORD *)(v53 + 40) + 16LL * v54;
    v58 = *(_BYTE *)v57;
    v59 = *(_QWORD *)(v57 + 8);
    LOBYTE(v92) = v58;
    v93 = v59;
    if ( v58 )
    {
      v60 = sub_216FFF0(v58);
    }
    else
    {
      v82 = v55;
      v60 = sub_1F58D40((__int64)&v92);
      v61 = v82;
    }
    if ( v60 >= v56 )
      goto LABEL_5;
    if ( v61 )
      v62 = sub_216FFF0(v61);
    else
      v62 = sub_1F58D40((__int64)v89);
    LOBYTE(v92) = v58;
    v93 = v59;
    if ( v58 )
      v63 = sub_216FFF0(v58);
    else
      v63 = sub_1F58D40((__int64)&v92);
    *a3 = (unsigned int)(v62 - v63);
    return v53;
  }
  if ( (__int16)v6 > 143 )
  {
    if ( (_WORD)v6 == 185 )
    {
      if ( ((*(_BYTE *)(a1 + 27) ^ 0xC) & 0xC) != 0 )
        goto LABEL_5;
      if ( v8 )
      {
        v65 = sub_216FFF0(v8);
      }
      else
      {
        v79 = sub_1F58D40((__int64)v89);
        v64 = 0;
        v65 = v79;
      }
      v66 = *(_BYTE *)(a1 + 88);
      v85 = *(_QWORD *)(a1 + 96);
      LOBYTE(v92) = v66;
      v93 = v85;
      if ( v66 )
      {
        v67 = sub_216FFF0(v66);
      }
      else
      {
        v83 = v64;
        v67 = sub_1F58D40((__int64)&v92);
        v68 = v83;
      }
      if ( v67 >= v65 )
        goto LABEL_5;
      if ( v68 )
        v69 = sub_216FFF0(v68);
      else
        v69 = sub_1F58D40((__int64)v89);
      LOBYTE(v92) = v66;
      v93 = v85;
      if ( v66 )
        v70 = sub_216FFF0(v66);
      else
        v70 = sub_1F58D40((__int64)&v92);
      *a3 = (unsigned int)(v69 - v70);
      return a1;
    }
    if ( (unsigned __int16)(v6 - 659) > 5u )
    {
LABEL_5:
      *a3 = 0;
      return 0;
    }
    v21 = *(_BYTE *)(a1 + 88);
    v22 = *(_QWORD *)(a1 + 96);
    LOBYTE(v92) = v21;
    v93 = v22;
    if ( v21 )
    {
      switch ( v21 )
      {
        case 14:
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 56:
        case 57:
        case 58:
        case 59:
        case 60:
        case 61:
          v73 = 2;
          v72 = 0;
          break;
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
          v73 = 3;
          v72 = 0;
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
          v73 = 4;
          v72 = 0;
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
          v73 = 5;
          v72 = 0;
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
          v73 = 6;
          v72 = 0;
          break;
        case 55:
          v73 = 7;
          v72 = 0;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v73 = 8;
          v72 = 0;
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
          v73 = 9;
          v72 = 0;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v73 = 10;
          v72 = 0;
          break;
      }
    }
    else
    {
      v71 = sub_1F596B0((__int64)&v92);
      v8 = v89[0];
      v73 = v71;
    }
    LOBYTE(v91[0]) = v73;
    v91[1] = v72;
    if ( v8 )
    {
      v75 = sub_216FFF0(v8);
    }
    else
    {
      v86 = v73;
      v78 = sub_1F58D40((__int64)v89);
      v74 = v86;
      v75 = v78;
    }
    if ( v74 )
      v76 = sub_216FFF0(v74);
    else
      v76 = sub_1F58D40((__int64)v91);
    if ( v76 < v75 )
    {
      *a3 = v75 - v76;
      return a1;
    }
    v6 = *(__int16 *)(a1 + 24);
    goto LABEL_24;
  }
  if ( (_WORD)v6 == 50 )
  {
    v47 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL);
    if ( *(_WORD *)(v47 + 24) != 10 )
      goto LABEL_5;
    v48 = *(_QWORD *)(v47 + 88);
    v49 = *(_DWORD *)(v48 + 32);
    if ( !(v49 <= 0x40 ? *(_QWORD *)(v48 + 24) == 0 : v49 == (unsigned int)sub_16A57B0(v48 + 24)) )
      goto LABEL_5;
    v36 = *(_BYTE *)v7;
    v51 = *(_QWORD *)(v7 + 8);
    LOBYTE(v92) = v36;
    v93 = v51;
    if ( !v36 )
    {
LABEL_40:
      v38 = sub_1F58D40((__int64)&v92);
LABEL_59:
      *a3 = v38 >> 1;
      return **(_QWORD **)(a1 + 32);
    }
LABEL_58:
    v38 = sub_216FFF0(v36);
    goto LABEL_59;
  }
  if ( (_WORD)v6 == 118 )
  {
    v11 = *(_QWORD **)(a1 + 32);
    v12 = *v11;
    if ( *(_WORD *)(*v11 + 24LL) != 10 )
    {
      v12 = v11[5];
      if ( *(_WORD *)(v12 + 24) != 10 )
        goto LABEL_5;
    }
    v13 = *(_QWORD *)(v12 + 88);
    v14 = *(_DWORD *)(v13 + 32);
    v15 = *(_QWORD *)(v13 + 24);
    v16 = 1LL << ((unsigned __int8)v14 - 1);
    if ( v14 <= 0x40 )
    {
      v17 = *(_QWORD *)(v13 + 24);
      if ( (v16 & v15) != 0 )
        goto LABEL_5;
      if ( !v15 )
      {
        v18 = 1;
LABEL_17:
        _BitScanReverse64(&v18, v18);
        v19 = 63 - ((unsigned int)v18 ^ 0x3F);
        if ( v8 )
          v20 = sub_216FFF0(v8);
        else
          v20 = sub_1F58D40((__int64)v89);
        if ( (unsigned int)v19 >= v20 )
          goto LABEL_5;
        *a3 = v20 - v19;
        return a1;
      }
    }
    else
    {
      if ( (*(_QWORD *)(v15 + 8LL * ((v14 - 1) >> 6)) & v16) != 0 )
        goto LABEL_5;
      v84 = v8;
      if ( v14 - (unsigned int)sub_16A57B0(v13 + 24) > 0x40 )
        goto LABEL_5;
      v17 = *(_QWORD *)v15;
      v8 = v84;
    }
    v18 = v17 + 1;
    if ( v17 == -1 || (v17 & v18) != 0 )
      goto LABEL_5;
    goto LABEL_17;
  }
LABEL_24:
  if ( (v6 & 0x8000u) == 0 )
    goto LABEL_5;
  v23 = ~v6;
  if ( v23 == 3243 )
    goto LABEL_42;
  if ( v23 > 3243 )
  {
    if ( (unsigned int)(v23 - 4449) > 1 )
      goto LABEL_5;
    v32 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL);
    if ( *(_WORD *)(v32 + 24) != 10 )
      goto LABEL_5;
    v33 = *(_QWORD *)(v32 + 88);
    v34 = *(_DWORD *)(v33 + 32);
    if ( v34 <= 0x40 )
    {
      if ( *(_QWORD *)(v33 + 24) )
        goto LABEL_5;
    }
    else if ( v34 != (unsigned int)sub_16A57B0(v33 + 24) )
    {
      goto LABEL_5;
    }
    v35 = *(_QWORD *)(a1 + 40) + v3;
    v36 = *(_BYTE *)v35;
    v37 = *(_QWORD *)(v35 + 8);
    LOBYTE(v92) = v36;
    v93 = v37;
    if ( !v36 )
      goto LABEL_40;
    goto LABEL_58;
  }
  if ( v23 <= 165 )
  {
    if ( v23 <= 163 )
      goto LABEL_5;
LABEL_42:
    if ( !(unsigned __int8)sub_216FEC0(a1, &v87, &v88) )
      return 0;
    v39 = *(_WORD *)(a1 + 24) == 0xFF5A;
    v40 = 64;
    *a3 = 0;
    if ( !v39 )
      v40 = 32;
    if ( v87 + v88 < v40 )
    {
      v80 = *(_QWORD *)(a1 + 32);
      v92 = 0;
      if ( sub_2171180(*(_QWORD *)(v80 + 40), *(_QWORD *)(v80 + 48), &v92) )
      {
        v81 = v40 - v88 - v87;
        if ( v81 > v92 )
          v81 = v92;
        *a3 += v81;
      }
    }
    v41 = *(_QWORD **)(a1 + 32);
    v91[0] = 0;
    if ( sub_2171180(*v41, v41[1], v91) )
    {
      v42 = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 32) + 8LL);
      v43 = *(_BYTE *)v42;
      v44 = *(_QWORD *)(v42 + 8);
      LOBYTE(v92) = v43;
      v93 = v44;
      if ( v43 )
        v45 = sub_216FFF0(v43);
      else
        v45 = sub_1F58D40((__int64)&v92);
      v46 = *a3;
      if ( v88 > (unsigned __int64)v45 - v91[0] )
      {
        v46 = v88 + v91[0] + v46 - v45;
        *a3 = v46;
      }
    }
    else
    {
      v46 = *a3;
    }
    if ( !v46 )
      return 0;
    return a1;
  }
  if ( v23 != 614 )
    goto LABEL_5;
  v24 = *(__int64 **)(a1 + 32);
  v25 = *v24;
  v26 = *(unsigned __int16 *)(*v24 + 24);
  v27 = (unsigned int)(v26 - 185);
  if ( (unsigned __int16)(v26 - 185) > 0x35u )
  {
    if ( (unsigned __int16)(v26 - 44) <= 1u )
    {
      if ( (*(_BYTE *)(v25 + 26) & 2) == 0 )
        goto LABEL_5;
    }
    else if ( (__int16)v26 <= 658 )
    {
      goto LABEL_5;
    }
  }
  else
  {
    v28 = 0x3FFFFD00000003LL;
    if ( !_bittest64(&v28, v27) )
      goto LABEL_5;
  }
  v29 = *(_BYTE *)(v25 + 88);
  v30 = *(_QWORD *)(v25 + 96);
  LOBYTE(v92) = v29;
  v93 = v30;
  if ( v29 )
    v31 = sub_216FFF0(v29);
  else
    v31 = sub_1F58D40((__int64)&v92);
  if ( v31 != 8 )
    goto LABEL_5;
  *a3 = 8;
  return v25;
}
