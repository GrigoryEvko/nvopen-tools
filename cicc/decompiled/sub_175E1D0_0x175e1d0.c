// Function: sub_175E1D0
// Address: 0x175e1d0
//
__int64 __fastcall sub_175E1D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  _BYTE *v15; // rdi
  unsigned __int8 v16; // al
  __int64 v17; // r12
  int v18; // edi
  unsigned int v19; // r15d
  unsigned __int64 v23; // r8
  unsigned int v24; // eax
  unsigned int v25; // ecx
  __int64 v26; // r12
  __int64 v28; // rax
  __int64 *v29; // rbx
  __int64 v30; // rax
  double v31; // xmm4_8
  double v32; // xmm5_8
  unsigned int v33; // edx
  __int64 v34; // rsi
  __int64 v35; // rax
  unsigned int v36; // r12d
  unsigned __int64 v37; // rsi
  __int64 v38; // rax
  int v39; // eax
  __int64 v40; // r14
  unsigned int v41; // r12d
  bool v42; // al
  unsigned int v43; // eax
  unsigned __int64 v44; // rdx
  char v45; // cl
  unsigned __int64 *v46; // rsi
  bool v47; // al
  int v48; // r12d
  __int64 v49; // rdi
  __int64 v50; // r13
  _QWORD *v51; // rax
  __int64 v52; // rax
  __int64 v53; // r13
  _QWORD *v54; // rax
  unsigned int v55; // eax
  __int64 v56; // rax
  __int64 v57; // r13
  __int16 v58; // si
  char v59; // r9
  int v60; // eax
  int v61; // edx
  __int64 v62; // rax
  __int64 v63; // r13
  _QWORD *v64; // rax
  unsigned int v65; // ecx
  unsigned int v66; // eax
  unsigned __int64 v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // r13
  _QWORD *v71; // rax
  unsigned int v72; // eax
  bool v73; // al
  int v74; // eax
  int v75; // eax
  unsigned int v76; // eax
  unsigned int v77; // eax
  unsigned int v78; // eax
  int v79; // [rsp+8h] [rbp-E8h]
  bool v80; // [rsp+10h] [rbp-E0h]
  unsigned int v81; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v82; // [rsp+18h] [rbp-D8h]
  _BOOL4 v83; // [rsp+20h] [rbp-D0h]
  char v84; // [rsp+20h] [rbp-D0h]
  bool v86; // [rsp+34h] [rbp-BCh]
  char v87; // [rsp+38h] [rbp-B8h]
  _BOOL4 v88; // [rsp+38h] [rbp-B8h]
  __int64 v89; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v90; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v91; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v92; // [rsp+58h] [rbp-98h]
  unsigned __int64 v93; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v94; // [rsp+68h] [rbp-88h]
  unsigned __int64 v95; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v96; // [rsp+78h] [rbp-78h]
  unsigned __int64 v97; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v98; // [rsp+88h] [rbp-68h]
  unsigned __int64 v99; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v100; // [rsp+98h] [rbp-58h]
  unsigned __int64 v101; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v102; // [rsp+A8h] [rbp-48h]
  __int16 v103; // [rsp+B0h] [rbp-40h]

  v15 = *(_BYTE **)(a3 - 24);
  v16 = v15[16];
  v17 = (__int64)(v15 + 24);
  if ( v16 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) != 16 )
      return 0;
    if ( v16 > 0x10u )
      return 0;
    v28 = sub_15A1020(v15, a2, *(_QWORD *)v15, a4);
    if ( !v28 || *(_BYTE *)(v28 + 16) != 13 )
      return 0;
    v17 = v28 + 24;
  }
  v87 = *(_BYTE *)(a3 + 16);
  v86 = v87 == 42;
  v18 = *(_WORD *)(a2 + 18) & 0x7FFF;
  if ( (unsigned int)(v18 - 32) > 1 && v86 != sub_15FF7F0(v18) )
    return 0;
  v19 = *(_DWORD *)(v17 + 8);
  if ( v19 <= 0x40 ? *(_QWORD *)v17 == 0 : v19 == (unsigned int)sub_16A57B0(v17) )
    return 0;
  if ( v19 <= 0x40 ? *(_QWORD *)v17 == 1 : v19 - 1 == (unsigned int)sub_16A57B0(v17) )
    return 0;
  if ( v87 != 42 )
  {
    sub_16A7B50((__int64)&v89, a4, (__int64 *)v17);
    sub_16A9D70((__int64)&v101, (__int64)&v89, v17);
    v23 = v101;
    if ( v102 <= 0x40 )
      goto LABEL_14;
    goto LABEL_28;
  }
  if ( v19 <= 0x40
     ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v19) == *(_QWORD *)v17
     : v19 == (unsigned int)sub_16A58F0(v17) )
  {
    return 0;
  }
  sub_16A7B50((__int64)&v89, a4, (__int64 *)v17);
  sub_16A9F90((__int64)&v101, (__int64)&v89, v17);
  v23 = v101;
  if ( v102 <= 0x40 )
  {
LABEL_14:
    v80 = v23 != *(_QWORD *)a4;
    goto LABEL_15;
  }
LABEL_28:
  v82 = v23;
  v80 = !sub_16A5220((__int64)&v101, (const void **)a4);
  if ( v82 )
    j_j___libc_free_0_0(v82);
LABEL_15:
  v24 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v24) &= ~0x80u;
  v81 = v24;
  if ( sub_15F23D0(a3) )
  {
    v25 = *(_DWORD *)(v17 + 8);
    v92 = v25;
    if ( v25 > 0x40 )
      sub_16A4EF0((__int64)&v91, 1, 0);
    else
      v91 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v25) & 1;
  }
  else
  {
    v92 = *(_DWORD *)(v17 + 8);
    if ( v92 > 0x40 )
      sub_16A4FD0((__int64)&v91, (const void **)v17);
    else
      v91 = *(_QWORD *)v17;
  }
  v96 = 1;
  v94 = 1;
  v93 = 0;
  v95 = 0;
  if ( v87 != 42 )
  {
    if ( v90 <= 0x40 )
    {
      v94 = v90;
      v93 = v89 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v90);
    }
    else
    {
      sub_16A51C0((__int64)&v93, (__int64)&v89);
    }
    if ( v80 )
      goto LABEL_36;
    v45 = 0;
    v46 = &v93;
    goto LABEL_82;
  }
  v33 = *(_DWORD *)(v17 + 8);
  v34 = *(_QWORD *)v17;
  v35 = 1LL << ((unsigned __int8)v33 - 1);
  if ( v33 > 0x40 )
  {
    if ( (*(_QWORD *)(v34 + 8LL * ((v33 - 1) >> 6)) & v35) != 0 )
      goto LABEL_59;
    v79 = *(_DWORD *)(v17 + 8);
    if ( v79 == (unsigned int)sub_16A57B0(v17) )
      goto LABEL_80;
  }
  else
  {
    if ( (v35 & v34) != 0 )
    {
LABEL_59:
      if ( sub_15F23D0(a3) )
      {
        if ( v92 > 0x40 )
          sub_16A8F40((__int64 *)&v91);
        else
          v91 = ~v91 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v92);
        sub_16A7400((__int64)&v91);
      }
      if ( !sub_13D01C0(a4) )
      {
        v36 = *(_DWORD *)(a4 + 8);
        v37 = *(_QWORD *)a4;
        v38 = 1LL << ((unsigned __int8)v36 - 1);
        if ( v36 > 0x40 )
        {
          if ( (*(_QWORD *)(v37 + 8LL * ((v36 - 1) >> 6)) & v38) != 0 )
            goto LABEL_63;
          v73 = v36 == (unsigned int)sub_16A57B0(a4);
        }
        else
        {
          if ( (v38 & v37) != 0 )
            goto LABEL_63;
          v73 = v37 == 0;
        }
        if ( !v73 )
        {
          sub_13A38D0((__int64)&v99, (__int64)&v89);
          sub_16A7490((__int64)&v99, 1);
          v102 = v100;
          v100 = 0;
          v101 = v99;
          sub_17571D0((__int64 *)&v95, (__int64 *)&v101);
          sub_135E100((__int64 *)&v101);
          sub_135E100((__int64 *)&v99);
          if ( v80 )
          {
            v74 = sub_15FF5D0(v81);
            v40 = *(_QWORD *)(a3 - 48);
            switch ( v74 )
            {
              case ' ':
                v48 = -1;
                v61 = -1;
                goto LABEL_116;
              case '!':
                v60 = -1;
                v61 = -1;
                goto LABEL_109;
              case '"':
              case '&':
                goto LABEL_56;
              case '$':
              case '(':
                goto LABEL_42;
              default:
                goto LABEL_168;
            }
          }
          v83 = -sub_1757340((__int64 *)&v93, (__int64)&v95, (__int64 *)&v91, 1);
          v76 = sub_15FF5D0(v81);
          v40 = *(_QWORD *)(a3 - 48);
          v61 = v83;
          v81 = v76;
          switch ( v76 )
          {
            case ' ':
              goto LABEL_119;
            case '!':
              v60 = 0;
              goto LABEL_109;
            case '"':
            case '&':
              goto LABEL_89;
            case '#':
            case '$':
            case '%':
            case '\'':
            case '(':
              goto LABEL_41;
          }
        }
LABEL_63:
        if ( v94 <= 0x40 && v90 <= 0x40 )
        {
          v94 = v90;
          v93 = v89 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v90);
        }
        else
        {
          sub_16A51C0((__int64)&v93, (__int64)&v89);
        }
        if ( v80 )
        {
          v39 = sub_15FF5D0(v81);
          v40 = *(_QWORD *)(a3 - 48);
          switch ( v39 )
          {
            case ' ':
            case '"':
            case '&':
              goto LABEL_42;
            case '!':
              v60 = 1;
              v61 = 1;
              goto LABEL_109;
            case '$':
            case '(':
              goto LABEL_56;
            default:
              goto LABEL_168;
          }
        }
        sub_16A7620((__int64)&v101, (__int64)&v89, (__int64)&v91, &v99);
        if ( v96 > 0x40 && v95 )
          j_j___libc_free_0_0(v95);
        v95 = v101;
        v77 = v102;
        v102 = 0;
        v96 = v77;
        sub_135E100((__int64 *)&v101);
        v48 = (unsigned __int8)v99;
        v84 = v99;
        v78 = sub_15FF5D0(v81);
        v40 = *(_QWORD *)(a3 - 48);
        v81 = v78;
        switch ( v78 )
        {
          case ' ':
            goto LABEL_115;
          case '!':
            if ( !v84 )
              goto LABEL_103;
            goto LABEL_134;
          case '"':
          case '&':
LABEL_88:
            if ( v48 != 1 )
              goto LABEL_89;
            goto LABEL_42;
          case '#':
          case '$':
          case '%':
          case '\'':
          case '(':
            goto LABEL_92;
        }
      }
      sub_13A38D0((__int64)&v99, (__int64)&v91);
      sub_16A7490((__int64)&v99, 1);
      v102 = v100;
      v100 = 0;
      v101 = v99;
      sub_17571D0((__int64 *)&v93, (__int64 *)&v101);
      sub_135E100((__int64 *)&v101);
      sub_135E100((__int64 *)&v99);
      sub_13A38D0((__int64)&v99, (__int64)&v91);
      if ( v100 > 0x40 )
        sub_16A8F40((__int64 *)&v99);
      else
        v99 = ~v99 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v100);
      sub_16A7400((__int64)&v99);
      v102 = v100;
      v100 = 0;
      v101 = v99;
      sub_17571D0((__int64 *)&v95, (__int64 *)&v101);
      sub_135E100((__int64 *)&v101);
      sub_135E100((__int64 *)&v99);
      if ( !sub_1455820((__int64)&v95, (_QWORD *)v17) )
      {
        v55 = sub_15FF5D0(v81);
        v40 = *(_QWORD *)(a3 - 48);
        v81 = v55;
        switch ( v55 )
        {
          case ' ':
            v48 = 0;
LABEL_115:
            v61 = 0;
LABEL_116:
            if ( v61 && v48 )
              goto LABEL_42;
            if ( !v48 )
              goto LABEL_119;
            goto LABEL_99;
          case '!':
            v60 = 0;
            v61 = 0;
            goto LABEL_109;
          case '"':
          case '&':
            goto LABEL_89;
          case '$':
          case '(':
            goto LABEL_92;
          default:
            goto LABEL_168;
        }
      }
      v102 = 1;
      v101 = 0;
      sub_17571D0((__int64 *)&v95, (__int64 *)&v101);
      sub_135E100((__int64 *)&v101);
      v75 = sub_15FF5D0(v81);
      v40 = *(_QWORD *)(a3 - 48);
      LOWORD(v81) = v75;
      switch ( v75 )
      {
        case ' ':
LABEL_99:
          v56 = sub_15A1070(*(_QWORD *)a3, (__int64)&v93);
          v103 = 257;
          v57 = v56;
          v26 = (__int64)sub_1648A60(56, 2u);
          if ( !v26 )
            goto LABEL_44;
          v58 = 4 * (v87 == 42) + 35;
          goto LABEL_101;
        case '!':
          v60 = 1;
          v61 = 0;
LABEL_109:
          if ( v61 && v60 )
            goto LABEL_56;
          if ( !v60 )
            goto LABEL_112;
LABEL_134:
          v68 = sub_15A1070(*(_QWORD *)a3, (__int64)&v93);
          v103 = 257;
          v57 = v68;
          v26 = (__int64)sub_1648A60(56, 2u);
          if ( !v26 )
            goto LABEL_44;
          v58 = 4 * (v87 == 42) + 36;
LABEL_101:
          sub_17582E0(v26, v58, v40, v57, (__int64)&v101);
          break;
        case '"':
        case '&':
          goto LABEL_42;
        case '#':
        case '$':
        case '%':
        case '\'':
        case '(':
          goto LABEL_92;
      }
      goto LABEL_44;
    }
    if ( !v34 )
    {
LABEL_80:
      v40 = *(_QWORD *)(a3 - 48);
      switch ( v81 )
      {
        case ' ':
          goto LABEL_107;
        case '!':
          goto LABEL_103;
        case '"':
        case '&':
          goto LABEL_89;
        case '$':
        case '(':
          goto LABEL_92;
        default:
          goto LABEL_168;
      }
    }
  }
  v41 = *(_DWORD *)(a4 + 8);
  if ( v41 <= 0x40 )
    v42 = *(_QWORD *)a4 == 0;
  else
    v42 = v41 == (unsigned int)sub_16A57B0(a4);
  if ( !v42 )
  {
    if ( sub_13D0200((__int64 *)a4, v41 - 1) || sub_13D01C0(a4) )
    {
      sub_13A38D0((__int64)&v99, (__int64)&v89);
      sub_16A7490((__int64)&v99, 1);
      v43 = v100;
      v44 = v99;
      v100 = 0;
      v102 = v43;
      v101 = v99;
      if ( v96 > 0x40 && v95 )
      {
        j_j___libc_free_0_0(v95);
        v44 = v101;
        v43 = v102;
      }
      v95 = v44;
      v96 = v43;
      v102 = 0;
      sub_135E100((__int64 *)&v101);
      sub_135E100((__int64 *)&v99);
      if ( v80 )
      {
        switch ( v81 )
        {
          case ' ':
          case '$':
          case '(':
            goto LABEL_42;
          case '!':
          case '"':
          case '&':
LABEL_56:
            v29 = (__int64 *)a1;
            v30 = sub_159C4F0(*(__int64 **)(*(_QWORD *)(a1 + 8) + 24LL));
            goto LABEL_43;
          default:
            goto LABEL_168;
        }
      }
      sub_13A38D0((__int64)&v101, (__int64)&v91);
      if ( v102 > 0x40 )
        sub_16A8F40((__int64 *)&v101);
      else
        v101 = ~v101 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v102);
      sub_16A7400((__int64)&v101);
      v72 = v102;
      v102 = 0;
      v100 = v72;
      v99 = v101;
      sub_135E100((__int64 *)&v101);
      v88 = -sub_1757340((__int64 *)&v93, (__int64)&v95, (__int64 *)&v99, 1);
      sub_135E100((__int64 *)&v99);
      v40 = *(_QWORD *)(a3 - 48);
      v61 = v88;
      switch ( v81 )
      {
        case ' ':
LABEL_119:
          if ( !v61 )
            goto LABEL_107;
          v62 = sub_15A1070(*(_QWORD *)a3, (__int64)&v95);
          v103 = 257;
          v63 = v62;
          v64 = sub_1648A60(56, 2u);
          v26 = (__int64)v64;
          if ( v64 )
            sub_17582E0((__int64)v64, 40, v40, v63, (__int64)&v101);
          goto LABEL_44;
        case '!':
LABEL_112:
          if ( !v61 )
            goto LABEL_103;
          v49 = *(_QWORD *)a3;
          goto LABEL_90;
        case '"':
        case '&':
          goto LABEL_89;
        case '#':
        case '$':
        case '%':
        case '\'':
        case '(':
LABEL_41:
          if ( v61 != -1 )
            goto LABEL_92;
LABEL_42:
          v29 = (__int64 *)a1;
          v30 = sub_159C540(*(__int64 **)(*(_QWORD *)(a1 + 8) + 24LL));
          goto LABEL_43;
      }
    }
    if ( v90 <= 0x40 )
    {
      v94 = v90;
      v93 = v89 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v90);
    }
    else
    {
      sub_16A51C0((__int64)&v93, (__int64)&v89);
    }
    if ( v80 )
    {
LABEL_36:
      switch ( v81 )
      {
        case ' ':
        case '"':
        case '&':
          goto LABEL_42;
        case '!':
        case '$':
        case '(':
          goto LABEL_56;
        default:
          goto LABEL_168;
      }
    }
    v46 = (unsigned __int64 *)&v89;
    v45 = 1;
LABEL_82:
    v47 = sub_1757340((__int64 *)&v95, (__int64)v46, (__int64 *)&v91, v45);
    v40 = *(_QWORD *)(a3 - 48);
    v48 = v47;
    switch ( v81 )
    {
      case ' ':
        if ( v47 )
          goto LABEL_99;
        goto LABEL_107;
      case '!':
        if ( !v47 )
          goto LABEL_103;
        goto LABEL_134;
      case '"':
      case '&':
        goto LABEL_88;
      case '$':
      case '(':
        goto LABEL_92;
      default:
        goto LABEL_168;
    }
  }
  sub_13A38D0((__int64)&v97, (__int64)&v91);
  sub_16A7800((__int64)&v97, 1u);
  v65 = v98;
  v98 = 0;
  v100 = v65;
  v99 = v97;
  if ( v65 > 0x40 )
    sub_16A8F40((__int64 *)&v99);
  else
    v99 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v65) & ~v97;
  sub_16A7400((__int64)&v99);
  v66 = v100;
  v67 = v99;
  v100 = 0;
  v102 = v66;
  v101 = v99;
  if ( v94 > 0x40 && v93 )
  {
    j_j___libc_free_0_0(v93);
    v67 = v101;
    v66 = v102;
  }
  v93 = v67;
  v94 = v66;
  v102 = 0;
  sub_135E100((__int64 *)&v101);
  sub_135E100((__int64 *)&v99);
  sub_135E100((__int64 *)&v97);
  if ( v96 <= 0x40 && v92 <= 0x40 )
  {
    v96 = v92;
    v95 = v91 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v92);
  }
  else
  {
    sub_16A51C0((__int64)&v95, (__int64)&v91);
  }
  v40 = *(_QWORD *)(a3 - 48);
  switch ( v81 )
  {
    case ' ':
LABEL_107:
      v59 = 1;
      goto LABEL_104;
    case '!':
LABEL_103:
      v59 = 0;
LABEL_104:
      v29 = (__int64 *)a1;
      v30 = (__int64)sub_17288D0(a1, v40, (__int64)&v93, (__int64)&v95, v86, v59, *(double *)a5.m128_u64, a6, a7);
LABEL_43:
      v26 = sub_170E100(v29, a2, v30, a5, a6, a7, a8, v31, v32, a11, a12);
      break;
    case '"':
    case '&':
LABEL_89:
      v49 = *(_QWORD *)a3;
      if ( v81 == 34 )
      {
        v69 = sub_15A1070(v49, (__int64)&v95);
        v103 = 257;
        v70 = v69;
        v71 = sub_1648A60(56, 2u);
        v26 = (__int64)v71;
        if ( v71 )
          sub_17582E0((__int64)v71, 35, v40, v70, (__int64)&v101);
      }
      else
      {
LABEL_90:
        v50 = sub_15A1070(v49, (__int64)&v95);
        v103 = 257;
        v51 = sub_1648A60(56, 2u);
        v26 = (__int64)v51;
        if ( v51 )
          sub_17582E0((__int64)v51, 39, v40, v50, (__int64)&v101);
      }
      break;
    case '$':
    case '(':
LABEL_92:
      v52 = sub_15A1070(*(_QWORD *)a3, (__int64)&v93);
      v103 = 257;
      v53 = v52;
      v54 = sub_1648A60(56, 2u);
      v26 = (__int64)v54;
      if ( v54 )
        sub_17582E0((__int64)v54, v81, v40, v53, (__int64)&v101);
      break;
    default:
LABEL_168:
      ++*(_DWORD *)(a3 + 16);
      BUG();
  }
LABEL_44:
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0(v95);
  if ( v94 > 0x40 && v93 )
    j_j___libc_free_0_0(v93);
  if ( v92 > 0x40 && v91 )
    j_j___libc_free_0_0(v91);
  if ( v90 > 0x40 && v89 )
    j_j___libc_free_0_0(v89);
  return v26;
}
