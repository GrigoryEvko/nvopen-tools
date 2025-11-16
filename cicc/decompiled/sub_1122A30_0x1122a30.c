// Function: sub_1122A30
// Address: 0x1122a30
//
__int64 __fastcall sub_1122A30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  unsigned int v7; // r15d
  char v8; // bl
  __int64 v9; // rax
  int v10; // edi
  __int64 v11; // r12
  unsigned int v12; // r15d
  __int64 v16; // r9
  bool v17; // zf
  unsigned int v18; // eax
  void *v19; // r12
  __int64 v21; // rdx
  _BYTE *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rcx
  unsigned int v26; // r8d
  unsigned int **v27; // r13
  _BYTE *v28; // rax
  __int64 v29; // r14
  unsigned int **v30; // r13
  _BYTE *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rax
  unsigned int v35; // edx
  __int64 v36; // rsi
  unsigned int v37; // ecx
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  unsigned int v43; // ebx
  __int64 v44; // rdx
  __int64 v45; // rax
  int v46; // eax
  bool v47; // al
  unsigned int v48; // r12d
  _BOOL4 v49; // r12d
  unsigned int v50; // edx
  unsigned __int64 v51; // rax
  unsigned int v52; // eax
  __int64 v53; // r14
  _QWORD *v54; // rax
  __int64 v55; // rdi
  __int64 v56; // r13
  _QWORD *v57; // rax
  unsigned int v58; // eax
  unsigned __int64 v59; // rdx
  unsigned int v60; // eax
  char v61; // r9
  char v62; // r8
  __int64 v63; // r13
  __int16 v64; // si
  int v65; // ecx
  char v66; // dl
  bool v67; // r12
  __int64 v68; // r13
  _QWORD *v69; // rax
  unsigned int v70; // eax
  __int64 v71; // r13
  _QWORD *v72; // rax
  bool v73; // al
  unsigned int v74; // eax
  unsigned __int64 v75; // rdx
  unsigned int v76; // eax
  unsigned int v77; // [rsp+Ch] [rbp-104h]
  __int64 v78; // [rsp+10h] [rbp-100h]
  int v79; // [rsp+10h] [rbp-100h]
  _BOOL4 v80; // [rsp+18h] [rbp-F8h]
  __int64 v81; // [rsp+20h] [rbp-F0h]
  __int64 v83; // [rsp+30h] [rbp-E0h]
  bool v85; // [rsp+43h] [rbp-CDh]
  __int16 v86; // [rsp+44h] [rbp-CCh]
  _BOOL4 v87; // [rsp+44h] [rbp-CCh]
  bool v88; // [rsp+48h] [rbp-C8h]
  unsigned int v89; // [rsp+4Ch] [rbp-C4h]
  __int64 v90; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v91; // [rsp+58h] [rbp-B8h]
  __int64 v92; // [rsp+60h] [rbp-B0h] BYREF
  unsigned int v93; // [rsp+68h] [rbp-A8h]
  __int64 v94; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v95; // [rsp+78h] [rbp-98h]
  __int64 v96; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v97; // [rsp+88h] [rbp-88h]
  unsigned __int64 v98; // [rsp+90h] [rbp-80h] BYREF
  unsigned int v99; // [rsp+98h] [rbp-78h]
  unsigned __int64 v100; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v101; // [rsp+A8h] [rbp-68h]
  unsigned __int64 v102; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v103; // [rsp+B8h] [rbp-58h]
  __int16 v104; // [rsp+D0h] [rbp-40h]

  v6 = *(_QWORD *)(a3 - 32);
  v83 = *(_QWORD *)(a3 - 64);
  v7 = (*(_WORD *)(a2 + 2) & 0x3F) - 32;
  v89 = *(_WORD *)(a2 + 2) & 0x3F;
  v86 = *(_WORD *)(a2 + 2) & 0x3F;
  v81 = *(_QWORD *)(a3 + 8);
  v8 = *(_BYTE *)a3;
  v88 = *(_BYTE *)a3 == 49;
  if ( v7 <= 1 )
  {
    v9 = *(_QWORD *)(a3 + 16);
    if ( v9 )
    {
      if ( !*(_QWORD *)(v9 + 8) )
      {
        v23 = (unsigned int)(*(_DWORD *)(a4 + 8) - 1);
        if ( sub_986C60((__int64 *)a4, v23)
          && (v8 != 49 || (unsigned __int8)sub_986B30((__int64 *)a4, v23, v24, v25, v26)) )
        {
          v104 = 257;
          v27 = *(unsigned int ***)(a1 + 32);
          v28 = (_BYTE *)sub_AD8D80(v81, a4);
          v29 = sub_92B530(v27, v89, v83, v28, (__int64)&v102);
          v30 = *(unsigned int ***)(a1 + 32);
          v104 = 257;
          v31 = (_BYTE *)sub_AD64C0(v81, 1, 0);
          v32 = sub_92B530(v30, v89, v6, v31, (__int64)&v102);
          v104 = 257;
          return sub_B504D0((unsigned int)(v89 != 32) + 28, v29, v32, (__int64)&v102, 0, 0);
        }
      }
    }
  }
  if ( *(_BYTE *)v6 == 17 )
  {
    v10 = v89;
    v11 = v6 + 24;
  }
  else
  {
    v21 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17;
    if ( (unsigned int)v21 > 1 )
      return 0;
    if ( *(_BYTE *)v6 > 0x15u )
      return 0;
    v22 = sub_AD7630(v6, 0, v21);
    if ( !v22 || *v22 != 17 )
      return 0;
    v11 = (__int64)(v22 + 24);
    v10 = *(_WORD *)(a2 + 2) & 0x3F;
    v7 = v10 - 32;
  }
  if ( v7 > 1 && v88 != sub_B532B0(v10) )
    return 0;
  v12 = *(_DWORD *)(v11 + 8);
  if ( v12 <= 0x40 ? *(_QWORD *)v11 == 0 : v12 == (unsigned int)sub_C444A0(v11) )
    return 0;
  if ( v12 <= 0x40 ? *(_QWORD *)v11 == 1 : v12 - 1 == (unsigned int)sub_C444A0(v11) )
    return 0;
  if ( v8 != 49 )
  {
    sub_C472A0((__int64)&v90, a4, (__int64 *)v11);
    sub_C4A1D0((__int64)&v102, (__int64)&v90, v11);
    v16 = v102;
    if ( v103 <= 0x40 )
      goto LABEL_18;
    goto LABEL_37;
  }
  if ( !v12 )
    return 0;
  if ( v12 <= 0x40
     ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v12) == *(_QWORD *)v11
     : v12 == (unsigned int)sub_C445E0(v11) )
  {
    return 0;
  }
  sub_C472A0((__int64)&v90, a4, (__int64 *)v11);
  sub_C4A3E0((__int64)&v102, (__int64)&v90, v11);
  v16 = v102;
  if ( v103 <= 0x40 )
  {
LABEL_18:
    v85 = v16 != *(_QWORD *)a4;
    goto LABEL_19;
  }
LABEL_37:
  v78 = v16;
  v85 = !sub_C43C50((__int64)&v102, (const void **)a4);
  if ( v78 )
    j_j___libc_free_0_0(v78);
LABEL_19:
  v17 = !sub_B44E60(a3);
  v18 = *(_DWORD *)(v11 + 8);
  v93 = v18;
  if ( v17 )
  {
    if ( v18 > 0x40 )
      sub_C43780((__int64)&v92, (const void **)v11);
    else
      v92 = *(_QWORD *)v11;
  }
  else if ( v18 > 0x40 )
  {
    sub_C43690((__int64)&v92, 1, 0);
  }
  else
  {
    v92 = 1;
  }
  v95 = 1;
  v94 = 0;
  v97 = 1;
  v96 = 0;
  if ( v8 != 49 )
  {
    if ( v91 <= 0x40 )
    {
      v95 = v91;
      v94 = v90;
    }
    else
    {
      sub_C43990((__int64)&v94, (__int64)&v90);
    }
    if ( v85 )
    {
LABEL_26:
      switch ( v86 )
      {
        case ' ':
        case '"':
        case '&':
          goto LABEL_63;
        case '!':
        case '$':
        case '(':
          goto LABEL_48;
        default:
          goto LABEL_170;
      }
    }
    v49 = sub_1111050(&v96, (__int64)&v94, &v92, 0);
    goto LABEL_97;
  }
  v35 = *(_DWORD *)(v11 + 8);
  v36 = *(_QWORD *)v11;
  v37 = v35 - 1;
  v38 = 1LL << ((unsigned __int8)v35 - 1);
  if ( v35 > 0x40 )
  {
    if ( (*(_QWORD *)(v36 + 8LL * (v37 >> 6)) & v38) != 0 )
      goto LABEL_66;
    v77 = v35 - 1;
    v79 = *(_DWORD *)(v11 + 8);
    v46 = sub_C444A0(v11);
    v37 = v77;
    v47 = v79 == v46;
  }
  else
  {
    if ( (v38 & v36) != 0 )
      goto LABEL_66;
    v47 = v36 == 0;
  }
  if ( v47 )
  {
LABEL_66:
    v39 = v37;
    if ( sub_986C60((__int64 *)v11, v37) )
    {
      if ( sub_B44E60(a3) )
      {
        sub_987160((__int64)&v92, v39, v40, v41, v42);
        sub_C46250((__int64)&v92);
      }
      if ( sub_9867B0(a4) )
      {
        sub_9865C0((__int64)&v100, (__int64)&v92);
        sub_C46A40((__int64)&v100, 1);
        v58 = v101;
        v101 = 0;
        v103 = v58;
        v102 = v100;
        sub_1110A30(&v94, (__int64 *)&v102);
        sub_969240((__int64 *)&v102);
        sub_969240((__int64 *)&v100);
        sub_9865C0((__int64)&v100, (__int64)&v92);
        if ( v101 > 0x40 )
        {
          sub_C43D10((__int64)&v100);
        }
        else
        {
          v59 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v101;
          if ( !v101 )
            v59 = 0;
          v100 = v59 & ~v100;
        }
        sub_C46250((__int64)&v100);
        v60 = v101;
        v101 = 0;
        v103 = v60;
        v102 = v100;
        sub_1110A30(&v96, (__int64 *)&v102);
        sub_969240((__int64 *)&v102);
        sub_969240((__int64 *)&v100);
        if ( !sub_AAD8B0((__int64)&v96, (_QWORD *)v11) )
        {
          v89 = sub_B52F50(v89);
          switch ( v89 )
          {
            case ' ':
              v65 = 0;
LABEL_130:
              LOBYTE(v49) = 0;
              goto LABEL_131;
            case '!':
              goto LABEL_112;
            case '"':
            case '&':
              goto LABEL_101;
            case '$':
            case '(':
              goto LABEL_99;
            default:
              goto LABEL_170;
          }
        }
        v103 = 1;
        v102 = 0;
        sub_1110A30(&v96, (__int64 *)&v102);
        sub_969240((__int64 *)&v102);
        v89 = sub_B52F50(v89);
        switch ( v89 )
        {
          case ' ':
            v65 = 0;
            LOBYTE(v49) = 1;
LABEL_131:
            v66 = v49;
            v67 = v65 != 0 && v49;
LABEL_124:
            if ( v67 )
              goto LABEL_63;
            if ( !v66 )
              goto LABEL_126;
            v63 = sub_AD8D80(v81, (__int64)&v94);
            v104 = 257;
            v19 = sub_BD2C40(72, unk_3F10FD0);
            if ( !v19 )
              goto LABEL_50;
            v64 = 39;
            break;
          case '!':
LABEL_161:
            v63 = sub_AD8D80(v81, (__int64)&v94);
            v104 = 257;
            v19 = sub_BD2C40(72, unk_3F10FD0);
            if ( !v19 )
              goto LABEL_50;
            v64 = 40;
            goto LABEL_118;
          case '"':
          case '&':
            goto LABEL_63;
          case '$':
          case '(':
            goto LABEL_99;
          default:
            goto LABEL_170;
        }
        goto LABEL_118;
      }
      v43 = *(_DWORD *)(a4 + 8);
      v44 = *(_QWORD *)a4;
      v45 = 1LL << ((unsigned __int8)v43 - 1);
      if ( v43 > 0x40 )
      {
        if ( (*(_QWORD *)(v44 + 8LL * ((v43 - 1) >> 6)) & v45) != 0 )
          goto LABEL_72;
        v73 = v43 == (unsigned int)sub_C444A0(a4);
      }
      else
      {
        if ( (v45 & v44) != 0 )
          goto LABEL_72;
        v73 = v44 == 0;
      }
      if ( !v73 )
      {
        sub_9865C0((__int64)&v100, (__int64)&v90);
        sub_C46A40((__int64)&v100, 1);
        v74 = v101;
        v101 = 0;
        v103 = v74;
        v102 = v100;
        sub_1110A30(&v96, (__int64 *)&v102);
        sub_969240((__int64 *)&v102);
        sub_969240((__int64 *)&v100);
        if ( v85 )
        {
          switch ( (unsigned int)sub_B52F50(v89) )
          {
            case ' ':
              v65 = -1;
LABEL_123:
              v66 = v65 != 0;
              v67 = v65 != 0;
              goto LABEL_124;
            case '!':
            case '"':
            case '&':
              goto LABEL_48;
            case '$':
            case '(':
              goto LABEL_63;
            default:
              goto LABEL_170;
          }
        }
        v87 = -sub_1111050(&v94, (__int64)&v96, &v92, 1);
        v89 = sub_B52F50(v89);
        v65 = v87;
        switch ( v89 )
        {
          case ' ':
            goto LABEL_130;
          case '!':
            goto LABEL_152;
          case '"':
          case '&':
            goto LABEL_101;
          case '$':
          case '(':
            goto LABEL_62;
          default:
            goto LABEL_170;
        }
      }
LABEL_72:
      if ( v95 <= 0x40 && v91 <= 0x40 )
      {
        v95 = v91;
        v94 = v90;
      }
      else
      {
        sub_C43990((__int64)&v94, (__int64)&v90);
      }
      if ( v85 )
      {
        switch ( (unsigned int)sub_B52F50(v89) )
        {
          case ' ':
            v65 = 1;
            goto LABEL_123;
          case '!':
          case '$':
          case '(':
            goto LABEL_48;
          case '"':
          case '&':
            goto LABEL_63;
          default:
            goto LABEL_170;
        }
      }
      v49 = sub_1110FE0(&v96, (__int64)&v90, &v92, 1);
      v89 = sub_B52F50(v89);
      switch ( v89 )
      {
        case ' ':
          v65 = 0;
          goto LABEL_131;
        case '!':
          if ( v49 )
            goto LABEL_161;
          goto LABEL_112;
        case '"':
        case '&':
LABEL_104:
          if ( v49 )
            goto LABEL_63;
          goto LABEL_101;
        case '$':
        case '(':
          goto LABEL_99;
        default:
          goto LABEL_170;
      }
    }
LABEL_95:
    switch ( v86 )
    {
      case ' ':
        goto LABEL_114;
      case '!':
        goto LABEL_112;
      case '"':
      case '&':
        goto LABEL_101;
      case '$':
      case '(':
        goto LABEL_99;
      default:
        goto LABEL_170;
    }
  }
  v48 = *(_DWORD *)(a4 + 8);
  if ( v48 <= 0x40 )
  {
    if ( *(_QWORD *)a4 )
    {
      if ( (*(_QWORD *)a4 & (1LL << ((unsigned __int8)v48 - 1))) == 0 )
        goto LABEL_84;
LABEL_136:
      sub_9865C0((__int64)&v100, (__int64)&v90);
      sub_C46A40((__int64)&v100, 1);
      v70 = v101;
      v101 = 0;
      v103 = v70;
      v102 = v100;
      sub_1110A30(&v96, (__int64 *)&v102);
      sub_969240((__int64 *)&v102);
      sub_969240((__int64 *)&v100);
      if ( v85 )
      {
        switch ( v86 )
        {
          case ' ':
          case '$':
          case '(':
            goto LABEL_63;
          case '!':
          case '"':
          case '&':
LABEL_48:
            v33 = a1;
            v34 = sub_ACD6D0(*(__int64 **)(*(_QWORD *)(a1 + 32) + 72LL));
            goto LABEL_49;
          default:
            goto LABEL_170;
        }
      }
      sub_9865C0((__int64)&v102, (__int64)&v92);
      if ( v103 > 0x40 )
      {
        sub_C43D10((__int64)&v102);
      }
      else
      {
        v75 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v103;
        if ( !v103 )
          v75 = 0;
        v102 = v75 & ~v102;
      }
      sub_C46250((__int64)&v102);
      v76 = v103;
      v103 = 0;
      v101 = v76;
      v100 = v102;
      sub_969240((__int64 *)&v102);
      v80 = -sub_1111050(&v94, (__int64)&v96, (__int64 *)&v100, 1);
      sub_969240((__int64 *)&v100);
      v65 = v80;
      switch ( v86 )
      {
        case ' ':
LABEL_126:
          if ( v65 )
          {
            v68 = sub_AD8D80(v81, (__int64)&v96);
            v104 = 257;
            v69 = sub_BD2C40(72, unk_3F10FD0);
            v19 = v69;
            if ( v69 )
              sub_1113300((__int64)v69, 40, v83, v68, (__int64)&v102);
            goto LABEL_50;
          }
LABEL_114:
          v61 = 1;
          v62 = v88;
LABEL_113:
          v33 = a1;
          v34 = (__int64)sub_10BE880(a1, v83, (__int64)&v94, (__int64)&v96, v62, v61);
          goto LABEL_49;
        case '!':
LABEL_152:
          v55 = v81;
          if ( v65 )
            goto LABEL_102;
LABEL_112:
          v61 = 0;
          v62 = v88;
          goto LABEL_113;
        case '"':
        case '&':
LABEL_101:
          v55 = v81;
          if ( v89 == 34 )
          {
            v71 = sub_AD8D80(v81, (__int64)&v96);
            v104 = 257;
            v72 = sub_BD2C40(72, unk_3F10FD0);
            v19 = v72;
            if ( v72 )
              sub_1113300((__int64)v72, 35, v83, v71, (__int64)&v102);
          }
          else
          {
LABEL_102:
            v56 = sub_AD8D80(v55, (__int64)&v96);
            v104 = 257;
            v57 = sub_BD2C40(72, unk_3F10FD0);
            v19 = v57;
            if ( v57 )
              sub_1113300((__int64)v57, 39, v83, v56, (__int64)&v102);
          }
          goto LABEL_50;
        case '$':
        case '(':
LABEL_62:
          if ( v65 == -1 )
          {
LABEL_63:
            v33 = a1;
            v34 = sub_ACD720(*(__int64 **)(*(_QWORD *)(a1 + 32) + 72LL));
LABEL_49:
            v19 = sub_F162A0(v33, a2, v34);
          }
          else
          {
LABEL_99:
            v53 = sub_AD8D80(v81, (__int64)&v94);
            v104 = 257;
            v54 = sub_BD2C40(72, unk_3F10FD0);
            v19 = v54;
            if ( v54 )
              sub_1113300((__int64)v54, v89, v83, v53, (__int64)&v102);
          }
          break;
        default:
LABEL_170:
          BUG();
      }
      goto LABEL_50;
    }
LABEL_88:
    sub_9865C0((__int64)&v98, (__int64)&v92);
    sub_C46F20((__int64)&v98, 1u);
    v50 = v99;
    v99 = 0;
    v101 = v50;
    v100 = v98;
    if ( v50 > 0x40 )
    {
      sub_C43D10((__int64)&v100);
    }
    else
    {
      v51 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v50) & ~v98;
      if ( !v50 )
        v51 = 0;
      v100 = v51;
    }
    sub_C46250((__int64)&v100);
    v52 = v101;
    v101 = 0;
    v103 = v52;
    v102 = v100;
    sub_1110A30(&v94, (__int64 *)&v102);
    sub_969240((__int64 *)&v102);
    sub_969240((__int64 *)&v100);
    sub_969240((__int64 *)&v98);
    if ( v97 <= 0x40 && v93 <= 0x40 )
    {
      v97 = v93;
      v96 = v92;
    }
    else
    {
      sub_C43990((__int64)&v96, (__int64)&v92);
    }
    goto LABEL_95;
  }
  if ( v48 == (unsigned int)sub_C444A0(a4) )
    goto LABEL_88;
  if ( (*(_QWORD *)(*(_QWORD *)a4 + 8LL * ((v48 - 1) >> 6)) & (1LL << ((unsigned __int8)v48 - 1))) != 0 )
    goto LABEL_136;
LABEL_84:
  if ( v91 <= 0x40 )
  {
    v95 = v91;
    v94 = v90;
  }
  else
  {
    sub_C43990((__int64)&v94, (__int64)&v90);
  }
  if ( v85 )
    goto LABEL_26;
  v49 = sub_1111050(&v96, (__int64)&v90, &v92, 1);
LABEL_97:
  switch ( v86 )
  {
    case ' ':
      if ( !v49 )
        goto LABEL_114;
      v63 = sub_AD8D80(v81, (__int64)&v94);
      v104 = 257;
      v19 = sub_BD2C40(72, unk_3F10FD0);
      if ( !v19 )
        goto LABEL_50;
      v64 = 4 * (v8 == 49) + 35;
      break;
    case '!':
      if ( !v49 )
        goto LABEL_112;
      v63 = sub_AD8D80(v81, (__int64)&v94);
      v104 = 257;
      v19 = sub_BD2C40(72, unk_3F10FD0);
      if ( !v19 )
        goto LABEL_50;
      v64 = 4 * (v8 == 49) + 36;
      break;
    case '"':
    case '&':
      goto LABEL_104;
    case '$':
    case '(':
      goto LABEL_99;
    default:
      goto LABEL_170;
  }
LABEL_118:
  sub_1113300((__int64)v19, v64, v83, v63, (__int64)&v102);
LABEL_50:
  if ( v97 > 0x40 && v96 )
    j_j___libc_free_0_0(v96);
  if ( v95 > 0x40 && v94 )
    j_j___libc_free_0_0(v94);
  if ( v93 > 0x40 && v92 )
    j_j___libc_free_0_0(v92);
  if ( v91 > 0x40 && v90 )
    j_j___libc_free_0_0(v90);
  return (__int64)v19;
}
