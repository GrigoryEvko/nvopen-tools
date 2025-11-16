// Function: sub_14C3BA0
// Address: 0x14c3ba0
//
__int64 __fastcall sub_14C3BA0(__int64 a1)
{
  __int64 v1; // r15
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r13
  int v6; // r12d
  unsigned __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // r12
  _QWORD *v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rax
  _QWORD *v16; // r14
  __int64 v17; // rcx
  __int64 v18; // r15
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // r9
  unsigned __int64 v23; // rdi
  signed __int64 v24; // rax
  char v25; // dl
  __int64 v26; // r15
  unsigned int v27; // eax
  unsigned __int64 v28; // r9
  unsigned __int64 v29; // r12
  __int64 v30; // rsi
  unsigned __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v35; // rax
  __int64 v36; // r14
  unsigned int v37; // eax
  __int64 v38; // r10
  unsigned __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rsi
  int v42; // eax
  __int64 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // r14
  unsigned int v47; // eax
  _QWORD *v48; // rax
  unsigned int v49; // r12d
  bool v50; // al
  __int64 v51; // rax
  unsigned int v52; // r12d
  unsigned int v53; // r14d
  __int64 v54; // rax
  char v55; // dl
  unsigned int v56; // r12d
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rsi
  int v61; // eax
  __int64 v62; // rax
  unsigned __int64 v63; // rax
  _QWORD *v64; // rax
  __int64 v65; // [rsp+0h] [rbp-80h]
  __int64 v66; // [rsp+8h] [rbp-78h]
  __int64 v67; // [rsp+8h] [rbp-78h]
  __int64 v68; // [rsp+10h] [rbp-70h]
  __int64 v69; // [rsp+10h] [rbp-70h]
  unsigned __int64 v70; // [rsp+18h] [rbp-68h]
  unsigned __int64 v71; // [rsp+18h] [rbp-68h]
  unsigned __int64 v72; // [rsp+18h] [rbp-68h]
  __int64 v73; // [rsp+20h] [rbp-60h]
  __int64 v74; // [rsp+20h] [rbp-60h]
  __int64 v75; // [rsp+20h] [rbp-60h]
  unsigned __int64 v76; // [rsp+28h] [rbp-58h]
  __int64 v77; // [rsp+28h] [rbp-58h]
  __int64 v78; // [rsp+30h] [rbp-50h]
  unsigned __int64 v79; // [rsp+38h] [rbp-48h]
  __int64 v80; // [rsp+38h] [rbp-48h]
  __int64 v81; // [rsp+38h] [rbp-48h]
  int v82; // [rsp+38h] [rbp-48h]
  unsigned int v83; // [rsp+40h] [rbp-40h]
  __int64 v84; // [rsp+40h] [rbp-40h]
  unsigned int v85; // [rsp+4Ch] [rbp-34h]

  v1 = 1;
  v3 = sub_15F2050(a1);
  v4 = sub_1632FA0(v3);
  v5 = *(_QWORD *)(a1 + 64);
  v78 = v4;
  v6 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v85 = v6 - 1;
  v7 = (unsigned int)sub_15A9FE0(v4, v5);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v5 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v32 = *(_QWORD *)(v5 + 32);
        v5 = *(_QWORD *)(v5 + 24);
        v1 *= v32;
        continue;
      case 1:
        v8 = 16;
        break;
      case 2:
        v8 = 32;
        break;
      case 3:
      case 9:
        v8 = 64;
        break;
      case 4:
        v8 = 80;
        break;
      case 5:
      case 6:
        v8 = 128;
        break;
      case 7:
        v8 = 8 * (unsigned int)sub_15A9520(v78, 0);
        break;
      case 0xB:
        v8 = *(_DWORD *)(v5 + 8) >> 8;
        break;
      case 0xD:
        v8 = 8LL * *(_QWORD *)sub_15A9930(v78, v5);
        break;
      case 0xE:
        v80 = *(_QWORD *)(v5 + 32);
        v30 = *(_QWORD *)(v5 + 24);
        v84 = 1;
        v31 = (unsigned int)sub_15A9FE0(v78, v30);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v30 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v59 = v84 * *(_QWORD *)(v30 + 32);
              v30 = *(_QWORD *)(v30 + 24);
              v84 = v59;
              continue;
            case 1:
              v58 = 16;
              break;
            case 2:
              v58 = 32;
              break;
            case 3:
            case 9:
              v58 = 64;
              break;
            case 4:
              v58 = 80;
              break;
            case 5:
            case 6:
              v58 = 128;
              break;
            case 7:
              v58 = 8 * (unsigned int)sub_15A9520(v78, 0);
              break;
            case 0xB:
              v58 = *(_DWORD *)(v30 + 8) >> 8;
              break;
            case 0xD:
              v58 = 8LL * *(_QWORD *)sub_15A9930(v78, v30);
              break;
            case 0xE:
              v77 = *(_QWORD *)(v30 + 32);
              v58 = 8 * v77 * sub_12BE0A0(v78, *(_QWORD *)(v30 + 24));
              break;
            case 0xF:
              v58 = 8 * (unsigned int)sub_15A9520(v78, *(_DWORD *)(v30 + 8) >> 8);
              break;
          }
          break;
        }
        v8 = 8 * v31 * v80 * ((v31 + ((unsigned __int64)(v84 * v58 + 7) >> 3) - 1) / v31);
        break;
      case 0xF:
        v8 = 8 * (unsigned int)sub_15A9520(v78, *(_DWORD *)(v5 + 8) >> 8);
        break;
    }
    break;
  }
  v76 = v7 * ((v7 + ((unsigned __int64)(v8 * v1 + 7) >> 3) - 1) / v7);
  if ( v85 <= 1 )
    return v85;
  v9 = v85;
  v83 = v6 - 3;
LABEL_6:
  v85 = v9;
  v10 = v9 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v11 = *(_QWORD *)(a1 + 24 * v10);
  if ( *(_BYTE *)(v11 + 16) > 0x10u )
    return v85;
  if ( (unsigned __int8)sub_1593BB0(*(_QWORD *)(a1 + 24 * v10)) )
  {
LABEL_8:
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v12 = *(_QWORD *)(a1 - 8);
    else
      v12 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v13 = (_QWORD *)(v12 + 24);
    v14 = sub_16348C0(a1);
    v15 = v14 | 4;
    if ( !v83 )
    {
      v21 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      v22 = v21;
      LOBYTE(v19) = (v15 >> 2) & 1;
      goto LABEL_21;
    }
    v16 = v13;
    v17 = v14 | 4;
    v18 = v83;
    while ( 1 )
    {
      while ( 1 )
      {
        v23 = v17 & 0xFFFFFFFFFFFFFFF8LL;
        v24 = v17 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v17 & 4) == 0 || !v23 )
          v24 = sub_1643D30(v23, *v16);
        v25 = *(_BYTE *)(v24 + 8);
        if ( ((v25 - 14) & 0xFD) != 0 )
          break;
        LOBYTE(v19) = 1;
        v20 = *(_QWORD *)(v24 + 24);
        v21 = v20 & 0xFFFFFFFFFFFFFFF8LL;
        v17 = v20 | 4;
        v22 = v21;
LABEL_14:
        v16 += 3;
        if ( !--v18 )
          goto LABEL_20;
      }
      if ( v25 != 13 )
      {
        v22 = 0;
        v21 = 0;
        LOBYTE(v19) = 0;
        v17 = 0;
        goto LABEL_14;
      }
      v17 = v24;
      v16 += 3;
      v21 = v24 & 0xFFFFFFFFFFFFFFF8LL;
      v19 = (v24 >> 2) & 1;
      v22 = v24 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !--v18 )
      {
LABEL_20:
        v13 += 3 * v83;
LABEL_21:
        if ( !(_BYTE)v19 || !v21 )
          v22 = sub_1643D30(v22, *v13);
        v79 = v22;
        v26 = 1;
        v27 = sub_15A9FE0(v78, v22);
        v28 = v79;
        v29 = v27;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v28 + 8) )
          {
            case 1:
              v33 = 16;
              goto LABEL_39;
            case 2:
              v33 = 32;
              goto LABEL_39;
            case 3:
            case 9:
              v33 = 64;
              goto LABEL_39;
            case 4:
              v33 = 80;
              goto LABEL_39;
            case 5:
            case 6:
              v33 = 128;
              goto LABEL_39;
            case 7:
              v33 = 8 * (unsigned int)sub_15A9520(v78, 0);
              goto LABEL_39;
            case 0xB:
              v33 = *(_DWORD *)(v28 + 8) >> 8;
              goto LABEL_39;
            case 0xD:
              v33 = 8LL * *(_QWORD *)sub_15A9930(v78, v28);
              goto LABEL_39;
            case 0xE:
              v36 = *(_QWORD *)(v28 + 24);
              v81 = *(_QWORD *)(v28 + 32);
              v37 = sub_15A9FE0(v78, v36);
              v38 = 1;
              v39 = v37;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v36 + 8) )
                {
                  case 1:
                    v40 = 16;
                    goto LABEL_54;
                  case 2:
                    v40 = 32;
                    goto LABEL_54;
                  case 3:
                  case 9:
                    v40 = 64;
                    goto LABEL_54;
                  case 4:
                    v40 = 80;
                    goto LABEL_54;
                  case 5:
                  case 6:
                    v40 = 128;
                    goto LABEL_54;
                  case 7:
                    v70 = v39;
                    v41 = 0;
                    v73 = v38;
                    goto LABEL_57;
                  case 0xB:
                    v40 = *(_DWORD *)(v36 + 8) >> 8;
                    goto LABEL_54;
                  case 0xD:
                    v72 = v39;
                    v75 = v38;
                    v48 = (_QWORD *)sub_15A9930(v78, v36);
                    v38 = v75;
                    v39 = v72;
                    v40 = 8LL * *v48;
                    goto LABEL_54;
                  case 0xE:
                    v44 = *(_QWORD *)(v36 + 24);
                    v45 = *(_QWORD *)(v36 + 32);
                    v65 = v39;
                    v46 = 1;
                    v66 = v38;
                    v74 = v45;
                    v47 = sub_15A9FE0(v78, v44);
                    v39 = v65;
                    v38 = v66;
                    v71 = v47;
                    while ( 2 )
                    {
                      switch ( *(_BYTE *)(v44 + 8) )
                      {
                        case 1:
                          v57 = 16;
                          goto LABEL_91;
                        case 2:
                          v57 = 32;
                          goto LABEL_91;
                        case 3:
                        case 9:
                          v57 = 64;
                          goto LABEL_91;
                        case 4:
                          v57 = 80;
                          goto LABEL_91;
                        case 5:
                        case 6:
                          v57 = 128;
                          goto LABEL_91;
                        case 7:
                          v67 = v65;
                          v60 = 0;
                          v68 = v38;
                          goto LABEL_103;
                        case 0xB:
                          v57 = *(_DWORD *)(v44 + 8) >> 8;
                          goto LABEL_91;
                        case 0xD:
                          v64 = (_QWORD *)sub_15A9930(v78, v44);
                          v38 = v66;
                          v39 = v65;
                          v57 = 8LL * *v64;
                          goto LABEL_91;
                        case 0xE:
                          v69 = *(_QWORD *)(v44 + 32);
                          v63 = sub_12BE0A0(v78, *(_QWORD *)(v44 + 24));
                          v38 = v66;
                          v39 = v65;
                          v57 = 8 * v69 * v63;
                          goto LABEL_91;
                        case 0xF:
                          v67 = v65;
                          v68 = v38;
                          v60 = *(_DWORD *)(v44 + 8) >> 8;
LABEL_103:
                          v61 = sub_15A9520(v78, v60);
                          v38 = v68;
                          v39 = v67;
                          v57 = (unsigned int)(8 * v61);
LABEL_91:
                          v40 = 8 * v71 * v74 * ((v71 + ((unsigned __int64)(v46 * v57 + 7) >> 3) - 1) / v71);
                          goto LABEL_54;
                        case 0x10:
                          v62 = *(_QWORD *)(v44 + 32);
                          v44 = *(_QWORD *)(v44 + 24);
                          v46 *= v62;
                          continue;
                        default:
                          goto LABEL_12;
                      }
                    }
                  case 0xF:
                    v70 = v39;
                    v73 = v38;
                    v41 = *(_DWORD *)(v36 + 8) >> 8;
LABEL_57:
                    v42 = sub_15A9520(v78, v41);
                    v38 = v73;
                    v39 = v70;
                    v40 = (unsigned int)(8 * v42);
LABEL_54:
                    v33 = 8 * v39 * v81 * ((v39 + ((unsigned __int64)(v40 * v38 + 7) >> 3) - 1) / v39);
                    goto LABEL_39;
                  case 0x10:
                    v43 = *(_QWORD *)(v36 + 32);
                    v36 = *(_QWORD *)(v36 + 24);
                    v38 *= v43;
                    continue;
                  default:
                    goto LABEL_12;
                }
              }
            case 0xF:
              v33 = 8 * (unsigned int)sub_15A9520(v78, *(_DWORD *)(v28 + 8) >> 8);
LABEL_39:
              if ( (unsigned int)v76 != v29 * ((v29 + ((unsigned __int64)(v33 * v26 + 7) >> 3) - 1) / v29) )
                return v85;
              --v9;
              --v85;
              --v83;
              if ( v9 == 1 )
                return v85;
              goto LABEL_6;
            case 0x10:
              v35 = *(_QWORD *)(v28 + 32);
              v28 = *(_QWORD *)(v28 + 24);
              v26 *= v35;
              continue;
            default:
LABEL_12:
              BUG();
          }
        }
      }
    }
  }
  if ( *(_BYTE *)(v11 + 16) == 13 )
  {
    v49 = *(_DWORD *)(v11 + 32);
    if ( v49 <= 0x40 )
      v50 = *(_QWORD *)(v11 + 24) == 0;
    else
      v50 = v49 == (unsigned int)sub_16A57B0(v11 + 24);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 16 )
      return v85;
    v51 = sub_15A1020(v11);
    if ( !v51 || *(_BYTE *)(v51 + 16) != 13 )
    {
      v82 = *(_QWORD *)(*(_QWORD *)v11 + 32LL);
      if ( !v82 )
        goto LABEL_8;
      v53 = 0;
      while ( 1 )
      {
        v54 = sub_15A0A60(v11, v53);
        if ( !v54 )
          return v85;
        v55 = *(_BYTE *)(v54 + 16);
        if ( v55 != 9 )
        {
          if ( v55 != 13 )
            return v85;
          v56 = *(_DWORD *)(v54 + 32);
          if ( v56 <= 0x40 )
          {
            if ( *(_QWORD *)(v54 + 24) )
              return v85;
          }
          else if ( v56 != (unsigned int)sub_16A57B0(v54 + 24) )
          {
            return v85;
          }
        }
        if ( v82 == ++v53 )
          goto LABEL_8;
      }
    }
    v52 = *(_DWORD *)(v51 + 32);
    if ( v52 <= 0x40 )
      v50 = *(_QWORD *)(v51 + 24) == 0;
    else
      v50 = v52 == (unsigned int)sub_16A57B0(v51 + 24);
  }
  if ( v50 )
    goto LABEL_8;
  return v85;
}
