// Function: sub_1F41BE0
// Address: 0x1f41be0
//
__int64 __fastcall sub_1F41BE0(__int64 a1, __int64 a2)
{
  __int64 i; // rax
  bool v4; // zf
  unsigned int v5; // esi
  unsigned int v6; // ecx
  __int64 v7; // rdx
  __int64 j; // rax
  _BYTE *v9; // rdx
  char v10; // di
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // ebx
  unsigned __int16 *v15; // r13
  _BYTE *v16; // r15
  char v17; // r14
  __int64 (__fastcall *v18)(__int64, __int64, __int64); // rax
  signed int v19; // r13d
  char v20; // al
  unsigned int v21; // ebx
  signed int v22; // eax
  char v23; // dl
  __int64 k; // rbx
  __int64 result; // rax
  char v26; // dl
  unsigned __int8 v27; // r12
  unsigned int v28; // r14d
  int v29; // ebx
  unsigned __int8 v30; // al
  __int64 v31; // rsi
  unsigned __int8 v32; // al
  unsigned __int8 v33; // dl
  int v34; // r14d
  unsigned int v35; // eax
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rdx
  unsigned __int64 v39; // rcx
  char v40; // r14
  unsigned int v41; // eax
  __int64 v42; // rsi
  char v43; // al
  int v44; // eax
  char v45; // di
  char v46; // al
  char v47; // si
  char v48; // si
  char v49; // cl
  char v50; // al
  char v51; // al
  char v52; // al
  unsigned int v54; // [rsp+8h] [rbp-98h]
  unsigned __int8 v55; // [rsp+8h] [rbp-98h]
  unsigned int v56; // [rsp+8h] [rbp-98h]
  unsigned __int16 *v57; // [rsp+10h] [rbp-90h]
  __int64 v58; // [rsp+18h] [rbp-88h]
  unsigned int v59; // [rsp+20h] [rbp-80h]
  char v60; // [rsp+26h] [rbp-7Ah]
  char v61; // [rsp+26h] [rbp-7Ah]
  char v62; // [rsp+27h] [rbp-79h]
  unsigned __int16 v63; // [rsp+28h] [rbp-78h]
  int v64; // [rsp+28h] [rbp-78h]
  unsigned int v65; // [rsp+2Ch] [rbp-74h]
  __int64 v66; // [rsp+30h] [rbp-70h]
  __int64 v67; // [rsp+38h] [rbp-68h]
  char v68; // [rsp+4Dh] [rbp-53h] BYREF
  unsigned __int8 v69; // [rsp+4Eh] [rbp-52h] BYREF
  char v70; // [rsp+4Fh] [rbp-51h] BYREF
  char v71[8]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v72; // [rsp+58h] [rbp-48h]
  char v73[8]; // [rsp+60h] [rbp-40h] BYREF
  __int64 v74; // [rsp+68h] [rbp-38h]

  for ( i = 0; i != 115; ++i )
  {
    *(_BYTE *)(a1 + i + 1040) = 1;
    *(_BYTE *)(a1 + i + 2307) = i;
    *(_BYTE *)(a1 + i + 1155) = i;
  }
  v4 = *(_QWORD *)(a1 + 176) == 0;
  *(_BYTE *)(a1 + 1152) = 0;
  if ( v4 )
  {
    v5 = 7;
    do
    {
      v7 = v5--;
      v6 = v5;
      LODWORD(j) = v7 - 2;
    }
    while ( !*(_QWORD *)(a1 + 8LL * v5 + 120) );
    v9 = (_BYTE *)(a1 + v7);
    do
    {
      ++v9;
      v10 = *(_BYTE *)(a1 + v6 + 1040);
      v9[2306] = v6++;
      v9[1154] = v5;
      v9[1039] = 2 * v10;
      *(_BYTE *)(a1 + (int)v6 + 73900) = 2;
    }
    while ( v6 != 7 );
    if ( (unsigned int)j <= 1 )
      goto LABEL_14;
  }
  else
  {
    LODWORD(j) = 6;
    LOBYTE(v5) = 7;
  }
  for ( j = (unsigned int)j; j != 1; --j )
  {
    while ( 1 )
    {
      v11 = (unsigned __int8)j;
      if ( !(_BYTE)j || !*(_QWORD *)(a1 + 8LL * (unsigned __int8)j + 120) )
        break;
      LOBYTE(v5) = j--;
      if ( j == 1 )
        goto LABEL_14;
    }
    *(_BYTE *)(a1 + j + 2307) = v5;
    *(_BYTE *)(a1 + j + 1155) = v5;
    *(_BYTE *)(a1 + v11 + 73900) = 1;
  }
LABEL_14:
  v12 = *(_QWORD *)(a1 + 200);
  v13 = *(_QWORD *)(a1 + 216);
  if ( !*(_QWORD *)(a1 + 224) )
  {
    if ( !v12 )
    {
      v48 = *(_BYTE *)(a1 + 1047);
      v49 = *(_BYTE *)(a1 + 1162);
      *(_BYTE *)(a1 + 2320) = 7;
      *(_BYTE *)(a1 + 73913) = 3;
      *(_BYTE *)(a1 + 1053) = v48;
      *(_BYTE *)(a1 + 1168) = v49;
      if ( v13 )
        goto LABEL_128;
      goto LABEL_127;
    }
    v47 = *(_BYTE *)(a1 + 1050);
    *(_BYTE *)(a1 + 1168) = 10;
    *(_BYTE *)(a1 + 2320) = 10;
    *(_BYTE *)(a1 + 73913) = 4;
    *(_BYTE *)(a1 + 1053) = 2 * v47;
    if ( v13 )
      goto LABEL_17;
LABEL_126:
    v48 = *(_BYTE *)(a1 + 1047);
    v49 = *(_BYTE *)(a1 + 1162);
LABEL_127:
    *(_BYTE *)(a1 + 1052) = v48;
    *(_BYTE *)(a1 + 1167) = v49;
    *(_BYTE *)(a1 + 2319) = 7;
    *(_BYTE *)(a1 + 73912) = 3;
    if ( !v12 )
      goto LABEL_128;
LABEL_17:
    if ( *(_QWORD *)(a1 + 192) )
      goto LABEL_18;
LABEL_129:
    v51 = *(_BYTE *)(a1 + 1045);
    v4 = *(_QWORD *)(a1 + 184) == 0;
    *(_BYTE *)(a1 + 2316) = 5;
    *(_BYTE *)(a1 + 73909) = 3;
    *(_BYTE *)(a1 + 1049) = v51;
    *(_BYTE *)(a1 + 1164) = *(_BYTE *)(a1 + 1160);
    if ( !v4 )
      goto LABEL_19;
    goto LABEL_130;
  }
  if ( !v13 )
    goto LABEL_126;
  if ( v12 )
    goto LABEL_17;
LABEL_128:
  v50 = *(_BYTE *)(a1 + 1046);
  v4 = *(_QWORD *)(a1 + 192) == 0;
  *(_BYTE *)(a1 + 2317) = 6;
  *(_BYTE *)(a1 + 73910) = 3;
  *(_BYTE *)(a1 + 1050) = v50;
  *(_BYTE *)(a1 + 1165) = *(_BYTE *)(a1 + 1161);
  if ( v4 )
    goto LABEL_129;
LABEL_18:
  if ( *(_QWORD *)(a1 + 184) )
    goto LABEL_19;
LABEL_130:
  v52 = *(_BYTE *)(a1 + 1049);
  *(_BYTE *)(a1 + 2315) = 9;
  *(_BYTE *)(a1 + 73908) = 8;
  *(_BYTE *)(a1 + 1048) = v52;
  *(_BYTE *)(a1 + 1163) = *(_BYTE *)(a1 + 1164);
LABEL_19:
  v66 = a1;
  v14 = 15;
  v15 = word_42F2F80;
  v67 = -7 * a1;
  v16 = (_BYTE *)(a1 + 1054);
LABEL_21:
  while ( 2 )
  {
    if ( *(_QWORD *)(v67 + 8LL * (_QWORD)v16 - 8200) )
      goto LABEL_20;
    v62 = v14 - 1;
    v63 = *v15;
    v65 = *v15;
    switch ( (char)v14 )
    {
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 33:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
      case 68:
        v17 = 3;
        goto LABEL_24;
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 41:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
      case 74:
        v17 = 4;
        goto LABEL_24;
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 49:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
      case 80:
        v17 = 5;
        goto LABEL_24;
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 55:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
      case 86:
        v17 = 6;
        goto LABEL_24;
      case 56:
        v68 = 7;
        v17 = 7;
        v18 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v66 + 64LL);
        if ( v18 == sub_1F3D320 )
          goto LABEL_96;
        goto LABEL_103;
      case 87:
      case 88:
      case 89:
      case 99:
      case 100:
      case 101:
        v17 = 8;
        goto LABEL_24;
      case 90:
      case 91:
      case 92:
      case 93:
      case 94:
      case 102:
      case 103:
      case 104:
      case 105:
      case 106:
        v17 = 9;
        goto LABEL_24;
      case 95:
      case 96:
      case 97:
      case 98:
      case 107:
      case 108:
      case 109:
      case 110:
        v17 = 10;
        goto LABEL_24;
      default:
        v17 = 2;
LABEL_24:
        v68 = v17;
        v18 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v66 + 64LL);
        if ( v18 == sub_1F3D320 )
        {
          if ( v65 == 1 )
          {
LABEL_96:
            v61 = 5;
            goto LABEL_48;
          }
          if ( v14 > 0x55 )
            goto LABEL_108;
        }
        else
        {
LABEL_103:
          v42 = v58;
          LOBYTE(v42) = v14 - 1;
          v43 = v18(v66, v42, 0);
          v61 = v43;
          if ( v43 == 7 )
            goto LABEL_104;
          if ( v43 != 1 )
            goto LABEL_48;
          if ( v14 > 0x55 )
          {
LABEL_108:
            v61 = 1;
LABEL_104:
            if ( v14 != 110 )
            {
LABEL_34:
              v22 = v14;
              while ( 1 )
              {
                switch ( (char)v22 )
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
                    v23 = 3;
                    goto LABEL_46;
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
                    if ( v17 == 4 )
                      goto LABEL_50;
                    goto LABEL_47;
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
                    v23 = 5;
                    goto LABEL_46;
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
                    v23 = 6;
                    goto LABEL_46;
                  case 55:
                    v23 = 7;
                    goto LABEL_46;
                  case 86:
                  case 87:
                  case 88:
                  case 98:
                  case 99:
                  case 100:
                    v23 = 8;
                    goto LABEL_46;
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
                    v23 = 9;
                    goto LABEL_46;
                  case 94:
                  case 95:
                  case 96:
                  case 97:
                  case 106:
                  case 107:
                  case 108:
                  case 109:
                    v23 = 10;
                    goto LABEL_46;
                  default:
                    v23 = 2;
LABEL_46:
                    if ( v23 != v17 )
                      goto LABEL_47;
LABEL_50:
                    if ( word_42F2F80[(unsigned __int8)(v22 - 14)] > v63 && *(_QWORD *)(v66 + 8LL * v22 + 120) )
                    {
                      ++v14;
                      v16[115] = v22;
                      ++v15;
                      (++v16)[1266] = v22;
                      *(v16 - 1) = 1;
                      v16[72859] = 7;
                      if ( v14 == 111 )
                        goto LABEL_53;
                      goto LABEL_21;
                    }
LABEL_47:
                    if ( ++v22 == 110 )
                      goto LABEL_48;
                    break;
                }
              }
            }
LABEL_48:
            switch ( v62 )
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
                v27 = 3;
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
                v27 = 4;
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
                v27 = 5;
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
                v27 = 6;
                break;
              case 55:
                v27 = 7;
                break;
              case 86:
              case 87:
              case 88:
              case 98:
              case 99:
              case 100:
                v27 = 8;
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
                v27 = 9;
                break;
              case 94:
              case 95:
              case 96:
              case 97:
              case 106:
              case 107:
              case 108:
              case 109:
                v27 = 10;
                break;
              default:
                v27 = 2;
                break;
            }
            if ( v65 )
            {
              if ( ((v65 - 1) & v65) != 0 )
              {
                v28 = 1;
                v64 = v65;
              }
              else if ( v65 == 1 )
              {
                v64 = 1;
                v28 = 1;
              }
              else
              {
                v28 = v65;
                v54 = v14;
                v29 = 1;
                do
                {
                  v30 = sub_1D15020(v27, v28);
                  if ( v30 && *(_QWORD *)(v66 + 8LL * v30 + 120) )
                    break;
                  v28 >>= 1;
                  v29 *= 2;
                }
                while ( v28 != 1 );
                v64 = v29;
                v14 = v54;
              }
            }
            else
            {
              v64 = 0;
              v28 = 1;
            }
            v31 = v28;
            v32 = sub_1D15020(v27, v28);
            v69 = v32;
            v33 = v32;
            if ( !v32 || (v31 = v66, v34 = v32, !*(_QWORD *)(v66 + 8LL * v32 + 120)) )
            {
              v69 = v27;
              v34 = v27;
              v33 = v27;
            }
            v55 = v33;
            v35 = sub_1F3E310(&v69);
            v38 = v55;
            v39 = v35;
            if ( !v35 || (v35 & (v35 - 1)) != 0 )
              v39 = (((((((((v35 | ((unsigned __int64)v35 >> 1)) >> 2) | v35 | ((unsigned __int64)v35 >> 1)) >> 4)
                       | ((v35 | ((unsigned __int64)v35 >> 1)) >> 2)
                       | v35
                       | ((unsigned __int64)v35 >> 1)) >> 8)
                     | ((((v35 | ((unsigned __int64)v35 >> 1)) >> 2) | v35 | ((unsigned __int64)v35 >> 1)) >> 4)
                     | ((v35 | ((unsigned __int64)v35 >> 1)) >> 2)
                     | v35
                     | ((unsigned __int64)v35 >> 1)) >> 16)
                   | ((((((v35 | ((unsigned __int64)v35 >> 1)) >> 2) | v35 | ((unsigned __int64)v35 >> 1)) >> 4)
                     | ((v35 | ((unsigned __int64)v35 >> 1)) >> 2)
                     | v35
                     | ((unsigned __int64)v35 >> 1)) >> 8)
                   | ((((v35 | ((unsigned __int64)v35 >> 1)) >> 2) | v35 | ((unsigned __int64)v35 >> 1)) >> 4)
                   | ((v35 | ((unsigned __int64)v35 >> 1)) >> 2)
                   | v35
                   | (v35 >> 1))
                  + 1;
            v73[0] = v55;
            v72 = 0;
            v40 = *(_BYTE *)(v66 + v34 + 1155);
            v74 = 0;
            v70 = v40;
            v71[0] = v40;
            if ( v40 != v55 )
            {
              v56 = v39;
              v41 = v40 ? sub_1F3E310(v71) : sub_1F58D40(v71, v31, v38, v39, v36, v37);
              if ( (unsigned int)sub_1F3E310(v73) > v41 )
                v64 *= v56 / (unsigned int)sub_1F3E310(&v70);
            }
            v16[115] = v40;
            *v16 = v64;
            if ( ((v65 - 1) & v65) == 0 )
              goto LABEL_81;
            _BitScanReverse((unsigned int *)&v44, v65 - 1);
            switch ( v62 )
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
                v45 = 3;
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
                v45 = 4;
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
                v45 = 5;
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
                v45 = 6;
                break;
              case 55:
                v45 = 7;
                break;
              case 86:
              case 87:
              case 88:
              case 98:
              case 99:
              case 100:
                v45 = 8;
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
                v45 = 9;
                break;
              case 94:
              case 95:
              case 96:
              case 97:
              case 106:
              case 107:
              case 108:
              case 109:
                v45 = 10;
                break;
              default:
                v45 = 2;
                break;
            }
            v46 = sub_1D15020(v45, 1 << (32 - (v44 ^ 0x1F)));
            if ( v62 == v46 )
            {
LABEL_81:
              v16[1267] = 1;
              if ( v61 == 5 )
              {
                v16[72860] = 5;
              }
              else if ( v61 == 6 )
              {
                v16[72860] = 6;
              }
              else
              {
                v16[72860] = (v65 != 1) + 5;
              }
            }
            else
            {
              v16[1267] = v46;
              v16[72860] = 7;
            }
            goto LABEL_20;
          }
        }
        v59 = v14;
        v57 = v15;
        v19 = v14;
        while ( 1 )
        {
          v60 = v19;
          switch ( (char)v19 )
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
              v20 = 3;
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
              v20 = 4;
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
              v20 = 5;
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
              v20 = 6;
              break;
            case 55:
              v20 = 7;
              break;
            default:
              v20 = 2;
              break;
          }
          v73[0] = v20;
          v21 = sub_1F3E310(v73);
          if ( v21 > (unsigned int)sub_1F3E310(&v68)
            && word_42F2F80[(unsigned __int8)(v19 - 14)] == v63
            && *(_QWORD *)(v66 + 8LL * v19 + 120) )
          {
            break;
          }
          if ( ++v19 == 86 )
          {
            v14 = v59;
            v61 = 1;
            v15 = v57;
            goto LABEL_34;
          }
        }
        v15 = v57;
        v14 = v59;
        v16[1267] = v60;
        v16[115] = v60;
        *v16 = 1;
        v16[72860] = 1;
LABEL_20:
        ++v14;
        ++v16;
        ++v15;
        if ( v14 != 111 )
          continue;
LABEL_53:
        for ( k = 0; k != 115; ++k )
        {
          result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v66 + 720LL))(
                     v66,
                     a2,
                     (unsigned int)k);
          *(_QWORD *)(v66 + 8 * k + 1272) = result;
          *(_BYTE *)(v66 + k + 2192) = v26;
        }
        return result;
    }
  }
}
