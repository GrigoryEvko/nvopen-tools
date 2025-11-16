// Function: sub_39F6040
// Address: 0x39f6040
//
__int64 __fastcall sub_39F6040(char *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // ebx
  int v7; // esi
  char *v8; // r8
  char v9; // dl
  __int64 v10; // rdx
  unsigned int v11; // ecx
  char v12; // di
  unsigned __int64 v13; // rax
  int v14; // esi
  _QWORD *v15; // r9
  unsigned __int64 v16; // r9
  int v17; // esi
  __int64 result; // rax
  int v19; // esi
  __int64 v20; // r9
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdi
  char *v33; // rax
  unsigned __int8 v34; // al
  int v35; // ecx
  char v36; // dl
  unsigned __int64 v37; // rax
  unsigned int v38; // ecx
  char v39; // dl
  unsigned __int64 v40; // rax
  int v41; // esi
  int v42; // ecx
  char v43; // dl
  unsigned __int64 v44; // rax
  __int64 v45; // r8
  unsigned int v46; // ecx
  char v47; // dl
  unsigned __int64 v48; // rax
  _QWORD *v49; // r9
  int v50; // esi
  int v51; // ecx
  char v52; // dl
  unsigned __int64 v53; // rax
  int v54; // ecx
  __int64 v55; // r8
  char v56; // dl
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // [rsp+8h] [rbp-240h] BYREF
  _QWORD v59[71]; // [rsp+10h] [rbp-238h]

  v59[0] = a4;
  if ( (unsigned __int64)a1 >= a2 )
    return a4;
  v6 = 1;
  while ( 2 )
  {
    v7 = (unsigned __int8)*a1;
    v8 = a1 + 1;
    v9 = *a1;
    switch ( (char)v7 )
    {
      case 3:
      case 14:
      case 15:
        v16 = *(_QWORD *)(a1 + 1);
        v17 = v6;
        a1 += 9;
        goto LABEL_16;
      case 6:
      case 25:
      case 31:
      case 32:
      case 35:
      case -108:
        if ( !v6 )
          goto LABEL_124;
        v17 = v6 - 1;
        v22 = v59[v6 - 1];
        if ( (unsigned __int8)v9 <= 0x23u )
        {
          switch ( v9 )
          {
            case 6:
              v16 = *(_QWORD *)v22;
              ++a1;
              goto LABEL_16;
            case 25:
              ++a1;
              v16 = abs64(v22);
              goto LABEL_16;
            case 31:
              v16 = -v22;
              ++a1;
              goto LABEL_16;
            case 32:
              v16 = ~v22;
              ++a1;
              goto LABEL_16;
            case 35:
              ++a1;
              v54 = 0;
              v55 = 0;
              do
              {
                v56 = *a1++;
                v57 = (unsigned __int64)(v56 & 0x7F) << v54;
                v54 += 7;
                v55 |= v57;
              }
              while ( v56 < 0 );
              v16 = v55 + v22;
              goto LABEL_16;
            default:
              goto LABEL_125;
          }
        }
        if ( v9 != -108 )
          goto LABEL_125;
        v34 = a1[1];
        if ( v34 == 4 )
        {
          v16 = *(unsigned int *)v22;
          a1 += 2;
        }
        else if ( v34 > 4u )
        {
          if ( v34 != 8 )
LABEL_125:
            abort();
          v16 = *(_QWORD *)v22;
          a1 += 2;
        }
        else if ( v34 == 1 )
        {
          v16 = *(unsigned __int8 *)v22;
          a1 += 2;
        }
        else
        {
          if ( v34 != 2 )
            goto LABEL_125;
          v16 = *(unsigned __int16 *)v22;
          a1 += 2;
        }
LABEL_16:
        if ( v17 > 63 )
          goto LABEL_124;
        v6 = v17 + 1;
        v59[v17] = v16;
LABEL_18:
        if ( a2 > (unsigned __int64)a1 )
          continue;
        if ( !v6 )
LABEL_124:
          abort();
        result = v59[v6 - 1];
        break;
      case 8:
        v16 = (unsigned __int8)a1[1];
        v17 = v6;
        a1 += 2;
        goto LABEL_16;
      case 9:
        v16 = a1[1];
        v17 = v6;
        a1 += 2;
        goto LABEL_16;
      case 10:
        v16 = *(unsigned __int16 *)(a1 + 1);
        v17 = v6;
        a1 += 3;
        goto LABEL_16;
      case 11:
        v16 = *(__int16 *)(a1 + 1);
        v17 = v6;
        a1 += 3;
        goto LABEL_16;
      case 12:
        v16 = *(unsigned int *)(a1 + 1);
        v17 = v6;
        a1 += 5;
        goto LABEL_16;
      case 13:
        v16 = *(int *)(a1 + 1);
        v17 = v6;
        a1 += 5;
        goto LABEL_16;
      case 16:
        ++a1;
        v16 = 0;
        v35 = 0;
        do
        {
          v36 = *a1++;
          v37 = (unsigned __int64)(v36 & 0x7F) << v35;
          v35 += 7;
          v16 |= v37;
        }
        while ( v36 < 0 );
        goto LABEL_65;
      case 17:
        ++a1;
        v16 = 0;
        v38 = 0;
        do
        {
          v39 = *a1++;
          v40 = (unsigned __int64)(v39 & 0x7F) << v38;
          v38 += 7;
          v16 |= v40;
        }
        while ( v39 < 0 );
        v17 = v6;
        if ( v38 <= 0x3F && (v39 & 0x40) != 0 )
          v16 |= -1LL << v38;
        goto LABEL_16;
      case 18:
        if ( !v6 )
          goto LABEL_124;
        v17 = v6;
        ++a1;
        v16 = v59[v6 - 1];
        goto LABEL_16;
      case 19:
        if ( !v6 )
          goto LABEL_124;
        --v6;
        ++a1;
        goto LABEL_18;
      case 20:
        if ( v6 <= 1 )
          goto LABEL_124;
        v17 = v6;
        ++a1;
        v16 = v59[v6 - 2];
        goto LABEL_16;
      case 21:
        v26 = (unsigned __int8)a1[1];
        v27 = v6 - 1;
        if ( v26 >= v27 )
          goto LABEL_124;
        v17 = v6;
        a1 += 2;
        v16 = v59[v27 - v26];
        goto LABEL_16;
      case 22:
        if ( v6 <= 1 )
          goto LABEL_124;
        ++a1;
        v23 = v6 - 1;
        v24 = v6 - 2;
        v25 = v59[v23];
        v59[v23] = v59[v24];
        v59[v24] = v25;
        goto LABEL_18;
      case 23:
        if ( v6 <= 2 )
          goto LABEL_124;
        v28 = v6 - 1;
        v29 = v6 - 2;
        v30 = v6 - 3;
        v31 = v59[v28];
        v32 = v59[v30];
        v59[v28] = v59[v29];
        v59[v29] = v32;
        a1 = v8;
        v59[v30] = v31;
        goto LABEL_18;
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 33:
      case 34:
      case 36:
      case 37:
      case 38:
      case 39:
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
        if ( v6 > 1 )
        {
          v17 = v6 - 2;
          v20 = v59[v6 - 2];
          v21 = v59[v6 - 1];
          switch ( v9 )
          {
            case 26:
              v16 = v21 & v20;
              ++a1;
              goto LABEL_16;
            case 27:
              ++a1;
              v16 = v20 / v21;
              goto LABEL_16;
            case 28:
              v16 = v20 - v21;
              ++a1;
              goto LABEL_16;
            case 29:
              ++a1;
              v16 = v20 % (unsigned __int64)v21;
              goto LABEL_16;
            case 30:
              v16 = v21 * v20;
              ++a1;
              goto LABEL_16;
            case 33:
              v16 = v21 | v20;
              ++a1;
              goto LABEL_16;
            case 34:
              v16 = v21 + v20;
              ++a1;
              goto LABEL_16;
            case 36:
              v16 = v20 << v21;
              ++a1;
              goto LABEL_16;
            case 37:
              v16 = (unsigned __int64)v20 >> v21;
              ++a1;
              goto LABEL_16;
            case 38:
              v16 = v20 >> v21;
              ++a1;
              goto LABEL_16;
            case 39:
              v16 = v21 ^ v20;
              ++a1;
              goto LABEL_16;
            case 41:
              ++a1;
              v16 = v20 == v21;
              goto LABEL_16;
            case 42:
              ++a1;
              v16 = v20 >= v21;
              goto LABEL_16;
            case 43:
              ++a1;
              v16 = v20 > v21;
              goto LABEL_16;
            case 44:
              ++a1;
              v16 = v20 <= v21;
              goto LABEL_16;
            case 45:
              ++a1;
              v16 = v20 < v21;
              goto LABEL_16;
            case 46:
              ++a1;
              v16 = v20 != v21;
              goto LABEL_16;
            default:
              goto LABEL_125;
          }
        }
        goto LABEL_124;
      case 40:
        if ( !v6 )
          goto LABEL_124;
        if ( v59[--v6] )
          a1 += *(__int16 *)(a1 + 1) + 3;
        else
          a1 += 3;
        goto LABEL_18;
      case 47:
        a1 += *(__int16 *)(a1 + 1) + 3;
        goto LABEL_18;
      case 48:
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 55:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v16 = (unsigned int)(v7 - 48);
        ++a1;
        v17 = v6;
        goto LABEL_16;
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
      case 86:
      case 87:
      case 88:
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 94:
      case 95:
      case 96:
      case 97:
      case 98:
      case 99:
      case 100:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
      case 106:
      case 107:
      case 108:
      case 109:
      case 110:
      case 111:
        v19 = v7 - 80;
        if ( v19 > 17 )
          goto LABEL_124;
        v16 = *(_QWORD *)(a3 + 8LL * v19);
        if ( (*(_BYTE *)(a3 + 199) & 0x40) != 0 && *(_BYTE *)(a3 + v19 + 216) )
          goto LABEL_26;
        if ( byte_5057700[v19] != 8 )
          goto LABEL_124;
        v16 = *(_QWORD *)v16;
LABEL_26:
        v17 = v6;
        ++a1;
        goto LABEL_16;
      case 112:
      case 113:
      case 114:
      case 115:
      case 116:
      case 117:
      case 118:
      case 119:
      case 120:
      case 121:
      case 122:
      case 123:
      case 124:
      case 125:
      case 126:
      case 127:
      case -128:
      case -127:
      case -126:
      case -125:
      case -124:
      case -123:
      case -122:
      case -121:
      case -120:
      case -119:
      case -118:
      case -117:
      case -116:
      case -115:
      case -114:
      case -113:
        v10 = 0;
        v11 = 0;
        do
        {
          v12 = *v8++;
          v13 = (unsigned __int64)(v12 & 0x7F) << v11;
          v11 += 7;
          v10 |= v13;
        }
        while ( v12 < 0 );
        if ( v11 <= 0x3F && (v12 & 0x40) != 0 )
          v10 |= -1LL << v11;
        v14 = v7 - 112;
        if ( v14 > 17 )
          goto LABEL_124;
        v15 = *(_QWORD **)(a3 + 8LL * v14);
        if ( (*(_BYTE *)(a3 + 199) & 0x40) != 0 && *(_BYTE *)(a3 + v14 + 216) )
          goto LABEL_14;
        if ( byte_5057700[v14] != 8 )
          goto LABEL_124;
        v15 = (_QWORD *)*v15;
LABEL_14:
        v16 = (unsigned __int64)v15 + v10;
        v17 = v6;
        a1 = v8;
        goto LABEL_16;
      case -112:
        ++a1;
        v50 = 0;
        v51 = 0;
        do
        {
          v52 = *a1++;
          v53 = (unsigned __int64)(v52 & 0x7F) << v51;
          v51 += 7;
          v50 |= v53;
        }
        while ( v52 < 0 );
        if ( v50 > 17 )
          goto LABEL_124;
        v16 = *(_QWORD *)(a3 + 8LL * v50);
        if ( (*(_BYTE *)(a3 + 199) & 0x40) != 0 && *(_BYTE *)(a3 + v50 + 216) )
        {
LABEL_65:
          v17 = v6;
        }
        else
        {
          if ( byte_5057700[v50] != 8 )
            goto LABEL_124;
          v16 = *(_QWORD *)v16;
          v17 = v6;
        }
        goto LABEL_16;
      case -110:
        ++a1;
        v41 = 0;
        v42 = 0;
        do
        {
          v43 = *a1++;
          v44 = (unsigned __int64)(v43 & 0x7F) << v42;
          v42 += 7;
          v41 |= v44;
        }
        while ( v43 < 0 );
        v45 = 0;
        v46 = 0;
        do
        {
          v47 = *a1++;
          v48 = (unsigned __int64)(v47 & 0x7F) << v46;
          v46 += 7;
          v45 |= v48;
        }
        while ( v47 < 0 );
        if ( v46 <= 0x3F && (v47 & 0x40) != 0 )
          v45 |= -1LL << v46;
        if ( v41 > 17 )
          goto LABEL_124;
        v49 = *(_QWORD **)(a3 + 8LL * v41);
        if ( (*(_BYTE *)(a3 + 199) & 0x40) != 0 && *(_BYTE *)(a3 + v41 + 216) )
          goto LABEL_84;
        if ( byte_5057700[v41] != 8 )
          goto LABEL_124;
        v49 = (_QWORD *)*v49;
LABEL_84:
        v16 = (unsigned __int64)v49 + v45;
        v17 = v6;
        goto LABEL_16;
      case -106:
        ++a1;
        goto LABEL_18;
      case -15:
        v33 = sub_39F5E90((_QWORD *)a3, a1[1], a1 + 2, &v58);
        v16 = v58;
        v17 = v6;
        a1 = v33;
        goto LABEL_16;
      default:
        goto LABEL_125;
    }
    return result;
  }
}
