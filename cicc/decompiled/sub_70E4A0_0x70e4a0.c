// Function: sub_70E4A0
// Address: 0x70e4a0
//
__int64 __fastcall sub_70E4A0(__int64 *a1, __int64 a2, int a3, _DWORD *a4, unsigned int a5)
{
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned __int64 v11; // r14
  __int64 v12; // rdi
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 i; // rax
  __int64 result; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r14
  unsigned int v22; // edi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r14
  char v27; // al
  int v28; // r15d
  __int64 v29; // r14
  int v30; // r15d
  char v31; // al
  __int64 v32; // rax
  char j; // dl
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r14
  int v37; // r15d
  char v38; // al
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 v41; // rax
  char k; // dl
  __int64 v43; // rax
  char m; // dl
  __int64 v45; // rax
  __int64 v46; // r15
  char v47; // r14
  __int64 v48; // r13
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rdi
  int v54; // [rsp+10h] [rbp-60h]
  int v55; // [rsp+14h] [rbp-5Ch]
  __int64 v58; // [rsp+20h] [rbp-50h]
  __int64 v59; // [rsp+20h] [rbp-50h]
  int v60; // [rsp+28h] [rbp-48h]
  _BYTE v62[4]; // [rsp+38h] [rbp-38h] BYREF
  _DWORD v63[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v6 = a2;
  v7 = a1[8];
  v8 = *(_QWORD *)(v7 + 56);
  v60 = sub_8DBE70(v8);
  if ( v60 )
  {
    sub_724C70(a2, 12);
    sub_7249B0(a2, 1);
    *(_QWORD *)(a2 + 184) = a1;
    goto LABEL_22;
  }
  v54 = 0;
  v55 = 0;
  v11 = *((unsigned __int8 *)a1 + 56);
  if ( (*(_BYTE *)(v8 + 140) & 0xFB) == 8 )
  {
    a2 = dword_4F077C4 != 2;
    v55 = sub_8D4C10(v8, a2) & 1;
    v54 = v55;
    if ( (unsigned __int8)v11 > 0x1Bu )
    {
LABEL_4:
      if ( (_BYTE)v11 == 116 )
        goto LABEL_5;
LABEL_12:
      v12 = v8;
      goto LABEL_8;
    }
  }
  else if ( (unsigned __int8)v11 > 0x1Bu )
  {
    goto LABEL_4;
  }
  v15 = 260310964;
  if ( !_bittest64(&v15, v11) )
    goto LABEL_12;
LABEL_5:
  v12 = v8;
  if ( (unsigned int)sub_8D3410(v8) )
  {
    v19 = sub_8D4130(v8);
    v12 = v19;
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 )
        {
          if ( (unsigned __int8)v11 <= 0x1Bu )
          {
            v20 = 167774132;
            if ( _bittest64(&v20, v11) )
            {
              if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v12) )
                sub_8AE000(v12);
              if ( (unsigned int)sub_8D23B0(v12) )
              {
                v21 = (int)dword_4F077BC;
                goto LABEL_40;
              }
            }
          }
        }
      }
    }
  }
LABEL_8:
  while ( 1 )
  {
    v13 = *(unsigned __int8 *)(v12 + 140);
    if ( (_BYTE)v13 != 12 )
      break;
    v12 = *(_QWORD *)(v12 + 160);
  }
  v14 = (unsigned int)(v13 - 9);
  if ( (unsigned __int8)(v13 - 9) > 2u )
  {
    switch ( (char)v11 )
    {
      case 1:
      case 3:
      case 6:
        LOBYTE(v60) = 1;
        goto LABEL_44;
      case 2:
      case 5:
      case 8:
      case 9:
      case 27:
        if ( !(unsigned int)sub_8D2310(v12) )
          goto LABEL_42;
        goto LABEL_47;
      case 4:
      case 7:
      case 18:
        if ( (unsigned int)sub_8D2FB0(v12) || (unsigned int)sub_8D2310(v12) )
          goto LABEL_47;
LABEL_42:
        v21 = (unsigned int)sub_8D2600(v12) == 0;
        goto LABEL_40;
      case 10:
      case 11:
      case 12:
      case 14:
      case 16:
      case 19:
      case 20:
      case 40:
      case 105:
      case 106:
      case 107:
        goto LABEL_39;
      case 17:
        v21 = 0;
        if ( (_BYTE)v13 == 2 )
          v21 = (*(_BYTE *)(v12 + 161) & 8) != 0;
        goto LABEL_40;
      case 23:
      case 93:
        v21 = (int)sub_8D2530(v12);
        goto LABEL_40;
      case 24:
        v21 = 0;
        if ( (unsigned int)sub_8D2530(v12) )
          v21 = (unsigned int)sub_8D2BE0(v12) == 0;
        goto LABEL_40;
      case 25:
        goto LABEL_55;
      case 26:
        goto LABEL_56;
      case 28:
      case 29:
LABEL_44:
        if ( (unsigned int)sub_8D2FB0(v12) || (unsigned int)sub_8D2310(v12) || (unsigned int)sub_8D2600(v12) | v54 )
          goto LABEL_47;
        if ( !(unsigned int)sub_8D3410(v12) )
          goto LABEL_60;
        v53 = sub_8D40F0(v12);
        if ( (unsigned __int8)(*(_BYTE *)(v53 + 140) - 9) > 2u )
          goto LABEL_60;
        v21 = ((unsigned int)sub_8E4550(v53) != 0) | (unsigned __int8)((v60 ^ 1) & 1);
        goto LABEL_40;
      case 61:
        goto LABEL_51;
      case 64:
        v23 = sub_8D4130(v8);
        v21 = (int)sub_70DD40(v23, 0, 1);
        goto LABEL_40;
      case 65:
        if ( (_BYTE)v13 != 8 )
        {
          if ( (unsigned int)sub_8D2B80(v12) )
            goto LABEL_60;
          v21 = (unsigned int)sub_8D2B50(v12) != 0;
          goto LABEL_40;
        }
        v39 = sub_8D4050(v12);
        if ( !(unsigned int)sub_8D23B0(v39) )
          goto LABEL_60;
        sub_724C70(v6, 0);
        if ( a4 )
          goto LABEL_160;
        goto LABEL_36;
      case 69:
        v21 = (int)sub_8D2310(v12);
        goto LABEL_40;
      case 79:
        v21 = (int)sub_8D3410(v12);
        goto LABEL_40;
      case 82:
        v21 = (int)sub_8D2DD0(v12);
        goto LABEL_40;
      case 83:
        goto LABEL_70;
      case 84:
        v21 = !(unsigned int)sub_8D2600(v12) && !(unsigned int)sub_8D2DD0(v12) && (unsigned int)sub_8D2660(v12) == 0;
        goto LABEL_40;
      case 85:
        v21 = v54;
        goto LABEL_40;
      case 86:
        v21 = (int)sub_8D2A90(v12);
        goto LABEL_40;
      case 87:
        v21 = (unsigned int)sub_8D2600(v12) || (unsigned int)sub_8D2DD0(v12) || (unsigned int)sub_8D2660(v12) != 0;
        goto LABEL_40;
      case 88:
        v21 = (int)sub_8D2780(v12);
        goto LABEL_40;
      case 89:
        v21 = (int)sub_8D3070(v12);
        goto LABEL_40;
      case 90:
        v21 = 0;
        if ( (unsigned int)sub_8D3D10(v12) )
        {
          v24 = sub_8D4870(v12);
          v21 = (unsigned int)sub_8D2310(v24) != 0;
        }
        goto LABEL_40;
      case 91:
        v21 = 0;
        if ( (unsigned int)sub_8D3D10(v12) )
        {
          v25 = sub_8D4870(v12);
          v21 = (unsigned int)sub_8D2310(v25) == 0;
        }
        goto LABEL_40;
      case 92:
        v21 = (int)sub_8D3D10(v12);
        goto LABEL_40;
      case 94:
        v21 = (int)sub_8D2E30(v12);
        goto LABEL_40;
      case 95:
        v21 = (int)sub_8D2FB0(v12);
        goto LABEL_40;
      case 96:
        v21 = (int)sub_8D3110(v12);
        goto LABEL_40;
      case 97:
        v21 = (int)sub_8D3350(v12);
        goto LABEL_40;
      case 98:
        if ( !(unsigned int)sub_8D2DD0(v12) || (unsigned int)sub_8D29A0(v12) )
        {
          v21 = 0;
        }
        else
        {
          v21 = 1;
          if ( !(unsigned int)sub_8D27E0(v12) )
            v21 = (unsigned int)sub_8D2A90(v12) != 0;
        }
        goto LABEL_40;
      case 99:
        v21 = 1;
        if ( !(unsigned int)sub_8D29A0(v12) )
          v21 = (unsigned int)sub_8D2DD0(v12) && !(unsigned int)sub_8D27E0(v12) && (unsigned int)sub_8D2A90(v12) == 0;
        goto LABEL_40;
      case 100:
        v21 = (int)sub_8D2600(v12);
        goto LABEL_40;
      case 101:
        v21 = 0;
        if ( (*(_BYTE *)(v8 + 140) & 0xFB) == 8 )
          goto LABEL_98;
        goto LABEL_40;
      case 102:
        if ( (unsigned int)sub_8D3410(v8) )
          goto LABEL_95;
        goto LABEL_47;
      case 103:
        if ( (unsigned int)sub_8D3410(v8) )
          goto LABEL_102;
        goto LABEL_47;
      case 104:
        goto LABEL_63;
      case 114:
        v21 = (int)sub_8D28B0(v12);
        goto LABEL_40;
      case 115:
        goto LABEL_65;
      case 116:
        goto LABEL_68;
      default:
        goto LABEL_38;
    }
  }
  v16 = a5;
  if ( a5 )
  {
    if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v12) )
      sub_8AE000(v12);
    if ( (unsigned int)sub_8D23B0(v12) )
    {
      sub_724C70(v6, 0);
      if ( a4 )
      {
        v22 = 1440;
        if ( (_BYTE)v11 == 65 )
LABEL_160:
          v22 = 2873;
        sub_6851C0(v22, a4);
      }
      goto LABEL_36;
    }
    *(_BYTE *)(v7 + 26) |= 0x40u;
    for ( i = v12; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v9 = *(_QWORD *)(*(_QWORD *)i + 96LL);
  }
  else
  {
    v9 = 0;
  }
  switch ( (char)v11 )
  {
    case 1:
    case 3:
      v21 = 0;
      if ( !v55 )
        v21 = (unsigned int)sub_70D130(v12) != 0;
      goto LABEL_40;
    case 2:
    case 5:
      v21 = (int)sub_70D330(v12);
      goto LABEL_40;
    case 4:
      v26 = *(_QWORD *)(v9 + 8);
      if ( !v26 )
      {
LABEL_60:
        v21 = 1;
        goto LABEL_40;
      }
      v27 = *(_BYTE *)(v26 + 80);
      v28 = 0;
      if ( v27 == 17 )
      {
        v26 = *(_QWORD *)(v26 + 88);
        if ( v26 )
        {
          v27 = *(_BYTE *)(v26 + 80);
          v28 = 1;
          goto LABEL_105;
        }
LABEL_47:
        v21 = 0;
        goto LABEL_40;
      }
LABEL_105:
      while ( 2 )
      {
        if ( v27 == 10 )
        {
          v40 = *(_QWORD *)(v26 + 88);
          if ( (unsigned int)sub_72F310(v40, 1, v14, v9, v10, v16) )
          {
            v60 = sub_8D7760(v40, 1, v14, v9, v10, v16);
            if ( !v60 )
              goto LABEL_47;
          }
        }
        if ( v28 )
        {
          v26 = *(_QWORD *)(v26 + 8);
          if ( v26 )
          {
            v27 = *(_BYTE *)(v26 + 80);
            continue;
          }
          v21 = v60;
        }
        else
        {
          v21 = v60;
        }
        goto LABEL_40;
      }
    case 6:
      v21 = 0;
      if ( v55 )
        goto LABEL_40;
      v29 = *(_QWORD *)(v9 + 32);
      v30 = 0;
      v31 = *(_BYTE *)(v29 + 80);
      if ( v31 != 17 )
        goto LABEL_112;
      v29 = *(_QWORD *)(v29 + 88);
      if ( !v29 )
        goto LABEL_114;
      v31 = *(_BYTE *)(v29 + 80);
      v30 = 1;
LABEL_112:
      while ( 2 )
      {
        if ( v31 != 10 )
          goto LABEL_113;
        v58 = *(_QWORD *)(v29 + 88);
        if ( !(unsigned int)sub_72F790(v58, v62, v63) || v63[0] )
          goto LABEL_113;
        if ( (*(_BYTE *)(v58 + 194) & 4) != 0 )
        {
          v60 = 1;
LABEL_113:
          if ( v30 )
          {
            v29 = *(_QWORD *)(v29 + 8);
            if ( v29 )
            {
              v31 = *(_BYTE *)(v29 + 80);
              continue;
            }
          }
LABEL_114:
          v21 = v60;
        }
        else
        {
          v21 = 0;
        }
        goto LABEL_40;
      }
    case 7:
      v21 = 1;
      if ( *(char *)(v9 + 178) >= 0 )
        v21 = *(_QWORD *)(v9 + 16) != 0;
      goto LABEL_40;
    case 8:
      v21 = 1;
      if ( (*(_BYTE *)(v9 + 177) & 0x40) != 0 )
        goto LABEL_40;
      v36 = *(_QWORD *)(v9 + 8);
      v37 = 0;
      v38 = *(_BYTE *)(v36 + 80);
      if ( v38 != 17 )
        goto LABEL_149;
      v36 = *(_QWORD *)(v36 + 88);
      if ( !v36 )
        goto LABEL_151;
      v38 = *(_BYTE *)(v36 + 80);
      v37 = 1;
LABEL_149:
      while ( 2 )
      {
        if ( v38 != 10 )
          goto LABEL_150;
        v59 = *(_QWORD *)(v36 + 88);
        if ( !(unsigned int)sub_72F500(v59, *(_QWORD *)(*(_QWORD *)(v59 + 40) + 32LL), v63, 0, 1) )
          goto LABEL_150;
        if ( (*(_BYTE *)(v59 + 194) & 4) != 0 )
        {
          v60 = 1;
LABEL_150:
          if ( v37 )
          {
            v36 = *(_QWORD *)(v36 + 8);
            if ( v36 )
            {
              v38 = *(_BYTE *)(v36 + 80);
              continue;
            }
          }
LABEL_151:
          v21 = v60;
        }
        else
        {
          v21 = 0;
        }
        goto LABEL_40;
      }
    case 9:
      v21 = (*(_BYTE *)(v9 + 177) & 2) != 0;
      goto LABEL_40;
    case 10:
      v35 = *(_QWORD *)(v9 + 24);
      v21 = 0;
      if ( v35 )
        v21 = ((*(_BYTE *)(*(_QWORD *)(v35 + 88) + 193LL) >> 4) ^ 1) & 1;
      goto LABEL_40;
    case 11:
      v34 = *(_QWORD *)(v9 + 24);
      v21 = 0;
      if ( v34 )
        v21 = (*(_BYTE *)(*(_QWORD *)(v34 + 88) + 192LL) & 2) != 0;
      goto LABEL_40;
    case 12:
      v21 = (*(_BYTE *)(v12 + 176) & 0x20) != 0;
      goto LABEL_40;
    case 14:
      v21 = (unsigned __int8)(*(_BYTE *)(v12 + 140) - 9) <= 1u;
      goto LABEL_40;
    case 16:
      v21 = 0;
      if ( !(unsigned int)sub_8D3B10(v12) )
        v21 = (unsigned int)sub_7A80B0(v12) != 0;
      goto LABEL_40;
    case 17:
    case 69:
    case 79:
    case 82:
    case 86:
    case 87:
    case 88:
    case 89:
    case 90:
    case 91:
    case 92:
    case 94:
    case 95:
    case 96:
    case 97:
    case 98:
    case 99:
    case 100:
    case 114:
LABEL_39:
      v21 = 0;
      goto LABEL_40;
    case 18:
      v21 = (int)sub_8E4550(v12);
      goto LABEL_40;
    case 19:
      v21 = (int)sub_8D3E60(v12);
      goto LABEL_40;
    case 20:
      v21 = *(_BYTE *)(v12 + 140) == 11;
      goto LABEL_40;
    case 23:
      v21 = (int)sub_8E43E0(v12, a2, v14, v9);
      goto LABEL_40;
    case 24:
      v21 = *(_BYTE *)(v9 + 181) >> 7;
      goto LABEL_40;
    case 25:
LABEL_55:
      v21 = (int)sub_8E3AD0(v8);
      goto LABEL_40;
    case 26:
LABEL_56:
      v21 = (int)sub_8D4160(v12);
      goto LABEL_40;
    case 27:
      v32 = sub_8D4130(v12);
      for ( j = *(_BYTE *)(v32 + 140); j == 12; j = *(_BYTE *)(v32 + 140) )
        v32 = *(_QWORD *)(v32 + 160);
      if ( (unsigned __int8)(j - 9) > 2u )
        goto LABEL_47;
      v21 = (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v32 + 96LL) + 177LL) & 0x40) != 0;
      goto LABEL_40;
    case 28:
      if ( v55 )
        goto LABEL_116;
      v41 = sub_8D4130(v12);
      for ( k = *(_BYTE *)(v41 + 140); k == 12; k = *(_BYTE *)(v41 + 140) )
        v41 = *(_QWORD *)(v41 + 160);
      if ( (unsigned __int8)(k - 9) > 2u )
LABEL_116:
        v21 = 0;
      else
        v21 = (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v41 + 96LL) + 177LL) & 0x20) != 0;
      goto LABEL_40;
    case 29:
      if ( v55 )
        goto LABEL_132;
      v43 = sub_8D4130(v12);
      for ( m = *(_BYTE *)(v43 + 140); m == 12; m = *(_BYTE *)(v43 + 140) )
        v43 = *(_QWORD *)(v43 + 160);
      if ( (unsigned __int8)(m - 9) > 2u )
        goto LABEL_132;
      v21 = 1;
      v45 = *(_QWORD *)(*(_QWORD *)v43 + 96LL);
      if ( (*(_BYTE *)(v45 + 177) & 0x20) != 0 )
        goto LABEL_40;
      v46 = *(_QWORD *)(v45 + 32);
      v47 = *(_BYTE *)(v46 + 80);
      if ( v47 == 17 )
      {
        v46 = *(_QWORD *)(v46 + 88);
        if ( !v46 )
          goto LABEL_197;
      }
      break;
    case 40:
      v21 = *(_BYTE *)(v12 + 176) & 1;
      goto LABEL_40;
    case 61:
      v21 = (*(_BYTE *)(v9 + 177) & 0x30) == 32;
      goto LABEL_40;
    case 64:
      v21 = (int)sub_70DD40(v12, 0, 1);
      goto LABEL_40;
    case 65:
      v21 = (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v12 + 96LL) + 178LL) & 0x40) != 0;
      goto LABEL_40;
    case 83:
LABEL_70:
      v21 = (int)sub_8D25A0(v12);
      goto LABEL_40;
    case 84:
    case 93:
LABEL_51:
      v21 = 1;
      goto LABEL_40;
    case 85:
      v21 = v54;
      goto LABEL_40;
    case 101:
      v21 = 0;
      if ( (*(_BYTE *)(v8 + 140) & 0xFB) == 8 )
LABEL_98:
        v21 = (sub_8D4C10(v8, dword_4F077C4 != 2) & 2) != 0;
      goto LABEL_40;
    case 102:
      if ( !(unsigned int)sub_8D3410(v8) )
        goto LABEL_47;
LABEL_95:
      v21 = (unsigned int)sub_8D23E0(v8) == 0;
      goto LABEL_40;
    case 103:
      if ( !(unsigned int)sub_8D3410(v8) )
        goto LABEL_47;
LABEL_102:
      v21 = (unsigned int)sub_8D23E0(v8) != 0;
      goto LABEL_40;
    case 104:
LABEL_63:
      v21 = (int)sub_8D3FE0(v8, a2, v14, v9);
      goto LABEL_40;
    case 105:
      v21 = *(_BYTE *)(*(_QWORD *)(v12 + 168) + 92LL) & 1;
      goto LABEL_40;
    case 106:
      v21 = (*(_BYTE *)(*(_QWORD *)(v12 + 168) + 92LL) & 4) != 0;
      goto LABEL_40;
    case 107:
      v21 = (*(_BYTE *)(*(_QWORD *)(v12 + 168) + 92LL) & 2) != 0;
      goto LABEL_40;
    case 115:
LABEL_65:
      v21 = (int)sub_70E160(v8, 1);
      goto LABEL_40;
    case 116:
LABEL_68:
      v21 = (int)sub_70DB60(v12);
      goto LABEL_40;
    default:
LABEL_38:
      sub_721090(v12);
  }
  do
  {
    if ( *(_BYTE *)(v46 + 80) == 10 )
    {
      v48 = *(_QWORD *)(v46 + 88);
      if ( (unsigned int)sub_72F850(v48) )
      {
        if ( !(unsigned int)sub_8D7760(v48, a2, v49, v50, v51, v52) )
        {
LABEL_132:
          v21 = 0;
          goto LABEL_40;
        }
        v60 = 1;
      }
    }
    if ( v47 != 17 )
      break;
    v46 = *(_QWORD *)(v46 + 8);
  }
  while ( v46 );
LABEL_197:
  v21 = v60;
LABEL_40:
  sub_724C70(v6, 1);
  sub_620D80((_WORD *)(v6 + 176), v21);
LABEL_36:
  if ( a3 )
    *(_QWORD *)(v6 + 144) = a1;
LABEL_22:
  result = *a1;
  *(_QWORD *)(v6 + 128) = *a1;
  return result;
}
