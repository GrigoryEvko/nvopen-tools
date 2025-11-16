// Function: sub_149FA60
// Address: 0x149fa60
//
void __fastcall sub_149FA60(__int64 a1, _DWORD *a2)
{
  char v4; // di
  char v5; // cl
  char v6; // dl
  char v7; // cl
  char v8; // dl
  int v9; // eax
  bool v10; // si
  bool v11; // r8
  int v12; // eax
  int v13; // edx
  unsigned int v14; // eax
  char v15; // dl
  int v16; // edx
  int v17; // eax
  int v18; // eax
  char v19; // al
  bool v20; // al
  bool v21; // al
  bool v22; // al
  int v23; // eax
  char v24; // al
  char v25; // dl
  char v26; // cl
  char v27; // si
  char v28; // di
  char v29; // r8
  bool v30; // al
  char v31; // r8
  char v32; // di
  char v33; // si
  char v34; // cl
  char v35; // dl
  char v36; // al
  unsigned int v37; // [rsp+4h] [rbp-3Ch] BYREF
  unsigned int v38; // [rsp+8h] [rbp-38h] BYREF
  _BYTE v39[52]; // [rsp+Ch] [rbp-34h] BYREF

  v4 = *(_BYTE *)(a1 + 62);
  v5 = *(_BYTE *)(a1 + 79);
  v6 = *(_BYTE *)(a1 + 80);
  *(_BYTE *)(a1 + 55) &= 0xCFu;
  v7 = v5 & 0x3F;
  *(_BYTE *)(a1 + 61) &= 0xCFu;
  v8 = v6 & 0xF3;
  *(_BYTE *)(a1 + 79) = v7;
  *(_BYTE *)(a1 + 56) &= 0xCCu;
  *(_BYTE *)(a1 + 49) &= 0x3Cu;
  *(_BYTE *)(a1 + 62) = v4 & 0xCF;
  *(_BYTE *)(a1 + 80) = v8;
  v9 = a2[8];
  if ( v9 == 17 )
  {
    v10 = 1;
    goto LABEL_29;
  }
  v10 = v9 == 24 || (v9 & 0xFFFFFFF7) == 18;
  v11 = 1;
  if ( v9 != 10 )
LABEL_29:
    v11 = (unsigned int)(v9 - 11) <= 2;
  *(_BYTE *)(a1 + 144) = v10;
  *(_BYTE *)(a1 + 145) = v10;
  *(_BYTE *)(a1 + 146) = v11;
  if ( (unsigned int)(a2[8] - 19) > 1
    || (*(_BYTE *)(a1 + 66) &= 0xC0u,
        *(_BYTE *)(a1 + 43) &= 3u,
        *(_BYTE *)(a1 + 67) &= 3u,
        (unsigned int)(a2[8] - 19) > 1) )
  {
    v12 = a2[11];
    if ( (v12 & 0xFFFFFFF7) == 3 )
    {
      *(_BYTE *)(a1 + 62) = v4 | 0x33;
      *(_BYTE *)(a1 + 79) = v7 | 0xC0;
      *(_BYTE *)(a1 + 80) = v8 | 0xC;
      if ( a2[11] == 11 )
      {
        sub_16E2390(a2, &v37, &v38, v39);
        if ( v37 == 10 )
        {
          if ( v38 == 5 )
            goto LABEL_7;
          v22 = v38 <= 4;
        }
        else
        {
          v22 = v37 <= 9;
        }
        if ( !v22 )
        {
LABEL_7:
          v12 = a2[11];
          if ( v12 != 3 )
          {
LABEL_8:
            if ( ((v12 - 7) & 0xFFFFFFFB) != 0 && (unsigned int)(v12 - 29) > 1 )
              goto LABEL_36;
            goto LABEL_10;
          }
LABEL_34:
          if ( a2[8] == 31 )
            goto LABEL_36;
          sub_16E2390(a2, &v37, &v38, v39);
          if ( v37 <= 0xC )
            goto LABEL_36;
          goto LABEL_75;
        }
      }
      else
      {
        sub_16E2390(a2, &v37, &v38, v39);
        if ( v37 == 9 || v37 > 8 )
          goto LABEL_7;
      }
    }
    else if ( v12 == 29 || v12 == 7 )
    {
      sub_16E2390(a2, &v37, &v38, v39);
      if ( v37 > 2 )
        goto LABEL_7;
    }
    else if ( v12 == 30 )
    {
LABEL_10:
      if ( a2[8] == 31 )
        goto LABEL_36;
      if ( v12 != 11 )
      {
LABEL_12:
        if ( v12 != 29 && v12 != 7 )
        {
LABEL_14:
          v13 = a2[8];
          if ( (v12 & 0xFFFFFFF7) != 3 || v13 != 31 )
          {
LABEL_16:
            if ( v13 != 33 && v13 != 27 )
            {
              *(_BYTE *)(a1 + 64) &= 0x3Fu;
              *(_BYTE *)(a1 + 87) &= 0xCFu;
              *(_BYTE *)(a1 + 50) &= 0xF3u;
            }
            v14 = a2[11];
            if ( v14 == 15 )
            {
              v23 = a2[12];
              if ( v23 == 16 || v23 == 1 )
                goto LABEL_50;
              *(_BYTE *)(a1 + 35) &= 0xFCu;
              *(_BYTE *)(a1 + 33) &= 0x3Fu;
              *(_BYTE *)(a1 + 39) &= 0xF3u;
              *(_BYTE *)(a1 + 41) &= 0xFCu;
              *(_BYTE *)(a1 + 42) &= 0xC3u;
              *(_BYTE *)(a1 + 46) &= 0xC3u;
              *(_BYTE *)(a1 + 51) &= 0xF3u;
              *(_BYTE *)(a1 + 52) &= 0x3Fu;
              *(_BYTE *)(a1 + 53) &= 0xCFu;
              *(_BYTE *)(a1 + 54) &= 0xF3u;
              *(_BYTE *)(a1 + 57) &= 0xCFu;
              *(_BYTE *)(a1 + 66) &= 0xC3u;
              *(_BYTE *)(a1 + 78) &= 0x3Fu;
              *(_BYTE *)(a1 + 87) &= 0xF0u;
              *(_BYTE *)(a1 + 88) &= 0x3Fu;
              *(_BYTE *)(a1 + 99) &= 0xC3u;
              *(_BYTE *)(a1 + 30) &= 0xFu;
              *(_BYTE *)(a1 + 31) &= 0xF0u;
              *(_BYTE *)(a1 + 34) &= 3u;
              *(_BYTE *)(a1 + 37) &= 0xC0u;
              *(_BYTE *)(a1 + 38) &= 0xC0u;
              *(_BYTE *)(a1 + 44) &= 0xC0u;
              *(_BYTE *)(a1 + 70) &= 0xCCu;
              *(_BYTE *)(a1 + 83) &= 3u;
              *(_BYTE *)(a1 + 84) &= 3u;
              *(_BYTE *)(a1 + 100) &= 0x3Fu;
              *(_BYTE *)(a1 + 101) &= 0xF0u;
              *(_BYTE *)(a1 + 32) = 0;
              *(_BYTE *)(a1 + 45) = 0;
              *(_WORD *)(a1 + 68) = 0;
              *(_BYTE *)(a1 + 75) = 0;
              sub_149D4C0(a1, 162, "_copysign", 9u);
              if ( a2[8] == 31 )
              {
                v31 = *(_BYTE *)(a1 + 39);
                v32 = *(_BYTE *)(a1 + 40);
                v33 = *(_BYTE *)(a1 + 42);
                v34 = *(_BYTE *)(a1 + 70);
                v35 = *(_BYTE *)(a1 + 74);
                *(_BYTE *)(a1 + 30) &= 0xF3u;
                v29 = v31 & 0xFC;
                v28 = v32 & 0x3F;
                *(_BYTE *)(a1 + 31) &= 0x3Fu;
                v27 = v33 & 0xFC;
                v26 = v34 & 0xF3;
                *(_BYTE *)(a1 + 34) &= 0xFCu;
                v25 = v35 & 0x3F;
                *(_BYTE *)(a1 + 33) &= 0xCFu;
                *(_BYTE *)(a1 + 41) &= 0xCFu;
                *(_BYTE *)(a1 + 44) &= 0x3Fu;
                *(_BYTE *)(a1 + 51) &= 0xFCu;
                *(_BYTE *)(a1 + 53) &= 0xF3u;
                *(_BYTE *)(a1 + 52) &= 0xCFu;
                *(_BYTE *)(a1 + 54) &= 0xFCu;
                *(_BYTE *)(a1 + 67) &= 0xCFu;
                *(_BYTE *)(a1 + 78) &= 0xCFu;
                *(_BYTE *)(a1 + 86) &= 0x33u;
                *(_BYTE *)(a1 + 88) &= 0xCFu;
                v36 = *(_BYTE *)(a1 + 99);
                *(_BYTE *)(a1 + 98) &= 0xCFu;
                v24 = v36 & 0xFC;
              }
              else
              {
                v24 = *(_BYTE *)(a1 + 99);
                v25 = *(_BYTE *)(a1 + 74);
                v26 = *(_BYTE *)(a1 + 70);
                v27 = *(_BYTE *)(a1 + 42);
                v28 = *(_BYTE *)(a1 + 40);
                v29 = *(_BYTE *)(a1 + 39);
              }
              *(_BYTE *)(a1 + 29) &= 0x3Fu;
              *(_BYTE *)(a1 + 47) &= 0xFCu;
              *(_BYTE *)(a1 + 48) &= 0xFCu;
              *(_BYTE *)(a1 + 50) &= 0xCCu;
              *(_BYTE *)(a1 + 58) &= 0xF3u;
              *(_BYTE *)(a1 + 59) &= 0xCCu;
              *(_BYTE *)(a1 + 60) &= 0x33u;
              *(_BYTE *)(a1 + 61) &= 0xFCu;
              *(_BYTE *)(a1 + 63) &= 0xC0u;
              *(_BYTE *)(a1 + 64) &= 0xC0u;
              *(_BYTE *)(a1 + 65) &= 0x3Fu;
              *(_BYTE *)(a1 + 71) &= 0x3Fu;
              *(_BYTE *)(a1 + 76) &= 0xC0u;
              *(_BYTE *)(a1 + 39) = v29 & 0xF;
              *(_BYTE *)(a1 + 40) = v28 & 0xF3;
              *(_BYTE *)(a1 + 42) = v27 & 0x3F;
              *(_BYTE *)(a1 + 70) = v26 & 0x3F;
              *(_BYTE *)(a1 + 74) = v25 & 0xFC;
              *(_BYTE *)(a1 + 77) &= 0x30u;
              *(_BYTE *)(a1 + 79) &= 0xFCu;
              *(_BYTE *)(a1 + 80) &= 0x3Fu;
              *(_BYTE *)(a1 + 81) &= 0xC3u;
              *(_BYTE *)(a1 + 82) &= 0xF3u;
              *(_BYTE *)(a1 + 84) &= 0xFCu;
              *(_BYTE *)(a1 + 85) &= 0xCFu;
              *(_BYTE *)(a1 + 89) &= 0x33u;
              *(_BYTE *)(a1 + 90) &= 3u;
              *(_BYTE *)(a1 + 93) &= 0xFCu;
              *(_BYTE *)(a1 + 101) &= 0xCFu;
              *(_BYTE *)(a1 + 105) &= 0xF3u;
              *(_BYTE *)(a1 + 57) &= 0xF3u;
              *(_BYTE *)(a1 + 99) = v24 & 0x3F;
              *(_BYTE *)(a1 + 66) &= 0x3Fu;
              *(_BYTE *)(a1 + 102) = 0;
              *(_BYTE *)(a1 + 36) = 0;
              v14 = a2[11];
            }
            v15 = *(_BYTE *)(a1 + 43);
            if ( v14 != 11 )
            {
              if ( v14 > 0xB )
              {
                if ( v14 - 29 <= 1 )
                  goto LABEL_23;
              }
              else if ( v14 == 7 )
              {
LABEL_23:
                *(_BYTE *)(a1 + 43) = v15 & 0x3F;
                if ( a2[11] == 30 )
                  goto LABEL_26;
                sub_16E2390(a2, &v37, &v38, v39);
                if ( v37 > 6 )
                {
                  sub_16E2390(a2, &v37, &v38, v39);
                  if ( v37 > 8 || (unsigned int)(a2[8] - 31) > 1 )
                    goto LABEL_26;
                }
                goto LABEL_64;
              }
LABEL_50:
              *(_BYTE *)(a1 + 43) &= 3u;
              goto LABEL_27;
            }
            *(_BYTE *)(a1 + 43) = v15 & 0x3F;
            if ( a2[11] == 11 )
            {
              sub_16E2390(a2, &v37, &v38, v39);
              if ( v37 == 10 )
              {
                if ( v38 == 9 )
                  goto LABEL_26;
                v21 = v38 <= 8;
              }
              else
              {
                v21 = v37 <= 9;
              }
              if ( v21 )
                goto LABEL_64;
            }
            else
            {
              sub_16E2390(a2, &v37, &v38, v39);
              if ( v37 <= 0xC )
              {
LABEL_64:
                *(_BYTE *)(a1 + 43) &= 0xC3u;
LABEL_27:
                v16 = a2[11];
                v17 = v16 - 3;
                switch ( v16 )
                {
                  case 3:
                  case 5:
                  case 7:
                  case 9:
                  case 11:
                  case 29:
                  case 30:
                    break;
                  default:
                    *(_BYTE *)(a1 + 48) &= 0xF3u;
                    v16 = a2[11];
                    v17 = v16 - 3;
                    break;
                }
                switch ( v17 )
                {
                  case 0:
                  case 2:
                  case 4:
                  case 6:
                  case 8:
                  case 26:
                  case 27:
                    break;
                  default:
                    *(_BYTE *)(a1 + 48) &= 0xCFu;
                    v16 = a2[11];
                    break;
                }
                if ( v16 != 5 )
                {
                  *(_BYTE *)(a1 + 51) &= 0xFu;
                  *(_BYTE *)(a1 + 52) &= 0xFCu;
                  if ( a2[11] == 9 )
                  {
                    v18 = a2[12];
                    if ( v18 != 10 )
                      goto LABEL_44;
                  }
                }
                *(_BYTE *)(a1 + 28) &= 0xCFu;
                v19 = *(_BYTE *)(a1 + 71);
                *(_BYTE *)(a1 + 29) &= 0xF3u;
                *(_BYTE *)(a1 + 20) &= 0x3Fu;
                *(_BYTE *)(a1 + 21) &= 0xFCu;
                *(_BYTE *)(a1 + 5) &= 0xF0u;
                if ( a2[12] != 10 )
                  v19 &= 0xCFu;
                *(_BYTE *)(a1 + 54) &= 0x3Fu;
                *(_BYTE *)(a1 + 58) &= 0xCFu;
                *(_BYTE *)(a1 + 59) &= 0x33u;
                *(_BYTE *)(a1 + 60) &= 0xCFu;
                *(_BYTE *)(a1 + 76) &= 0x3Fu;
                *(_BYTE *)(a1 + 89) &= 0xCFu;
                *(_BYTE *)(a1 + 90) &= 0xFCu;
                *(_BYTE *)(a1 + 100) &= 0xF3u;
                *(_BYTE *)(a1 + 12) &= 0xFu;
                *(_BYTE *)(a1 + 18) &= 0xFu;
                *(_BYTE *)(a1 + 23) &= 0xF0u;
                *(_BYTE *)(a1 + 24) &= 0xFu;
                *(_BYTE *)(a1 + 25) &= 0x3Cu;
                *(_BYTE *)(a1 + 26) &= 0xF0u;
                *(_BYTE *)(a1 + 71) = v19 & 0xFC;
                *(_DWORD *)(a1 + 13) = 0;
                *(_DWORD *)(a1 + 19) = 0;
                v18 = a2[12];
                if ( a2[11] == 9 )
                {
LABEL_44:
                  if ( (unsigned int)(v18 - 1) <= 5 )
                    goto LABEL_58;
                  if ( v18 != 10 )
                  {
LABEL_46:
                    if ( (unsigned int)(a2[8] - 34) > 1 )
                    {
LABEL_59:
                      *(_BYTE *)(a1 + 24) &= 0xF3u;
                      goto LABEL_48;
                    }
LABEL_47:
                    sub_149CBC0((_QWORD *)a1);
                    *(_BYTE *)(a1 + 24) |= 0xCu;
LABEL_48:
                    sub_149E420(a1, dword_4F9D200);
                    return;
                  }
                }
                else if ( v18 != 10 )
                {
                  goto LABEL_46;
                }
                sub_16E22F0(a2, &v37, &v38, v39);
                if ( (!(unsigned __int8)sub_16E2900(a2) || v37 > 0x14) && v37 > 0x1B )
                {
LABEL_58:
                  *(_BYTE *)(a1 + 62) |= 0x33u;
                  *(_BYTE *)(a1 + 79) |= 0xC0u;
                  *(_BYTE *)(a1 + 80) |= 0xCu;
                  *(_BYTE *)(a1 + 55) |= 0x30u;
                  *(_BYTE *)(a1 + 61) |= 0x30u;
                  *(_BYTE *)(a1 + 56) |= 0x33u;
                  *(_BYTE *)(a1 + 49) |= 0xC3u;
                  if ( (unsigned int)(a2[8] - 34) > 1 )
                    goto LABEL_59;
                  goto LABEL_47;
                }
                goto LABEL_46;
              }
            }
LABEL_26:
            sub_149D4C0(a1, 173, "__exp10", 7u);
            sub_149D4C0(a1, 174, "__exp10f", 8u);
            goto LABEL_27;
          }
          if ( v12 != 11 )
          {
            sub_16E2390(a2, &v37, &v38, v39);
            if ( v37 != 11 && v37 <= 0xA )
              goto LABEL_93;
            goto LABEL_92;
          }
          sub_16E2390(a2, &v37, &v38, v39);
          if ( v37 == 10 )
          {
            if ( v38 == 7 )
            {
LABEL_92:
              sub_149D4C0(a1, 245, "fwrite$UNIX2003", 0xFu);
              sub_149D4C0(a1, 223, "fputs$UNIX2003", 0xEu);
LABEL_93:
              v13 = a2[8];
              goto LABEL_16;
            }
            v30 = v38 <= 6;
          }
          else
          {
            v30 = v37 <= 9;
          }
          if ( v30 )
            goto LABEL_93;
          goto LABEL_92;
        }
        sub_16E2390(a2, &v37, &v38, v39);
        if ( v37 > 6 )
        {
LABEL_37:
          v12 = a2[11];
          goto LABEL_14;
        }
LABEL_36:
        *(_BYTE *)(a1 + 26) &= 0xFu;
        *(_BYTE *)(a1 + 17) &= 0xF0u;
        *(_BYTE *)(a1 + 25) &= 0xC3u;
        goto LABEL_37;
      }
      sub_16E2390(a2, &v37, &v38, v39);
      if ( v37 != 10 )
      {
        v20 = v37 <= 9;
LABEL_74:
        if ( v20 )
          goto LABEL_36;
        goto LABEL_75;
      }
      if ( v38 != 9 )
      {
        v20 = v38 <= 8;
        goto LABEL_74;
      }
LABEL_75:
      v12 = a2[11];
      goto LABEL_12;
    }
    *(_BYTE *)(a1 + 73) &= 0x3Fu;
    v12 = a2[11];
    if ( v12 != 3 )
      goto LABEL_8;
    goto LABEL_34;
  }
  *(_BYTE *)(a1 + 72) &= 0xCFu;
  *(_BYTE *)(a1 + 73) &= 0xFu;
}
