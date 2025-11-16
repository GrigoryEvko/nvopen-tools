// Function: sub_982C80
// Address: 0x982c80
//
void __fastcall sub_982C80(__int64 a1, _DWORD *a2)
{
  char v4; // di
  int v5; // esi
  char v6; // cl
  char v7; // dl
  char v8; // di
  unsigned __int64 v9; // rax
  unsigned int v10; // edx
  unsigned int v11; // ecx
  unsigned int v12; // eax
  __int64 v13; // rdx
  int v14; // eax
  char v15; // r15
  char v16; // di
  char v17; // r10
  char v18; // r11
  char v19; // bl
  char v20; // r9
  char v21; // cl
  char v22; // r14
  char v23; // dl
  char v24; // al
  char v25; // bl
  char v26; // r8
  char v27; // cl
  char v28; // al
  char v29; // dl
  char v30; // r14
  char v31; // r8
  char v32; // si
  char v33; // di
  char v34; // r14
  char v35; // si
  char v36; // di
  char v37; // r10
  char v38; // r9
  char v39; // r15
  char v40; // r11
  unsigned __int64 v41; // rcx
  __int64 v42; // rdx
  char v43; // dl
  int v44; // eax
  int v45; // eax
  char v46; // si
  char v47; // cl
  bool v48; // zf
  int v49; // edx
  bool v50; // cc
  char v51; // al
  char v52; // al
  char v53; // r11
  char v54; // r15
  char v55; // [rsp-37h] [rbp-3Fh]
  char v56; // [rsp-36h] [rbp-3Eh]
  char v57; // [rsp-35h] [rbp-3Dh]
  char v58; // [rsp-34h] [rbp-3Ch]
  char v59; // [rsp-33h] [rbp-3Bh]
  char v60; // [rsp-33h] [rbp-3Bh]
  bool v61; // [rsp-32h] [rbp-3Ah]
  char v62; // [rsp-32h] [rbp-3Ah]
  char v63; // [rsp-31h] [rbp-39h]

  *(_WORD *)(a1 + 128) = -1;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_BYTE *)(a1 + 130) = -1;
  *(_OWORD *)a1 = -1;
  *(_OWORD *)(a1 + 16) = -1;
  *(_OWORD *)(a1 + 32) = -1;
  *(_OWORD *)(a1 + 48) = -1;
  *(_OWORD *)(a1 + 64) = -1;
  *(_OWORD *)(a1 + 80) = -1;
  *(_OWORD *)(a1 + 96) = -1;
  *(_OWORD *)(a1 + 112) = -1;
  sub_97DEE0(a1, (__int64)a2);
  v4 = *(_BYTE *)(a1 + 77);
  v5 = a2[8];
  v6 = *(_BYTE *)(a1 + 78) & 0xFC;
  v7 = *(_BYTE *)(a1 + 98) & 0xCC;
  *(_BYTE *)(a1 + 62) &= 0x3Fu;
  v8 = v4 & 0xCC;
  *(_BYTE *)(a1 + 72) &= 0xFCu;
  *(_BYTE *)(a1 + 71) &= 0xCCu;
  *(_BYTE *)(a1 + 63) &= 0xCFu;
  *(_BYTE *)(a1 + 78) = v6;
  *(_BYTE *)(a1 + 98) = v7;
  *(_BYTE *)(a1 + 77) = v8;
  if ( (unsigned int)(v5 - 26) <= 1 )
  {
    sub_97F7E0((_QWORD *)a1);
    *(_BYTE *)(a1 + 27) |= 0x3Cu;
    return;
  }
  v9 = (unsigned int)a2[11];
  if ( (a2[11] & 0xFFFFFFF7) == 1 )
  {
    *(_BYTE *)(a1 + 90) &= 0xF3u;
    *(_BYTE *)(a1 + 77) = v8 | 0x30;
    *(_BYTE *)(a1 + 78) = v6 | 3;
    *(_BYTE *)(a1 + 98) = v7 | 0x33;
    if ( !(unsigned __int8)sub_CC8200(a2, 10, 5, 0) )
    {
LABEL_5:
      v9 = (unsigned int)a2[11];
      v5 = a2[8];
      goto LABEL_6;
    }
LABEL_30:
    *(_BYTE *)(a1 + 91) &= 0xF0u;
    *(_BYTE *)(a1 + 90) &= 0x3Fu;
    goto LABEL_5;
  }
  if ( (_DWORD)v9 == 27 || (_DWORD)v9 == 5 )
  {
    if ( (unsigned int)sub_CC78E0(a2) > 2 )
      goto LABEL_5;
    goto LABEL_30;
  }
  if ( (_DWORD)v9 == 28 )
    goto LABEL_24;
  *(_BYTE *)(a1 + 91) &= 0xF0u;
  *(_BYTE *)(a1 + 90) &= 0x3Fu;
LABEL_6:
  if ( (_DWORD)v9 != 1 )
  {
    if ( (unsigned int)v9 > 0x1F )
      goto LABEL_8;
    v13 = 3623879200LL;
    if ( !_bittest64(&v13, v9) )
      goto LABEL_8;
LABEL_24:
    if ( v5 == 38 )
      goto LABEL_8;
    if ( (_DWORD)v9 != 9 )
      goto LABEL_26;
    goto LABEL_98;
  }
  if ( v5 == 38 )
    goto LABEL_8;
LABEL_98:
  if ( (unsigned __int8)sub_CC8200(a2, 10, 9, 0) )
  {
    v5 = a2[8];
    LODWORD(v9) = a2[11];
    goto LABEL_8;
  }
  LODWORD(v9) = a2[11];
LABEL_26:
  if ( (_DWORD)v9 != 5 && (_DWORD)v9 != 27 )
  {
    v5 = a2[8];
    goto LABEL_9;
  }
  LODWORD(v9) = sub_CC78E0(a2);
  v5 = a2[8];
  v50 = (unsigned int)v9 <= 6;
  LODWORD(v9) = a2[11];
  if ( v50 )
  {
LABEL_8:
    *(_BYTE *)(a1 + 33) &= 0xFu;
    *(_BYTE *)(a1 + 21) &= 0xFu;
    *(_BYTE *)(a1 + 32) &= 0xC3u;
  }
LABEL_9:
  if ( (_DWORD)v9 == 7 )
  {
    v41 = (unsigned int)a2[12];
    if ( (unsigned int)v41 <= 0x31 )
    {
      v42 = 0x2000003FC1FFELL;
      if ( _bittest64(&v42, v41) )
      {
        if ( v5 == 34 || v5 == 40 )
        {
          *(_BYTE *)(a1 + 34) &= 0xC0u;
          goto LABEL_54;
        }
        goto LABEL_102;
      }
    }
  }
  else if ( (_DWORD)v9 == 12 || (_DWORD)v9 == 3 )
  {
    goto LABEL_13;
  }
  *(_BYTE *)(a1 + 46) &= 0xCFu;
LABEL_13:
  if ( (v9 & 0xFFFFFFF7) == 1 && v5 == 38 )
  {
    if ( !(unsigned __int8)sub_CC8200(a2, 10, 7, 0) )
    {
      sub_981A80(a1, 0x133u, "fwrite$UNIX2003", 0xFu);
      sub_981A80(a1, 0x11Du, "fputs$UNIX2003", 0xEu);
    }
    v5 = a2[8];
    LODWORD(v9) = a2[11];
  }
  if ( v5 == 34 || v5 == 40 )
  {
    if ( (_DWORD)v9 == 37 )
      goto LABEL_17;
    goto LABEL_36;
  }
  if ( (_DWORD)v9 == 37 )
  {
LABEL_17:
    *(_DWORD *)a1 = 0;
    *(_BYTE *)(a1 + 4) = 0;
LABEL_18:
    v10 = a2[11];
    v11 = a2[12];
LABEL_19:
    v12 = v11;
    goto LABEL_20;
  }
LABEL_102:
  *(_BYTE *)(a1 + 81) &= 0xFCu;
  *(_BYTE *)(a1 + 111) &= 0xF3u;
  *(_BYTE *)(a1 + 64) &= 0xFCu;
LABEL_36:
  *(_BYTE *)(a1 + 34) &= 0xC0u;
  if ( (_DWORD)v9 != 14 )
    goto LABEL_54;
  v11 = a2[12];
  v12 = v11;
  if ( v11 == 29 )
  {
    v10 = a2[11];
    goto LABEL_112;
  }
  if ( v11 == 1 )
  {
    v10 = a2[11];
    goto LABEL_46;
  }
  v61 = 1;
  if ( v11 == 27 )
  {
    v14 = sub_CC7810(a2);
    v5 = a2[8];
    v61 = (unsigned int)(v14 - 1) > 0x11;
  }
  v55 = 1;
  v15 = *(_BYTE *)(a1 + 92);
  v16 = *(_BYTE *)(a1 + 80);
  v58 = *(_BYTE *)(a1 + 123);
  v17 = *(_BYTE *)(a1 + 69);
  v59 = *(_BYTE *)(a1 + 112);
  v18 = *(_BYTE *)(a1 + 52);
  v19 = *(_BYTE *)(a1 + 49);
  v56 = *(_BYTE *)(a1 + 102);
  v20 = *(_BYTE *)(a1 + 44);
  v21 = *(_BYTE *)(a1 + 58);
  v57 = *(_BYTE *)(a1 + 101);
  v22 = *(_BYTE *)(a1 + 43);
  v23 = *(_BYTE *)(a1 + 87);
  v63 = *(_BYTE *)(a1 + 42);
  v24 = *(_BYTE *)(a1 + 60);
  if ( (v5 & 0xFFFFFFFD) != 1 )
  {
    if ( v5 != 39 )
    {
      v63 &= 0xFCu;
      v22 &= 0x3Fu;
      v20 &= 0xF3u;
      *(_BYTE *)(a1 + 40) &= 0xF3u;
      v19 &= 0xF3u;
      v18 &= 0xF3u;
      v21 &= 0xCFu;
      *(_BYTE *)(a1 + 51) &= 0x3Fu;
      v17 &= 0xF3u;
      v16 &= 0xCFu;
      v23 &= 0xF3u;
      *(_BYTE *)(a1 + 64) &= 0x3Fu;
      v15 &= 0xF3u;
      *(_BYTE *)(a1 + 83) &= 0x3Fu;
      *(_BYTE *)(a1 + 96) &= 0x3Fu;
      v57 &= 0xF3u;
      v56 &= 0x3Cu;
      *(_BYTE *)(a1 + 109) &= 0x33u;
      v59 &= 0xF3u;
      *(_BYTE *)(a1 + 122) &= 0x3Fu;
      v58 &= 0xF3u;
      *(_BYTE *)(a1 + 42) = v63;
      v55 = 0;
    }
    v24 &= 0xFCu;
  }
  *(_BYTE *)(a1 + 69) = v17 & 0xCF;
  v25 = v19 & 0xCF;
  v26 = *(_BYTE *)(a1 + 41);
  v27 = v21 & 0x3F;
  v28 = v24 & 0xF3;
  *(_BYTE *)(a1 + 43) = v22 & 0xFC;
  v29 = v23 & 0xCF;
  v30 = *(_BYTE *)(a1 + 45);
  *(_BYTE *)(a1 + 44) = v20 & 0xFC;
  v31 = v26 & 0xF3;
  v32 = *(_BYTE *)(a1 + 73);
  *(_BYTE *)(a1 + 80) = v16 & 0x3F;
  *(_BYTE *)(a1 + 101) = v57 & 0xCF;
  v33 = *(_BYTE *)(a1 + 84);
  v34 = v30 & 0xF3;
  v35 = v32 & 0xFC;
  *(_BYTE *)(a1 + 72) &= 0x3Fu;
  v36 = v33 & 0xFC;
  *(_BYTE *)(a1 + 65) &= 0xFCu;
  *(_BYTE *)(a1 + 82) &= 0xFu;
  *(_BYTE *)(a1 + 97) &= 0xFCu;
  *(_BYTE *)(a1 + 103) &= 0xFCu;
  *(_BYTE *)(a1 + 102) = v56 & 0xF3;
  *(_BYTE *)(a1 + 41) = v31;
  *(_BYTE *)(a1 + 45) = v34;
  *(_BYTE *)(a1 + 49) = v25;
  *(_BYTE *)(a1 + 52) = v18 & 0xF;
  *(_BYTE *)(a1 + 58) = v27;
  *(_BYTE *)(a1 + 60) = v28;
  *(_BYTE *)(a1 + 73) = v35;
  *(_BYTE *)(a1 + 84) = v36;
  *(_BYTE *)(a1 + 87) = v29;
  *(_BYTE *)(a1 + 92) = v15 & 0xCF;
  *(_BYTE *)(a1 + 110) &= 0xF0u;
  *(_BYTE *)(a1 + 112) = v59 & 0xCF;
  *(_BYTE *)(a1 + 123) = v58 & 0xF;
  if ( v61 )
  {
    v37 = *(_BYTE *)(a1 + 85);
    v62 = *(_BYTE *)(a1 + 125);
    v38 = *(_BYTE *)(a1 + 86);
    v60 = *(_BYTE *)(a1 + 104) & 0xF;
    v39 = *(_BYTE *)(a1 + 106) & 0xF3;
    v40 = *(_BYTE *)(a1 + 107) & 0x3C;
  }
  else
  {
    *(_BYTE *)(a1 + 40) &= 0xFu;
    *(_BYTE *)(a1 + 44) = v20 & 0xC;
    *(_BYTE *)(a1 + 42) = v63 & 0xC3;
    sub_981A80(a1, 0xBDu, "_cabs", 5u);
    *(_BYTE *)(a1 + 47) &= 0xCFu;
    *(_BYTE *)(a1 + 48) &= 0xC3u;
    sub_981A80(a1, 0xCBu, "_copysign", 9u);
    sub_981A80(a1, 0xCCu, "_copysignf", 0xAu);
    *(_BYTE *)(a1 + 57) &= 0x3Fu;
    *(_BYTE *)(a1 + 58) &= 0xFCu;
    *(_BYTE *)(a1 + 59) &= 0xF0u;
    *(_BYTE *)(a1 + 66) &= 0x30u;
    *(_BYTE *)(a1 + 67) &= 0xFCu;
    *(_BYTE *)(a1 + 84) &= 0xC3u;
    *(_BYTE *)(a1 + 85) &= 0xF0u;
    sub_981A80(a1, 0x15Au, "_logb", 5u);
    v52 = *(_BYTE *)(a1 + 86);
    v37 = *(_BYTE *)(a1 + 85) & 0x3F;
    v38 = v52 & 0x3C;
    *(_BYTE *)(a1 + 85) = v37;
    *(_BYTE *)(a1 + 86) = v52 & 0xFC;
    if ( v55 )
    {
      sub_981A80(a1, 0x15Bu, "_logbf", 6u);
      v38 = *(_BYTE *)(a1 + 86);
      v37 = *(_BYTE *)(a1 + 85);
    }
    v53 = *(_BYTE *)(a1 + 125);
    v54 = *(_BYTE *)(a1 + 42);
    v60 = 0;
    *(_BYTE *)(a1 + 105) &= 0xFCu;
    v35 = *(_BYTE *)(a1 + 73);
    v63 = v54;
    v39 = 0;
    v28 = *(_BYTE *)(a1 + 60);
    v25 = *(_BYTE *)(a1 + 49);
    v29 = *(_BYTE *)(a1 + 87);
    v62 = v53 & 0xF0;
    v40 = 0;
    v36 = *(_BYTE *)(a1 + 84);
    v27 = *(_BYTE *)(a1 + 58);
    v34 = *(_BYTE *)(a1 + 45);
    v31 = *(_BYTE *)(a1 + 41);
  }
  *(_BYTE *)(a1 + 107) = v40;
  *(_BYTE *)(a1 + 41) = v31 & 0xFC;
  *(_BYTE *)(a1 + 47) &= 0x3Fu;
  *(_BYTE *)(a1 + 48) &= 0x3Fu;
  *(_BYTE *)(a1 + 51) &= 0xF3u;
  *(_BYTE *)(a1 + 59) &= 0xCFu;
  *(_BYTE *)(a1 + 66) &= 0xCFu;
  *(_BYTE *)(a1 + 67) &= 0xF3u;
  *(_BYTE *)(a1 + 39) &= 0x3Fu;
  *(_BYTE *)(a1 + 50) &= 0xCFu;
  *(_BYTE *)(a1 + 87) = v29 & 0xFC;
  *(_BYTE *)(a1 + 60) = v28 & 0x3F;
  *(_BYTE *)(a1 + 42) = v63 & 0x3F;
  *(_BYTE *)(a1 + 45) = v34 & 0xFC;
  *(_BYTE *)(a1 + 58) = v27 & 0xF3;
  *(_BYTE *)(a1 + 84) = v36 & 0x3F;
  *(_BYTE *)(a1 + 85) = v37 & 0xCF;
  *(_BYTE *)(a1 + 86) = v38 & 0xF3;
  *(_BYTE *)(a1 + 106) = v39;
  *(_BYTE *)(a1 + 125) = v62 & 0xCF;
  *(_BYTE *)(a1 + 49) = v25 & 0x3F;
  *(_BYTE *)(a1 + 63) &= 0x3Fu;
  *(_BYTE *)(a1 + 74) &= 0xCFu;
  *(_BYTE *)(a1 + 75) &= 0x3Fu;
  *(_BYTE *)(a1 + 79) &= 0xCFu;
  *(_BYTE *)(a1 + 88) &= 0x3Fu;
  *(_BYTE *)(a1 + 91) &= 0xCFu;
  *(_BYTE *)(a1 + 94) &= 0x3Cu;
  *(_BYTE *)(a1 + 95) &= 0xC3u;
  *(_BYTE *)(a1 + 96) &= 0xFCu;
  *(_BYTE *)(a1 + 99) &= 0xCFu;
  *(_BYTE *)(a1 + 113) &= 0xFCu;
  *(_BYTE *)(a1 + 114) &= 0xCFu;
  *(_BYTE *)(a1 + 104) = v60;
  *(_BYTE *)(a1 + 117) &= 0xF3u;
  *(_BYTE *)(a1 + 73) = v35 & 0x3F;
  *(_BYTE *)(a1 + 126) &= 0x33u;
  *(_BYTE *)(a1 + 130) &= 0xCFu;
  v10 = a2[11];
  LODWORD(v9) = v10;
  if ( v10 != 14 )
  {
LABEL_54:
    *(_DWORD *)a1 = 0;
    v43 = *(_BYTE *)(a1 + 57);
    *(_BYTE *)(a1 + 4) = 0;
    if ( (_DWORD)v9 == 9 )
    {
      *(_BYTE *)(a1 + 57) = v43 & 0xCF;
      if ( (unsigned __int8)sub_CC8200(a2, 10, 9, 0) )
      {
LABEL_56:
        *(_BYTE *)(a1 + 57) &= 0xF0u;
        v12 = a2[12];
        v10 = a2[11];
        goto LABEL_57;
      }
LABEL_86:
      sub_981A80(a1, 0xE4u, "__exp10", 7u);
      sub_981A80(a1, 0xE5u, "__exp10f", 8u);
      v12 = a2[12];
      v10 = a2[11];
LABEL_57:
      switch ( v10 )
      {
        case 1u:
        case 3u:
        case 5u:
        case 7u:
        case 9u:
        case 0x1Bu:
        case 0x1Cu:
        case 0x1Fu:
          goto LABEL_22;
        default:
          goto LABEL_21;
      }
      goto LABEL_22;
    }
    if ( (unsigned int)v9 <= 9 )
    {
      if ( (_DWORD)v9 != 5 )
        goto LABEL_18;
    }
    else
    {
      if ( (unsigned int)v9 <= 0x1C )
      {
        if ( (unsigned int)v9 <= 0x1A )
          goto LABEL_18;
        *(_BYTE *)(a1 + 57) = v43 & 0xCF;
        if ( (_DWORD)v9 == 28 )
          goto LABEL_86;
        goto LABEL_83;
      }
      if ( (_DWORD)v9 != 31 )
        goto LABEL_18;
    }
    *(_BYTE *)(a1 + 57) = v43 & 0xCF;
LABEL_83:
    if ( (unsigned int)sub_CC78E0(a2) <= 6 || (unsigned int)sub_CC78E0(a2) <= 8 && (unsigned int)(a2[8] - 38) <= 1 )
      goto LABEL_56;
    goto LABEL_86;
  }
  v11 = a2[12];
  v12 = v11;
  if ( v11 == 29 )
  {
LABEL_112:
    *(_DWORD *)a1 = 0;
    *(_BYTE *)(a1 + 4) = 0;
    goto LABEL_19;
  }
LABEL_46:
  *(_BYTE *)(a1 + 46) &= 0xFu;
  *(_BYTE *)(a1 + 47) &= 0xFCu;
  *(_BYTE *)(a1 + 50) &= 0xFCu;
  *(_BYTE *)(a1 + 53) &= 0xFCu;
  *(_BYTE *)(a1 + 61) &= 0x3Fu;
  *(_BYTE *)(a1 + 64) &= 0xF3u;
  *(_BYTE *)(a1 + 75) &= 0xFCu;
  *(_BYTE *)(a1 + 76) &= 0xC3u;
  *(_BYTE *)(a1 + 78) &= 0xFu;
  *(_BYTE *)(a1 + 79) &= 0x3Cu;
  *(_BYTE *)(a1 + 80) &= 0xFCu;
  *(_BYTE *)(a1 + 87) &= 0x3Fu;
  *(_BYTE *)(a1 + 90) &= 0xF3u;
  *(_BYTE *)(a1 + 94) &= 0xC3u;
  *(_BYTE *)(a1 + 97) &= 0xF3u;
  *(_BYTE *)(a1 + 99) &= 0x3Cu;
  *(_BYTE *)(a1 + 100) &= 0x3Fu;
  *(_BYTE *)(a1 + 108) &= 0xCFu;
  *(_BYTE *)(a1 + 113) &= 0xCFu;
  *(_BYTE *)(a1 + 114) &= 0xF0u;
  *(_BYTE *)(a1 + 124) &= 0xFCu;
  *(_BYTE *)(a1 + 125) &= 0x3Fu;
  *(_BYTE *)(a1 + 126) &= 0xCFu;
  *(_BYTE *)(a1 + 127) &= 0xFCu;
  *(_BYTE *)(a1 + 82) &= 0xCCu;
  if ( v11 != 27 && v11 > 1 && v11 != 28 )
    goto LABEL_112;
  *(_BYTE *)(a1 + 5) &= 0xFu;
  *(_BYTE *)(a1 + 16) &= 0xF0u;
  *(_WORD *)(a1 + 6) = 0;
  *(_QWORD *)(a1 + 8) = 0;
LABEL_20:
  *(_BYTE *)(a1 + 57) &= 0xC0u;
  if ( v10 <= 0x1F )
    goto LABEL_57;
LABEL_21:
  *(_BYTE *)(a1 + 62) &= 0xFCu;
LABEL_22:
  switch ( v10 )
  {
    case 1u:
    case 3u:
    case 5u:
    case 7u:
    case 9u:
    case 0x1Bu:
    case 0x1Cu:
    case 0x1Fu:
      break;
    default:
      *(_BYTE *)(a1 + 62) &= 0xF3u;
      break;
  }
  if ( v10 != 3 )
  {
    *(_BYTE *)(a1 + 65) &= 3u;
    if ( v10 == 7 && v12 - 1 <= 0xB )
      goto LABEL_62;
  }
  *(_BYTE *)(a1 + 37) &= 0xFCu;
  v46 = *(_BYTE *)(a1 + 27);
  *(_BYTE *)(a1 + 38) &= 0x3Fu;
  v47 = *(_BYTE *)(a1 + 88);
  *(_BYTE *)(a1 + 5) &= 0xF0u;
  if ( v12 != 49 && v10 != 39 && v12 - 17 > 8 )
    v47 &= 0xCFu;
  *(_BYTE *)(a1 + 17) = 0;
  *(_BYTE *)(a1 + 88) = v47 & 0xFC;
  *(_BYTE *)(a1 + 70) &= 0xFCu;
  *(_BYTE *)(a1 + 74) &= 0x3Cu;
  *(_BYTE *)(a1 + 75) &= 0xF3u;
  *(_BYTE *)(a1 + 76) &= 0xFCu;
  *(_BYTE *)(a1 + 95) &= 0xFCu;
  *(_BYTE *)(a1 + 113) &= 0x33u;
  *(_BYTE *)(a1 + 124) &= 0xCFu;
  *(_BYTE *)(a1 + 16) &= 0xFu;
  *(_BYTE *)(a1 + 20) &= 0x3Cu;
  *(_BYTE *)(a1 + 21) &= 0xF0u;
  *(_BYTE *)(a1 + 24) &= 0xFu;
  *(_BYTE *)(a1 + 31) &= 0xFu;
  *(_BYTE *)(a1 + 32) &= 0x3Cu;
  *(_WORD *)(a1 + 18) = 0;
  *(_WORD *)(a1 + 25) = 0;
  *(_BYTE *)(a1 + 27) = v46 & 0x3C;
  *(_WORD *)(a1 + 28) = 0;
  *(_BYTE *)(a1 + 33) &= 0xF0u;
  *(_BYTE *)(a1 + 35) &= 3u;
  if ( v10 == 7 && v12 - 1 <= 0xB )
    goto LABEL_62;
  if ( v12 != 17 )
    goto LABEL_66;
  v48 = !sub_97EFB0((__int64)a2, 0x1Cu);
  v12 = a2[12];
  if ( v48 )
  {
LABEL_62:
    *(_BYTE *)(a1 + 78) |= 3u;
    *(_BYTE *)(a1 + 98) |= 0x33u;
    *(_BYTE *)(a1 + 62) |= 0xC0u;
    *(_BYTE *)(a1 + 72) |= 3u;
    *(_BYTE *)(a1 + 77) |= 0x33u;
    *(_BYTE *)(a1 + 71) |= 0x33u;
    *(_BYTE *)(a1 + 63) |= 0x30u;
  }
  if ( v12 == 17 && sub_97EFB0((__int64)a2, 0x15u) )
    *(_BYTE *)(a1 + 114) &= 0xF0u;
LABEL_66:
  v44 = a2[8];
  if ( v44 == 39 )
  {
    v49 = a2[11];
    v45 = v49;
    if ( a2[10] == 3 && (unsigned int)(v49 - 24) <= 1 )
    {
      *(_BYTE *)(a1 + 6) &= 0xFu;
      *(_BYTE *)(a1 + 8) &= 0xFu;
      *(_BYTE *)(a1 + 9) &= 0xFu;
      *(_BYTE *)(a1 + 10) &= 0xF0u;
      *(_BYTE *)(a1 + 12) &= 0xFu;
      *(_BYTE *)(a1 + 13) &= 0xF0u;
      *(_BYTE *)(a1 + 31) &= 0xFCu;
      *(_BYTE *)(a1 + 34) &= 0x3Fu;
      *(_BYTE *)(a1 + 35) &= 0xFCu;
      *(_BYTE *)(a1 + 37) &= 3u;
      *(_BYTE *)(a1 + 39) &= 0x30u;
      *(_BYTE *)(a1 + 49) &= 0x3Fu;
      *(_BYTE *)(a1 + 50) &= 0xCCu;
      *(_BYTE *)(a1 + 53) &= 0xFCu;
      *(_BYTE *)(a1 + 54) &= 0x3Fu;
      *(_BYTE *)(a1 + 30) = 0;
      *(_BYTE *)(a1 + 36) = 0;
      *(_BYTE *)(a1 + 55) = 0;
      *(_BYTE *)(a1 + 56) &= 0xC0u;
      *(_BYTE *)(a1 + 70) &= 0xF3u;
      *(_BYTE *)(a1 + 74) &= 0xCFu;
      *(_BYTE *)(a1 + 78) &= 3u;
      *(_BYTE *)(a1 + 82) &= 0xFCu;
      *(_BYTE *)(a1 + 87) &= 0x3Fu;
      *(_BYTE *)(a1 + 91) &= 0xCFu;
      *(_BYTE *)(a1 + 95) &= 0xC3u;
      *(_BYTE *)(a1 + 96) &= 0xFCu;
      *(_BYTE *)(a1 + 97) &= 0xF3u;
      *(_BYTE *)(a1 + 99) &= 0xCu;
      *(_BYTE *)(a1 + 103) &= 0xCFu;
      *(_BYTE *)(a1 + 104) &= 0x3Fu;
      *(_BYTE *)(a1 + 108) &= 0xCFu;
      *(_BYTE *)(a1 + 113) &= 0xCCu;
      *(_BYTE *)(a1 + 122) &= 0xF3u;
      *(_BYTE *)(a1 + 125) &= 0x3Fu;
      *(_BYTE *)(a1 + 126) &= 3u;
      *(_BYTE *)(a1 + 47) = 0;
      *(_BYTE *)(a1 + 127) &= 0xF0u;
      *(_BYTE *)(a1 + 130) &= 0xCFu;
      *(_BYTE *)(a1 + 20) &= 0xC3u;
      *(_BYTE *)(a1 + 27) &= 0xC3u;
      *(_BYTE *)(a1 + 38) &= 0xC0u;
      *(_BYTE *)(a1 + 46) &= 0xFu;
      *(_BYTE *)(a1 + 61) &= 0x3Fu;
      *(_BYTE *)(a1 + 64) &= 0xF3u;
      *(_BYTE *)(a1 + 73) &= 0x3Fu;
      *(_BYTE *)(a1 + 75) &= 0x3Cu;
      *(_BYTE *)(a1 + 76) &= 0xC3u;
      *(_BYTE *)(a1 + 79) &= 0xCu;
      *(_BYTE *)(a1 + 80) &= 0xFCu;
      *(_BYTE *)(a1 + 81) &= 0xF3u;
      v51 = *(_BYTE *)(a1 + 88);
      *(_BYTE *)(a1 + 90) &= 0xF0u;
      *(_BYTE *)(a1 + 94) &= 3u;
      *(_BYTE *)(a1 + 100) &= 3u;
      *(_BYTE *)(a1 + 88) = v51 & 0xF | 0x30;
      v45 = v49;
      *(_BYTE *)(a1 + 105) &= 3u;
      *(_BYTE *)(a1 + 114) &= 0xF0u;
      *(_BYTE *)(a1 + 116) &= 0xFu;
      *(_BYTE *)(a1 + 118) &= 0xC3u;
      *(_BYTE *)(a1 + 124) &= 0x30u;
    }
    goto LABEL_101;
  }
  if ( v44 != 42 && v44 != 43 )
  {
    v45 = a2[11];
LABEL_101:
    *(_BYTE *)(a1 + 31) &= 0xF3u;
    goto LABEL_69;
  }
  sub_97F7E0((_QWORD *)a1);
  *(_BYTE *)(a1 + 31) |= 0xCu;
  v45 = a2[11];
  *(_BYTE *)(a1 + 88) |= 0xCu;
  *(_BYTE *)(a1 + 72) |= 0xCu;
  *(_BYTE *)(a1 + 27) |= 0x3Cu;
LABEL_69:
  if ( v45 == 19 )
  {
    *(_BYTE *)(a1 + 90) &= 0xF3u;
  }
  else
  {
    *(_BYTE *)(a1 + 128) &= 0xF0u;
    *(_BYTE *)(a1 + 127) &= 0xFu;
  }
  sub_982B20((__m128i **)a1, qword_4F7FCC8, (__int64)a2);
}
