// Function: sub_38CBC00
// Address: 0x38cbc00
//
__int64 __fastcall sub_38CBC00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r15d
  char v6; // r10
  char v7; // si
  char v8; // r9
  char v9; // dl
  char v10; // r8
  char v11; // cl
  char v12; // dl
  bool v13; // dl
  char v14; // r8
  bool v15; // r9
  bool v16; // r11
  bool v17; // r11
  bool v18; // dl
  char v19; // dl
  char v20; // r11
  bool v21; // r14
  char v22; // r11
  bool v23; // dl
  bool v24; // bl
  bool v25; // r8
  char v26; // r12
  char v27; // dl
  char v28; // r11
  char v29; // bl
  bool v30; // r13
  char v31; // r12
  char v32; // bl
  char v33; // dl
  bool v34; // bl
  bool v35; // r11
  char v36; // r12
  char v37; // r13
  char v38; // bl
  char v39; // r11
  char v40; // r11
  char v41; // dl
  bool v42; // r13
  char v43; // dl
  unsigned __int8 v44; // r13
  bool v45; // r12
  char v46; // al
  bool v47; // r12
  char v48; // r11
  char v49; // al
  bool v50; // dl
  bool v52; // al
  char v53; // r11
  char v54; // r11
  char v55; // r11
  char v56; // al
  char v57; // bl
  char v58; // r11
  char v59; // bl
  char v60; // dl
  char v61; // r9
  char v62; // dl
  char v63; // r12
  char v64; // [rsp+8h] [rbp-68h]
  char v65; // [rsp+9h] [rbp-67h]
  char v66; // [rsp+9h] [rbp-67h]
  char v67; // [rsp+Ah] [rbp-66h]
  char v68; // [rsp+Bh] [rbp-65h]
  char v69; // [rsp+Bh] [rbp-65h]
  bool v70; // [rsp+Ch] [rbp-64h]
  bool v71; // [rsp+Dh] [rbp-63h]
  char v72; // [rsp+Eh] [rbp-62h]
  bool v73; // [rsp+Fh] [rbp-61h]
  _QWORD v74[2]; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 v75; // [rsp+20h] [rbp-50h] BYREF
  __int64 v76; // [rsp+28h] [rbp-48h]
  __int64 v77; // [rsp+30h] [rbp-40h] BYREF

  v74[0] = a1;
  v74[1] = a2;
  sub_16D2060(&v75, v74, a3, a4, a5);
  if ( v76 == 6 )
  {
    if ( *(_DWORD *)v75 == 1919972452 && *(_WORD *)(v75 + 4) == 27749 )
    {
      v5 = 119;
    }
    else
    {
      if ( *(_DWORD *)v75 != 1869640804 || *(_WORD *)(v75 + 4) != 26214 )
      {
        if ( *(_DWORD *)v75 == 1869901671 && *(_WORD *)(v75 + 4) == 26214 )
        {
          v5 = 3;
        }
        else
        {
          if ( *(_DWORD *)v75 != 1920233319 || *(_WORD *)(v75 + 4) != 27749 )
          {
            v7 = 1;
            v6 = 0;
            goto LABEL_275;
          }
          v5 = 4;
        }
        v72 = 0;
        v6 = 0;
LABEL_389:
        v10 = 1;
        v11 = 1;
        v7 = v76 == 6;
        v60 = 0;
        goto LABEL_289;
      }
      v5 = 15;
    }
    v6 = 0;
LABEL_388:
    v72 = v76 == 9;
    goto LABEL_389;
  }
  if ( v76 != 3 )
  {
    v6 = 0;
    if ( v76 == 8 )
    {
      if ( *(_QWORD *)v75 == 0x6C65726370746F67LL )
      {
        v5 = 5;
      }
      else
      {
        if ( *(_QWORD *)v75 != 0x66666F7074746F67LL )
        {
          v72 = 0;
          v60 = 1;
          v10 = 0;
          v7 = 0;
          v11 = 0;
          goto LABEL_289;
        }
        v5 = 6;
      }
      v72 = 0;
      v60 = 0;
      v10 = 1;
      v7 = 0;
      v11 = 1;
      goto LABEL_289;
    }
    goto LABEL_4;
  }
  v6 = 1;
  if ( *(_WORD *)v75 == 28519 && *(_BYTE *)(v75 + 2) == 116 )
  {
    v5 = 2;
    goto LABEL_388;
  }
LABEL_4:
  v7 = v76 == 6;
  if ( v76 == 9 )
  {
    if ( *(_QWORD *)v75 == 0x666F70746E646E69LL && *(_BYTE *)(v75 + 8) == 102 )
    {
      v8 = 0;
      v9 = 0;
      v10 = 1;
      v11 = 1;
    }
    else
    {
      v8 = 1;
      v9 = 1;
      v10 = 0;
      v11 = 0;
    }
    v72 = 1;
    v5 = 7;
    v12 = v6 & v9;
    if ( !v8 )
      goto LABEL_8;
LABEL_290:
    if ( *(_QWORD *)v75 == 0x666F70746E746F67LL && *(_BYTE *)(v75 + 8) == 102 )
    {
      v5 = 9;
      v11 = 1;
      goto LABEL_168;
    }
    goto LABEL_9;
  }
LABEL_275:
  if ( !v7 )
  {
    v72 = 0;
    v10 = 0;
    v60 = 1;
    v11 = 0;
LABEL_289:
    v61 = v60 & v72;
    v12 = v6 & v60;
    if ( !v61 )
      goto LABEL_8;
    goto LABEL_290;
  }
  if ( *(_DWORD *)v75 == 1869640814 && *(_WORD *)(v75 + 4) == 26214 )
  {
    v72 = 0;
    v5 = 8;
    v11 = 1;
    goto LABEL_168;
  }
  v72 = 0;
  v12 = v6;
  v10 = 0;
  v11 = 0;
LABEL_8:
  if ( v12 )
  {
    if ( *(_WORD *)v75 != 27760 || *(_BYTE *)(v75 + 2) != 116 )
    {
LABEL_11:
      v10 = v11;
      if ( v76 != 5 )
        goto LABEL_12;
      if ( *(_DWORD *)v75 == 1735617652 && *(_BYTE *)(v75 + 4) == 100 )
      {
        v5 = 11;
LABEL_449:
        v71 = 0;
        v11 = 1;
        goto LABEL_169;
      }
      if ( *(_DWORD *)v75 == 1819503732 && *(_BYTE *)(v75 + 4) == 100 )
      {
        v5 = 12;
        goto LABEL_449;
      }
      if ( v11 )
      {
        v71 = 0;
        goto LABEL_169;
      }
      if ( *(_DWORD *)v75 == 1718579316 && *(_BYTE *)(v75 + 4) == 102 )
      {
        v5 = 14;
      }
      else
      {
        if ( *(_DWORD *)v75 != 1701998708 || *(_BYTE *)(v75 + 4) != 108 )
        {
          v14 = 0;
          v13 = 1;
LABEL_361:
          v71 = 0;
          v15 = 0;
          v73 = 0;
          goto LABEL_170;
        }
        v5 = 118;
      }
      v14 = 1;
      v13 = 0;
      v11 = 1;
      goto LABEL_361;
    }
    v5 = 10;
    v11 = 1;
LABEL_168:
    v71 = v76 == 4;
    goto LABEL_169;
  }
LABEL_9:
  if ( v10 )
    goto LABEL_168;
  if ( v76 != 7 )
    goto LABEL_11;
  if ( *(_DWORD *)v75 == 1668508788 && *(_WORD *)(v75 + 4) == 27745 && *(_BYTE *)(v75 + 6) == 108 )
  {
    v5 = 16;
    goto LABEL_167;
  }
  if ( *(_DWORD *)v75 == 1685286004 && *(_WORD *)(v75 + 4) == 29541 && *(_BYTE *)(v75 + 6) == 99 )
  {
    v5 = 17;
LABEL_167:
    v11 = 1;
    goto LABEL_168;
  }
LABEL_12:
  if ( ((unsigned __int8)v7 & ((unsigned __int8)v10 ^ 1)) == 0 )
  {
    v71 = v76 == 4;
    if ( !v10 )
      goto LABEL_15;
LABEL_169:
    v14 = 1;
    v73 = v76 == 8;
    v15 = v76 == 11;
    v13 = 0;
    goto LABEL_170;
  }
  if ( *(_DWORD *)v75 == 1819503732 && *(_WORD *)(v75 + 4) == 28004 )
  {
    v7 &= v10 ^ 1;
    v11 = 1;
    v5 = 13;
    v71 = v76 == 4;
    goto LABEL_169;
  }
  v7 &= v10 ^ 1;
LABEL_15:
  v71 = v76 == 4;
  v73 = v76 == 8;
  if ( v76 == 4 )
  {
    v71 = 1;
    if ( *(_DWORD *)v75 == 1886809204 )
      v11 = 1;
    v13 = *(_DWORD *)v75 != 1886809204;
    v14 = *(_DWORD *)v75 == 1886809204;
    if ( *(_DWORD *)v75 == 1886809204 )
      v5 = 18;
    v15 = 0;
LABEL_21:
    v16 = v13 && v71;
    goto LABEL_22;
  }
  v15 = v76 == 11;
  if ( v76 != 8 )
  {
    v13 = 1;
    v14 = 0;
LABEL_170:
    if ( v15 && v13 )
    {
      if ( *(_QWORD *)v75 == 0x6567617070766C74LL && *(_WORD *)(v75 + 8) == 26223 && *(_BYTE *)(v75 + 10) == 102 )
      {
        v11 = 1;
        v14 = v15 && v13;
        v5 = 20;
      }
      goto LABEL_23;
    }
    goto LABEL_21;
  }
  v16 = 0;
  v73 = 1;
  if ( *(_QWORD *)v75 != 0x6567617070766C74LL )
    v16 = v76 == 4;
  v13 = *(_QWORD *)v75 != 0x6567617070766C74LL;
  v14 = *(_QWORD *)v75 == 0x6567617070766C74LL;
  if ( *(_QWORD *)v75 == 0x6567617070766C74LL )
  {
    v11 = 1;
    v5 = 19;
  }
LABEL_22:
  if ( v16 )
  {
    if ( *(_DWORD *)v75 != 1701273968 )
      goto LABEL_25;
    v11 = 1;
    v5 = 21;
    v70 = v76 == 10;
LABEL_150:
    v13 = 0;
    v14 = 1;
    goto LABEL_151;
  }
LABEL_23:
  v70 = v76 == 10;
  if ( v14 )
    goto LABEL_150;
  v13 = 1;
  if ( v76 == 7 )
  {
    if ( *(_DWORD *)v75 == 1701273968 && *(_WORD *)(v75 + 4) == 26223 && *(_BYTE *)(v75 + 6) == 102 )
    {
      v5 = 22;
    }
    else
    {
      if ( *(_DWORD *)v75 != 1886678887 || *(_WORD *)(v75 + 4) != 26465 || *(_BYTE *)(v75 + 6) != 101 )
      {
        v70 = 0;
        v54 = v7;
        v13 = 1;
LABEL_152:
        if ( v54 )
        {
          if ( *(_DWORD *)v75 == 1919380841 && *(_WORD *)(v75 + 4) == 27749 )
          {
            v11 = 1;
            v14 = v54;
            v5 = 100;
          }
LABEL_30:
          if ( !v14 )
          {
            if ( !v71 )
              goto LABEL_32;
            if ( *(_DWORD *)v75 == 1702521203 )
            {
              v5 = 26;
            }
            else
            {
              if ( *(_DWORD *)v75 != 947085921 )
                goto LABEL_33;
              v5 = 28;
            }
            v11 = 1;
            v67 = v76 == 2;
LABEL_140:
            v19 = 0;
            v14 = 1;
            goto LABEL_141;
          }
LABEL_139:
          v67 = v76 == 2;
          goto LABEL_140;
        }
        goto LABEL_153;
      }
      v5 = 23;
    }
    v70 = 0;
    v13 = 0;
    v14 = 1;
    v11 = 1;
LABEL_153:
    v18 = v73 && v13;
    goto LABEL_29;
  }
LABEL_25:
  v70 = v76 == 10;
  v17 = v13 && v76 == 10;
  if ( !v17 )
  {
LABEL_151:
    v54 = v7 & v13;
    goto LABEL_152;
  }
  if ( *(_QWORD *)v75 == 0x6F65676170746F67LL && *(_WORD *)(v75 + 8) == 26214 )
  {
    v14 = v13 && v76 == 10;
    v18 = 0;
    v11 = 1;
    v5 = 24;
  }
  else
  {
    v18 = v73;
  }
  v70 = v17;
LABEL_29:
  if ( !v18 )
    goto LABEL_30;
  if ( *(_QWORD *)v75 == 0x32336C6572636573LL )
  {
    v11 = 1;
    v5 = 25;
    goto LABEL_139;
  }
LABEL_32:
  v14 = v11;
  if ( v76 == 1 )
  {
    if ( *(_BYTE *)v75 == 108 )
    {
      v5 = 44;
    }
    else
    {
      if ( *(_BYTE *)v75 != 104 )
      {
        v67 = 0;
        v14 = v11;
        v19 = v11 ^ 1;
        v53 = (v11 ^ 1) & v71;
LABEL_142:
        v21 = v76 == 5;
        if ( v53 )
        {
          if ( *(_DWORD *)v75 == 1751607656 )
          {
            v14 = v53;
            v11 = 1;
            v5 = 47;
            goto LABEL_42;
          }
          v22 = v7 & v19;
          goto LABEL_41;
        }
        goto LABEL_143;
      }
      v5 = 45;
    }
    v67 = 0;
    v19 = 0;
    v14 = 1;
    v21 = 0;
    v11 = 1;
LABEL_143:
    if ( ((unsigned __int8)v19 & v21) != 0 )
    {
      if ( *(_DWORD *)v75 == 1751607656 && *(_BYTE *)(v75 + 4) == 97 )
      {
        v11 = 1;
        v14 = v19 & v21;
        v5 = 48;
      }
      goto LABEL_42;
    }
    goto LABEL_40;
  }
LABEL_33:
  v19 = v14 ^ 1;
  v67 = v76 == 2;
  v20 = (v14 ^ 1) & (v76 == 2);
  if ( !v20 )
  {
LABEL_141:
    v53 = v19 & v71;
    goto LABEL_142;
  }
  v19 = 0;
  v67 = (v14 ^ 1) & (v76 == 2);
  if ( *(_WORD *)v75 == 24936 )
    v11 = 1;
  v14 = 0;
  if ( *(_WORD *)v75 == 24936 )
  {
    v14 = v20;
    v5 = 46;
  }
  else
  {
    v19 = v20;
  }
  v21 = 0;
LABEL_40:
  v22 = v19 & v7;
LABEL_41:
  if ( v22 )
  {
    if ( *(_DWORD *)v75 == 1751607656 && *(_WORD *)(v75 + 4) == 29285 )
    {
      v5 = 49;
      v11 = 1;
      goto LABEL_182;
    }
    goto LABEL_44;
  }
LABEL_42:
  if ( v14 )
    goto LABEL_182;
  v19 = 1;
  if ( v76 == 7 )
  {
    if ( *(_DWORD *)v75 == 1751607656 && *(_WORD *)(v75 + 4) == 29285 && *(_BYTE *)(v75 + 6) == 97 )
    {
      v5 = 50;
    }
    else
    {
      if ( *(_DWORD *)v75 != 1751607656 || *(_WORD *)(v75 + 4) != 29541 || *(_BYTE *)(v75 + 6) != 116 )
        goto LABEL_255;
      v5 = 51;
    }
    v11 = 1;
    goto LABEL_182;
  }
LABEL_44:
  v23 = v73 & v19;
  if ( v23 )
  {
    v73 = v23;
    if ( *(_QWORD *)v75 != 0x6174736568676968LL )
    {
      v24 = v21;
      v21 = v14;
      goto LABEL_47;
    }
    v5 = 52;
    v11 = 1;
LABEL_182:
    v28 = 1;
    v25 = v76 == 7;
    v27 = 0;
    goto LABEL_183;
  }
  if ( v14 )
    goto LABEL_182;
LABEL_255:
  v24 = 0;
  if ( v21 )
  {
    if ( *(_DWORD *)v75 == 1081372519 && *(_BYTE *)(v75 + 4) == 108 )
    {
      v5 = 53;
    }
    else
    {
      if ( *(_DWORD *)v75 != 1081372519 || *(_BYTE *)(v75 + 4) != 104 )
      {
        v25 = v76 == 7;
        v21 = 0;
        goto LABEL_259;
      }
      v5 = 54;
    }
    v28 = v21;
    v11 = 1;
    v25 = v76 == 7;
    v27 = 0;
LABEL_183:
    v26 = v25 & v27;
    v29 = v6 & v27;
    if ( (v25 & (unsigned __int8)v27) == 0 )
      goto LABEL_51;
LABEL_184:
    if ( *(_DWORD *)v75 == 1650683764 && *(_WORD *)(v75 + 4) == 29537 && *(_BYTE *)(v75 + 6) == 101 )
    {
      v11 = 1;
      v28 = v26;
      v5 = 56;
    }
    goto LABEL_52;
  }
LABEL_47:
  if ( v7 )
  {
    v25 = v76 == 7;
    if ( *(_DWORD *)v75 == 1081372519 && *(_WORD *)(v75 + 4) == 24936 )
    {
      v28 = v7;
      v26 = 0;
      v27 = 0;
      v11 = 1;
      v5 = 55;
    }
    else
    {
      v26 = v76 == 7;
      v27 = v7;
      v28 = v21;
    }
    v21 = v24;
    v29 = v6 & v27;
    if ( !v26 )
    {
LABEL_51:
      if ( !v29 )
        goto LABEL_52;
LABEL_262:
      if ( *(_WORD *)v75 != 28532 || *(_BYTE *)(v75 + 2) != 99 )
      {
        v30 = v21;
        v21 = v28;
        v28 = v27;
        goto LABEL_55;
      }
      v5 = 57;
      v11 = 1;
LABEL_218:
      v33 = 1;
      v28 = 0;
LABEL_219:
      v57 = v6 & v28;
      goto LABEL_220;
    }
    goto LABEL_184;
  }
  v25 = v76 == 7;
  if ( !v24 )
  {
    v28 = v21;
    v27 = 1;
    v21 = 0;
    goto LABEL_183;
  }
LABEL_259:
  if ( *(_DWORD *)v75 == 1633906540 && *(_BYTE *)(v75 + 4) == 108 )
  {
    v59 = 0;
    v27 = 0;
    v28 = 1;
    v11 = 1;
    v5 = 99;
  }
  else
  {
    v59 = v6;
    v28 = v21;
    v27 = 1;
  }
  v21 = 1;
  if ( v59 )
    goto LABEL_262;
LABEL_52:
  if ( v28 )
    goto LABEL_218;
  if ( !v21 )
  {
    v30 = 0;
    v28 = 1;
LABEL_55:
    v31 = v7 & v28;
    if ( ((unsigned __int8)v7 & (unsigned __int8)v28) != 0 )
    {
      if ( *(_DWORD *)v75 == 1080258420 && *(_WORD *)(v75 + 4) == 24936 )
      {
        v33 = v7 & v28;
        v32 = 0;
        v28 = 0;
        v11 = 1;
        v5 = 60;
      }
      else
      {
        v32 = v7 & v28;
        v28 &= v7;
        v33 = v21;
      }
      v7 = v31;
      v21 = v30;
      if ( !v32 )
        goto LABEL_59;
LABEL_222:
      if ( *(_DWORD *)v75 != 1836086372 || *(_WORD *)(v75 + 4) != 25711 )
      {
        v34 = v25;
        v25 = v33;
        v33 = v28;
        goto LABEL_62;
      }
      v5 = 61;
      v11 = 1;
LABEL_199:
      v36 = 1;
      v33 = 0;
LABEL_200:
      v55 = v33 & v70;
      goto LABEL_201;
    }
    v33 = v21;
    v21 = v30;
    goto LABEL_219;
  }
  if ( *(_DWORD *)v75 == 1080258420 && *(_BYTE *)(v75 + 4) == 108 )
  {
    v5 = 58;
LABEL_491:
    v33 = v21;
    v11 = 1;
LABEL_221:
    if ( ((unsigned __int8)v28 & (unsigned __int8)v7) == 0 )
      goto LABEL_59;
    goto LABEL_222;
  }
  if ( *(_DWORD *)v75 == 1080258420 && *(_BYTE *)(v75 + 4) == 104 )
  {
    v5 = 59;
    goto LABEL_491;
  }
  v57 = v6;
  v33 = 0;
  v28 = v21;
LABEL_220:
  if ( !v57 )
    goto LABEL_221;
  if ( *(_WORD *)v75 == 27764 && *(_BYTE *)(v75 + 2) == 115 )
  {
    v11 = 1;
    v33 = v57;
    v5 = 88;
  }
LABEL_59:
  if ( v33 )
    goto LABEL_199;
  if ( v25 )
  {
    if ( *(_DWORD *)v75 == 1701998708 && *(_WORD *)(v75 + 4) == 16492 && *(_BYTE *)(v75 + 6) == 108 )
    {
      v5 = 62;
    }
    else
    {
      if ( *(_DWORD *)v75 != 1701998708 || *(_WORD *)(v75 + 4) != 16492 || *(_BYTE *)(v75 + 6) != 104 )
      {
        v55 = v70;
        v36 = 0;
        v33 = v25;
LABEL_201:
        if ( v55 )
        {
          v38 = v76 == 12;
          if ( *(_QWORD *)v75 == 0x6968406C65727074LL && *(_WORD *)(v75 + 8) == 26727 )
          {
            v36 = v55;
            v11 = 1;
            v5 = 65;
            goto LABEL_68;
          }
          v39 = v38 & v33;
          goto LABEL_67;
        }
        goto LABEL_202;
      }
      v5 = 63;
    }
    v36 = v25;
    v11 = 1;
LABEL_202:
    v37 = v15 & v33;
    goto LABEL_66;
  }
  v34 = 0;
  v33 = 1;
LABEL_62:
  v35 = v33 & v73;
  if ( ((unsigned __int8)v33 & v73) == 0 )
  {
    v36 = v25;
    v25 = v34;
    goto LABEL_200;
  }
  if ( *(_QWORD *)v75 == 0x6168406C65727074LL )
  {
    v36 = v33 & v73;
    v37 = 0;
    v33 = 0;
    v11 = 1;
    v5 = 64;
  }
  else
  {
    v37 = v15;
    v33 &= v73;
    v36 = v25;
  }
  v73 = v35;
  v25 = v34;
LABEL_66:
  v38 = v76 == 12;
  v39 = (v76 == 12) & v33;
  if ( v37 )
  {
    if ( *(_QWORD *)v75 == 0x6968406C65727074LL && *(_WORD *)(v75 + 8) == 26727 && *(_BYTE *)(v75 + 10) == 97 )
    {
      v11 = 1;
      v36 = v37;
      v5 = 66;
    }
    goto LABEL_68;
  }
LABEL_67:
  if ( v39 )
  {
    if ( *(_QWORD *)v75 == 0x6968406C65727074LL && *(_DWORD *)(v75 + 8) == 1919248487 )
    {
      v5 = 67;
      v11 = 1;
      v40 = v76 == 14;
      goto LABEL_193;
    }
    goto LABEL_70;
  }
LABEL_68:
  v40 = v76 == 14;
  if ( v36 )
    goto LABEL_193;
  v33 = 1;
  if ( v76 == 13 )
  {
    if ( *(_QWORD *)v75 == 0x6968406C65727074LL && *(_DWORD *)(v75 + 8) == 1919248487 && *(_BYTE *)(v75 + 12) == 97 )
    {
      v5 = 68;
    }
    else
    {
      if ( *(_QWORD *)v75 != 0x6968406C65727074LL || *(_DWORD *)(v75 + 8) != 1936025703 || *(_BYTE *)(v75 + 12) != 116 )
      {
        v40 = 0;
LABEL_250:
        v42 = 0;
        if ( v73 )
        {
          if ( *(_QWORD *)v75 == 0x6C406C6572707464LL )
          {
            v5 = 71;
          }
          else
          {
            if ( *(_QWORD *)v75 != 0x68406C6572707464LL )
            {
              v66 = v15;
              v43 = v73;
              v68 = 0;
LABEL_195:
              if ( v66 )
              {
                v44 = v76 == 13;
                if ( *(_QWORD *)v75 == 0x68406C6572707464LL
                  && *(_WORD *)(v75 + 8) == 26473
                  && *(_BYTE *)(v75 + 10) == 104 )
                {
                  v5 = 74;
                  v68 = v66;
                  v11 = 1;
                  goto LABEL_79;
                }
                v65 = v43 & v44;
                goto LABEL_78;
              }
              goto LABEL_196;
            }
            v5 = 72;
          }
          v43 = 0;
          v68 = v73;
          v11 = 1;
LABEL_196:
          v64 = v38 & v43;
          goto LABEL_77;
        }
        goto LABEL_73;
      }
      v5 = 69;
    }
    v40 = 0;
    v11 = 1;
LABEL_193:
    v68 = 1;
    v43 = 0;
LABEL_194:
    v66 = v43 & v15;
    goto LABEL_195;
  }
LABEL_70:
  v40 = v76 == 14;
  v41 = (v76 == 14) & v33;
  if ( !v41 )
  {
    if ( !v36 )
      goto LABEL_250;
    goto LABEL_193;
  }
  if ( *(_QWORD *)v75 == 0x6968406C65727074LL && *(_DWORD *)(v75 + 8) == 1936025703 && *(_WORD *)(v75 + 12) == 24948 )
  {
    v40 = v41;
    v5 = 70;
    v11 = 1;
    goto LABEL_193;
  }
  v42 = v73;
  v73 = v36;
  v40 = v41;
LABEL_73:
  if ( !v72 )
  {
    v62 = v73;
    v73 = v42;
    v68 = v62;
    v43 = 1;
    goto LABEL_194;
  }
  if ( *(_QWORD *)v75 == 0x68406C6572707464LL && *(_BYTE *)(v75 + 8) == 97 )
  {
    v64 = 0;
    v43 = 0;
    v5 = 73;
    v68 = v72;
    v11 = 1;
  }
  else
  {
    v43 = v72;
    v64 = v38;
    v68 = v73;
  }
  v73 = v42;
LABEL_77:
  v44 = v76 == 13;
  v65 = (v76 == 13) & v43;
  if ( v64 )
  {
    if ( *(_QWORD *)v75 == 0x68406C6572707464LL )
    {
      v63 = v64;
      if ( *(_DWORD *)(v75 + 8) == 1634232169 )
        v11 = 1;
      else
        v63 = v68;
      v68 = v63;
      if ( *(_DWORD *)(v75 + 8) == 1634232169 )
        v5 = 75;
    }
    goto LABEL_79;
  }
LABEL_78:
  if ( v65 )
  {
    if ( *(_QWORD *)v75 != 0x68406C6572707464LL || *(_DWORD *)(v75 + 8) != 1701341033 || *(_BYTE *)(v75 + 12) != 114 )
    {
      v40 = v68;
LABEL_81:
      if ( ((unsigned __int8)v43 & (v76 == 15)) != 0 )
      {
        if ( *(_QWORD *)v75 == 0x68406C6572707464LL
          && *(_DWORD *)(v75 + 8) == 1701341033
          && *(_WORD *)(v75 + 12) == 29811
          && *(_BYTE *)(v75 + 14) == 97 )
        {
          v40 = v43 & (v76 == 15);
          v11 = 1;
          v5 = 79;
        }
        goto LABEL_83;
      }
      goto LABEL_158;
    }
    v5 = 76;
    v11 = 1;
LABEL_157:
    v43 = 0;
    v40 = 1;
LABEL_158:
    v69 = v43 & v72;
    goto LABEL_159;
  }
LABEL_79:
  if ( v68 )
    goto LABEL_157;
  v43 = 1;
  if ( !v40 )
    goto LABEL_81;
  if ( *(_QWORD *)v75 == 0x68406C6572707464LL && *(_DWORD *)(v75 + 8) == 1701341033 && *(_WORD *)(v75 + 12) == 24946 )
  {
    v5 = 77;
LABEL_521:
    v11 = 1;
    goto LABEL_83;
  }
  if ( *(_QWORD *)v75 == 0x68406C6572707464LL && *(_DWORD *)(v75 + 8) == 1701341033 && *(_WORD *)(v75 + 12) == 29811 )
  {
    v5 = 78;
    goto LABEL_521;
  }
  v43 = v40;
  v40 = 0;
  v69 = v72;
LABEL_159:
  if ( v69 )
  {
    if ( *(_QWORD *)v75 != 0x6572707440746F67LL || *(_BYTE *)(v75 + 8) != 108 )
    {
      v45 = v15;
      v15 = v40;
      goto LABEL_85;
    }
    v5 = 80;
    v11 = 1;
LABEL_233:
    v46 = 1;
    v43 = 0;
LABEL_234:
    v58 = v43 & v70;
    goto LABEL_235;
  }
LABEL_83:
  if ( v40 )
    goto LABEL_233;
  v45 = 0;
  v43 = 1;
  if ( v15 )
  {
    if ( *(_QWORD *)v75 == 0x6572707440746F67LL && *(_WORD *)(v75 + 8) == 16492 && *(_BYTE *)(v75 + 10) == 108 )
    {
      v5 = 81;
    }
    else
    {
      if ( *(_QWORD *)v75 != 0x6572707440746F67LL || *(_WORD *)(v75 + 8) != 16492 || *(_BYTE *)(v75 + 10) != 104 )
      {
        v58 = v70;
        v43 = v15;
        v46 = 0;
LABEL_235:
        if ( v58 )
        {
          if ( *(_QWORD *)v75 != 0x7270746440746F67LL || *(_WORD *)(v75 + 8) != 27749 )
          {
LABEL_238:
            if ( (v44 & (unsigned __int8)v43) != 0 )
            {
              if ( *(_QWORD *)v75 == 0x7270746440746F67LL
                && *(_DWORD *)(v75 + 8) == 1749052517
                && *(_BYTE *)(v75 + 12) == 97 )
              {
                v11 = 1;
                v46 = v44 & v43;
                v5 = 87;
              }
              v44 &= v43;
              goto LABEL_92;
            }
            goto LABEL_228;
          }
          v11 = 1;
          v5 = 84;
LABEL_227:
          v46 = 1;
          v43 = 0;
LABEL_228:
          if ( ((unsigned __int8)v43 & (unsigned __int8)v72) == 0 )
            goto LABEL_92;
LABEL_229:
          if ( *(_QWORD *)v75 == 0x67736C7440746F67LL && *(_BYTE *)(v75 + 8) == 100 )
          {
            v5 = 89;
            v11 = 1;
            goto LABEL_206;
          }
          v47 = v15;
          v15 = v46;
          goto LABEL_94;
        }
LABEL_236:
        if ( !v46 )
        {
          v43 = 1;
          if ( v38 )
            goto LABEL_89;
          goto LABEL_238;
        }
        goto LABEL_227;
      }
      v5 = 82;
    }
    v46 = v15;
    v11 = 1;
    goto LABEL_236;
  }
LABEL_85:
  if ( ((unsigned __int8)v38 & (unsigned __int8)v43) == 0 )
  {
    v46 = v15;
    v15 = v45;
    goto LABEL_234;
  }
  if ( *(_QWORD *)v75 == 0x6572707440746F67LL && *(_DWORD *)(v75 + 8) == 1634222188 )
  {
    v38 &= v43;
    v11 = 1;
    v5 = 83;
  }
  else
  {
    v38 = v15;
  }
  v15 = v45;
  if ( v38 )
    goto LABEL_227;
LABEL_89:
  if ( *(_QWORD *)v75 == 0x7270746440746F67LL && *(_DWORD *)(v75 + 8) == 1816161381 )
  {
    v5 = 85;
LABEL_205:
    v38 = 1;
    v11 = 1;
    goto LABEL_206;
  }
  if ( *(_QWORD *)v75 == 0x7270746440746F67LL && *(_DWORD *)(v75 + 8) == 1749052517 )
  {
    v5 = 86;
    goto LABEL_205;
  }
  v43 = 1;
  v46 = 0;
  v38 = 1;
  if ( v72 )
    goto LABEL_229;
LABEL_92:
  if ( v46 )
  {
LABEL_206:
    v48 = 1;
    v43 = 0;
LABEL_207:
    v56 = v43 & v72;
    goto LABEL_208;
  }
  v47 = 0;
  v43 = 1;
  if ( !v15 )
  {
LABEL_94:
    if ( ((unsigned __int8)v38 & (unsigned __int8)v43) != 0 )
    {
      if ( *(_QWORD *)v75 == 0x67736C7440746F67LL && *(_DWORD *)(v75 + 8) == 1634222180 )
      {
        v48 = v38 & v43;
        v11 = 1;
        v5 = 92;
      }
      else
      {
        v48 = v15;
      }
      v15 = v47;
      v38 &= v43;
      goto LABEL_98;
    }
    v48 = v15;
    v15 = v47;
    goto LABEL_207;
  }
  if ( *(_QWORD *)v75 == 0x67736C7440746F67LL && *(_WORD *)(v75 + 8) == 16484 && *(_BYTE *)(v75 + 10) == 108 )
  {
    v5 = 90;
LABEL_440:
    v48 = v15;
    v11 = 1;
    goto LABEL_98;
  }
  if ( *(_QWORD *)v75 == 0x67736C7440746F67LL && *(_WORD *)(v75 + 8) == 16484 && *(_BYTE *)(v75 + 10) == 104 )
  {
    v5 = 91;
    goto LABEL_440;
  }
  v56 = v72;
  v43 = v15;
  v48 = 0;
LABEL_208:
  if ( !v56 )
  {
LABEL_98:
    if ( v48 )
      goto LABEL_103;
    if ( v15 )
    {
      if ( *(_QWORD *)v75 == 0x6C736C7440746F67LL && *(_WORD *)(v75 + 8) == 16484 && *(_BYTE *)(v75 + 10) == 108 )
      {
        v5 = 95;
      }
      else
      {
        if ( *(_QWORD *)v75 != 0x6C736C7440746F67LL || *(_WORD *)(v75 + 8) != 16484 || *(_BYTE *)(v75 + 10) != 104 )
          goto LABEL_212;
        v5 = 96;
      }
      v11 = 1;
      goto LABEL_103;
    }
LABEL_100:
    if ( !v38 )
      goto LABEL_211;
    if ( *(_QWORD *)v75 == 0x6C736C7440746F67LL && *(_DWORD *)(v75 + 8) == 1634222180 )
    {
      v5 = 97;
      v11 = 1;
      goto LABEL_103;
    }
    if ( !v15 )
      goto LABEL_213;
LABEL_103:
    v25 = 1;
    v49 = 0;
LABEL_104:
    v50 = v49 & v21;
LABEL_105:
    if ( !v50 )
    {
LABEL_106:
      v7 &= v49;
      goto LABEL_107;
    }
    if ( *(_DWORD *)v75 == 1701995123 && *(_BYTE *)(v75 + 4) == 108 )
    {
      v11 = 1;
      v25 = v50;
      v5 = 34;
    }
LABEL_108:
    if ( !v25 )
    {
      if ( v6 )
      {
        if ( *(_WORD *)v75 == 28524 && *(_BYTE *)(v75 + 2) == 56 )
        {
          v5 = 38;
        }
        else
        {
          if ( *(_WORD *)v75 != 26984 || *(_BYTE *)(v75 + 2) != 56 )
          {
            v25 = v6;
            v52 = v73;
            v6 = 0;
LABEL_128:
            if ( v52 )
            {
              if ( *(_QWORD *)v75 == 0x6E6F6974636E7566LL )
              {
                v5 = 111;
                goto LABEL_114;
              }
              goto LABEL_131;
            }
            goto LABEL_129;
          }
          v5 = 39;
        }
        v11 = 1;
LABEL_129:
        v72 &= v25;
        goto LABEL_130;
      }
      v25 = 1;
      goto LABEL_111;
    }
LABEL_126:
    v6 = 1;
    v25 = 0;
    goto LABEL_127;
  }
  if ( *(_QWORD *)v75 == 0x6C736C7440746F67LL && *(_BYTE *)(v75 + 8) == 100 )
  {
    v5 = 94;
    v11 = 1;
    goto LABEL_103;
  }
  v15 = v48;
  if ( v43 )
    goto LABEL_100;
LABEL_211:
  if ( v15 )
    goto LABEL_103;
LABEL_212:
  if ( v21 )
  {
    if ( *(_DWORD *)v75 == 1869046887 && *(_BYTE *)(v75 + 4) == 116 )
    {
      v11 = 1;
      v5 = 105;
      goto LABEL_103;
    }
    if ( *(_DWORD *)v75 == 1819305063 && *(_BYTE *)(v75 + 4) == 116 )
    {
      v11 = 1;
      v5 = 107;
      goto LABEL_103;
    }
    if ( *(_DWORD *)v75 == 1869047145 && *(_BYTE *)(v75 + 4) == 116 )
    {
      v11 = 1;
      v5 = 110;
      goto LABEL_103;
    }
  }
LABEL_213:
  if ( v67 )
  {
    if ( *(_WORD *)v75 == 25961 )
    {
      v5 = 109;
      v11 = 1;
      goto LABEL_103;
    }
  }
  else if ( v21 )
  {
    if ( *(_DWORD *)v75 == 1869046892 && *(_BYTE *)(v75 + 4) == 116 )
    {
      v11 = 1;
      v5 = 106;
      goto LABEL_103;
    }
    if ( *(_DWORD *)v75 == 1819305068 && *(_BYTE *)(v75 + 4) == 116 )
    {
      v11 = 1;
      v5 = 108;
      goto LABEL_103;
    }
    if ( *(_DWORD *)v75 == 1701995376 && *(_BYTE *)(v75 + 4) == 108 )
    {
      v11 = 1;
      v5 = 101;
      goto LABEL_103;
    }
  }
  if ( v71 )
  {
    if ( *(_DWORD *)v75 == 1701736302 )
    {
      v11 = 1;
      v5 = 29;
      goto LABEL_103;
    }
    goto LABEL_245;
  }
  if ( !v73 )
  {
LABEL_245:
    if ( v25 )
    {
      if ( *(_DWORD *)v75 == 1735549300 && *(_WORD *)(v75 + 4) == 29797 && *(_BYTE *)(v75 + 6) == 49 )
      {
        v5 = 31;
      }
      else
      {
        if ( *(_DWORD *)v75 != 1735549300 || *(_WORD *)(v75 + 4) != 29797 || *(_BYTE *)(v75 + 6) != 50 )
        {
          v49 = v25;
          v50 = v21;
          v25 = 0;
          goto LABEL_105;
        }
        v5 = 32;
      }
      v11 = 1;
      v49 = 0;
      goto LABEL_106;
    }
    goto LABEL_246;
  }
  if ( *(_QWORD *)v75 == 0x6C6572705F746F67LL )
  {
    v5 = 30;
    v11 = 1;
    goto LABEL_103;
  }
LABEL_246:
  v25 = 0;
  v49 = 1;
  if ( !v7 )
    goto LABEL_104;
  if ( *(_DWORD *)v75 == 1818587760 && *(_WORD *)(v75 + 4) == 12595 )
  {
    v25 = v7;
    v11 = 1;
    v5 = 33;
    goto LABEL_108;
  }
  v49 = v7;
  v25 = 0;
LABEL_107:
  if ( !v7 )
    goto LABEL_108;
  if ( *(_DWORD *)v75 == 1819503732 && *(_WORD *)(v75 + 4) == 28516 )
  {
    v11 = 1;
    v5 = 35;
    goto LABEL_126;
  }
  v6 = v25;
  v25 = v49;
  if ( !v49 )
    goto LABEL_127;
LABEL_111:
  if ( !v71 )
  {
LABEL_127:
    v52 = v25 && v73;
    goto LABEL_128;
  }
  if ( *(_DWORD *)v75 == 946826344 )
  {
    v5 = 40;
    goto LABEL_114;
  }
LABEL_130:
  if ( v72 )
  {
    if ( *(_QWORD *)v75 == 0x65646E6965707974LL && *(_BYTE *)(v75 + 8) == 120 )
    {
      v5 = 112;
      goto LABEL_114;
    }
    goto LABEL_133;
  }
LABEL_131:
  if ( v6 )
    goto LABEL_114;
  if ( v44 )
  {
    if ( *(_QWORD *)v75 == 0x6C65726370746F67LL && *(_DWORD *)(v75 + 8) == 1816146483 )
    {
      v5 = 113;
      if ( *(_BYTE *)(v75 + 12) == 111 )
        goto LABEL_114;
    }
    if ( *(_QWORD *)v75 == 0x6C65726370746F67LL && *(_DWORD *)(v75 + 8) == 1749037619 )
    {
      v5 = 114;
      if ( *(_BYTE *)(v75 + 12) == 105 )
        goto LABEL_114;
    }
LABEL_134:
    if ( v6 != 1 && v21 && *(_DWORD *)v75 == 913073522 )
    {
      v5 = 117;
      if ( *(_BYTE *)(v75 + 4) == 52 )
        goto LABEL_114;
    }
LABEL_136:
    v5 = 1;
    goto LABEL_114;
  }
LABEL_133:
  v6 = v11;
  if ( !v73 )
    goto LABEL_134;
  if ( *(_QWORD *)v75 == 0x6F6C4032336C6572LL )
  {
    v5 = 115;
    goto LABEL_114;
  }
  if ( *(_QWORD *)v75 == 0x69684032336C6572LL )
  {
    v5 = 116;
    goto LABEL_114;
  }
  if ( !v11 )
    goto LABEL_136;
LABEL_114:
  if ( (__int64 *)v75 != &v77 )
    j_j___libc_free_0(v75);
  return v5;
}
