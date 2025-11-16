// Function: sub_8AA320
// Address: 0x8aa320
//
__int64 __fastcall sub_8AA320(_QWORD *a1, int a2, unsigned int a3)
{
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r10
  __int64 v8; // r10
  char v9; // al
  __int64 v10; // r12
  unsigned __int64 v11; // rax
  char v12; // al
  __int64 **v13; // rax
  unsigned int v14; // eax
  char v15; // al
  __int64 v16; // rdx
  char v17; // al
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 *v25; // r9
  __int64 v26; // rsi
  __int64 v27; // rbx
  __int64 v28; // r10
  __int64 v29; // r11
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  unsigned __int64 v37; // rdi
  __int64 v38; // rcx
  _QWORD *v39; // rdx
  __int16 v40; // di
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 *v44; // r9
  __int64 v45; // rax
  __int64 result; // rax
  char v47; // al
  __int64 v48; // rsi
  __int64 v49; // rcx
  __int64 v50; // rdi
  __int64 v51; // rdi
  __int64 v52; // rax
  char v53; // al
  __int64 v54; // rdi
  char v55; // al
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rdi
  __int64 i; // rdx
  __int64 v60; // [rsp+8h] [rbp-388h]
  __int64 v61; // [rsp+10h] [rbp-380h]
  __int64 v62; // [rsp+10h] [rbp-380h]
  int v63; // [rsp+18h] [rbp-378h]
  char v65; // [rsp+20h] [rbp-370h]
  __int64 v66; // [rsp+20h] [rbp-370h]
  __int64 v67; // [rsp+28h] [rbp-368h]
  __int64 v68; // [rsp+30h] [rbp-360h]
  __int64 *v69; // [rsp+30h] [rbp-360h]
  __int64 v70; // [rsp+38h] [rbp-358h]
  _BOOL4 v71; // [rsp+38h] [rbp-358h]
  int v72; // [rsp+40h] [rbp-350h]
  unsigned int v73; // [rsp+40h] [rbp-350h]
  unsigned int v74; // [rsp+40h] [rbp-350h]
  char v76; // [rsp+4Fh] [rbp-341h]
  __int64 v77; // [rsp+50h] [rbp-340h]
  __int64 v78; // [rsp+58h] [rbp-338h]
  int v79; // [rsp+6Ch] [rbp-324h] BYREF
  __m128i v80[4]; // [rsp+70h] [rbp-320h] BYREF
  _QWORD v81[12]; // [rsp+B0h] [rbp-2E0h] BYREF
  _BYTE v82[112]; // [rsp+110h] [rbp-280h] BYREF
  _QWORD v83[66]; // [rsp+180h] [rbp-210h] BYREF

  v4 = a1[3];
  v5 = a1[4];
  v79 = 0;
  v76 = *(_BYTE *)(v4 + 80);
  switch ( *(_BYTE *)(v5 + 80) )
  {
    case 4:
    case 5:
      v77 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 80LL);
      break;
    case 6:
      v77 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v77 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v77 = *(_QWORD *)(v5 + 88);
      break;
    default:
      v77 = 0;
      break;
  }
  v72 = 256;
  v78 = v77;
  if ( v76 == 7 )
  {
    if ( a1[5] )
    {
      v5 = a1[5];
    }
    else
    {
      v5 = sub_8A9D50(v4, *(_QWORD *)(v5 + 88), 0);
      if ( !v5 )
        v5 = a1[4];
      a1[5] = v5;
      a1[4] = v5;
    }
    v47 = *(_BYTE *)(v5 + 80);
    v48 = *(_QWORD *)(v5 + 88);
    v49 = *(_QWORD *)(v48 + 88);
    switch ( v47 )
    {
      case 4:
      case 5:
        v77 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 80LL);
        goto LABEL_155;
      case 6:
        v52 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 32LL);
        v77 = v52;
        if ( !v49 )
          goto LABEL_172;
        if ( (*(_BYTE *)(v48 + 160) & 1) == 0 )
          goto LABEL_167;
        v49 = v5;
LABEL_211:
        v52 = *(_QWORD *)(*(_QWORD *)(v49 + 96) + 32LL);
        goto LABEL_172;
      case 9:
      case 10:
        v52 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 56LL);
        v77 = v52;
        if ( !v49 )
          goto LABEL_172;
        if ( (*(_BYTE *)(v48 + 160) & 1) == 0 )
          goto LABEL_167;
        v49 = v5;
LABEL_171:
        v52 = *(_QWORD *)(*(_QWORD *)(v49 + 96) + 56LL);
LABEL_172:
        v78 = v52;
        v72 = 0;
        break;
      case 19:
      case 20:
      case 21:
      case 22:
        v77 = *(_QWORD *)(v5 + 88);
        goto LABEL_155;
      default:
        v77 = 0;
LABEL_155:
        if ( v49 && (*(_BYTE *)(v48 + 160) & 1) == 0 )
LABEL_167:
          v47 = *(_BYTE *)(v49 + 80);
        else
          v49 = v5;
        switch ( v47 )
        {
          case 4:
          case 5:
            v72 = 0;
            v78 = *(_QWORD *)(*(_QWORD *)(v49 + 96) + 80LL);
            break;
          case 6:
            goto LABEL_211;
          case 9:
          case 10:
            goto LABEL_171;
          case 19:
          case 20:
          case 21:
          case 22:
            v72 = 0;
            v78 = *(_QWORD *)(v49 + 88);
            break;
          default:
            v72 = 0;
            v78 = 0;
            break;
        }
        break;
    }
  }
  v70 = sub_892400(v77);
  v6 = *(_QWORD *)(v77 + 88);
  if ( !v6 || (*(_BYTE *)(v77 + 160) & 1) != 0 )
  {
    v7 = v77 + 200;
    if ( *(_QWORD *)(v77 + 232) )
      goto LABEL_7;
  }
  else
  {
    v7 = *(_QWORD *)(v6 + 88) + 200LL;
    if ( *(_QWORD *)(*(_QWORD *)(v6 + 88) + 232LL) )
      goto LABEL_7;
  }
  if ( *(_QWORD *)(v77 + 32) )
    v7 = v77;
LABEL_7:
  v68 = v7;
  v67 = *(_QWORD *)(v78 + 192);
  sub_892270(a1);
  v8 = v68;
  if ( v76 != 7 )
    *(_BYTE *)(a1[2] + 28LL) |= 1u;
  v9 = *(_BYTE *)(v4 + 80);
  if ( v9 == 9 || v9 == 7 )
  {
    v10 = *(_QWORD *)(v4 + 88);
  }
  else
  {
    v10 = 0;
    if ( v9 == 21 )
      v10 = *(_QWORD *)(*(_QWORD *)(v4 + 88) + 192LL);
  }
  v11 = *(unsigned int *)(v78 + 40);
  if ( v11 < unk_4D042F0 )
  {
    *(_DWORD *)(v78 + 40) = v11 + 1;
    if ( unk_4D04734 == 3 )
    {
      *(_BYTE *)(v10 + 168) &= 0xF8u;
      v12 = *(_BYTE *)(v10 + 88);
      *(_BYTE *)(v10 + 136) = 2;
      *(_BYTE *)(v10 + 88) = v12 & 0x8F | 0x10;
    }
    v13 = *(__int64 ***)(v10 + 216);
    v69 = v13[1];
    if ( !v69 )
      v69 = *v13;
    v61 = v8;
    v14 = v72 | 0x400000;
    if ( (*(_BYTE *)(v77 + 160) & 1) == 0 )
      v14 = v72;
    v73 = v14;
    sub_864700(*(_QWORD *)(v8 + 32), 0, 0, v4, v5, (__int64)v69, 1, v14);
    *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 768) = *(_DWORD *)(v78 + 240);
    sub_87E3B0((__int64)v82);
    memset(v81, 0, 0x58u);
    memset(v83, 0, 0x1D8u);
    v83[19] = v83;
    v83[3] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
      BYTE2(v83[22]) |= 1u;
    v83[0] = v4;
    BYTE3(v83[15]) = (8 * (word_4D04430 & 1)) | BYTE3(v83[15]) & 0xF7;
    sub_7BC160(v61);
    sub_898140((__int64)v83, 0, 0, a1[4], 0, 0, 0, v80, (unsigned __int64)v82, 0, (__int64)a1, v81);
    v15 = *(_BYTE *)(v4 + 80);
    if ( v15 == 9 )
    {
      v16 = *(_QWORD *)(v4 + 88);
    }
    else if ( v15 == 7 )
    {
      v16 = *(_QWORD *)(v4 + 88);
      if ( (*(_BYTE *)(v4 + 81) & 0x10) == 0 && BYTE5(v83[33]) == 2 )
      {
        v53 = *(_BYTE *)(v16 + 88);
        *(_BYTE *)(v16 + 168) &= 0xF8u;
        *(_BYTE *)(v16 + 136) = 2;
        *(_BYTE *)(v16 + 88) = v53 & 0x8F | 0x10;
      }
    }
    else
    {
      v16 = 0;
      if ( v15 == 21 )
        v16 = *(_QWORD *)(*(_QWORD *)(v4 + 88) + 192LL);
    }
    if ( *(_QWORD *)(v16 + 120) )
      sub_64E420(v4, v83[36], &v80[0].m128i_u32[2]);
    else
      *(_QWORD *)(v16 + 120) = v83[36];
    if ( (v83[15] & 0x20000000000LL) != 0 )
    {
      *(_BYTE *)(v10 + 175) |= 2u;
    }
    else if ( (v83[15] & 0x10000000000LL) != 0 )
    {
      *(_BYTE *)(v10 + 175) |= 1u;
    }
    else if ( (v83[15] & 0x40000000000LL) != 0 )
    {
      *(_BYTE *)(v10 + 175) |= 4u;
    }
    if ( (*(_BYTE *)(v10 + 176) & 1) != 0 && dword_4F077BC )
      sub_5EB3F0((_QWORD *)v10);
    if ( dword_4F077C4 == 2 )
    {
      v54 = *(_QWORD *)(v10 + 120);
      if ( (unsigned int)sub_8D23B0(v54) )
        sub_8AE000(v54);
    }
    if ( a2 && v76 == 7 )
    {
      sub_735E40(v10, -1);
      *(_BYTE *)(v10 + 176) = *(_BYTE *)(v67 + 176) & 1 | *(_BYTE *)(v10 + 176) & 0xFE;
      *(_BYTE *)(v10 + 172) = *(_BYTE *)(v67 + 172) & 0x20 | *(_BYTE *)(v10 + 172) & 0xDF;
      v17 = *(_BYTE *)(v10 + 156);
      if ( (v17 & 3) == 1 )
      {
        sub_88E010(v10, 0, **(__int64 ***)(v10 + 216));
        v17 = *(_BYTE *)(v10 + 156);
      }
      if ( (v17 & 1) != 0 && dword_4F04C44 == -1 )
      {
        v56 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v56 + 6) & 6) == 0 && *(_BYTE *)(v56 + 4) != 12 )
        {
          if ( *(_QWORD *)(v10 + 8) )
          {
            if ( (*(_BYTE *)(v10 + 170) & 0x60) == 0 )
            {
              v57 = *(_QWORD *)(v10 + 120);
              if ( *(_BYTE *)(v10 + 177) != 5 )
              {
                if ( v57 )
                {
                  v58 = *(_QWORD *)(v10 + 120);
                  if ( *(_BYTE *)(v57 + 140) == 12 )
                  {
                    do
                      v58 = *(_QWORD *)(v58 + 160);
                    while ( *(_BYTE *)(v58 + 140) == 12 );
                  }
                  v66 = *(_QWORD *)(v10 + 120);
                  if ( !(unsigned int)sub_8D23B0(v58) )
                  {
                    for ( i = v66; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
                      ;
                    if ( !*(_QWORD *)(i + 128) )
                      sub_6851A0(0xE54u, (_DWORD *)(v67 + 64), *(_QWORD *)(v10 + 8));
                  }
                }
              }
            }
          }
        }
      }
    }
    else if ( v76 != 7 )
    {
      if ( (*(_BYTE *)(v5 + 81) & 0x10) != 0 )
      {
        sub_854C10(*(const __m128i **)(v78 + 56));
        goto LABEL_46;
      }
      goto LABEL_43;
    }
    if ( !*(_QWORD *)(v10 + 256) && v83[35] )
      *(_QWORD *)(v10 + 256) = v83[35];
    if ( (*(_BYTE *)(v5 + 81) & 0x10) != 0 )
    {
      sub_854C10(*(const __m128i **)(v78 + 56));
      goto LABEL_107;
    }
LABEL_43:
    if ( HIDWORD(qword_4F077B4)
      && !(_DWORD)qword_4F077B4
      && qword_4F077A8
      && BYTE4(v83[33]) != 1
      && (v83[1] & 2) == 0
      && (v50 = *(_QWORD *)(v10 + 120), (*(_BYTE *)(v50 + 140) & 0xFB) == 8)
      && (sub_8D4C10(v50, dword_4F077C4 != 2) & 1) != 0
      && ((v51 = *(_QWORD *)(v10 + 120), (*(_BYTE *)(v51 + 140) & 0xFB) != 8)
       || (sub_8D4C10(v51, dword_4F077C4 != 2) & 2) == 0)
      || dword_4D047F8 && (unsigned int)sub_88DB10(v69) )
    {
      *(_BYTE *)(v10 + 168) &= 0xF8u;
      v55 = *(_BYTE *)(v10 + 88);
      *(_BYTE *)(v10 + 136) = 2;
      *(_BYTE *)(v10 + 88) = v55 & 0x8F | 0x10;
    }
    sub_854C10(*(const __m128i **)(v78 + 56));
    if ( v76 != 7 )
    {
LABEL_46:
      v18 = 1;
      *(_BYTE *)(a1[2] + 28LL) |= 1u;
      BYTE2(v83[15]) |= 1u;
      sub_644920(v83, 1);
      if ( *(_QWORD *)(v70 + 8) )
        goto LABEL_48;
      goto LABEL_114;
    }
LABEL_107:
    if ( !a3 )
    {
      v18 = 1;
      sub_644920(v83, 1);
      v65 = 0;
      if ( *(_BYTE *)(v10 + 177) )
        goto LABEL_82;
      if ( v76 != 7 )
        goto LABEL_116;
LABEL_124:
      if ( (*(_BYTE *)(v5 + 81) & 2) == 0 )
        goto LABEL_82;
      v18 = a3;
      if ( a3 )
        goto LABEL_116;
      if ( (*(_BYTE *)(v10 + 168) & 7) != 0 )
      {
        sub_643EB0((__int64)v83, 0);
        goto LABEL_128;
      }
LABEL_83:
      v19 = *(_BYTE *)(v67 + 168) & 7;
      *(_BYTE *)(v10 + 168) = v19 | *(_BYTE *)(v10 + 168) & 0xF8;
LABEL_84:
      v38 = a3;
      if ( !a3 )
        goto LABEL_85;
      goto LABEL_133;
    }
    if ( (*(_BYTE *)(v5 + 81) & 2) == 0 && (*(_BYTE *)(v10 + 176) & 1) == 0
      || (*(_BYTE *)(v10 + 89) & 4) != 0 && (*(_BYTE *)(v10 + 172) & 0x20) == 0 && (*(_BYTE *)(v78 + 168) & 1) == 0 )
    {
      sub_644920(v83, 1);
      v18 = (__int64)v83;
      sub_6581B0(v10, (__int64)v83, 0);
      if ( *(_BYTE *)(v10 + 177) )
      {
        v65 = 0;
        if ( (*(_BYTE *)(v10 + 168) & 7) != 0 )
        {
LABEL_133:
          sub_65C470((__int64)v83, v18, v19, v38, v20);
LABEL_85:
          sub_643EB0((__int64)v83, 0);
          if ( v76 != 7 )
          {
            if ( v65 )
            {
LABEL_87:
              if ( *(_BYTE *)(v10 + 136) == 1 )
                *(_BYTE *)(v10 + 136) = 0;
              v39 = (_QWORD *)(v5 + 48);
              v40 = -32766;
              goto LABEL_90;
            }
LABEL_131:
            v39 = (_QWORD *)(v5 + 48);
            v40 = 0x8000;
LABEL_90:
            sub_8756F0(v40, v4, v39, 0);
            *(_BYTE *)(v10 + 88) |= 4u;
LABEL_91:
            sub_854980(v4, 0);
            sub_863FE0(v4, 0, v41, v42, v43, v44);
            --*(_DWORD *)(v78 + 40);
            v45 = *(_QWORD *)(v10 + 216);
            *(_BYTE *)(v10 + 170) |= 0x10u;
            *(_QWORD *)(v45 + 16) = *(_QWORD *)(v77 + 104);
            return sub_8CB9C0(v10);
          }
LABEL_128:
          if ( !v79 && (unsigned int)sub_8B1260(*(_QWORD *)(v10 + 120), *(unsigned __int8 *)(v10 + 136), v83, a3) )
            *(_QWORD *)(v10 + 120) = sub_72C930();
          if ( v65 )
            goto LABEL_87;
          if ( !a3 )
            goto LABEL_91;
          goto LABEL_131;
        }
        goto LABEL_83;
      }
      v65 = 0;
      if ( *(_QWORD *)(v70 + 8) )
        goto LABEL_50;
      goto LABEL_115;
    }
    *(_BYTE *)(a1[2] + 28LL) |= 1u;
    BYTE2(v83[15]) |= 1u;
    sub_644920(v83, 1);
    v18 = (__int64)v83;
    sub_6581B0(v10, (__int64)v83, 1);
    if ( *(_QWORD *)(v70 + 8) )
    {
LABEL_48:
      v65 = 1;
      if ( *(_BYTE *)(v10 + 136) == 1 )
        *(_BYTE *)(v10 + 136) = 0;
LABEL_50:
      v63 = 0;
      if ( (*(_BYTE *)(v10 + 176) & 1) != 0 )
      {
        v21 = *(_QWORD *)(v10 + 120);
        if ( (*(_BYTE *)(v21 + 140) & 0xFB) == 8 )
          v63 = sub_8D4C10(v21, dword_4F077C4 != 2) & 1;
      }
      v22 = v70;
      sub_7BC160(v70);
      v26 = *(_QWORD *)(v70 + 32);
      if ( *(_QWORD *)(v77 + 232) != v26 )
      {
        v27 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        v28 = *(_QWORD *)(v27 + 440);
        v29 = *(_QWORD *)(v27 + 432);
        *(_QWORD *)(v27 + 440) = 0;
        *(_QWORD *)(v27 + 432) = 0;
        v60 = v28;
        v62 = v29;
        sub_863FE0((__int64)qword_4F04C68, v26, v23, (__int64)&dword_4F04C64, v24, v25);
        v26 = 0;
        v22 = *(_QWORD *)(v70 + 32);
        sub_864700(v22, 0, 0, v4, v5, (__int64)v69, 1, v73);
        *(_QWORD *)(v27 + 440) = v60;
        *(_QWORD *)(v27 + 432) = v62;
      }
      v30 = HIBYTE(v83[15]);
      v31 = word_4F06418[0];
      LOBYTE(v26) = word_4F06418[0] == 27;
      v71 = word_4F06418[0] == 27;
      if ( word_4F06418[0] == 73 )
      {
        v30 = HIBYTE(v83[15]) | 0xCu;
        HIBYTE(v83[15]) |= 0xCu;
      }
      else
      {
        v26 = (unsigned int)(8 * v26);
        HIBYTE(v83[15]) = v26 | HIBYTE(v83[15]) & 0xF3 | 4;
        if ( v76 != 7 || (*(_BYTE *)(v67 + 176) & 1) == 0 )
          sub_7B8B50(v22, (unsigned int *)v26, v30, word_4F06418[0], v24, (__int64)v25);
      }
      v32 = *(_QWORD *)(v10 + 120);
      v83[0] = v4;
      v83[36] = v32;
      if ( (v83[15] & 0x800000000000000LL) == 0 && word_4F06418[0] == 56 && (*(_BYTE *)(v67 + 176) & 1) != 0 )
        sub_7B8B50(v22, (unsigned int *)v26, v30, v31, v24, (__int64)v25);
      v74 = 0;
      if ( (v83[15] & 0x8000000000LL) != 0 )
      {
        v26 = v71;
        v22 = (__int64)v83;
        sub_6BDE10((__int64)v83, v71);
        if ( dword_4F077C4 == 2 )
        {
          v22 = v83[36];
          if ( (unsigned int)sub_8D23B0(v83[36]) )
            sub_8AE000(v22);
        }
        BYTE4(v83[15]) &= ~0x80u;
        v74 = 1;
      }
      if ( (v83[1] & 2) != 0 )
      {
        if ( dword_4D04820
          || (v26 = (__int64)&v83[11], dword_4F077BC)
          && (v26 = (__int64)&v83[11], (unsigned int)sub_657F30((unsigned int *)&v83[11])) )
        {
          v26 = 1;
          v22 = v10;
          sub_658080((_BYTE *)v10, 1);
        }
        else
        {
          v22 = 325;
          sub_6851C0(0x145u, &v83[11]);
        }
      }
      if ( (*(_BYTE *)(v10 + 176) & 1) != 0 && (*(_BYTE *)(v10 + 172) & 0x20) == 0 )
      {
        v22 = *(_QWORD *)(v10 + 120);
        v26 = v10;
        if ( !(unsigned int)sub_5F2750(v22, v10, v63, v76 == 7, 0) )
        {
          v18 = *(_QWORD *)(v10 + 120);
          *(_QWORD *)(v10 + 120) = sub_5F2840((__int64)v83, v18, (__int64)&dword_4F063F8);
          sub_6BBA30((__int64)v83);
          goto LABEL_76;
        }
      }
      if ( (v83[15] & 0x800000000000000LL) == 0 && word_4F06418[0] == 56 )
      {
        if ( (*(_BYTE *)(v67 + 176) & 1) == 0 )
          goto LABEL_74;
        sub_7B8B50(v22, (unsigned int *)v26, v30, v31, v24, (__int64)v25);
      }
      if ( v76 == 7 && (*(_BYTE *)(v67 + 176) & 1) != 0 && (*(_BYTE *)(v10 + 172) & 0x20) == 0 )
      {
        v18 = v10;
        sub_5F2700((__int64)v83, v10, v30, v31, v24, (__int64)v25);
        goto LABEL_75;
      }
LABEL_74:
      v18 = v5 + 48;
      sub_638AC0((__int64)v83, (_QWORD *)(v5 + 48), 2u, v71, &v79, 0);
LABEL_75:
      sub_649FB0((__int64)v83, v18);
LABEL_76:
      v37 = v74;
      if ( v74 )
        BYTE4(v83[15]) |= 0x80u;
      if ( word_4F06418[0] != 9 )
      {
        v18 = (__int64)&dword_4F063F8;
        v37 = 65;
        sub_6851C0(0x41u, &dword_4F063F8);
        while ( word_4F06418[0] != 9 )
          sub_7B8B50(0x41u, &dword_4F063F8, v33, v34, v35, v36);
      }
      sub_7B8B50(v37, (unsigned int *)v18, v33, v34, v35, v36);
      goto LABEL_82;
    }
LABEL_114:
    v65 = 1;
    if ( *(_BYTE *)(v10 + 177) )
      goto LABEL_82;
LABEL_115:
    if ( v76 != 7 )
    {
LABEL_116:
      if ( *(_BYTE *)(v10 + 136) == 1 )
        *(_BYTE *)(v10 + 136) = 0;
      if ( (v83[1] & 0x200000LL) != 0 )
        *(_BYTE *)(v10 + 172) |= 0x10u;
      v18 = v5 + 48;
      if ( !(unsigned int)sub_63BB10(v4, v5 + 48) )
      {
        v18 = *(_QWORD *)(v10 + 120);
        *(_QWORD *)dword_4F07508 = v83[6];
        sub_640330(v4, v18, 0, 0);
      }
LABEL_82:
      if ( (*(_BYTE *)(v10 + 168) & 7) != 0 )
        goto LABEL_84;
      goto LABEL_83;
    }
    goto LABEL_124;
  }
  sub_6854E0(0x1C8u, v4);
  *(_QWORD *)(v10 + 120) = sub_72C930();
  result = a1[2];
  *(_BYTE *)(result + 28) |= 1u;
  return result;
}
