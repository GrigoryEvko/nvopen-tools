// Function: sub_21C5A60
// Address: 0x21c5a60
//
__int64 __fastcall sub_21C5A60(
        __int64 *a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        int a8)
{
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rsi
  int v12; // eax
  __int64 v13; // r12
  char v14; // al
  __int64 v15; // r9
  unsigned int v16; // r15d
  _BYTE *v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // rax
  int v20; // r14d
  __int64 v21; // r12
  __int64 *v22; // rax
  _QWORD *v23; // rax
  unsigned __int8 *v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // rax
  int v27; // edx
  __int64 v28; // r14
  __int16 v29; // ax
  int v30; // r8d
  int v31; // edx
  int v32; // ecx
  __int64 v33; // r9
  __int64 *v34; // r11
  unsigned int v35; // r12d
  char v36; // dl
  __int16 v37; // ax
  __int64 v39; // rdx
  _QWORD *v40; // rax
  __int16 v41; // si
  _QWORD *v42; // rdi
  __int64 v43; // rdx
  __int64 *v44; // rax
  _QWORD *v45; // rax
  unsigned __int8 *v46; // rax
  __int64 v47; // r14
  unsigned __int8 v48; // si
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rdx
  unsigned int v54; // ebx
  unsigned int v55; // r12d
  unsigned int v56; // r10d
  __int64 v57; // r15
  __int128 v58; // rax
  unsigned int v59; // r10d
  __int64 v60; // r9
  int v61; // r8d
  int v62; // edx
  unsigned __int8 v63; // si
  int v64; // ecx
  __int64 *v65; // r11
  int v66; // r8d
  int v67; // edx
  unsigned __int8 v68; // si
  int v69; // ecx
  __int128 v70; // [rsp-30h] [rbp-1C0h]
  int v71; // [rsp-10h] [rbp-1A0h]
  int v72; // [rsp-10h] [rbp-1A0h]
  int v73; // [rsp-10h] [rbp-1A0h]
  unsigned int v74; // [rsp+8h] [rbp-188h]
  unsigned int v75; // [rsp+10h] [rbp-180h]
  char v76; // [rsp+14h] [rbp-17Ch]
  __int16 v77; // [rsp+14h] [rbp-17Ch]
  __int64 v78; // [rsp+18h] [rbp-178h]
  __int64 v79; // [rsp+18h] [rbp-178h]
  __int32 v80; // [rsp+20h] [rbp-170h]
  __int64 v81; // [rsp+20h] [rbp-170h]
  char v82; // [rsp+28h] [rbp-168h]
  __int64 v83; // [rsp+30h] [rbp-160h]
  int v84; // [rsp+30h] [rbp-160h]
  _QWORD *v85; // [rsp+30h] [rbp-160h]
  unsigned __int32 v86; // [rsp+38h] [rbp-158h]
  unsigned int v87; // [rsp+38h] [rbp-158h]
  __int64 v88; // [rsp+40h] [rbp-150h]
  __int64 v89; // [rsp+40h] [rbp-150h]
  __int64 v90; // [rsp+50h] [rbp-140h]
  unsigned __int64 v91; // [rsp+58h] [rbp-138h]
  __int64 v92; // [rsp+60h] [rbp-130h] BYREF
  __int64 v93; // [rsp+68h] [rbp-128h] BYREF
  __int64 v94; // [rsp+70h] [rbp-120h] BYREF
  __int64 v95; // [rsp+78h] [rbp-118h] BYREF
  __int64 v96; // [rsp+80h] [rbp-110h] BYREF
  int v97; // [rsp+88h] [rbp-108h]
  __m128i v98; // [rsp+90h] [rbp-100h] BYREF
  __m128i v99; // [rsp+A0h] [rbp-F0h] BYREF
  __m128i v100; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v101; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v102; // [rsp+C8h] [rbp-C8h]
  __m128 v103; // [rsp+D0h] [rbp-C0h] BYREF
  __m128i v104; // [rsp+E0h] [rbp-B0h]
  __int64 v105; // [rsp+F0h] [rbp-A0h]
  __int32 v106; // [rsp+F8h] [rbp-98h]
  _BYTE *v107; // [rsp+100h] [rbp-90h] BYREF
  __int64 v108; // [rsp+108h] [rbp-88h]
  _BYTE v109[128]; // [rsp+110h] [rbp-80h] BYREF

  v9 = a2;
  v10 = *(_QWORD *)(a2 + 32);
  v78 = *(_QWORD *)v10;
  v80 = *(_DWORD *)(v10 + 8);
  if ( *(_WORD *)(a2 + 24) == 44 )
  {
    v39 = *(_QWORD *)(*(_QWORD *)(v10 + 40) + 88LL);
    v88 = *(_QWORD *)(v10 + 80);
    v86 = *(_DWORD *)(v10 + 88);
    v40 = *(_QWORD **)(v39 + 24);
    if ( *(_DWORD *)(v39 + 32) > 0x40u )
      v40 = (_QWORD *)*v40;
    if ( (unsigned int)v40 <= 0xFDE )
    {
      v35 = 0;
      if ( (unsigned int)v40 <= 0xFDB )
        return v35;
      v76 = 1;
    }
    else
    {
      v35 = 0;
      if ( (unsigned int)((_DWORD)v40 - 4069) > 2 )
        return v35;
      v76 = 0;
    }
  }
  else
  {
    v88 = *(_QWORD *)(v10 + 40);
    v76 = 1;
    v86 = *(_DWORD *)(v10 + 48);
  }
  v11 = *(_QWORD *)(a2 + 72);
  v96 = v11;
  if ( v11 )
    sub_1623A60((__int64)&v96, v11, 2);
  v12 = *(_DWORD *)(v9 + 64);
  v13 = *(_QWORD *)(v9 + 96);
  v98.m128i_i64[0] = 0;
  v98.m128i_i32[2] = 0;
  v97 = v12;
  v14 = *(_BYTE *)(v9 + 88);
  v99.m128i_i64[0] = 0;
  v99.m128i_i32[2] = 0;
  v100.m128i_i64[0] = 0;
  v100.m128i_i32[2] = 0;
  LOBYTE(v101) = v14;
  v102 = v13;
  if ( v14 )
  {
    v16 = 1;
    if ( (unsigned __int8)(v14 - 14) <= 0x5Fu )
    {
      v16 = word_433D980[(unsigned __int8)(v14 - 14)];
      switch ( v14 )
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
          LOBYTE(v101) = 3;
          v102 = 0;
          goto LABEL_21;
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
          LOBYTE(v101) = 4;
          v102 = 0;
          goto LABEL_75;
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
          LOBYTE(v101) = 5;
          v102 = 0;
          goto LABEL_75;
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
          LOBYTE(v101) = 6;
          v102 = 0;
          goto LABEL_75;
        case 55:
          LOBYTE(v101) = 7;
          v102 = 0;
          goto LABEL_75;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          LOBYTE(v101) = 8;
          v102 = 0;
          goto LABEL_73;
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
          LOBYTE(v101) = 9;
          v102 = 0;
          goto LABEL_75;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          LOBYTE(v101) = 10;
          v102 = 0;
          goto LABEL_75;
        default:
          LOBYTE(v101) = 2;
          v102 = 0;
          goto LABEL_75;
      }
    }
  }
  else
  {
    if ( !sub_1F58D20((__int64)&v101) )
    {
      v15 = v101;
      v16 = 1;
      v107 = v109;
      v108 = 0x500000000LL;
      goto LABEL_8;
    }
    v16 = sub_1F58D30((__int64)&v101);
    v14 = sub_1F596B0((__int64)&v101);
    LOBYTE(v101) = v14;
    v102 = v53;
    if ( v14 == 8 )
    {
LABEL_73:
      if ( **(_BYTE **)(v9 + 40) == 86 )
      {
        LOBYTE(v101) = 86;
        v16 >>= 1;
        v102 = 0;
      }
      goto LABEL_75;
    }
  }
  if ( v14 == 3 )
  {
LABEL_21:
    v15 = 4;
    v13 = 0;
    goto LABEL_22;
  }
LABEL_75:
  v15 = v101;
  v13 = v102;
LABEL_22:
  v107 = v109;
  v108 = 0x500000000LL;
  if ( !v16 )
  {
    v19 = 0;
    goto LABEL_15;
  }
LABEL_8:
  v83 = v9;
  v17 = v109;
  v18 = v13;
  v19 = 0;
  v20 = 0;
  v21 = v15;
  while ( 1 )
  {
    ++v20;
    v22 = (__int64 *)&v17[16 * v19];
    *v22 = v21;
    v22[1] = v18;
    v19 = (unsigned int)(v108 + 1);
    LODWORD(v108) = v108 + 1;
    if ( v20 == v16 )
      break;
    if ( HIDWORD(v108) <= (unsigned int)v19 )
    {
      sub_16CD150((__int64)&v107, v109, 0, 16, a8, v15);
      v19 = (unsigned int)v108;
    }
    v17 = v107;
  }
  v9 = v83;
  if ( (unsigned int)v19 >= HIDWORD(v108) )
  {
    sub_16CD150((__int64)&v107, v109, 0, 16, a8, v15);
    v19 = (unsigned int)v108;
  }
LABEL_15:
  v23 = &v107[16 * v19];
  *v23 = 1;
  v24 = v107;
  v23[1] = 0;
  v25 = a1[34];
  LODWORD(v108) = v108 + 1;
  v26 = sub_1D25C30(v25, v24, (unsigned int)v108);
  v84 = v27;
  v28 = v26;
  if ( !sub_21C2A00((__int64)a1, v88, v86, (__int64)&v100) )
  {
    if ( *(_BYTE *)(a1[58] + 936) )
      v35 = sub_21C2F80((__int64)a1, v88, v88, v86, (__int64)&v98, (__int64)&v99, a3, *(double *)a4.m128i_i64, a5);
    else
      v35 = sub_21C2F60((__int64)a1, v88, v88, v86, (__int64)&v98, (__int64)&v99, a3, *(double *)a4.m128i_i64, a5);
    v36 = *(_BYTE *)(a1[58] + 936);
    v37 = *(_WORD *)(v9 + 24);
    if ( (_BYTE)v35 )
    {
      if ( v36 )
      {
        if ( v37 > 664 )
          goto LABEL_36;
        if ( v37 > 658 )
        {
          switch ( v37 )
          {
            case 660:
            case 662:
              BYTE4(v95) = 0;
              v94 = 0x1000005AALL;
              BYTE4(v92) = 0;
              v93 = 0x1000005A5LL;
              sub_21BD570(
                (__int64)&v103,
                v101,
                1470,
                1460,
                1465,
                (__int64)&v92,
                (__int64)&v93,
                (__int64)&v94,
                1455,
                (__int64)&v95);
              break;
            case 663:
              v95 = 0x100000604LL;
              v94 = 0x1000005FALL;
              v93 = 0x1000005F5LL;
              v92 = 0x100000613LL;
              sub_21BD570(
                (__int64)&v103,
                v101,
                1560,
                1545,
                1550,
                (__int64)&v92,
                (__int64)&v93,
                (__int64)&v94,
                1535,
                (__int64)&v95);
              break;
            case 664:
              BYTE4(v95) = 0;
              v94 = 0x100000622LL;
              BYTE4(v92) = 0;
              v93 = 0x10000061DLL;
              sub_21BD570(
                (__int64)&v103,
                v101,
                1590,
                1580,
                1585,
                (__int64)&v92,
                (__int64)&v93,
                (__int64)&v94,
                1575,
                (__int64)&v95);
              break;
            default:
              v95 = 0x10000058CLL;
              v94 = 0x100000582LL;
              v93 = 0x10000057DLL;
              v92 = 0x10000059BLL;
              sub_21BD570(
                (__int64)&v103,
                v101,
                1440,
                1425,
                1430,
                (__int64)&v92,
                (__int64)&v93,
                (__int64)&v94,
                1415,
                (__int64)&v95);
              break;
          }
          goto LABEL_102;
        }
        if ( v37 != 44 && v37 != 185 )
          goto LABEL_36;
        if ( v76 )
        {
          v61 = 1380;
          v94 = 0x100000550LL;
          v62 = 1390;
          v92 = 0x100000569LL;
          v63 = v101;
          v72 = 1365;
          v95 = 0x10000055ALL;
          v93 = 0x10000054BLL;
          v64 = 1375;
        }
        else
        {
          v61 = 1500;
          v94 = 0x1000005C8LL;
          v62 = 1510;
          v92 = 0x1000005E1LL;
          v63 = v101;
          v72 = 1485;
          v95 = 0x1000005D2LL;
          v93 = 0x1000005C3LL;
          v64 = 1495;
        }
      }
      else
      {
        if ( v37 > 664 )
          goto LABEL_36;
        if ( v37 > 658 )
        {
          switch ( v37 )
          {
            case 660:
            case 662:
              BYTE4(v95) = 0;
              v94 = 0x1000005A9LL;
              BYTE4(v92) = 0;
              v93 = 0x1000005A4LL;
              sub_21BD570(
                (__int64)&v103,
                v101,
                1469,
                1459,
                1464,
                (__int64)&v92,
                (__int64)&v93,
                (__int64)&v94,
                1454,
                (__int64)&v95);
              break;
            case 663:
              v95 = 0x100000603LL;
              v94 = 0x1000005F9LL;
              v93 = 0x1000005F4LL;
              v92 = 0x100000612LL;
              sub_21BD570(
                (__int64)&v103,
                v101,
                1559,
                1544,
                1549,
                (__int64)&v92,
                (__int64)&v93,
                (__int64)&v94,
                1534,
                (__int64)&v95);
              break;
            case 664:
              BYTE4(v95) = 0;
              v94 = 0x100000621LL;
              BYTE4(v92) = 0;
              v93 = 0x10000061CLL;
              sub_21BD570(
                (__int64)&v103,
                v101,
                1589,
                1579,
                1584,
                (__int64)&v92,
                (__int64)&v93,
                (__int64)&v94,
                1574,
                (__int64)&v95);
              break;
            default:
              v95 = 0x10000058BLL;
              v94 = 0x100000581LL;
              v93 = 0x10000057CLL;
              v92 = 0x10000059ALL;
              sub_21BD570(
                (__int64)&v103,
                v101,
                1439,
                1424,
                1429,
                (__int64)&v92,
                (__int64)&v93,
                (__int64)&v94,
                1414,
                (__int64)&v95);
              break;
          }
LABEL_102:
          if ( v103.m128_i8[4] )
          {
            a4 = _mm_load_si128(&v98);
            v43 = 3;
            a5 = _mm_load_si128(&v99);
            v41 = v103.m128_i16[0];
            v105 = v78;
            v42 = (_QWORD *)a1[34];
            v103 = (__m128)a4;
            v106 = v80;
            v44 = v65;
            v104 = a5;
LABEL_58:
            v89 = sub_1D23DE0(v42, v41, (__int64)&v96, v28, v84, v33, v44, v43);
            v45 = (_QWORD *)sub_1E0A240(a1[32], 1);
            *v45 = *(_QWORD *)(v9 + 104);
            *(_QWORD *)(v89 + 88) = v45;
            *(_QWORD *)(v89 + 96) = v45 + 1;
            v46 = *(unsigned __int8 **)(v9 + 40);
            v82 = *v46;
            v47 = *v46;
            v81 = *((_QWORD *)v46 + 1);
            if ( *(_WORD *)(v9 + 24) == 185 )
            {
              v48 = v101;
              if ( (_BYTE)v101 == v82 )
              {
                if ( v81 == v102 || (_BYTE)v101 )
                  goto LABEL_62;
                v48 = 0;
              }
              v77 = sub_21C5950(v82, v48, ((*(_BYTE *)(v9 + 27) >> 2) & 3) == 2);
              if ( v16 )
              {
                v75 = v16;
                v79 = v9;
                v54 = v47;
                v55 = 0;
                v56 = v74;
                do
                {
                  LOBYTE(v56) = 5;
                  v57 = v55++;
                  v85 = (_QWORD *)a1[34];
                  v91 = v57 | v91 & 0xFFFFFFFF00000000LL;
                  *(_QWORD *)&v58 = sub_1D38BB0(
                                      (__int64)v85,
                                      0,
                                      (__int64)&v96,
                                      v56,
                                      0,
                                      1,
                                      a3,
                                      *(double *)a4.m128i_i64,
                                      a5,
                                      0);
                  LOBYTE(v54) = v82;
                  *((_QWORD *)&v70 + 1) = v91;
                  *(_QWORD *)&v70 = v89;
                  v87 = v59;
                  v90 = sub_1D2CCE0(v85, v77, (__int64)&v96, v54, v81, v60, v70, v58);
                  sub_1D44C70(a1[34], v79, v57, v90, 0);
                  sub_1D49010(v90);
                  v56 = v87;
                }
                while ( v55 != v75 );
                v9 = v79;
              }
            }
LABEL_62:
            v35 = 1;
            sub_1D444E0(a1[34], v9, v89);
            sub_1D49010(v89);
            sub_1D2DC70((const __m128i *)a1[34], v9, v49, v50, v51, v52);
            goto LABEL_37;
          }
          goto LABEL_36;
        }
        if ( v37 != 44 && v37 != 185 )
          goto LABEL_36;
        if ( v76 )
        {
          v61 = 1379;
          v94 = 0x10000054FLL;
          v62 = 1389;
          v92 = 0x100000568LL;
          v63 = v101;
          v72 = 1364;
          v95 = 0x100000559LL;
          v93 = 0x10000054ALL;
          v64 = 1374;
        }
        else
        {
          v61 = 1499;
          v94 = 0x1000005C7LL;
          v62 = 1509;
          v92 = 0x1000005E0LL;
          v63 = v101;
          v72 = 1484;
          v95 = 0x1000005D1LL;
          v93 = 0x1000005C2LL;
          v64 = 1494;
        }
      }
      sub_21BD570((__int64)&v103, v63, v62, v64, v61, (__int64)&v92, (__int64)&v93, (__int64)&v94, v72, (__int64)&v95);
      goto LABEL_102;
    }
    if ( v36 )
    {
      if ( v37 > 664 )
        goto LABEL_37;
      if ( v37 > 658 )
      {
        switch ( v37 )
        {
          case 660:
          case 662:
            BYTE4(v95) = 0;
            v94 = 0x1000005A8LL;
            BYTE4(v92) = 0;
            v93 = 0x1000005A3LL;
            sub_21BD570(
              (__int64)&v103,
              v101,
              1468,
              1458,
              1463,
              (__int64)&v92,
              (__int64)&v93,
              (__int64)&v94,
              1453,
              (__int64)&v95);
            break;
          case 663:
            v95 = 0x100000602LL;
            v94 = 0x1000005F8LL;
            v93 = 0x1000005F3LL;
            v92 = 0x100000611LL;
            sub_21BD570(
              (__int64)&v103,
              v101,
              1558,
              1543,
              1548,
              (__int64)&v92,
              (__int64)&v93,
              (__int64)&v94,
              1533,
              (__int64)&v95);
            break;
          case 664:
            BYTE4(v95) = 0;
            v94 = 0x100000620LL;
            BYTE4(v92) = 0;
            v93 = 0x10000061BLL;
            sub_21BD570(
              (__int64)&v103,
              v101,
              1588,
              1578,
              1583,
              (__int64)&v92,
              (__int64)&v93,
              (__int64)&v94,
              1573,
              (__int64)&v95);
            break;
          default:
            v95 = 0x10000058ALL;
            v94 = 0x100000580LL;
            v93 = 0x10000057BLL;
            v92 = 0x100000599LL;
            sub_21BD570(
              (__int64)&v103,
              v101,
              1438,
              1423,
              1428,
              (__int64)&v92,
              (__int64)&v93,
              (__int64)&v94,
              1413,
              (__int64)&v95);
            break;
        }
        goto LABEL_105;
      }
      if ( v37 != 44 && v37 != 185 )
        goto LABEL_37;
      if ( v76 )
      {
        v66 = 1378;
        v94 = 0x10000054ELL;
        v67 = 1388;
        v92 = 0x100000567LL;
        v68 = v101;
        v73 = 1363;
        v95 = 0x100000558LL;
        v93 = 0x100000549LL;
        v69 = 1373;
      }
      else
      {
        v66 = 1498;
        v94 = 0x1000005C6LL;
        v67 = 1508;
        v92 = 0x1000005DFLL;
        v68 = v101;
        v73 = 1483;
        v95 = 0x1000005D0LL;
        v93 = 0x1000005C1LL;
        v69 = 1493;
      }
    }
    else
    {
      if ( v37 > 664 )
        goto LABEL_37;
      if ( v37 > 658 )
      {
        switch ( v37 )
        {
          case 660:
          case 662:
            BYTE4(v95) = 0;
            v94 = 0x1000005A7LL;
            BYTE4(v92) = 0;
            v93 = 0x1000005A2LL;
            sub_21BD570(
              (__int64)&v103,
              v101,
              1467,
              1457,
              1462,
              (__int64)&v92,
              (__int64)&v93,
              (__int64)&v94,
              1452,
              (__int64)&v95);
            break;
          case 663:
            v95 = 0x100000601LL;
            v94 = 0x1000005F7LL;
            v93 = 0x1000005F2LL;
            v92 = 0x100000610LL;
            sub_21BD570(
              (__int64)&v103,
              v101,
              1557,
              1542,
              1547,
              (__int64)&v92,
              (__int64)&v93,
              (__int64)&v94,
              1532,
              (__int64)&v95);
            break;
          case 664:
            BYTE4(v95) = 0;
            v94 = 0x10000061FLL;
            BYTE4(v92) = 0;
            v93 = 0x10000061ALL;
            sub_21BD570(
              (__int64)&v103,
              v101,
              1587,
              1577,
              1582,
              (__int64)&v92,
              (__int64)&v93,
              (__int64)&v94,
              1572,
              (__int64)&v95);
            break;
          default:
            v95 = 0x100000589LL;
            v94 = 0x10000057FLL;
            v93 = 0x10000057ALL;
            v92 = 0x100000598LL;
            sub_21BD570(
              (__int64)&v103,
              v101,
              1437,
              1422,
              1427,
              (__int64)&v92,
              (__int64)&v93,
              (__int64)&v94,
              1412,
              (__int64)&v95);
            break;
        }
        goto LABEL_105;
      }
      if ( v37 != 44 && v37 != 185 )
        goto LABEL_37;
      if ( v76 )
      {
        v66 = 1377;
        v94 = 0x10000054DLL;
        v67 = 1387;
        v92 = 0x100000566LL;
        v68 = v101;
        v73 = 1362;
        v95 = 0x100000557LL;
        v93 = 0x100000548LL;
        v69 = 1372;
      }
      else
      {
        v66 = 1497;
        v94 = 0x1000005C5LL;
        v67 = 1507;
        v92 = 0x1000005DELL;
        v68 = v101;
        v73 = 1482;
        v95 = 0x1000005CFLL;
        v93 = 0x1000005C0LL;
        v69 = 1492;
      }
    }
    sub_21BD570((__int64)&v103, v68, v67, v69, v66, (__int64)&v92, (__int64)&v93, (__int64)&v94, v73, (__int64)&v95);
LABEL_105:
    if ( !v103.m128_i8[4] )
      goto LABEL_37;
    v41 = v103.m128_i16[0];
    v103.m128_u64[0] = v88;
    v103.m128_i32[2] = v86;
    goto LABEL_57;
  }
  v29 = *(_WORD *)(v9 + 24);
  if ( v29 <= 664 )
  {
    if ( v29 > 658 )
    {
      switch ( v29 )
      {
        case 660:
        case 662:
          BYTE4(v95) = 0;
          v94 = 0x1000005ABLL;
          BYTE4(v92) = 0;
          v93 = 0x1000005A6LL;
          sub_21BD570(
            (__int64)&v103,
            v101,
            1471,
            1461,
            1466,
            (__int64)&v92,
            (__int64)&v93,
            (__int64)&v94,
            1456,
            (__int64)&v95);
          break;
        case 663:
          v95 = 0x100000605LL;
          v94 = 0x1000005FBLL;
          v93 = 0x1000005F6LL;
          v92 = 0x100000614LL;
          sub_21BD570(
            (__int64)&v103,
            v101,
            1561,
            1546,
            1551,
            (__int64)&v92,
            (__int64)&v93,
            (__int64)&v94,
            1536,
            (__int64)&v95);
          break;
        case 664:
          BYTE4(v95) = 0;
          v94 = 0x100000623LL;
          BYTE4(v92) = 0;
          v93 = 0x10000061ELL;
          sub_21BD570(
            (__int64)&v103,
            v101,
            1591,
            1581,
            1586,
            (__int64)&v92,
            (__int64)&v93,
            (__int64)&v94,
            1576,
            (__int64)&v95);
          break;
        default:
          v95 = 0x10000058DLL;
          v94 = 0x100000583LL;
          v93 = 0x10000057ELL;
          v92 = 0x10000059CLL;
          sub_21BD570(
            (__int64)&v103,
            v101,
            1441,
            1426,
            1431,
            (__int64)&v92,
            (__int64)&v93,
            (__int64)&v94,
            1416,
            (__int64)&v95);
          break;
      }
LABEL_55:
      if ( v103.m128_i8[4] )
      {
        a3 = _mm_load_si128(&v100);
        v41 = v103.m128_i16[0];
        v103 = (__m128)a3;
LABEL_57:
        v42 = (_QWORD *)a1[34];
        v43 = 2;
        v104.m128i_i64[0] = v78;
        v104.m128i_i32[2] = v80;
        v44 = v34;
        goto LABEL_58;
      }
      goto LABEL_36;
    }
    if ( v29 == 44 || v29 == 185 )
    {
      if ( v76 )
      {
        v30 = 1381;
        v31 = 1391;
        v95 = 0x10000055BLL;
        v94 = 0x100000551LL;
        v71 = 1366;
        v93 = 0x10000054CLL;
        v92 = 0x10000056ALL;
        v32 = 1376;
      }
      else
      {
        v30 = 1501;
        v31 = 1511;
        v95 = 0x1000005D3LL;
        v94 = 0x1000005C9LL;
        v71 = 1486;
        v93 = 0x1000005C4LL;
        v92 = 0x1000005E2LL;
        v32 = 1496;
      }
      sub_21BD570((__int64)&v103, v101, v31, v32, v30, (__int64)&v92, (__int64)&v93, (__int64)&v94, v71, (__int64)&v95);
      goto LABEL_55;
    }
  }
LABEL_36:
  v35 = 0;
LABEL_37:
  if ( v107 != v109 )
    _libc_free((unsigned __int64)v107);
  if ( v96 )
    sub_161E7C0((__int64)&v96, v96);
  return v35;
}
