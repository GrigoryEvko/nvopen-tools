// Function: sub_29FD8C0
// Address: 0x29fd8c0
//
__int64 __fastcall sub_29FD8C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v9; // rbx
  __int64 v10; // rdx
  _BYTE *v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // r15
  int v15; // eax
  __int64 *v16; // r15
  char v17; // r13
  __int64 *v18; // rbx
  __int64 v19; // r15
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  _QWORD *v24; // rbx
  _QWORD *v25; // r14
  void (__fastcall *v26)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v27; // rax
  __m128i v28; // xmm1
  float v29; // xmm0_4
  double v30; // xmm0_8
  __int64 v31; // rsi
  __int64 v32; // r8
  unsigned __int64 v33; // rax
  int v34; // esi
  unsigned __int64 *v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rdx
  int v39; // eax
  __m128i v40; // xmm1
  __int64 v41; // rax
  _QWORD *v42; // rdi
  _QWORD *v43; // rbx
  unsigned __int64 v44; // r9
  unsigned __int64 v45; // rax
  __int64 v46; // [rsp-10h] [rbp-500h]
  __int64 v47; // [rsp-8h] [rbp-4F8h]
  __int64 *v48; // [rsp+0h] [rbp-4F0h]
  __int64 v49; // [rsp+18h] [rbp-4D8h]
  unsigned __int64 v50; // [rsp+28h] [rbp-4C8h]
  _DWORD *v51; // [rsp+30h] [rbp-4C0h]
  __int64 v52; // [rsp+30h] [rbp-4C0h]
  __int64 v53; // [rsp+30h] [rbp-4C0h]
  __int64 v54; // [rsp+40h] [rbp-4B0h]
  unsigned __int8 v55; // [rsp+60h] [rbp-490h]
  __int64 v56; // [rsp+60h] [rbp-490h]
  __int64 *v57; // [rsp+68h] [rbp-488h]
  unsigned int v58; // [rsp+7Ch] [rbp-474h] BYREF
  void *v59; // [rsp+80h] [rbp-470h] BYREF
  _QWORD *v60; // [rsp+88h] [rbp-468h]
  __int64 v61[4]; // [rsp+A0h] [rbp-450h] BYREF
  __int16 v62; // [rsp+C0h] [rbp-430h]
  unsigned __int64 *v63; // [rsp+D0h] [rbp-420h] BYREF
  __int64 v64; // [rsp+D8h] [rbp-418h]
  _BYTE v65[32]; // [rsp+E0h] [rbp-410h] BYREF
  __int64 v66; // [rsp+100h] [rbp-3F0h]
  __int64 v67; // [rsp+108h] [rbp-3E8h]
  __int16 v68; // [rsp+110h] [rbp-3E0h]
  __int64 *v69; // [rsp+118h] [rbp-3D8h]
  void **v70; // [rsp+120h] [rbp-3D0h]
  void **v71; // [rsp+128h] [rbp-3C8h]
  __int64 v72; // [rsp+130h] [rbp-3C0h]
  int v73; // [rsp+138h] [rbp-3B8h]
  __int16 v74; // [rsp+13Ch] [rbp-3B4h]
  char v75; // [rsp+13Eh] [rbp-3B2h]
  __int64 v76; // [rsp+140h] [rbp-3B0h]
  __int64 v77; // [rsp+148h] [rbp-3A8h]
  void *v78; // [rsp+150h] [rbp-3A0h] BYREF
  void *v79; // [rsp+158h] [rbp-398h] BYREF
  _QWORD v80[2]; // [rsp+160h] [rbp-390h] BYREF
  __int64 *v81; // [rsp+170h] [rbp-380h]
  __int64 v82; // [rsp+178h] [rbp-378h]
  _BYTE v83[128]; // [rsp+180h] [rbp-370h] BYREF
  unsigned __int64 v84[2]; // [rsp+200h] [rbp-2F0h] BYREF
  _BYTE v85[512]; // [rsp+210h] [rbp-2E0h] BYREF
  __int64 v86; // [rsp+410h] [rbp-E0h]
  __int64 v87; // [rsp+418h] [rbp-D8h]
  __int64 v88; // [rsp+420h] [rbp-D0h]
  __int64 v89; // [rsp+428h] [rbp-C8h]
  char v90; // [rsp+430h] [rbp-C0h]
  __int64 v91; // [rsp+438h] [rbp-B8h]
  _BYTE *v92; // [rsp+440h] [rbp-B0h]
  __int64 v93; // [rsp+448h] [rbp-A8h]
  int v94; // [rsp+450h] [rbp-A0h]
  char v95; // [rsp+454h] [rbp-9Ch]
  _BYTE v96[64]; // [rsp+458h] [rbp-98h] BYREF
  __int16 v97; // [rsp+498h] [rbp-58h]
  _QWORD *v98; // [rsp+4A0h] [rbp-50h]
  _QWORD *v99; // [rsp+4A8h] [rbp-48h]
  __int64 v100; // [rsp+4B0h] [rbp-40h]

  v5 = 47;
  v55 = sub_B2D610(a1, 47);
  if ( v55 )
    return 0;
  v80[0] = a2;
  v9 = *(_QWORD *)(a1 + 80);
  v10 = (__int64)v96;
  v84[0] = (unsigned __int64)v85;
  v88 = a3;
  v80[1] = v84;
  v11 = v83;
  v84[1] = 0x1000000000LL;
  v86 = 0;
  v87 = 0;
  v89 = 0;
  v90 = 1;
  v91 = 0;
  v92 = v96;
  v93 = 8;
  v94 = 0;
  v95 = 1;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v81 = (__int64 *)v83;
  v82 = 0x1000000000LL;
  if ( a1 + 72 != v9 )
  {
    do
    {
      v12 = v9;
      v9 = *(_QWORD *)(v9 + 8);
      v13 = *(_QWORD *)(v12 + 32);
      v14 = v12 + 24;
LABEL_7:
      while ( v14 != v13 )
      {
        while ( 1 )
        {
          v5 = v13;
          v13 = *(_QWORD *)(v13 + 8);
          v15 = *(unsigned __int8 *)(v5 - 24);
          if ( v15 == 85 )
          {
            v5 -= 24;
            sub_29FCBF0((__int64)v80, (unsigned __int8 *)v5);
            goto LABEL_7;
          }
          if ( (unsigned int)(v15 - 29) <= 0x38 )
            break;
          if ( (unsigned int)(v15 - 86) > 0xA )
            goto LABEL_131;
          if ( v14 == v13 )
            goto LABEL_12;
        }
        if ( (unsigned int)(v15 - 30) > 0x36 )
LABEL_131:
          BUG();
      }
LABEL_12:
      ;
    }
    while ( a1 + 72 != v9 );
    v10 = (unsigned int)v82;
    v16 = &v81[(unsigned int)v82];
    if ( v81 != v16 )
    {
      v57 = &v81[(unsigned int)v82];
      v17 = 0;
      v18 = v81;
      while ( 1 )
      {
        v19 = *v18;
        v5 = *(_QWORD *)(*v18 - 32);
        if ( v5 )
        {
          if ( *(_BYTE *)v5 )
          {
            v5 = 0;
          }
          else if ( *(_QWORD *)(v5 + 24) != *(_QWORD *)(v19 + 80) )
          {
            v5 = 0;
          }
        }
        sub_981210(*(_QWORD *)v80[0], v5, &v58);
        if ( v58 > 0xD3 )
        {
          if ( v58 <= 0x1C2 )
          {
            if ( v58 > 0x1BF )
            {
              v10 = sub_29FD4E0(v19, 4u, (__m128i)0LL);
              goto LABEL_28;
            }
            if ( v58 > 0x1B5 )
            {
              if ( v58 == 441 )
              {
LABEL_49:
                v10 = sub_29FCE30(
                        v19,
                        *(_QWORD *)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)),
                        1u,
                        *(_QWORD *)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)),
                        1u,
                        INFINITY,
                        (__m128i)0xFF800000);
              }
              else
              {
LABEL_54:
                if ( v58 - 438 > 2 )
                  goto LABEL_55;
                switch ( v58 )
                {
                  case 0x1B7u:
LABEL_69:
                    v28 = (__m128i)0xC2B20000;
                    v29 = 89.0;
                    break;
                  case 0x1B8u:
LABEL_65:
                    v28 = (__m128i)0xC6317400;
                    v29 = 11357.0;
                    break;
                  case 0x1B6u:
LABEL_68:
                    v28 = (__m128i)0xC4318000;
                    v29 = 710.0;
                    break;
                  default:
                    goto LABEL_131;
                }
LABEL_66:
                v10 = sub_29FCE30(
                        v19,
                        *(_QWORD *)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)),
                        2u,
                        *(_QWORD *)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)),
                        4u,
                        v29,
                        v28);
              }
LABEL_28:
              v5 = v19;
              v17 = 1;
              sub_29FCB10((__int64)v80, (_QWORD *)v19, v10);
              goto LABEL_29;
            }
            if ( v58 > 0x1B3 )
              goto LABEL_49;
          }
          if ( v58 > 0xEE )
            goto LABEL_54;
LABEL_26:
          if ( v58 > 0xEB )
          {
            v10 = sub_29FD4E0(v19, 2u, (__m128i)dword_439B458[v58 - 236]);
            goto LABEL_28;
          }
        }
        else if ( v58 > 0x9F )
        {
          v10 = v58 - 160;
          switch ( v58 )
          {
            case 0xA0u:
            case 0xA1u:
            case 0xA5u:
            case 0xA7u:
            case 0xA8u:
            case 0xACu:
              v10 = sub_29FCE30(
                      v19,
                      *(_QWORD *)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)),
                      4u,
                      *(_QWORD *)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)),
                      2u,
                      -1.0,
                      (__m128i)0x3F800000u);
              goto LABEL_28;
            case 0xA2u:
            case 0xA3u:
            case 0xA4u:
              v10 = sub_29FD4E0(v19, 4u, (__m128i)0x3F800000u);
              goto LABEL_28;
            case 0xCEu:
            case 0xCFu:
            case 0xD3u:
              goto LABEL_49;
            default:
              goto LABEL_26;
          }
        }
        if ( v58 > 0xD2 )
        {
          if ( v58 - 227 <= 8 )
          {
LABEL_62:
            switch ( v58 )
            {
              case 0xD0u:
                goto LABEL_68;
              case 0xD1u:
                goto LABEL_69;
              case 0xD2u:
                goto LABEL_65;
              case 0xE3u:
                v28 = (__m128i)0xC43A4000;
                v29 = 709.0;
                goto LABEL_66;
              case 0xE4u:
                v28 = (__m128i)0xC3A18000;
                v29 = 308.0;
                goto LABEL_66;
              case 0xE5u:
                v28 = (__m128i)0xC2340000;
                v29 = 38.0;
                goto LABEL_66;
              case 0xE6u:
                v28 = (__m128i)0xC59AB000;
                v29 = 4932.0;
                goto LABEL_66;
              case 0xE7u:
                v28 = (__m128i)0xC4864000;
                v29 = 1023.0;
                goto LABEL_66;
              case 0xE8u:
                v28 = (__m128i)0xC3150000;
                v29 = 127.0;
                goto LABEL_66;
              case 0xE9u:
                v28 = (__m128i)0xC6807A00;
                v29 = 11383.0;
                goto LABEL_66;
              case 0xEAu:
                v28 = (__m128i)0xC2CE0000;
                v29 = 88.0;
                goto LABEL_66;
              case 0xEBu:
                v28 = (__m128i)0xC6321C00;
                v29 = 11356.0;
                goto LABEL_66;
              default:
                goto LABEL_131;
            }
          }
LABEL_55:
          v10 = v58 - 333;
          switch ( v58 )
          {
            case 0x14Du:
            case 0x14Eu:
            case 0x14Fu:
            case 0x150u:
            case 0x154u:
            case 0x155u:
            case 0x156u:
            case 0x15Au:
            case 0x15Bu:
            case 0x15Cu:
            case 0x15Du:
            case 0x15Eu:
              v10 = sub_29FD4E0(v19, 5u, (__m128i)0LL);
              goto LABEL_28;
            case 0x151u:
            case 0x152u:
            case 0x153u:
              v10 = sub_29FD4E0(v19, 5u, (__m128i)0xBF800000);
              goto LABEL_28;
            case 0x182u:
            case 0x183u:
            case 0x184u:
              if ( v58 != 386 )
                goto LABEL_29;
              v10 = *(_DWORD *)(v19 + 4) & 0x7FFFFFF;
              v5 = *(_QWORD *)(v19 - 32 * v10);
              v54 = *(_QWORD *)(v19 + 32 * (1 - v10));
              if ( *(_BYTE *)v5 == 18 )
              {
                v30 = sub_C41B00((__int64 *)(v5 + 24));
                if ( v30 < 1.0 || v30 > 255.0 )
                  goto LABEL_29;
                v69 = (__int64 *)sub_BD5C60(v19);
                v70 = &v78;
                v71 = &v79;
                v74 = 512;
                v63 = (unsigned __int64 *)v65;
                v78 = &unk_49DA100;
                v66 = 0;
                v67 = 0;
                v64 = 0x200000000LL;
                v72 = 0;
                v73 = 0;
                v75 = 7;
                v76 = 0;
                v77 = 0;
                v68 = 0;
                v79 = &unk_49DA0B0;
                v66 = *(_QWORD *)(v19 + 40);
                v67 = v19 + 24;
                v31 = *(_QWORD *)sub_B46C60(v19);
                v61[0] = v31;
                if ( !v31 || (sub_B96E90((__int64)v61, v31, 1), (v32 = v61[0]) == 0) )
                {
                  sub_93FB40((__int64)&v63, 0);
                  v32 = v61[0];
                  goto LABEL_116;
                }
                v33 = (unsigned __int64)v63;
                v34 = v64;
                v35 = &v63[2 * (unsigned int)v64];
                if ( v63 == v35 )
                {
LABEL_112:
                  if ( (unsigned int)v64 >= (unsigned __int64)HIDWORD(v64) )
                  {
                    v44 = (unsigned int)v64 + 1LL;
                    v45 = v49 & 0xFFFFFFFF00000000LL;
                    v49 &= 0xFFFFFFFF00000000LL;
                    if ( HIDWORD(v64) < v44 )
                    {
                      v50 = v45;
                      v53 = v61[0];
                      sub_C8D5F0((__int64)&v63, v65, v44, 0x10u, v61[0], v44);
                      v45 = v50;
                      v32 = v53;
                      v35 = &v63[2 * (unsigned int)v64];
                    }
                    *v35 = v45;
                    v35[1] = v32;
                    v32 = v61[0];
                    LODWORD(v64) = v64 + 1;
                  }
                  else
                  {
                    if ( v35 )
                    {
                      *(_DWORD *)v35 = 0;
                      v35[1] = v32;
                      v34 = v64;
                      v32 = v61[0];
                    }
                    LODWORD(v64) = v34 + 1;
                  }
LABEL_116:
                  if ( v32 )
LABEL_83:
                    sub_B91220((__int64)v61, v32);
                  v51 = sub_C33310();
                  sub_C3B170((__int64)v61, (__m128i)0x42FE0000u);
                  sub_C407B0(&v59, v61, v51);
                  sub_C338F0((__int64)v61);
                  v52 = sub_AC8EA0(v69, (__int64 *)&v59);
                  if ( v59 == sub_C33340() )
                  {
                    if ( v60 )
                    {
                      v41 = *(v60 - 1);
                      v42 = &v60[3 * v41];
                      if ( v60 != v42 )
                      {
                        v48 = v18;
                        v43 = &v60[3 * v41];
                        do
                        {
                          v43 -= 3;
                          sub_91D830(v43);
                        }
                        while ( v60 != v43 );
                        v42 = v43;
                        v18 = v48;
                      }
                      j_j_j___libc_free_0_0((unsigned __int64)(v42 - 1));
                    }
                  }
                  else
                  {
                    sub_C338F0((__int64)&v59);
                  }
                  v36 = *(_QWORD *)(v54 + 8);
                  if ( *(_BYTE *)(v36 + 8) != 2 )
                    v52 = sub_AA93C0(0x2Eu, v52, v36);
                  if ( (unsigned __int8)sub_B2D610(*(_QWORD *)(v66 + 72), 72) )
                    LOBYTE(v74) = 1;
                  HIDWORD(v59) = 0;
                  v62 = 257;
                  v56 = sub_B35C90((__int64)&v63, 2u, v54, v52, (__int64)v61, 0, (unsigned int)v59, 0);
                  nullsub_61();
                  v78 = &unk_49DA100;
                  nullsub_63();
                  v5 = v46;
                  v10 = v56;
                  v6 = v47;
                  if ( v63 != (unsigned __int64 *)v65 )
                  {
                    _libc_free((unsigned __int64)v63);
                    v10 = v56;
                  }
                  goto LABEL_109;
                }
                while ( *(_DWORD *)v33 )
                {
                  v33 += 16LL;
                  if ( v35 == (unsigned __int64 *)v33 )
                    goto LABEL_112;
                }
                *(_QWORD *)(v33 + 8) = v61[0];
                goto LABEL_83;
              }
              if ( (unsigned __int8)(*(_BYTE *)v5 - 72) > 1u )
                goto LABEL_29;
              if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
                v37 = *(_QWORD *)(v5 - 8);
              else
                v37 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
              v63 = (unsigned __int64 *)sub_BCAE30(*(_QWORD *)(*(_QWORD *)v37 + 8LL));
              v64 = v38;
              v39 = sub_CA1930(&v63);
              switch ( v39 )
              {
                case 8:
                  v40 = (__m128i)0x43000000u;
                  break;
                case 16:
                  v40 = (__m128i)0x42800000u;
                  break;
                case 32:
                  v40 = (__m128i)0x42000000u;
                  break;
                default:
                  goto LABEL_29;
              }
              v10 = sub_29FCE30(v19, v5, 5u, v54, 2u, 0.0, v40);
LABEL_109:
              if ( v10 )
                goto LABEL_28;
              break;
            default:
              goto LABEL_29;
          }
          goto LABEL_29;
        }
        if ( v58 > 0xCF )
          goto LABEL_62;
        if ( v58 <= 0xB4 && v58 > 0xB1 )
        {
          v10 = sub_29FCE30(
                  v19,
                  *(_QWORD *)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)),
                  5u,
                  *(_QWORD *)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)),
                  3u,
                  -1.0,
                  (__m128i)0x3F800000u);
          goto LABEL_28;
        }
LABEL_29:
        if ( v57 == ++v18 )
        {
          v55 = v17;
          v16 = v81;
          break;
        }
      }
    }
    if ( v16 != (__int64 *)v83 )
      _libc_free((unsigned __int64)v16);
  }
  sub_FFCE90((__int64)v84, v5, v10, (__int64)v11, v6, v7);
  sub_FFD870((__int64)v84, v5, v20, v21, v22, v23);
  sub_FFBC40((__int64)v84, v5);
  v24 = v99;
  v25 = v98;
  if ( v99 != v98 )
  {
    do
    {
      v26 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v25[7];
      *v25 = &unk_49E5048;
      if ( v26 )
        v26(v25 + 5, v25 + 5, 3);
      *v25 = &unk_49DB368;
      v27 = v25[3];
      if ( v27 != -4096 && v27 != 0 && v27 != -8192 )
        sub_BD60C0(v25 + 1);
      v25 += 9;
    }
    while ( v24 != v25 );
    v25 = v98;
  }
  if ( v25 )
    j_j___libc_free_0((unsigned __int64)v25);
  if ( !v95 )
    _libc_free((unsigned __int64)v92);
  if ( (_BYTE *)v84[0] != v85 )
    _libc_free(v84[0]);
  return v55;
}
