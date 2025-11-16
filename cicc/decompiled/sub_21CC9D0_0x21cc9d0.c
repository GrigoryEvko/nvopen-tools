// Function: sub_21CC9D0
// Address: 0x21cc9d0
//
void __fastcall sub_21CC9D0(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128 a5, __m128i a6)
{
  char *v7; // rdx
  __int64 v8; // rsi
  char v9; // al
  const void **v10; // rdx
  unsigned int v11; // r12d
  __int64 v12; // r13
  __int64 v13; // rax
  char v14; // r12
  __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // r9
  __int64 v18; // rax
  int v19; // r9d
  __int64 v20; // rdx
  __int64 v21; // r12
  unsigned __int16 v22; // r13
  __int64 v23; // rdx
  const __m128i *v24; // rax
  int v25; // esi
  __int64 v26; // rcx
  const __m128i *v27; // r8
  __m128 *v28; // rdx
  unsigned __int64 v29; // r15
  int v30; // r10d
  unsigned __int8 v31; // si
  __int64 v32; // rdx
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // r10
  __int64 v36; // r11
  __int64 v37; // rax
  __int64 *v38; // rax
  __int64 *v39; // rsi
  __int64 v40; // r9
  int v41; // r8d
  __int64 v42; // r9
  unsigned int v43; // r14d
  __int64 v44; // r12
  __int128 v45; // rax
  __m128i v46; // rax
  __int128 v47; // rax
  int v48; // r8d
  int v49; // r9d
  __int64 *v50; // r12
  __int64 *v51; // rdx
  __int64 *v52; // r13
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 **v55; // rax
  __int64 v56; // rax
  __m128i *v57; // rdx
  __int64 v58; // r12
  int v59; // r8d
  int v60; // r9d
  __int64 *v61; // r13
  __int64 *v62; // rdx
  __int64 *v63; // r14
  __int64 v64; // rdx
  __int64 **v65; // rdx
  __int64 v66; // rax
  __int64 *v67; // rax
  __m128i *v68; // rdi
  __int64 v69; // rbx
  unsigned int v70; // r14d
  unsigned int v71; // r15d
  __int64 v72; // r12
  unsigned __int64 v73; // r13
  __int64 v74; // rax
  __int16 v75; // si
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 *v78; // rax
  unsigned int v79; // eax
  const void **v80; // rdx
  const void **v81; // r8
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // rdx
  unsigned __int8 v88; // [rsp-10h] [rbp-210h]
  __int128 v89; // [rsp-10h] [rbp-210h]
  __int128 v90; // [rsp-10h] [rbp-210h]
  __int64 v91; // [rsp-10h] [rbp-210h]
  __int128 v92; // [rsp-10h] [rbp-210h]
  __int64 v93; // [rsp-8h] [rbp-208h]
  int v94; // [rsp-8h] [rbp-208h]
  unsigned int v95; // [rsp+0h] [rbp-200h]
  unsigned int v96; // [rsp+8h] [rbp-1F8h]
  const __m128i *v97; // [rsp+10h] [rbp-1F0h]
  const __m128i *v98; // [rsp+20h] [rbp-1E0h]
  __int64 v99; // [rsp+20h] [rbp-1E0h]
  __int64 v100; // [rsp+28h] [rbp-1D8h]
  char v101; // [rsp+38h] [rbp-1C8h]
  __int64 *v102; // [rsp+38h] [rbp-1C8h]
  __m128i v104; // [rsp+50h] [rbp-1B0h] BYREF
  __int64 v105; // [rsp+60h] [rbp-1A0h]
  __int64 v106; // [rsp+68h] [rbp-198h]
  __int64 v107; // [rsp+70h] [rbp-190h]
  __int64 v108; // [rsp+78h] [rbp-188h]
  __int64 v109; // [rsp+80h] [rbp-180h] BYREF
  const void **v110; // [rsp+88h] [rbp-178h]
  __int64 v111; // [rsp+90h] [rbp-170h] BYREF
  int v112; // [rsp+98h] [rbp-168h]
  __m128i v113; // [rsp+A0h] [rbp-160h] BYREF
  __int64 *v114; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v115; // [rsp+B8h] [rbp-148h]
  _BYTE v116[128]; // [rsp+C0h] [rbp-140h] BYREF
  __m128i v117; // [rsp+140h] [rbp-C0h] BYREF
  __m128i v118; // [rsp+150h] [rbp-B0h] BYREF
  __m128i v119; // [rsp+160h] [rbp-A0h]
  __m128i v120; // [rsp+170h] [rbp-90h]
  char v121; // [rsp+180h] [rbp-80h]
  __int64 v122; // [rsp+188h] [rbp-78h]

  v7 = *(char **)(a1 + 40);
  v8 = *(_QWORD *)(a1 + 72);
  v9 = *v7;
  v10 = (const void **)*((_QWORD *)v7 + 1);
  v111 = v8;
  LOBYTE(v109) = v9;
  v110 = v10;
  if ( v8 )
  {
    sub_1623A60((__int64)&v111, v8, 2);
    v9 = v109;
  }
  v112 = *(_DWORD *)(a1 + 64);
  switch ( v9 )
  {
    case 25:
    case 26:
    case 34:
    case 35:
    case 42:
    case 43:
    case 50:
    case 86:
    case 87:
    case 88:
    case 90:
    case 91:
    case 95:
      v11 = sub_1E34390(*(_QWORD *)(a1 + 104));
      v12 = sub_1E0A0C0(*(_QWORD *)(a2 + 32));
      v13 = sub_1F58E60((__int64)&v109, *(_QWORD **)(a2 + 48));
      if ( (unsigned int)sub_15AAE50(v12, v13) > v11 )
        goto LABEL_5;
      if ( (_BYTE)v109 )
      {
        switch ( (char)v109 )
        {
          case 14:
          case 15:
          case 16:
          case 17:
          case 18:
          case 19:
          case 20:
          case 21:
          case 22:
          case 23:
          case 56:
          case 57:
          case 58:
          case 59:
          case 60:
          case 61:
            v113.m128i_i8[0] = 2;
            goto LABEL_49;
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
            v113.m128i_i8[0] = 3;
            goto LABEL_49;
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
            v113.m128i_i8[0] = 4;
            goto LABEL_49;
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
            v113.m128i_i8[0] = 5;
            goto LABEL_49;
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
            v113.m128i_i8[0] = 6;
            goto LABEL_49;
          case 55:
            v113.m128i_i8[0] = 7;
            v113.m128i_i64[1] = 0;
            LODWORD(v105) = 1;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            v113.m128i_i8[0] = 8;
            goto LABEL_49;
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
            v113.m128i_i8[0] = 9;
            goto LABEL_49;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            v113.m128i_i8[0] = 10;
LABEL_49:
            v113.m128i_i64[1] = 0;
            LODWORD(v105) = (unsigned __int16)word_435D740[(unsigned __int8)(v109 - 14)];
            break;
        }
      }
      else
      {
        v14 = sub_1F596B0((__int64)&v109);
        v113.m128i_i64[1] = v15;
        v113.m128i_i8[0] = v14;
        if ( (_BYTE)v109 )
          LODWORD(v105) = (unsigned __int16)word_435D740[(unsigned __int8)(v109 - 14)];
        else
          LODWORD(v105) = sub_1F58D30((__int64)&v109);
        if ( !v14 )
        {
          v16 = sub_1F58D40((__int64)&v113);
          goto LABEL_13;
        }
      }
      v16 = sub_1F3E310(&v113);
LABEL_13:
      v104.m128i_i8[0] = 0;
      if ( v16 <= 0xF )
      {
        v113.m128i_i8[0] = 4;
        v113.m128i_i64[1] = 0;
        v104.m128i_i8[0] = 1;
      }
      switch ( (_DWORD)v105 )
      {
        case 4:
          a4 = _mm_load_si128(&v113);
          v121 = 1;
          v122 = 0;
          v22 = 660;
          v117 = a4;
          v118 = a4;
          v119 = a4;
          v120 = a4;
          v86 = sub_1D25C30(a2, (unsigned __int8 *)&v117, 5);
          v101 = 0;
          v106 = v87;
          v21 = v86;
          break;
        case 8:
          v117.m128i_i8[0] = 86;
          v118.m128i_i8[0] = 86;
          v22 = 660;
          v117.m128i_i64[1] = 0;
          v118.m128i_i64[1] = 0;
          v119.m128i_i8[0] = 86;
          v119.m128i_i64[1] = 0;
          v120.m128i_i8[0] = 86;
          v120.m128i_i64[1] = 0;
          v121 = 1;
          v122 = 0;
          v84 = sub_1D25C30(a2, (unsigned __int8 *)&v117, 5);
          v101 = 1;
          v106 = v85;
          v21 = v84;
          break;
        case 2:
          v18 = sub_1D25E70(a2, v113.m128i_u32[0], v113.m128i_i64[1], v113.m128i_u32[0], v113.m128i_i64[1], v17, 1, 0);
          v101 = 0;
          v106 = v20;
          v21 = v18;
          v22 = 659;
          break;
        default:
          goto LABEL_5;
      }
      v23 = *(unsigned int *)(a1 + 56);
      v24 = *(const __m128i **)(a1 + 32);
      v115 = 0x800000000LL;
      v25 = 0;
      v26 = 40 * v23;
      v114 = (__int64 *)v116;
      v27 = (const __m128i *)((char *)v24 + 40 * v23);
      v28 = (__m128 *)v116;
      v29 = 0xCCCCCCCCCCCCCCCDLL * (v26 >> 3);
      if ( (unsigned __int64)v26 > 0x140 )
      {
        v97 = v24;
        v98 = v27;
        sub_16CD150((__int64)&v114, v116, 0xCCCCCCCCCCCCCCCDLL * (v26 >> 3), 16, (int)v27, v19);
        v25 = v115;
        v24 = v97;
        v27 = v98;
        v28 = (__m128 *)&v114[2 * (unsigned int)v115];
      }
      if ( v24 != v27 )
      {
        do
        {
          if ( v28 )
          {
            a5 = (__m128)_mm_loadu_si128(v24);
            *v28 = a5;
          }
          v24 = (const __m128i *)((char *)v24 + 40);
          ++v28;
        }
        while ( v27 != v24 );
        v25 = v115;
      }
      v30 = v29 + v25;
      v31 = *(_BYTE *)(a1 + 27);
      LODWORD(v115) = v30;
      v35 = sub_1D38E70(a2, (v31 >> 2) & 3, (__int64)&v111, 0, a4, *(double *)a5.m128_u64, a6);
      v36 = v32;
      v37 = (unsigned int)v115;
      if ( (unsigned int)v115 >= HIDWORD(v115) )
      {
        v100 = v32;
        v99 = v35;
        sub_16CD150((__int64)&v114, v116, 0, 16, v33, v34);
        v37 = (unsigned int)v115;
        v35 = v99;
        v36 = v100;
      }
      v38 = &v114[2 * v37];
      *v38 = v35;
      v39 = v114;
      v38[1] = v36;
      v93 = *(_QWORD *)(a1 + 96);
      v88 = *(_BYTE *)(a1 + 88);
      v40 = *(_QWORD *)(a1 + 104);
      LODWORD(v115) = v115 + 1;
      v106 = sub_1D24DC0((_QWORD *)a2, v22, (__int64)&v111, v21, v106, v40, v39, (unsigned int)v115, v88, v93);
      v117.m128i_i64[0] = (__int64)&v118;
      v117.m128i_i64[1] = 0x800000000LL;
      if ( v101 )
      {
        LODWORD(v105) = (unsigned int)v105 >> 1;
        if ( (_DWORD)v105 )
        {
          v43 = 0;
          do
          {
            v44 = v106;
            *(_QWORD *)&v45 = sub_1D38E70(a2, 0, (__int64)&v111, 0, a4, *(double *)a5.m128_u64, a6);
            v46.m128i_i64[0] = (__int64)sub_1D332F0(
                                          (__int64 *)a2,
                                          106,
                                          (__int64)&v111,
                                          v113.m128i_u32[0],
                                          (const void **)v113.m128i_i64[1],
                                          0,
                                          *(double *)a4.m128i_i64,
                                          *(double *)a5.m128_u64,
                                          a6,
                                          v44,
                                          v43,
                                          v45);
            v104 = v46;
            *(_QWORD *)&v47 = sub_1D38E70(a2, 1, (__int64)&v111, 0, a4, *(double *)a5.m128_u64, a6);
            v50 = sub_1D332F0(
                    (__int64 *)a2,
                    106,
                    (__int64)&v111,
                    v113.m128i_u32[0],
                    (const void **)v113.m128i_i64[1],
                    0,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128_u64,
                    a6,
                    v44,
                    v43,
                    v47);
            v52 = v51;
            v53 = v117.m128i_u32[2];
            if ( v117.m128i_i32[2] >= (unsigned __int32)v117.m128i_i32[3] )
            {
              sub_16CD150((__int64)&v117, &v118, 0, 16, v48, v49);
              v53 = v117.m128i_u32[2];
            }
            a6 = _mm_load_si128(&v104);
            *(__m128i *)(v117.m128i_i64[0] + 16 * v53) = a6;
            v54 = (unsigned int)(v117.m128i_i32[2] + 1);
            v117.m128i_i32[2] = v54;
            if ( v117.m128i_i32[3] <= (unsigned int)v54 )
            {
              sub_16CD150((__int64)&v117, &v118, 0, 16, v48, v49);
              v54 = v117.m128i_u32[2];
            }
            v55 = (__int64 **)(v117.m128i_i64[0] + 16 * v54);
            ++v43;
            *v55 = v50;
            v55[1] = v52;
            v56 = (unsigned int)++v117.m128i_i32[2];
          }
          while ( (_DWORD)v105 != v43 );
          v57 = (__m128i *)v117.m128i_i64[0];
          v105 = (unsigned int)v105;
          goto LABEL_37;
        }
      }
      else if ( (_DWORD)v105 )
      {
        v105 = (unsigned int)v105;
        v102 = (__int64 *)a2;
        v69 = 0;
        v70 = v95;
        v71 = v96;
        do
        {
          v72 = v106;
          v73 = (unsigned int)v69;
          if ( byte_4FD3A00 )
          {
            v74 = *(_QWORD *)(v106 + 40) + 16 * v69;
            switch ( *(_BYTE *)v74 )
            {
              case 2:
                v75 = 616;
                break;
              case 4:
                v75 = 614;
                break;
              case 5:
                v75 = 618;
                break;
              case 6:
                v75 = 620;
                break;
              case 9:
                v75 = 479;
                break;
              case 0xA:
                v75 = 481;
                break;
              default:
                if ( (__m128i *)v117.m128i_i64[0] != &v118 )
                  _libc_free(v117.m128i_u64[0]);
                if ( v114 != (__int64 *)v116 )
                  _libc_free((unsigned __int64)v114);
                goto LABEL_5;
            }
            *((_QWORD *)&v90 + 1) = (unsigned int)v69;
            LOBYTE(v70) = *(_BYTE *)v74;
            *(_QWORD *)&v90 = v106;
            v76 = sub_1D2CC80(v102, v75, (__int64)&v111, v70, *(_QWORD *)(v74 + 8), v42, v90);
            v42 = v91;
            v72 = v76;
            v73 = 0;
          }
          if ( v104.m128i_i8[0] )
          {
            if ( (_BYTE)v109 )
            {
              switch ( (char)v109 )
              {
                case 14:
                case 15:
                case 16:
                case 17:
                case 18:
                case 19:
                case 20:
                case 21:
                case 22:
                case 23:
                case 56:
                case 57:
                case 58:
                case 59:
                case 60:
                case 61:
                  LOBYTE(v79) = 2;
                  break;
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
                  LOBYTE(v79) = 3;
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
                  LOBYTE(v79) = 4;
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
                  LOBYTE(v79) = 5;
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
                  LOBYTE(v79) = 6;
                  break;
                case 55:
                  LOBYTE(v79) = 7;
                  break;
                case 86:
                case 87:
                case 88:
                case 98:
                case 99:
                case 100:
                  LOBYTE(v79) = 8;
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
                  LOBYTE(v79) = 9;
                  break;
                case 94:
                case 95:
                case 96:
                case 97:
                case 106:
                case 107:
                case 108:
                case 109:
                  LOBYTE(v79) = 10;
                  break;
                default:
                  BUG();
              }
              v81 = 0;
            }
            else
            {
              LOBYTE(v79) = sub_1F596B0((__int64)&v109);
              v71 = v79;
              v81 = v80;
            }
            *((_QWORD *)&v92 + 1) = v73;
            LOBYTE(v71) = v79;
            *(_QWORD *)&v92 = v72;
            v82 = sub_1D309E0(
                    v102,
                    145,
                    (__int64)&v111,
                    v71,
                    v81,
                    0,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128_u64,
                    *(double *)a6.m128i_i64,
                    v92);
            v41 = v94;
            v107 = v82;
            v72 = v82;
            v108 = v83;
            v73 = (unsigned int)v83 | v73 & 0xFFFFFFFF00000000LL;
          }
          v77 = v117.m128i_u32[2];
          if ( v117.m128i_i32[2] >= (unsigned __int32)v117.m128i_i32[3] )
          {
            sub_16CD150((__int64)&v117, &v118, 0, 16, v41, v42);
            v77 = v117.m128i_u32[2];
          }
          v78 = (__int64 *)(v117.m128i_i64[0] + 16 * v77);
          ++v69;
          *v78 = v72;
          v78[1] = v73;
          v56 = (unsigned int)++v117.m128i_i32[2];
        }
        while ( v105 != v69 );
        a2 = (__int64)v102;
        v57 = (__m128i *)v117.m128i_i64[0];
        goto LABEL_37;
      }
      v105 = 0;
      v57 = &v118;
      v56 = 0;
LABEL_37:
      *((_QWORD *)&v89 + 1) = v56;
      *(_QWORD *)&v89 = v57;
      v58 = v105;
      v61 = sub_1D359D0(
              (__int64 *)a2,
              104,
              (__int64)&v111,
              v109,
              v110,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128_u64,
              a6,
              v89);
      v63 = v62;
      v64 = *(unsigned int *)(a3 + 8);
      if ( (unsigned int)v64 >= *(_DWORD *)(a3 + 12) )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v59, v60);
        v64 = *(unsigned int *)(a3 + 8);
      }
      v65 = (__int64 **)(*(_QWORD *)a3 + 16 * v64);
      *v65 = v61;
      v65[1] = v63;
      LODWORD(v105) = *(_DWORD *)(a3 + 8);
      v66 = (unsigned int)(v105 + 1);
      *(_DWORD *)(a3 + 8) = v66;
      if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v66 )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v59, v60);
        v66 = *(unsigned int *)(a3 + 8);
      }
      v67 = (__int64 *)(*(_QWORD *)a3 + 16 * v66);
      *v67 = v106;
      v67[1] = v58;
      v68 = (__m128i *)v117.m128i_i64[0];
      ++*(_DWORD *)(a3 + 8);
      if ( v68 != &v118 )
        _libc_free((unsigned __int64)v68);
      if ( v114 != (__int64 *)v116 )
        _libc_free((unsigned __int64)v114);
      if ( v111 )
        sub_161E7C0((__int64)&v111, v111);
      return;
    default:
LABEL_5:
      if ( v111 )
        sub_161E7C0((__int64)&v111, v111);
      return;
  }
}
