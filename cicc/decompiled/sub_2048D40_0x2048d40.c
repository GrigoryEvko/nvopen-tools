// Function: sub_2048D40
// Address: 0x2048d40
//
void __fastcall sub_2048D40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        int a6,
        __m128i a7,
        double a8,
        __m128i a9,
        unsigned __int8 a10,
        __int64 a11,
        unsigned int *a12)
{
  unsigned int v12; // ebx
  unsigned int v13; // r14d
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r10
  char v18; // r8
  __int64 v19; // rsi
  bool v20; // zf
  __int64 (__fastcall *v21)(__int64, __int64, __int64, unsigned int, __int64, __int64, unsigned int *, char *); // rax
  unsigned int v22; // eax
  char v23; // al
  unsigned __int8 v24; // si
  int v25; // ecx
  unsigned int v26; // eax
  int v27; // r9d
  const void **v28; // r8
  __int64 v29; // rdx
  __int64 v30; // r14
  _BYTE *v31; // rax
  unsigned int v32; // r8d
  __int64 v33; // r14
  _BYTE *i; // rdx
  __int64 v35; // r14
  __int64 v36; // rax
  _BYTE *v37; // r15
  __int64 v38; // rax
  unsigned int v39; // edx
  char v40; // al
  __int128 v41; // rax
  __int64 *v42; // rax
  int v43; // edx
  int v44; // esi
  __int64 *v45; // rdx
  __int64 v46; // rax
  _BYTE *v47; // rax
  __int64 v48; // rax
  unsigned int v49; // edx
  char v50; // al
  __int64 v51; // rcx
  __int64 v52; // rsi
  __int64 v53; // r15
  __int128 v54; // rax
  __int64 *v55; // rax
  int v56; // edx
  unsigned int v57; // r9d
  int v58; // ebx
  int v59; // r14d
  _QWORD *v60; // rax
  __int64 v61; // rax
  unsigned int v62; // edx
  int v63; // ebx
  char v64; // r8
  int v65; // eax
  __int64 v66; // r9
  char v67; // r8
  unsigned __int8 v68; // r14
  char v69; // al
  __int64 v70; // rdx
  unsigned __int8 v71; // dl
  char v72; // bl
  char v73; // si
  char v74; // r8
  __int64 v75; // rdx
  __int64 v76; // rax
  unsigned int v77; // eax
  unsigned int v78; // ecx
  char v79; // bl
  unsigned int v80; // eax
  char v81; // bl
  const void **v82; // rdx
  const void **v83; // rcx
  bool v84; // al
  char v85; // al
  const void **v86; // rdx
  __int64 v87; // r14
  __int64 v88; // rbx
  __int64 v89; // rcx
  __int64 v90; // rdx
  int v91; // r8d
  const void **v92; // rdx
  unsigned int v93; // ebx
  char v94; // r8
  unsigned int v95; // eax
  unsigned int v96; // esi
  unsigned int v97; // eax
  const void **v98; // rdx
  unsigned int v99; // ebx
  unsigned int v100; // edx
  __int64 v101; // rbx
  unsigned __int64 v102; // rdx
  unsigned int v103; // edx
  __int64 (__fastcall *v104)(__int64, __int64); // rbx
  __int64 v105; // rax
  unsigned int v106; // edx
  unsigned __int8 v107; // al
  __int128 v108; // rax
  unsigned int v109; // edx
  char v110; // al
  char v111; // r14
  const void **v112; // rdx
  int v113; // ebx
  char v114; // al
  unsigned int v115; // edx
  unsigned int v116; // eax
  int v117; // eax
  unsigned int v118; // ebx
  unsigned int v119; // r14d
  unsigned __int8 v120; // al
  __int128 v121; // rax
  __int64 *v122; // rax
  __int64 v123; // rdx
  __int64 *v124; // r8
  __int64 v125; // rdx
  __int64 **v126; // rdx
  __int64 v127; // rax
  unsigned int v128; // edx
  int v129; // ebx
  int v130; // r8d
  _QWORD *v131; // r14
  __int64 v132; // rdx
  __int64 v133; // r15
  __int64 v134; // rdx
  _QWORD *v135; // rdx
  unsigned int v136; // eax
  unsigned int v137; // edx
  __int128 v138; // [rsp-10h] [rbp-220h]
  int v139; // [rsp+0h] [rbp-210h]
  unsigned int v140; // [rsp+8h] [rbp-208h]
  _QWORD *v141; // [rsp+8h] [rbp-208h]
  const void **v142; // [rsp+10h] [rbp-200h]
  unsigned int v143; // [rsp+18h] [rbp-1F8h]
  unsigned int v144; // [rsp+1Ch] [rbp-1F4h]
  int v145; // [rsp+1Ch] [rbp-1F4h]
  unsigned int v147; // [rsp+30h] [rbp-1E0h]
  unsigned int v148; // [rsp+30h] [rbp-1E0h]
  __int64 (__fastcall *v149)(__int64, __int64); // [rsp+30h] [rbp-1E0h]
  __int64 (__fastcall *v150)(__int64, __int64); // [rsp+30h] [rbp-1E0h]
  __int64 (__fastcall *v151)(__int64, __int64); // [rsp+30h] [rbp-1E0h]
  __int64 *v152; // [rsp+30h] [rbp-1E0h]
  __int64 v153; // [rsp+38h] [rbp-1D8h]
  __int64 v154; // [rsp+40h] [rbp-1D0h]
  __int64 v155; // [rsp+40h] [rbp-1D0h]
  unsigned __int8 v156; // [rsp+40h] [rbp-1D0h]
  unsigned int v157; // [rsp+48h] [rbp-1C8h]
  __int64 v158; // [rsp+48h] [rbp-1C8h]
  unsigned int v159; // [rsp+48h] [rbp-1C8h]
  unsigned __int8 v160; // [rsp+48h] [rbp-1C8h]
  __int64 v161; // [rsp+50h] [rbp-1C0h]
  __int64 v162; // [rsp+50h] [rbp-1C0h]
  unsigned int v163; // [rsp+50h] [rbp-1C0h]
  int v164; // [rsp+50h] [rbp-1C0h]
  unsigned int v165; // [rsp+50h] [rbp-1C0h]
  __int64 v166; // [rsp+58h] [rbp-1B8h]
  const void **v167; // [rsp+58h] [rbp-1B8h]
  char v168; // [rsp+58h] [rbp-1B8h]
  char v169; // [rsp+58h] [rbp-1B8h]
  char v170; // [rsp+58h] [rbp-1B8h]
  unsigned __int128 v171; // [rsp+60h] [rbp-1B0h]
  unsigned int v172; // [rsp+60h] [rbp-1B0h]
  int v173; // [rsp+60h] [rbp-1B0h]
  char v174; // [rsp+9Bh] [rbp-175h] BYREF
  unsigned int v175; // [rsp+9Ch] [rbp-174h] BYREF
  __int64 v176; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v177; // [rsp+A8h] [rbp-168h]
  int v178; // [rsp+B0h] [rbp-160h] BYREF
  char v179; // [rsp+B4h] [rbp-15Ch]
  __int64 v180; // [rsp+B8h] [rbp-158h]
  __int64 v181; // [rsp+C0h] [rbp-150h] BYREF
  const void **v182; // [rsp+C8h] [rbp-148h]
  _BYTE *v183; // [rsp+D0h] [rbp-140h] BYREF
  __int64 v184; // [rsp+D8h] [rbp-138h]
  _BYTE v185[304]; // [rsp+E0h] [rbp-130h] BYREF

  v171 = __PAIR128__(a4, a3);
  v15 = *(_QWORD *)(a1 + 16);
  v166 = a3;
  v157 = a4;
  v161 = (unsigned int)a4;
  v154 = 16LL * (unsigned int)a4;
  v16 = *(_QWORD *)(a3 + 40) + v154;
  v17 = *(_QWORD *)(v16 + 8);
  v18 = *(_BYTE *)v16;
  v177 = v17;
  LOBYTE(v176) = v18;
  if ( a6 == 1 )
  {
    v180 = 0;
    LOBYTE(v178) = a10;
    if ( a10 == v18 && (!v17 || a10) )
      goto LABEL_64;
    v63 = sub_2045180(a10);
    if ( v64 )
    {
      v65 = sub_2045180(v64);
    }
    else
    {
      v65 = sub_1F58D40((__int64)&v176);
      v67 = 0;
    }
    if ( v63 == v65 )
    {
      v166 = sub_1D309E0((__int64 *)a1, 158, a2, a10, 0, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, v171);
      v157 = v115;
      goto LABEL_64;
    }
    v68 = a10 - 14;
    if ( (unsigned __int8)(a10 - 14) > 0x5Fu )
    {
LABEL_101:
      if ( v67 )
        goto LABEL_102;
LABEL_118:
      if ( (unsigned int)sub_1F58D30((__int64)&v176) != 1 )
      {
        v96 = sub_1F58D40((__int64)&v176);
LABEL_104:
        if ( v96 == 32 )
        {
          LOBYTE(v97) = 5;
        }
        else if ( v96 > 0x20 )
        {
          if ( v96 == 64 )
          {
            LOBYTE(v97) = 6;
          }
          else
          {
            if ( v96 != 128 )
            {
LABEL_112:
              v97 = sub_1F58CC0(*(_QWORD **)(a1 + 48), v96);
              v147 = v97;
              goto LABEL_109;
            }
            LOBYTE(v97) = 7;
          }
        }
        else if ( v96 == 8 )
        {
          LOBYTE(v97) = 3;
        }
        else
        {
          LOBYTE(v97) = 4;
          if ( v96 != 16 )
          {
            LOBYTE(v97) = 2;
            if ( v96 != 1 )
              goto LABEL_112;
          }
        }
        v98 = 0;
LABEL_109:
        v99 = v147;
        LOBYTE(v99) = v97;
        v101 = sub_1D32840(
                 (__int64 *)a1,
                 v99,
                 v98,
                 v171,
                 *((__int64 *)&v171 + 1),
                 *(double *)a7.m128i_i64,
                 a8,
                 *(double *)a9.m128i_i64);
        v102 = v100 | *((_QWORD *)&v171 + 1) & 0xFFFFFFFF00000000LL;
LABEL_110:
        v166 = sub_1D321C0((__int64 *)a1, v101, v102, a2, a10, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64);
        v157 = v103;
LABEL_64:
        *(_QWORD *)a5 = v166;
        *(_DWORD *)(a5 + 8) = v157;
        return;
      }
LABEL_120:
      v104 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v15 + 48LL);
      v105 = sub_1E0A0C0(*(_QWORD *)(a1 + 32));
      if ( v104 == sub_1D13A20 )
      {
        v106 = 8 * sub_15A9520(v105, 0);
        if ( v106 == 32 )
        {
          v107 = 5;
        }
        else if ( v106 > 0x20 )
        {
          v107 = 6;
          if ( v106 != 64 )
          {
            v107 = 0;
            if ( v106 == 128 )
              v107 = 7;
          }
        }
        else
        {
          v107 = 3;
          if ( v106 != 8 )
            v107 = 4 * (v106 == 16);
        }
      }
      else
      {
        v107 = v104(v15, v105);
      }
      *(_QWORD *)&v108 = sub_1D38BB0(a1, 0, a2, v107, 0, 0, a7, a8, a9, 0);
      v166 = (__int64)sub_1D332F0(
                        (__int64 *)a1,
                        106,
                        a2,
                        a10,
                        0,
                        0,
                        *(double *)a7.m128i_i64,
                        a8,
                        a9,
                        v171,
                        *((unsigned __int64 *)&v171 + 1),
                        v108);
      v157 = v109;
      goto LABEL_64;
    }
    if ( v67 )
    {
      switch ( v67 )
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
          v73 = 2;
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
          v73 = 3;
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
          v73 = 4;
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
          v73 = 5;
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
          v73 = 6;
          break;
        case 55:
          v73 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v73 = 8;
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
          v73 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v73 = 10;
          break;
      }
      v71 = a10;
    }
    else
    {
      v69 = sub_1F596B0((__int64)&v176);
      v155 = v70;
      v71 = v178;
      v72 = v69;
      v73 = v69;
      if ( !(_BYTE)v178 )
      {
        v74 = sub_1F596B0((__int64)&v178);
        v76 = v75;
        v71 = v178;
        if ( v72 != v74 || !v72 && v155 != v76 )
          goto LABEL_79;
        if ( !(_BYTE)v178 )
        {
          v77 = sub_1F58D30((__int64)&v178);
          v71 = 0;
          v78 = v77;
          goto LABEL_76;
        }
LABEL_142:
        v78 = word_4307B00[(unsigned __int8)(v71 - 14)];
LABEL_76:
        v79 = v176;
        if ( (_BYTE)v176 )
        {
          v80 = word_4307B00[(unsigned __int8)(v176 - 14)];
        }
        else
        {
          v156 = v71;
          v159 = v78;
          v80 = sub_1F58D30((__int64)&v176);
          v71 = v156;
          v78 = v159;
        }
        if ( v80 >= v78 )
        {
LABEL_79:
          if ( !v71 )
          {
            v81 = sub_1F596B0((__int64)&v178);
            v83 = v82;
LABEL_81:
            v67 = v176;
            LOBYTE(v181) = v81;
            v182 = v83;
            if ( (_BYTE)v176 )
            {
              switch ( (char)v176 )
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
                  v111 = 2;
                  goto LABEL_153;
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
                  v111 = 3;
                  goto LABEL_153;
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
                  v111 = 4;
                  goto LABEL_153;
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
                  v111 = 5;
                  goto LABEL_153;
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
                  v111 = 6;
                  goto LABEL_153;
                case 55:
                  v111 = 7;
                  goto LABEL_153;
                case 86:
                case 87:
                case 88:
                case 98:
                case 99:
                case 100:
                  v111 = 8;
                  goto LABEL_153;
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
                  v111 = 9;
                  goto LABEL_153;
                case 94:
                case 95:
                case 96:
                case 97:
                case 106:
                case 107:
                case 108:
                case 109:
                  v111 = 10;
LABEL_153:
                  v112 = 0;
                  goto LABEL_154;
                default:
                  goto LABEL_242;
              }
            }
            v167 = v83;
            v114 = sub_1F596B0((__int64)&v176);
            v67 = v176;
            v83 = v167;
            v111 = v114;
LABEL_154:
            LOBYTE(v183) = v111;
            v184 = (__int64)v112;
            if ( v81 == v111 )
            {
              if ( v111 || v112 == v83 )
                goto LABEL_156;
            }
            else if ( v81 )
            {
              v93 = sub_2045180(v81);
              goto LABEL_98;
            }
            v169 = v67;
            v116 = sub_1F58D40((__int64)&v181);
            v94 = v169;
            v93 = v116;
LABEL_98:
            if ( v111 )
            {
              v95 = sub_2045180(v111);
            }
            else
            {
              v168 = v94;
              v95 = sub_1F58D40((__int64)&v183);
              v67 = v168;
            }
            if ( v95 > v93 )
              goto LABEL_101;
LABEL_156:
            if ( (_BYTE)v178 )
            {
              v113 = word_4307B00[(unsigned __int8)(v178 - 14)];
            }
            else
            {
              v170 = v67;
              v117 = sub_1F58D30((__int64)&v178);
              v67 = v170;
              v113 = v117;
            }
            if ( v67 )
            {
              if ( word_4307B00[(unsigned __int8)(v67 - 14)] != v113 )
              {
LABEL_102:
                if ( word_4307B00[(unsigned __int8)(v67 - 14)] != 1 )
                {
                  v96 = sub_2045180(v67);
                  goto LABEL_104;
                }
                goto LABEL_120;
              }
            }
            else if ( (unsigned int)sub_1F58D30((__int64)&v176) != v113 )
            {
              goto LABEL_118;
            }
            v102 = *((_QWORD *)&v171 + 1);
            v101 = v171;
            goto LABEL_110;
          }
LABEL_183:
          switch ( v71 )
          {
            case 0xEu:
            case 0xFu:
            case 0x10u:
            case 0x11u:
            case 0x12u:
            case 0x13u:
            case 0x14u:
            case 0x15u:
            case 0x16u:
            case 0x17u:
            case 0x38u:
            case 0x39u:
            case 0x3Au:
            case 0x3Bu:
            case 0x3Cu:
            case 0x3Du:
              v81 = 2;
              v83 = 0;
              goto LABEL_81;
            case 0x18u:
            case 0x19u:
            case 0x1Au:
            case 0x1Bu:
            case 0x1Cu:
            case 0x1Du:
            case 0x1Eu:
            case 0x1Fu:
            case 0x20u:
            case 0x3Eu:
            case 0x3Fu:
            case 0x40u:
            case 0x41u:
            case 0x42u:
            case 0x43u:
              v81 = 3;
              v83 = 0;
              goto LABEL_81;
            case 0x21u:
            case 0x22u:
            case 0x23u:
            case 0x24u:
            case 0x25u:
            case 0x26u:
            case 0x27u:
            case 0x28u:
            case 0x44u:
            case 0x45u:
            case 0x46u:
            case 0x47u:
            case 0x48u:
            case 0x49u:
              v81 = 4;
              v83 = 0;
              goto LABEL_81;
            case 0x29u:
            case 0x2Au:
            case 0x2Bu:
            case 0x2Cu:
            case 0x2Du:
            case 0x2Eu:
            case 0x2Fu:
            case 0x30u:
            case 0x4Au:
            case 0x4Bu:
            case 0x4Cu:
            case 0x4Du:
            case 0x4Eu:
            case 0x4Fu:
              v81 = 5;
              v83 = 0;
              goto LABEL_81;
            case 0x31u:
            case 0x32u:
            case 0x33u:
            case 0x34u:
            case 0x35u:
            case 0x36u:
            case 0x50u:
            case 0x51u:
            case 0x52u:
            case 0x53u:
            case 0x54u:
            case 0x55u:
              v81 = 6;
              v83 = 0;
              goto LABEL_81;
            case 0x37u:
              v81 = 7;
              v83 = 0;
              goto LABEL_81;
            case 0x56u:
            case 0x57u:
            case 0x58u:
            case 0x62u:
            case 0x63u:
            case 0x64u:
              v81 = 8;
              v83 = 0;
              goto LABEL_81;
            case 0x59u:
            case 0x5Au:
            case 0x5Bu:
            case 0x5Cu:
            case 0x5Du:
            case 0x65u:
            case 0x66u:
            case 0x67u:
            case 0x68u:
            case 0x69u:
              v81 = 9;
              v83 = 0;
              goto LABEL_81;
            case 0x5Eu:
            case 0x5Fu:
            case 0x60u:
            case 0x61u:
            case 0x6Au:
            case 0x6Bu:
            case 0x6Cu:
            case 0x6Du:
              v81 = 10;
              v83 = 0;
              goto LABEL_81;
            default:
              goto LABEL_242;
          }
        }
        switch ( a10 )
        {
          case 0x18u:
          case 0x19u:
          case 0x1Au:
          case 0x1Bu:
          case 0x1Cu:
          case 0x1Du:
          case 0x1Eu:
          case 0x1Fu:
          case 0x20u:
          case 0x3Eu:
          case 0x3Fu:
          case 0x40u:
          case 0x41u:
          case 0x42u:
          case 0x43u:
            v160 = 3;
            break;
          case 0x21u:
          case 0x22u:
          case 0x23u:
          case 0x24u:
          case 0x25u:
          case 0x26u:
          case 0x27u:
          case 0x28u:
          case 0x44u:
          case 0x45u:
          case 0x46u:
          case 0x47u:
          case 0x48u:
          case 0x49u:
            v160 = 4;
            break;
          case 0x29u:
          case 0x2Au:
          case 0x2Bu:
          case 0x2Cu:
          case 0x2Du:
          case 0x2Eu:
          case 0x2Fu:
          case 0x30u:
          case 0x4Au:
          case 0x4Bu:
          case 0x4Cu:
          case 0x4Du:
          case 0x4Eu:
          case 0x4Fu:
            v160 = 5;
            break;
          case 0x31u:
          case 0x32u:
          case 0x33u:
          case 0x34u:
          case 0x35u:
          case 0x36u:
          case 0x50u:
          case 0x51u:
          case 0x52u:
          case 0x53u:
          case 0x54u:
          case 0x55u:
            v160 = 6;
            break;
          case 0x37u:
            v160 = 7;
            break;
          case 0x56u:
          case 0x57u:
          case 0x58u:
          case 0x62u:
          case 0x63u:
          case 0x64u:
            v160 = 8;
            break;
          case 0x59u:
          case 0x5Au:
          case 0x5Bu:
          case 0x5Cu:
          case 0x5Du:
          case 0x65u:
          case 0x66u:
          case 0x67u:
          case 0x68u:
          case 0x69u:
            v160 = 9;
            break;
          case 0x5Eu:
          case 0x5Fu:
          case 0x60u:
          case 0x61u:
          case 0x6Au:
          case 0x6Bu:
          case 0x6Cu:
          case 0x6Du:
            v160 = 10;
            break;
          default:
            v160 = 2;
            break;
        }
        v183 = v185;
        v184 = 0x1000000000LL;
        if ( v79 )
        {
          v145 = word_4307B00[(unsigned __int8)(v79 - 14)];
          if ( !word_4307B00[(unsigned __int8)(v79 - 14)] )
          {
LABEL_226:
            v129 = word_4307B00[(unsigned __int8)(v79 - 14)];
LABEL_227:
            v173 = word_4307B00[v68];
            if ( v173 == v129 )
            {
              v136 = v184;
            }
            else
            {
              do
              {
                LODWORD(v182) = 0;
                v181 = 0;
                v131 = sub_1D2B300((_QWORD *)a1, 0x30u, (__int64)&v181, v160, 0, v66);
                v133 = v132;
                if ( v181 )
                  sub_161E7C0((__int64)&v181, v181);
                v134 = (unsigned int)v184;
                if ( (unsigned int)v184 >= HIDWORD(v184) )
                {
                  sub_16CD150((__int64)&v183, v185, 0, 16, v130, v66);
                  v134 = (unsigned int)v184;
                }
                v135 = &v183[16 * v134];
                ++v129;
                *v135 = v131;
                v135[1] = v133;
                v136 = v184 + 1;
                LODWORD(v184) = v184 + 1;
              }
              while ( v129 != v173 );
            }
            *((_QWORD *)&v138 + 1) = v136;
            *(_QWORD *)&v138 = v183;
            v166 = (__int64)sub_1D359D0((__int64 *)a1, 104, a2, a10, 0, 0, *(double *)a7.m128i_i64, a8, a9, v138);
            v157 = v137;
            if ( v183 != v185 )
              _libc_free((unsigned __int64)v183);
            goto LABEL_64;
          }
        }
        else
        {
          v145 = sub_1F58D30((__int64)&v176);
          if ( !v145 )
            goto LABEL_236;
        }
        v118 = 0;
        v119 = v140;
        do
        {
          v151 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v15 + 48LL);
          v127 = sub_1E0A0C0(*(_QWORD *)(a1 + 32));
          if ( v151 == sub_1D13A20 )
          {
            v128 = 8 * sub_15A9520(v127, 0);
            if ( v128 == 32 )
            {
              v120 = 5;
            }
            else if ( v128 <= 0x20 )
            {
              v120 = 3;
              if ( v128 != 8 )
                v120 = 4 * (v128 == 16);
            }
            else
            {
              v120 = 6;
              if ( v128 != 64 )
              {
                v120 = 0;
                if ( v128 == 128 )
                  v120 = 7;
              }
            }
          }
          else
          {
            v120 = v151(v15, v127);
          }
          *(_QWORD *)&v121 = sub_1D38BB0(a1, v118, a2, v120, 0, 0, a7, a8, a9, 0);
          LOBYTE(v119) = v160;
          *((_QWORD *)&v171 + 1) = v161 | *((_QWORD *)&v171 + 1) & 0xFFFFFFFF00000000LL;
          v122 = sub_1D332F0(
                   (__int64 *)a1,
                   106,
                   a2,
                   v119,
                   0,
                   0,
                   *(double *)a7.m128i_i64,
                   a8,
                   a9,
                   v166,
                   *((unsigned __int64 *)&v171 + 1),
                   v121);
          v66 = v123;
          v124 = v122;
          v125 = (unsigned int)v184;
          if ( (unsigned int)v184 >= HIDWORD(v184) )
          {
            v152 = v122;
            v153 = v66;
            sub_16CD150((__int64)&v183, v185, 0, 16, (int)v122, v66);
            v125 = (unsigned int)v184;
            v124 = v152;
            v66 = v153;
          }
          v126 = (__int64 **)&v183[16 * v125];
          ++v118;
          *v126 = v124;
          v126[1] = (__int64 *)v66;
          LODWORD(v184) = v184 + 1;
        }
        while ( v118 != v145 );
        v79 = v176;
        v68 = a10 - 14;
        if ( (_BYTE)v176 )
          goto LABEL_226;
LABEL_236:
        v129 = sub_1F58D30((__int64)&v176);
        goto LABEL_227;
      }
    }
    switch ( v71 )
    {
      case 0xEu:
      case 0xFu:
      case 0x10u:
      case 0x11u:
      case 0x12u:
      case 0x13u:
      case 0x14u:
      case 0x15u:
      case 0x16u:
      case 0x17u:
      case 0x38u:
      case 0x39u:
      case 0x3Au:
      case 0x3Bu:
      case 0x3Cu:
      case 0x3Du:
        v110 = 2;
        break;
      case 0x18u:
      case 0x19u:
      case 0x1Au:
      case 0x1Bu:
      case 0x1Cu:
      case 0x1Du:
      case 0x1Eu:
      case 0x1Fu:
      case 0x20u:
      case 0x3Eu:
      case 0x3Fu:
      case 0x40u:
      case 0x41u:
      case 0x42u:
      case 0x43u:
        v110 = 3;
        break;
      case 0x21u:
      case 0x22u:
      case 0x23u:
      case 0x24u:
      case 0x25u:
      case 0x26u:
      case 0x27u:
      case 0x28u:
      case 0x44u:
      case 0x45u:
      case 0x46u:
      case 0x47u:
      case 0x48u:
      case 0x49u:
        v110 = 4;
        break;
      case 0x29u:
      case 0x2Au:
      case 0x2Bu:
      case 0x2Cu:
      case 0x2Du:
      case 0x2Eu:
      case 0x2Fu:
      case 0x30u:
      case 0x4Au:
      case 0x4Bu:
      case 0x4Cu:
      case 0x4Du:
      case 0x4Eu:
      case 0x4Fu:
        v110 = 5;
        break;
      case 0x31u:
      case 0x32u:
      case 0x33u:
      case 0x34u:
      case 0x35u:
      case 0x36u:
      case 0x50u:
      case 0x51u:
      case 0x52u:
      case 0x53u:
      case 0x54u:
      case 0x55u:
        v110 = 6;
        break;
      case 0x37u:
        v110 = 7;
        break;
      case 0x56u:
      case 0x57u:
      case 0x58u:
      case 0x62u:
      case 0x63u:
      case 0x64u:
        v110 = 8;
        break;
      case 0x59u:
      case 0x5Au:
      case 0x5Bu:
      case 0x5Cu:
      case 0x5Du:
      case 0x65u:
      case 0x66u:
      case 0x67u:
      case 0x68u:
      case 0x69u:
        v110 = 9;
        break;
      case 0x5Eu:
      case 0x5Fu:
      case 0x60u:
      case 0x61u:
      case 0x6Au:
      case 0x6Bu:
      case 0x6Cu:
      case 0x6Du:
        v110 = 10;
        break;
      default:
LABEL_242:
        BUG();
    }
    if ( v73 != v110 )
      goto LABEL_183;
    goto LABEL_142;
  }
  LOBYTE(v181) = 0;
  v182 = 0;
  v19 = *(_QWORD *)(a1 + 48);
  v20 = *((_BYTE *)a12 + 4) == 0;
  v174 = 0;
  if ( v20 )
  {
    v143 = sub_1F426C0(v15, v19, (unsigned int)v176, v17, (__int64)&v181, &v175, &v174);
    v23 = v176;
    if ( (_BYTE)v176 )
    {
LABEL_6:
      v144 = word_4307B00[(unsigned __int8)(v23 - 14)];
      goto LABEL_7;
    }
  }
  else
  {
    v21 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int, __int64, __int64, unsigned int *, char *))(*(_QWORD *)v15 + 312LL);
    if ( v21 == sub_1F42D90 )
      v22 = sub_1F426C0(v15, v19, v176, v177, (__int64)&v181, &v175, &v174);
    else
      v22 = v21(v15, v19, *a12, v176, v17, (__int64)&v181, &v175, &v174);
    v143 = v22;
    v23 = v176;
    if ( (_BYTE)v176 )
      goto LABEL_6;
  }
  v144 = sub_1F58D30((__int64)&v176);
LABEL_7:
  v24 = v181;
  v25 = v175;
  if ( (_BYTE)v181 )
  {
    if ( (unsigned __int8)(v181 - 14) <= 0x5Fu )
    {
      switch ( (char)v181 )
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
          v24 = 3;
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
          v24 = 4;
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
          v24 = 5;
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
          v24 = 6;
          break;
        case 55:
          v24 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v24 = 8;
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
          v24 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v24 = 10;
          break;
        default:
          v24 = 2;
          break;
      }
      v142 = 0;
      v25 = word_4307B00[(unsigned __int8)(v181 - 14)] * v175;
      goto LABEL_10;
    }
  }
  else
  {
    v163 = v175;
    v84 = sub_1F58D20((__int64)&v181);
    v25 = v163;
    if ( v84 )
    {
      v164 = sub_1F58D30((__int64)&v181) * v163;
      v85 = sub_1F596B0((__int64)&v181);
      v25 = v164;
      v142 = v86;
      v24 = v85;
      goto LABEL_10;
    }
  }
  v142 = v182;
LABEL_10:
  v148 = v25;
  v141 = *(_QWORD **)(a1 + 48);
  LOBYTE(v26) = sub_1D15020(v24, v25);
  v28 = 0;
  if ( !(_BYTE)v26 )
  {
    v26 = sub_1F593D0(v141, v24, (__int64)v142, v148);
    v13 = v26;
    v28 = v92;
  }
  LOBYTE(v13) = v26;
  v29 = *(_QWORD *)(v166 + 40) + v154;
  if ( *(_BYTE *)v29 != (_BYTE)v26 || *(const void ***)(v29 + 8) != v28 && !(_BYTE)v26 )
  {
    v61 = sub_1D309E0((__int64 *)a1, 158, a2, v13, v28, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, v171);
    v157 = v62;
    v166 = v61;
  }
  v30 = v175;
  v31 = v185;
  v183 = v185;
  v32 = v175;
  v184 = 0x800000000LL;
  if ( v175 > 8 )
  {
    v165 = v175;
    sub_16CD150((__int64)&v183, v185, v175, 16, v175, v27);
    v31 = v183;
    v32 = v165;
  }
  v33 = 16 * v30;
  LODWORD(v184) = v32;
  for ( i = &v31[v33]; i != v31; v31 += 16 )
  {
    if ( v31 )
    {
      *(_QWORD *)v31 = 0;
      *((_DWORD *)v31 + 2) = 0;
    }
  }
  v35 = v175;
  if ( v175 )
  {
    v36 = v157;
    LODWORD(v35) = 0;
    v158 = v15;
    LODWORD(v37) = v139;
    v162 = v36;
    while ( 1 )
    {
      if ( (_BYTE)v181 )
      {
        if ( (unsigned __int8)(v181 - 14) <= 0x5Fu )
          goto LABEL_24;
LABEL_32:
        v150 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v158 + 48LL);
        v48 = sub_1E0A0C0(*(_QWORD *)(a1 + 32));
        if ( v150 == sub_1D13A20 )
        {
          v49 = 8 * sub_15A9520(v48, 0);
          if ( v49 == 32 )
          {
            v50 = 5;
          }
          else if ( v49 > 0x20 )
          {
            v50 = 6;
            if ( v49 != 64 )
            {
              v50 = 0;
              if ( v49 == 128 )
                v50 = 7;
            }
          }
          else
          {
            v50 = 3;
            if ( v49 != 8 )
              v50 = 4 * (v49 == 16);
          }
        }
        else
        {
          v50 = v150(v158, v48);
        }
        LOBYTE(v37) = v50;
        v51 = (unsigned int)v37;
        v52 = (unsigned int)v35;
        v53 = (unsigned int)v35;
        v35 = (unsigned int)(v35 + 1);
        *(_QWORD *)&v54 = sub_1D38BB0(a1, v52, a2, v51, 0, 0, a7, a8, a9, 0);
        *((_QWORD *)&v171 + 1) = v162 | *((_QWORD *)&v171 + 1) & 0xFFFFFFFF00000000LL;
        v55 = sub_1D332F0(
                (__int64 *)a1,
                106,
                a2,
                (unsigned int)v181,
                v182,
                0,
                *(double *)a7.m128i_i64,
                a8,
                a9,
                v166,
                *((unsigned __int64 *)&v171 + 1),
                v54);
        v37 = &v183[16 * v53];
        *(_QWORD *)v37 = v55;
        *((_DWORD *)v37 + 2) = v56;
        if ( v175 == (_DWORD)v35 )
          break;
      }
      else
      {
        if ( !sub_1F58D20((__int64)&v181) )
          goto LABEL_32;
LABEL_24:
        v149 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v158 + 48LL);
        v38 = sub_1E0A0C0(*(_QWORD *)(a1 + 32));
        if ( v149 == sub_1D13A20 )
        {
          v39 = 8 * sub_15A9520(v38, 0);
          if ( v39 == 32 )
          {
            v40 = 5;
          }
          else if ( v39 > 0x20 )
          {
            v40 = 6;
            if ( v39 != 64 )
            {
              v40 = 0;
              if ( v39 == 128 )
                v40 = 7;
            }
          }
          else
          {
            v40 = 3;
            if ( v39 != 8 )
              v40 = 4 * (v39 == 16);
          }
        }
        else
        {
          v40 = v149(v158, v38);
        }
        LOBYTE(v12) = v40;
        *(_QWORD *)&v41 = sub_1D38BB0(a1, (unsigned int)v35 * (v144 / v175), a2, v12, 0, 0, a7, a8, a9, 0);
        v12 = DWORD2(v41);
        *((_QWORD *)&v171 + 1) = v162 | *((_QWORD *)&v171 + 1) & 0xFFFFFFFF00000000LL;
        v42 = sub_1D332F0(
                (__int64 *)a1,
                109,
                a2,
                (unsigned int)v181,
                v182,
                0,
                *(double *)a7.m128i_i64,
                a8,
                a9,
                v166,
                *((unsigned __int64 *)&v171 + 1),
                v41);
        v44 = v43;
        v45 = v42;
        v46 = (unsigned int)v35;
        v35 = (unsigned int)(v35 + 1);
        v47 = &v183[16 * v46];
        *(_QWORD *)v47 = v45;
        *((_DWORD *)v47 + 2) = v44;
        if ( v175 == (_DWORD)v35 )
          break;
      }
    }
  }
  if ( (_DWORD)v35 == v143 )
  {
    if ( (_DWORD)v35 )
    {
      v87 = 16 * v35;
      v88 = 0;
      do
      {
        v179 = *((_BYTE *)a12 + 4);
        if ( v179 )
          v178 = *a12;
        v89 = *(_QWORD *)&v183[v88 + 8];
        v90 = *(_QWORD *)&v183[v88];
        v91 = a5 + v88;
        v88 += 16;
        sub_204A2F0(a1, a2, v90, v89, v91, 1, a10, a11, (__int64)&v178, 144);
      }
      while ( v88 != v87 );
    }
  }
  else if ( v143 && v175 )
  {
    v57 = v143 / (unsigned int)v35;
    v58 = 0;
    v59 = 0;
    do
    {
      v179 = *((_BYTE *)a12 + 4);
      if ( v179 )
        v178 = *a12;
      v60 = &v183[16 * v58++];
      v172 = v57;
      sub_204A2F0(a1, a2, *v60, v60[1], a5 + 16 * v59, v57, a10, a11, (__int64)&v178, 144);
      v57 = v172;
      v59 += v172;
    }
    while ( v175 != v58 );
  }
  if ( v183 != v185 )
    _libc_free((unsigned __int64)v183);
}
