// Function: sub_20B91E0
// Address: 0x20b91e0
//
__int64 __fastcall sub_20B91E0(__int64 *a1, __int64 a2, __int64 a3, __m128i a4)
{
  __int64 v6; // rsi
  const void **v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  __m128i v10; // xmm1
  __int64 v11; // rdi
  __m128i v12; // xmm2
  __int64 v13; // rbx
  unsigned int v14; // r15d
  char v15; // bl
  __int64 v16; // rax
  unsigned __int8 v17; // cl
  const void **v18; // rax
  const void **v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 (__fastcall *v22)(__int64, __int64); // rbx
  __int64 v23; // rax
  unsigned int v24; // edx
  unsigned __int8 v25; // al
  char v26; // bl
  unsigned int v27; // r14d
  unsigned int v28; // eax
  __int64 v29; // rax
  unsigned int v30; // r15d
  __int64 v31; // rax
  unsigned int v32; // r14d
  __int128 v33; // rax
  int v34; // edx
  __int64 v35; // rdx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 *v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned int v46; // edx
  __int64 v47; // rdi
  __int16 v48; // cx
  unsigned int v49; // eax
  __int64 v50; // rcx
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rdi
  __int64 v53; // rbx
  char v54; // si
  __int64 v55; // rdx
  bool v56; // zf
  _QWORD *v57; // rdx
  __int64 v58; // r14
  unsigned int v60; // esi
  __int64 v61; // rax
  const void **v62; // r15
  __int64 v63; // rcx
  unsigned int v64; // eax
  __int64 v65; // rdi
  __int64 v66; // r14
  unsigned __int64 v67; // rdx
  int v68; // esi
  __int128 v69; // rax
  __int128 v70; // rax
  unsigned int v71; // edx
  __int128 v72; // rax
  __int128 v73; // rax
  __int128 v74; // rax
  unsigned __int64 v75; // rdx
  int v76; // ebx
  bool v77; // al
  char v78; // al
  const void **v79; // rdx
  const void **v80; // rdx
  __int64 v81; // rdi
  unsigned int v82; // ebx
  __int128 v83; // [rsp-20h] [rbp-270h]
  unsigned int v84; // [rsp-18h] [rbp-268h]
  __int128 v85; // [rsp-10h] [rbp-260h]
  unsigned int v86; // [rsp+8h] [rbp-248h]
  __int64 v87; // [rsp+20h] [rbp-230h]
  __int64 v88; // [rsp+30h] [rbp-220h]
  __int64 v89; // [rsp+38h] [rbp-218h]
  unsigned int v90; // [rsp+40h] [rbp-210h]
  __int64 v91; // [rsp+50h] [rbp-200h]
  __int64 v92; // [rsp+58h] [rbp-1F8h]
  unsigned int v93; // [rsp+60h] [rbp-1F0h]
  unsigned int v94; // [rsp+68h] [rbp-1E8h]
  __int64 v95; // [rsp+70h] [rbp-1E0h]
  const void **v96; // [rsp+78h] [rbp-1D8h]
  __int64 v97; // [rsp+80h] [rbp-1D0h]
  __int64 *v98; // [rsp+80h] [rbp-1D0h]
  __int64 v99; // [rsp+88h] [rbp-1C8h]
  __int64 v100; // [rsp+90h] [rbp-1C0h]
  __int64 *v101; // [rsp+90h] [rbp-1C0h]
  __int64 v102; // [rsp+98h] [rbp-1B8h]
  unsigned int v103; // [rsp+A0h] [rbp-1B0h]
  __int16 v104; // [rsp+A0h] [rbp-1B0h]
  __int64 v105; // [rsp+A0h] [rbp-1B0h]
  unsigned int v106; // [rsp+A0h] [rbp-1B0h]
  __int64 v107; // [rsp+A8h] [rbp-1A8h]
  __int64 v108; // [rsp+B0h] [rbp-1A0h]
  __int64 v109; // [rsp+B8h] [rbp-198h]
  unsigned int v110; // [rsp+C0h] [rbp-190h]
  __int64 v111; // [rsp+C0h] [rbp-190h]
  unsigned __int64 v112; // [rsp+C8h] [rbp-188h]
  __int64 v114; // [rsp+D8h] [rbp-178h]
  __int64 v115; // [rsp+D8h] [rbp-178h]
  unsigned int v116; // [rsp+D8h] [rbp-178h]
  __int64 v117; // [rsp+E0h] [rbp-170h]
  unsigned __int64 v118; // [rsp+E8h] [rbp-168h]
  unsigned __int64 v119; // [rsp+E8h] [rbp-168h]
  unsigned __int64 v120; // [rsp+F8h] [rbp-158h]
  __int64 v121; // [rsp+110h] [rbp-140h] BYREF
  int v122; // [rsp+118h] [rbp-138h]
  char v123[8]; // [rsp+120h] [rbp-130h] BYREF
  const void **v124; // [rsp+128h] [rbp-128h]
  unsigned __int8 v125[8]; // [rsp+130h] [rbp-120h] BYREF
  const void **v126; // [rsp+138h] [rbp-118h]
  __int64 v127; // [rsp+140h] [rbp-110h] BYREF
  const void **v128; // [rsp+148h] [rbp-108h]
  __m128i v129; // [rsp+150h] [rbp-100h] BYREF
  __int64 v130; // [rsp+160h] [rbp-F0h]
  __int128 v131; // [rsp+170h] [rbp-E0h]
  __int64 v132; // [rsp+180h] [rbp-D0h]
  __m128i v133; // [rsp+190h] [rbp-C0h] BYREF
  _QWORD v134[22]; // [rsp+1A0h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a2 + 72);
  v121 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v121, v6, 2);
  v7 = *(const void ***)(a2 + 96);
  v122 = *(_DWORD *)(a2 + 64);
  v8 = *(_QWORD *)(a2 + 32);
  v9 = *(_QWORD *)(v8 + 40);
  v10 = _mm_loadu_si128((const __m128i *)(v8 + 80));
  v11 = *(_QWORD *)(v8 + 80);
  v12 = _mm_loadu_si128((const __m128i *)(v8 + 40));
  v100 = *(_QWORD *)v8;
  v13 = *(_QWORD *)(v8 + 8);
  v14 = *(_DWORD *)(v8 + 88);
  v124 = v7;
  v95 = v9;
  v97 = v13;
  v15 = *(_BYTE *)(a2 + 88);
  v92 = *(unsigned int *)(v8 + 48);
  v16 = *(_QWORD *)(v9 + 40) + 16 * v92;
  v123[0] = v15;
  v17 = *(_BYTE *)v16;
  v18 = *(const void ***)(v16 + 8);
  v109 = v11;
  v125[0] = v17;
  v96 = v18;
  v126 = v18;
  v120 = v12.m128i_u64[1];
  if ( v17 )
  {
    if ( (unsigned __int8)(v17 - 14) <= 0x5Fu )
    {
      switch ( v17 )
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
          v17 = 3;
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
          v17 = 4;
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
          v17 = 5;
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
          v17 = 6;
          break;
        case 0x37u:
          v17 = 7;
          break;
        case 0x56u:
        case 0x57u:
        case 0x58u:
        case 0x62u:
        case 0x63u:
        case 0x64u:
          v17 = 8;
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
          v17 = 9;
          break;
        case 0x5Eu:
        case 0x5Fu:
        case 0x60u:
        case 0x61u:
        case 0x6Au:
        case 0x6Bu:
        case 0x6Cu:
        case 0x6Du:
          v17 = 10;
          break;
        default:
          v17 = 2;
          break;
      }
      v96 = 0;
    }
  }
  else
  {
    v77 = sub_1F58D20((__int64)v125);
    v17 = 0;
    if ( v77 )
    {
      v78 = sub_1F596B0((__int64)v125);
      v15 = v123[0];
      v96 = v79;
      v17 = v78;
    }
  }
  v94 = v17;
  if ( v15 )
  {
    if ( (unsigned __int8)(v15 - 14) > 0x5Fu )
    {
LABEL_7:
      v19 = v124;
      goto LABEL_8;
    }
    switch ( v15 )
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
        v15 = 3;
        v19 = 0;
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
        v15 = 4;
        v19 = 0;
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
        v15 = 5;
        v19 = 0;
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
        v15 = 6;
        v19 = 0;
        break;
      case 55:
        v15 = 7;
        v19 = 0;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v15 = 8;
        v19 = 0;
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
        v15 = 9;
        v19 = 0;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v15 = 10;
        v19 = 0;
        break;
      default:
        v15 = 2;
        v19 = 0;
        break;
    }
  }
  else
  {
    if ( !sub_1F58D20((__int64)v123) )
      goto LABEL_7;
    v15 = sub_1F596B0((__int64)v123);
  }
LABEL_8:
  v20 = *a1;
  v21 = *(_QWORD *)(a3 + 32);
  LOBYTE(v127) = v15;
  v128 = v19;
  v22 = *(__int64 (__fastcall **)(__int64, __int64))(v20 + 48);
  v23 = sub_1E0A0C0(v21);
  if ( v22 == sub_1D13A20 )
  {
    v24 = 8 * sub_15A9520(v23, 0);
    if ( v24 == 32 )
    {
      v25 = 5;
    }
    else if ( v24 > 0x20 )
    {
      v25 = 6;
      if ( v24 != 64 )
      {
        v25 = 0;
        if ( v24 == 128 )
          v25 = 7;
      }
    }
    else
    {
      v25 = 3;
      if ( v24 != 8 )
        v25 = 4 * (v24 == 16);
    }
  }
  else
  {
    v25 = v22((__int64)a1, v23);
  }
  v93 = v25;
  v26 = v123[0];
  if ( v123[0] )
    v27 = word_430A1A0[(unsigned __int8)(v123[0] - 14)];
  else
    v27 = sub_1F58D30((__int64)v123);
  if ( (_BYTE)v127 )
  {
    v28 = sub_1F3E310(&v127);
    v110 = v28 & 7;
    if ( (v28 & 7) == 0 )
    {
LABEL_17:
      v90 = v28 >> 3;
      v133.m128i_i64[0] = (__int64)v134;
      v133.m128i_i64[1] = 0x800000000LL;
      if ( v27 )
      {
        v29 = v14;
        v30 = v86;
        v88 = v29;
        v108 = 16 * v29;
        v31 = v27;
        v32 = v103;
        v91 = v31;
        v115 = 0;
        v89 = v97;
        v87 = v100;
        for ( *(_QWORD *)&v33 = sub_1D38BB0(a3, 0, (__int64)&v121, v93, 0, 0, a4, *(double *)v10.m128i_i64, v12, 0);
              ;
              *(_QWORD *)&v33 = sub_1D38BB0(a3, v115, (__int64)&v121, v93, 0, 0, a4, *(double *)v10.m128i_i64, v12, 0) )
        {
          v120 = v92 | v120 & 0xFFFFFFFF00000000LL;
          v98 = sub_1D332F0(
                  (__int64 *)a3,
                  106,
                  (__int64)&v121,
                  v94,
                  v96,
                  0,
                  *(double *)a4.m128i_i64,
                  *(double *)v10.m128i_i64,
                  v12,
                  v95,
                  v120,
                  v33);
          v99 = v41;
          v42 = *(_QWORD *)(v109 + 40) + v108;
          LOBYTE(v32) = *(_BYTE *)v42;
          v43 = sub_1D38BB0(
                  a3,
                  v110,
                  (__int64)&v121,
                  v32,
                  *(const void ***)(v42 + 8),
                  0,
                  a4,
                  *(double *)v10.m128i_i64,
                  v12,
                  0);
          v44 = *(_QWORD *)(v109 + 40) + v108;
          LOBYTE(v30) = *(_BYTE *)v44;
          v118 = v88 | v118 & 0xFFFFFFFF00000000LL;
          *((_QWORD *)&v83 + 1) = v45;
          *(_QWORD *)&v83 = v43;
          v101 = sub_1D332F0(
                   (__int64 *)a3,
                   52,
                   (__int64)&v121,
                   v30,
                   *(const void ***)(v44 + 8),
                   3u,
                   *(double *)a4.m128i_i64,
                   *(double *)v10.m128i_i64,
                   v12,
                   v109,
                   v118,
                   v83);
          v102 = v46;
          v47 = *(_QWORD *)(a2 + 104);
          a4 = _mm_loadu_si128((const __m128i *)(v47 + 40));
          v130 = *(_QWORD *)(v47 + 56);
          v48 = *(_WORD *)(v47 + 32);
          v129 = a4;
          v104 = v48;
          v49 = sub_1E34390(v47);
          v50 = *(_QWORD *)(a2 + 104);
          v51 = -(__int64)(v110 | (unsigned __int64)v49) & (v110 | (unsigned __int64)v49);
          v52 = *(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v52 )
          {
            v53 = *(_QWORD *)(v50 + 8) + v110;
            v54 = *(_BYTE *)(v50 + 16);
            if ( (*(_QWORD *)v50 & 4) != 0 )
            {
              *((_QWORD *)&v131 + 1) = *(_QWORD *)(v50 + 8) + v110;
              LOBYTE(v132) = v54;
              *(_QWORD *)&v131 = v52 | 4;
              HIDWORD(v132) = *(_DWORD *)(v52 + 12);
            }
            else
            {
              v55 = *(_QWORD *)v52;
              *(_QWORD *)&v131 = *(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL;
              *((_QWORD *)&v131 + 1) = v53;
              v56 = *(_BYTE *)(v55 + 8) == 16;
              LOBYTE(v132) = v54;
              if ( v56 )
                v55 = **(_QWORD **)(v55 + 16);
              HIDWORD(v132) = *(_DWORD *)(v55 + 8) >> 8;
            }
          }
          else
          {
            v132 = 0;
            v34 = *(_DWORD *)(v50 + 20);
            v131 = 0u;
            HIDWORD(v132) = v34;
          }
          v36 = sub_1D2C750(
                  (_QWORD *)a3,
                  v87,
                  v89,
                  (__int64)&v121,
                  (__int64)v98,
                  v99,
                  (__int64)v101,
                  v102,
                  v131,
                  v132,
                  v127,
                  (__int64)v128,
                  v51,
                  v104,
                  (__int64)&v129);
          v37 = v35;
          v38 = v133.m128i_u32[2];
          if ( v133.m128i_i32[2] >= (unsigned __int32)v133.m128i_i32[3] )
          {
            v107 = v35;
            v105 = v36;
            sub_16CD150((__int64)&v133, v134, 0, 16, v36, v35);
            v38 = v133.m128i_u32[2];
            v36 = v105;
            v37 = v107;
          }
          v39 = (__int64 *)(v133.m128i_i64[0] + 16 * v38);
          ++v115;
          *v39 = v36;
          v39[1] = v37;
          v110 += v90;
          v40 = (unsigned int)++v133.m128i_i32[2];
          if ( v91 == v115 )
            break;
        }
        v57 = (_QWORD *)v133.m128i_i64[0];
      }
      else
      {
        v57 = v134;
        v40 = 0;
      }
      *((_QWORD *)&v85 + 1) = v40;
      *(_QWORD *)&v85 = v57;
      v58 = (__int64)sub_1D359D0(
                       (__int64 *)a3,
                       2,
                       (__int64)&v121,
                       1,
                       0,
                       0,
                       *(double *)a4.m128i_i64,
                       *(double *)v10.m128i_i64,
                       v12,
                       v85);
      if ( (_QWORD *)v133.m128i_i64[0] != v134 )
        _libc_free(v133.m128i_u64[0]);
      goto LABEL_37;
    }
  }
  else
  {
    v28 = sub_1F58D40((__int64)&v127);
    v110 = v28 & 7;
    if ( (v28 & 7) == 0 )
      goto LABEL_17;
  }
  if ( v26 )
    v60 = sub_1F3E310(v123);
  else
    v60 = sub_1F58D40((__int64)v123);
  if ( v60 == 32 )
  {
    LOBYTE(v61) = 5;
  }
  else if ( v60 > 0x20 )
  {
    if ( v60 == 64 )
    {
      LOBYTE(v61) = 6;
    }
    else
    {
      if ( v60 != 128 )
      {
LABEL_66:
        v61 = sub_1F58CC0(*(_QWORD **)(a3 + 48), v60);
        v114 = v61;
        v62 = v80;
        goto LABEL_49;
      }
      LOBYTE(v61) = 7;
    }
  }
  else if ( v60 == 8 )
  {
    LOBYTE(v61) = 3;
  }
  else
  {
    LOBYTE(v61) = 4;
    if ( v60 != 16 )
    {
      LOBYTE(v61) = 2;
      if ( v60 != 1 )
        goto LABEL_66;
    }
  }
  v62 = 0;
LABEL_49:
  v63 = v114;
  LOBYTE(v63) = v61;
  v116 = v63;
  v117 = sub_1D38BB0(a3, 0, (__int64)&v121, v63, v62, 0, a4, *(double *)v10.m128i_i64, v12, 0);
  v64 = v27;
  v106 = v27 - 1;
  v65 = v27;
  v66 = 0;
  v119 = v67;
  if ( v64 )
  {
    do
    {
      *(_QWORD *)&v72 = sub_1D38BB0(a3, v66, (__int64)&v121, v93, 0, 0, a4, *(double *)v10.m128i_i64, v12, 0);
      v120 = v92 | v120 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v73 = sub_1D332F0(
                          (__int64 *)a3,
                          106,
                          (__int64)&v121,
                          v94,
                          v96,
                          0,
                          *(double *)a4.m128i_i64,
                          *(double *)v10.m128i_i64,
                          v12,
                          v95,
                          v120,
                          v72);
      *(_QWORD *)&v74 = sub_1D309E0(
                          (__int64 *)a3,
                          145,
                          (__int64)&v121,
                          (unsigned int)v127,
                          v128,
                          0,
                          *(double *)a4.m128i_i64,
                          *(double *)v10.m128i_i64,
                          *(double *)v12.m128i_i64,
                          v73);
      v111 = sub_1D309E0(
               (__int64 *)a3,
               143,
               (__int64)&v121,
               v116,
               v62,
               0,
               *(double *)a4.m128i_i64,
               *(double *)v10.m128i_i64,
               *(double *)v12.m128i_i64,
               v74);
      v112 = v75;
      v76 = v106 - v66;
      if ( !*(_BYTE *)sub_1E0A0C0(*(_QWORD *)(a3 + 32)) )
        v76 = v66;
      if ( (_BYTE)v127 )
        v68 = sub_1F3E310(&v127);
      else
        v68 = sub_1F58D40((__int64)&v127);
      ++v66;
      *(_QWORD *)&v69 = sub_1D38BB0(
                          a3,
                          (unsigned int)(v76 * v68),
                          (__int64)&v121,
                          v116,
                          v62,
                          0,
                          a4,
                          *(double *)v10.m128i_i64,
                          v12,
                          0);
      *(_QWORD *)&v70 = sub_1D332F0(
                          (__int64 *)a3,
                          122,
                          (__int64)&v121,
                          v116,
                          v62,
                          0,
                          *(double *)a4.m128i_i64,
                          *(double *)v10.m128i_i64,
                          v12,
                          v111,
                          v112,
                          v69);
      v117 = (__int64)sub_1D332F0(
                        (__int64 *)a3,
                        119,
                        (__int64)&v121,
                        v116,
                        v62,
                        0,
                        *(double *)a4.m128i_i64,
                        *(double *)v10.m128i_i64,
                        v12,
                        v117,
                        v119,
                        v70);
      v119 = v71 | v119 & 0xFFFFFFFF00000000LL;
    }
    while ( v65 != v66 );
  }
  v81 = *(_QWORD *)(a2 + 104);
  v133 = _mm_loadu_si128((const __m128i *)(v81 + 40));
  v134[0] = *(_QWORD *)(v81 + 56);
  v82 = *(unsigned __int16 *)(v81 + 32);
  v84 = sub_1E34390(v81);
  v58 = sub_1D2BF40(
          (_QWORD *)a3,
          v100,
          v97,
          (__int64)&v121,
          v117,
          v119,
          v10.m128i_i64[0],
          v10.m128i_i64[1],
          *(_OWORD *)*(_QWORD *)(a2 + 104),
          *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
          v84,
          v82,
          (__int64)&v133);
LABEL_37:
  if ( v121 )
    sub_161E7C0((__int64)&v121, v121);
  return v58;
}
