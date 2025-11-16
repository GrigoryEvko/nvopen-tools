// Function: sub_2173C80
// Address: 0x2173c80
//
__int64 __fastcall sub_2173C80(__int64 a1, __int64 *a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  char v7; // cl
  unsigned int v8; // eax
  __m128i *v9; // rcx
  bool v10; // cc
  const void **v11; // rax
  unsigned __int8 v12; // di
  __m128i v13; // xmm1
  int v14; // eax
  __int64 v15; // rsi
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // r14
  __int64 v19; // r13
  const __m128i *v20; // r12
  __m128i v21; // xmm0
  char v22; // r14
  __m128i v23; // xmm2
  __int64 v24; // r12
  __int64 v25; // rsi
  __int128 v26; // rax
  __int64 v27; // r14
  __int64 v28; // rax
  const void **v29; // r8
  __int64 v30; // rsi
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int8 v33; // r14
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 *v38; // rdx
  __int64 v39; // rsi
  unsigned int v40; // eax
  __int64 v41; // rsi
  __int64 v42; // rdi
  unsigned int v43; // eax
  __int64 v44; // rax
  __int64 v45; // r8
  int v46; // r9d
  __int64 v47; // rcx
  const __m128i *v48; // rax
  __int32 v49; // edx
  __int64 v50; // rdx
  __m128i v51; // xmm3
  __m128i *v52; // rdx
  unsigned int v53; // eax
  unsigned __int8 v54; // r14
  __int64 v55; // r15
  __int64 v56; // r13
  __int64 *v57; // r12
  __int64 v58; // rax
  int v59; // edx
  __int64 v60; // r9
  __int64 v61; // rcx
  int v62; // r8d
  __int64 v63; // r12
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rdx
  const void **v69; // rdx
  bool v70; // al
  char v71; // al
  __int64 v72; // r13
  __int64 v73; // r12
  __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 *v76; // rdx
  char v77; // al
  __int64 v78; // rdx
  __int64 v79; // rsi
  __int64 v80; // rdx
  __int64 v81; // [rsp-10h] [rbp-1B0h]
  __int128 v82; // [rsp-10h] [rbp-1B0h]
  __int128 v83; // [rsp-10h] [rbp-1B0h]
  int v84; // [rsp-8h] [rbp-1A8h]
  unsigned __int16 v85; // [rsp+4h] [rbp-19Ch]
  __m128i v86; // [rsp+8h] [rbp-198h]
  __m128i *v87; // [rsp+20h] [rbp-180h]
  const void **v88; // [rsp+28h] [rbp-178h]
  __int64 v90; // [rsp+38h] [rbp-168h]
  __int64 v91; // [rsp+40h] [rbp-160h]
  const void **v92; // [rsp+40h] [rbp-160h]
  __int64 v93; // [rsp+50h] [rbp-150h]
  __int128 v94; // [rsp+50h] [rbp-150h]
  __int64 v95; // [rsp+50h] [rbp-150h]
  __int64 v96; // [rsp+50h] [rbp-150h]
  __int64 v97; // [rsp+50h] [rbp-150h]
  __int64 *v98; // [rsp+50h] [rbp-150h]
  __int64 v99; // [rsp+58h] [rbp-148h]
  __int64 v100; // [rsp+58h] [rbp-148h]
  __int64 v101; // [rsp+58h] [rbp-148h]
  __int64 v102; // [rsp+58h] [rbp-148h]
  char v103; // [rsp+68h] [rbp-138h]
  int v104; // [rsp+68h] [rbp-138h]
  __int64 v105; // [rsp+70h] [rbp-130h]
  unsigned int v106; // [rsp+78h] [rbp-128h]
  __int64 v107; // [rsp+78h] [rbp-128h]
  __m128i v108; // [rsp+80h] [rbp-120h] BYREF
  __int64 v109; // [rsp+90h] [rbp-110h] BYREF
  int v110; // [rsp+98h] [rbp-108h]
  __int64 v111; // [rsp+A0h] [rbp-100h] BYREF
  int v112; // [rsp+A8h] [rbp-F8h]
  __m128i v113; // [rsp+B0h] [rbp-F0h] BYREF
  __m128i v114; // [rsp+C0h] [rbp-E0h] BYREF
  __m128i v115; // [rsp+D0h] [rbp-D0h] BYREF
  __m128i v116; // [rsp+E0h] [rbp-C0h] BYREF
  _OWORD v117[11]; // [rsp+F0h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 120LL) + 40LL)
     + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 32) + 128LL);
  v7 = *(_BYTE *)v6;
  v103 = v7;
  v91 = *(_QWORD *)(v6 + 8);
  v108.m128i_i8[0] = *(_BYTE *)v6;
  v108.m128i_i64[1] = v91;
  if ( v7 )
  {
    if ( (unsigned __int8)(v7 - 14) <= 0x5Fu )
    {
      v91 = 0;
      switch ( v7 )
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
          v103 = 3;
          v8 = sub_216FFF0(3);
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
          v103 = 4;
          v8 = sub_216FFF0(4);
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
          v103 = 5;
          v8 = sub_216FFF0(5);
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
          v103 = 6;
          v8 = sub_216FFF0(6);
          break;
        case 55:
          v103 = 7;
          v8 = sub_216FFF0(7);
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v103 = 8;
          v8 = sub_216FFF0(8);
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
          v103 = 9;
          v8 = sub_216FFF0(9);
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v103 = 10;
          v8 = sub_216FFF0(10);
          break;
        default:
          v103 = 2;
          v8 = sub_216FFF0(2);
          break;
      }
    }
    else
    {
      v8 = sub_216FFF0(v7);
    }
    goto LABEL_4;
  }
  if ( sub_1F58D20((__int64)&v108) )
  {
    v77 = sub_1F596B0((__int64)&v108);
    v91 = v78;
    v116.m128i_i8[0] = v77;
    v116.m128i_i64[1] = v78;
    if ( v77 )
    {
      v103 = v77;
      v8 = sub_216FFF0(v77);
LABEL_4:
      v87 = v9;
      goto LABEL_5;
    }
  }
  else
  {
    v116.m128i_i8[0] = 0;
    v116.m128i_i64[1] = v91;
  }
  v87 = &v116;
  v8 = sub_1F58D40((__int64)&v116);
LABEL_5:
  v10 = v8 <= 0xF;
  v11 = 0;
  if ( !v10 )
    v11 = (const void **)v91;
  v12 = v103;
  v13 = _mm_load_si128(&v108);
  v92 = v11;
  if ( v10 )
    v12 = 4;
  v116 = v13;
  v90 = v12;
  if ( v108.m128i_i8[0] )
  {
    if ( (unsigned __int8)(v108.m128i_i8[0] - 14) <= 0x5Fu )
    {
      v14 = word_432BB60[(unsigned __int8)(v108.m128i_i8[0] - 14)];
LABEL_12:
      v85 = (v14 != 2) + 682;
      goto LABEL_13;
    }
  }
  else if ( sub_1F58D20((__int64)v87) )
  {
    v14 = sub_1F58D30((__int64)v87);
    goto LABEL_12;
  }
  v85 = 681;
LABEL_13:
  v15 = *(_QWORD *)(a1 + 72);
  v113.m128i_i64[0] = v15;
  if ( v15 )
    sub_1623A60((__int64)&v113, v15, 2);
  v113.m128i_i32[2] = *(_DWORD *)(a1 + 64);
  sub_2170130(
    (__int64)v87,
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 80LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 88LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 200LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 208LL),
    a3,
    *(double *)v13.m128i_i64,
    a5,
    (__int64)&v113,
    a2);
  v18 = v116.m128i_i64[0];
  v86.m128i_i64[0] = *(_QWORD *)&v117[0];
  v19 = v116.m128i_u32[2];
  v86.m128i_i64[1] = DWORD2(v117[0]);
  if ( v113.m128i_i64[0] )
    sub_161E7C0((__int64)&v113, v113.m128i_i64[0]);
  v20 = *(const __m128i **)(a1 + 32);
  v21 = _mm_loadu_si128(v20);
  v114.m128i_i64[0] = v18;
  v114.m128i_i64[1] = v19;
  v22 = v108.m128i_i8[0];
  v23 = _mm_load_si128(&v114);
  v116.m128i_i64[0] = (__int64)v117;
  v116.m128i_i64[1] = 0x800000002LL;
  v113 = v21;
  v117[0] = v21;
  v117[1] = v23;
  if ( v108.m128i_i8[0] )
  {
    if ( (unsigned __int8)(v108.m128i_i8[0] - 14) <= 0x5Fu )
      goto LABEL_19;
    v72 = v20[8].m128i_i64[0];
    v73 = v20[7].m128i_i64[1];
    goto LABEL_80;
  }
  if ( !sub_1F58D20((__int64)&v108) )
  {
    v72 = v20[8].m128i_i64[0];
    v73 = v20[7].m128i_i64[1];
    if ( sub_1F58D20((__int64)&v108) )
    {
      v22 = sub_1F596B0((__int64)&v108);
LABEL_81:
      if ( v12 != v22 || !v12 && v92 != (const void **)v74 )
      {
        v79 = *(_QWORD *)(a1 + 72);
        v113.m128i_i64[0] = v79;
        if ( v79 )
          sub_1623A60((__int64)&v113, v79, 2);
        *((_QWORD *)&v83 + 1) = v72;
        *(_QWORD *)&v83 = v73;
        v113.m128i_i32[2] = *(_DWORD *)(a1 + 64);
        v73 = sub_1D309E0(
                a2,
                144,
                (__int64)&v113,
                v12,
                v92,
                0,
                *(double *)v21.m128i_i64,
                *(double *)v13.m128i_i64,
                *(double *)v23.m128i_i64,
                v83);
        v72 = v80;
        v16 = v84;
        if ( v113.m128i_i64[0] )
          sub_161E7C0((__int64)&v113, v113.m128i_i64[0]);
      }
      v75 = v116.m128i_u32[2];
      if ( v116.m128i_i32[2] >= (unsigned __int32)v116.m128i_i32[3] )
      {
        sub_16CD150((__int64)v87, v117, 0, 16, v16, v17);
        v75 = v116.m128i_u32[2];
      }
      v76 = (__int64 *)(v116.m128i_i64[0] + 16 * v75);
      *v76 = v73;
      v76[1] = v72;
      ++v116.m128i_i32[2];
      goto LABEL_46;
    }
LABEL_80:
    v74 = v108.m128i_i64[1];
    goto LABEL_81;
  }
LABEL_19:
  v106 = 0;
  v24 = v93;
  if ( !v22 )
    goto LABEL_45;
LABEL_20:
  v25 = *(_QWORD *)(a1 + 72);
  if ( v106 < word_432BB60[(unsigned __int8)(v22 - 14)] )
  {
    do
    {
      v111 = v25;
      if ( v25 )
        sub_1623A60((__int64)&v111, v25, 2);
      v112 = *(_DWORD *)(a1 + 64);
      *(_QWORD *)&v26 = sub_1D38E70((__int64)a2, v106, (__int64)&v111, 0, v21, *(double *)v13.m128i_i64, v23);
      v27 = *(_QWORD *)(a1 + 32);
      v94 = v26;
      if ( v108.m128i_i8[0] )
      {
        switch ( v108.m128i_i8[0] )
        {
          case 0xE:
          case 0xF:
          case 0x10:
          case 0x11:
          case 0x12:
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
          case 0x17:
          case 0x38:
          case 0x39:
          case 0x3A:
          case 0x3B:
          case 0x3C:
          case 0x3D:
            LOBYTE(v28) = 2;
            break;
          case 0x18:
          case 0x19:
          case 0x1A:
          case 0x1B:
          case 0x1C:
          case 0x1D:
          case 0x1E:
          case 0x1F:
          case 0x20:
          case 0x3E:
          case 0x3F:
          case 0x40:
          case 0x41:
          case 0x42:
          case 0x43:
            LOBYTE(v28) = 3;
            break;
          case 0x21:
          case 0x22:
          case 0x23:
          case 0x24:
          case 0x25:
          case 0x26:
          case 0x27:
          case 0x28:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x47:
          case 0x48:
          case 0x49:
            LOBYTE(v28) = 4;
            break;
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x2C:
          case 0x2D:
          case 0x2E:
          case 0x2F:
          case 0x30:
          case 0x4A:
          case 0x4B:
          case 0x4C:
          case 0x4D:
          case 0x4E:
          case 0x4F:
            LOBYTE(v28) = 5;
            break;
          case 0x31:
          case 0x32:
          case 0x33:
          case 0x34:
          case 0x35:
          case 0x36:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x53:
          case 0x54:
          case 0x55:
            LOBYTE(v28) = 6;
            break;
          case 0x37:
            LOBYTE(v28) = 7;
            break;
          case 0x56:
          case 0x57:
          case 0x58:
          case 0x62:
          case 0x63:
          case 0x64:
            LOBYTE(v28) = 8;
            break;
          case 0x59:
          case 0x5A:
          case 0x5B:
          case 0x5C:
          case 0x5D:
          case 0x65:
          case 0x66:
          case 0x67:
          case 0x68:
          case 0x69:
            LOBYTE(v28) = 9;
            break;
          case 0x5E:
          case 0x5F:
          case 0x60:
          case 0x61:
          case 0x6A:
          case 0x6B:
          case 0x6C:
          case 0x6D:
            LOBYTE(v28) = 10;
            break;
          default:
            ++*(_DWORD *)(v24 + 16);
            BUG();
        }
        v29 = 0;
      }
      else
      {
        LOBYTE(v28) = sub_1F596B0((__int64)&v108);
        v24 = v28;
        v29 = v69;
      }
      v30 = *(_QWORD *)(a1 + 72);
      LOBYTE(v24) = v28;
      v109 = v30;
      if ( v30 )
      {
        v88 = v29;
        sub_1623A60((__int64)&v109, v30, 2);
        v29 = v88;
      }
      v110 = *(_DWORD *)(a1 + 64);
      v31 = sub_1D332F0(
              a2,
              106,
              (__int64)&v109,
              (unsigned int)v24,
              v29,
              0,
              *(double *)v21.m128i_i64,
              *(double *)v13.m128i_i64,
              v23,
              *(_QWORD *)(v27 + 120),
              *(_QWORD *)(v27 + 128),
              v94);
      v33 = v108.m128i_i8[0];
      v34 = (__int64)v31;
      v35 = v32;
      if ( v108.m128i_i8[0] )
      {
        if ( (unsigned __int8)(v108.m128i_i8[0] - 14) > 0x5Fu )
          goto LABEL_36;
        switch ( v108.m128i_i8[0] )
        {
          case 0x18:
          case 0x19:
          case 0x1A:
          case 0x1B:
          case 0x1C:
          case 0x1D:
          case 0x1E:
          case 0x1F:
          case 0x20:
          case 0x3E:
          case 0x3F:
          case 0x40:
          case 0x41:
          case 0x42:
          case 0x43:
            v33 = 3;
            v36 = 0;
            break;
          case 0x21:
          case 0x22:
          case 0x23:
          case 0x24:
          case 0x25:
          case 0x26:
          case 0x27:
          case 0x28:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x47:
          case 0x48:
          case 0x49:
            v33 = 4;
            v36 = 0;
            break;
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x2C:
          case 0x2D:
          case 0x2E:
          case 0x2F:
          case 0x30:
          case 0x4A:
          case 0x4B:
          case 0x4C:
          case 0x4D:
          case 0x4E:
          case 0x4F:
            v33 = 5;
            v36 = 0;
            break;
          case 0x31:
          case 0x32:
          case 0x33:
          case 0x34:
          case 0x35:
          case 0x36:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x53:
          case 0x54:
          case 0x55:
            v33 = 6;
            v36 = 0;
            break;
          case 0x37:
            v33 = 7;
            v36 = 0;
            break;
          case 0x56:
          case 0x57:
          case 0x58:
          case 0x62:
          case 0x63:
          case 0x64:
            v33 = 8;
            v36 = 0;
            break;
          case 0x59:
          case 0x5A:
          case 0x5B:
          case 0x5C:
          case 0x5D:
          case 0x65:
          case 0x66:
          case 0x67:
          case 0x68:
          case 0x69:
            v33 = 9;
            v36 = 0;
            break;
          case 0x5E:
          case 0x5F:
          case 0x60:
          case 0x61:
          case 0x6A:
          case 0x6B:
          case 0x6C:
          case 0x6D:
            v33 = 10;
            v36 = 0;
            break;
          default:
            v33 = 2;
            v36 = 0;
            break;
        }
      }
      else
      {
        v98 = v31;
        v102 = v32;
        v70 = sub_1F58D20((__int64)&v108);
        v34 = (__int64)v98;
        v35 = v102;
        if ( !v70 )
        {
LABEL_36:
          v36 = v108.m128i_i64[1];
          goto LABEL_37;
        }
        v71 = sub_1F596B0((__int64)&v108);
        v34 = (__int64)v98;
        v35 = v102;
        v33 = v71;
      }
LABEL_37:
      if ( v12 == v33 && (v12 || v92 == (const void **)v36) )
        goto LABEL_39;
      v65 = *(_QWORD *)(a1 + 72);
      v113.m128i_i64[0] = v65;
      if ( v65 )
      {
        v95 = v34;
        v99 = v35;
        sub_1623A60((__int64)&v113, v65, 2);
        v34 = v95;
        v35 = v99;
      }
      *((_QWORD *)&v82 + 1) = v35;
      *(_QWORD *)&v82 = v34;
      v113.m128i_i32[2] = *(_DWORD *)(a1 + 64);
      v66 = v90;
      LOBYTE(v66) = v12;
      v90 = v66;
      v67 = sub_1D309E0(
              a2,
              144,
              (__int64)&v113,
              (unsigned int)v66,
              v92,
              0,
              *(double *)v21.m128i_i64,
              *(double *)v13.m128i_i64,
              *(double *)v23.m128i_i64,
              v82);
      v34 = v67;
      v35 = v68;
      if ( !v113.m128i_i64[0] )
      {
LABEL_39:
        v37 = v116.m128i_u32[2];
        if ( v116.m128i_i32[2] < (unsigned __int32)v116.m128i_i32[3] )
          goto LABEL_40;
      }
      else
      {
        v100 = v68;
        v96 = v67;
        sub_161E7C0((__int64)&v113, v113.m128i_i64[0]);
        v34 = v96;
        v35 = v100;
        v37 = v116.m128i_u32[2];
        if ( v116.m128i_i32[2] < (unsigned __int32)v116.m128i_i32[3] )
          goto LABEL_40;
      }
      v97 = v34;
      v101 = v35;
      sub_16CD150((__int64)v87, v117, 0, 16, v34, v35);
      v37 = v116.m128i_u32[2];
      v34 = v97;
      v35 = v101;
LABEL_40:
      v38 = (__int64 *)(v116.m128i_i64[0] + 16 * v37);
      *v38 = v34;
      v39 = v109;
      v38[1] = v35;
      ++v116.m128i_i32[2];
      if ( v39 )
        sub_161E7C0((__int64)&v109, v39);
      if ( v111 )
        sub_161E7C0((__int64)&v111, v111);
      v22 = v108.m128i_i8[0];
      ++v106;
      if ( v108.m128i_i8[0] )
        goto LABEL_20;
LABEL_45:
      v40 = sub_1F58D30((__int64)&v108);
      v25 = *(_QWORD *)(a1 + 72);
    }
    while ( v106 < v40 );
  }
LABEL_46:
  v41 = *(_QWORD *)(a1 + 72);
  v111 = v41;
  if ( v41 )
    sub_1623A60((__int64)&v111, v41, 2);
  v42 = *(_QWORD *)(a1 + 104);
  v112 = *(_DWORD *)(a1 + 64);
  v43 = sub_1E340A0(v42);
  v44 = sub_1D38BB0((__int64)a2, v43, (__int64)&v111, 3, 0, 1, v21, *(double *)v13.m128i_i64, v23, 0);
  v47 = v81;
  v113.m128i_i64[0] = v44;
  v48 = *(const __m128i **)(a1 + 32);
  v113.m128i_i32[2] = v49;
  v50 = v116.m128i_u32[2];
  v51 = _mm_loadu_si128(v48 + 10);
  v115 = v86;
  v114 = v51;
  if ( v116.m128i_u32[3] - (unsigned __int64)v116.m128i_u32[2] <= 2 )
  {
    sub_16CD150((__int64)v87, v117, v116.m128i_u32[2] + 3LL, 16, v45, v46);
    v50 = v116.m128i_u32[2];
  }
  v52 = (__m128i *)(v116.m128i_i64[0] + 16 * v50);
  *v52 = _mm_load_si128(&v113);
  v52[1] = _mm_load_si128(&v114);
  v52[2] = _mm_load_si128(&v115);
  v53 = v116.m128i_i32[2] + 3;
  v116.m128i_i32[2] += 3;
  if ( v111 )
  {
    sub_161E7C0((__int64)&v111, v111);
    v53 = v116.m128i_u32[2];
  }
  v54 = *(_BYTE *)(a1 + 88);
  v55 = *(_QWORD *)(a1 + 96);
  v56 = v53;
  v107 = *(_QWORD *)(a1 + 104);
  v57 = (__int64 *)v116.m128i_i64[0];
  v58 = sub_1D29190((__int64)a2, 1u, 0, v47, v45, v107);
  v60 = v107;
  v61 = v58;
  v62 = v59;
  v113.m128i_i64[0] = *(_QWORD *)(a1 + 72);
  if ( v113.m128i_i64[0] )
  {
    v104 = v59;
    v105 = v58;
    sub_1623A60((__int64)&v113, v113.m128i_i64[0], 2);
    v62 = v104;
    v61 = v105;
    v60 = v107;
  }
  v113.m128i_i32[2] = *(_DWORD *)(a1 + 64);
  v63 = sub_1D24DC0(a2, v85, (__int64)&v113, v61, v62, v60, v57, v56, v54, v55);
  if ( v113.m128i_i64[0] )
    sub_161E7C0((__int64)&v113, v113.m128i_i64[0]);
  if ( (_OWORD *)v116.m128i_i64[0] != v117 )
    _libc_free(v116.m128i_u64[0]);
  return v63;
}
