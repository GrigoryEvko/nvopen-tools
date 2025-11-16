// Function: sub_1F40D10
// Address: 0x1f40d10
//
__int64 __fastcall sub_1F40D10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  unsigned __int8 v7; // dl
  char v8; // al
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v14; // eax
  unsigned int v15; // eax
  char v16; // al
  __int64 v17; // rdx
  unsigned int v18; // r13d
  unsigned int v19; // edx
  unsigned int v20; // edx
  unsigned int v21; // ecx
  unsigned __int8 v22; // al
  __int64 v23; // rdx
  __int64 v24; // r14
  __int64 v25; // rbx
  unsigned int v26; // ebx
  char v27; // al
  __int64 v28; // rdx
  unsigned __int64 v29; // r14
  __int8 v30; // al
  char v31; // r15
  __int64 v32; // rdx
  char v33; // al
  int v34; // eax
  __int64 v35; // r13
  unsigned __int64 v36; // rcx
  unsigned __int32 v37; // r14d
  unsigned int v38; // ebx
  char v39; // al
  __int64 v40; // rdx
  unsigned __int8 v41; // al
  __int64 v42; // rsi
  unsigned int v43; // eax
  unsigned int v44; // ecx
  __int64 v45; // rax
  __m128i v46; // xmm1
  unsigned __int8 v47; // si
  unsigned int v48; // eax
  unsigned int v49; // ebx
  __int64 v50; // r14
  char v51; // di
  unsigned int v52; // r13d
  unsigned int v53; // eax
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int8 i; // al
  int v59; // eax
  __int64 v60; // rsi
  unsigned int v61; // eax
  char v62; // di
  unsigned __int8 v63; // al
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  int v68; // eax
  int v69; // ecx
  __int64 v70; // rdx
  __m128i v71; // kr00_16
  unsigned int v72; // r13d
  unsigned int v73; // edx
  unsigned __int8 v74; // al
  __int64 v75; // rdx
  __m128i v76; // kr10_16
  char v77; // al
  __int64 v78; // rdx
  unsigned __int32 v79; // ebx
  unsigned int v80; // r14d
  __int64 v81; // r13
  __int64 v82; // [rsp+0h] [rbp-A0h]
  _QWORD v84[2]; // [rsp+20h] [rbp-80h] BYREF
  __m128i v85; // [rsp+30h] [rbp-70h] BYREF
  char v86[8]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v87; // [rsp+48h] [rbp-58h]
  __m128i v88; // [rsp+50h] [rbp-50h] BYREF
  __int64 v89; // [rsp+60h] [rbp-40h]

  v5 = a2;
  v84[0] = a4;
  v84[1] = a5;
  if ( (_BYTE)a4 )
  {
    v7 = *(_BYTE *)(a2 + (unsigned __int8)a4 + 73900);
    if ( v7 != 6 )
    {
      if ( v7 == 5 )
      {
        switch ( (char)a4 )
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
            v7 = 2;
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
            v7 = 3;
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
            v7 = 4;
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
            v7 = 6;
            break;
          case 55:
            v7 = 7;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            v7 = 8;
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
            v7 = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            v7 = 10;
            break;
        }
        *(_BYTE *)a1 = 5;
        *(_BYTE *)(a1 + 8) = v7;
        *(_QWORD *)(a1 + 16) = 0;
      }
      else
      {
        v8 = *(_BYTE *)(a2 + (unsigned __int8)a4 + 2307);
        *(_BYTE *)a1 = v7;
        *(_QWORD *)(a1 + 16) = 0;
        *(_BYTE *)(a1 + 8) = v8;
      }
      return a1;
    }
    v18 = word_42F2F80[(unsigned __int8)(a4 - 14)] >> 1;
    switch ( (char)a4 )
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
        v7 = 3;
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
        v7 = 4;
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
        v7 = 5;
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
        break;
      case 55:
        v7 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v7 = 8;
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
        v7 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v7 = 10;
        break;
      default:
        v7 = 2;
        break;
    }
    v26 = v7;
    v27 = sub_1D15020(v7, v18);
    v28 = 0;
    if ( !v27 )
      v27 = sub_1F593D0(a3, v26, 0, v18);
LABEL_32:
    *(_BYTE *)a1 = 6;
    *(_BYTE *)(a1 + 8) = v27;
    *(_QWORD *)(a1 + 16) = v28;
    return a1;
  }
  if ( !(unsigned __int8)sub_1F58D20(v84) )
  {
    v14 = sub_1F58D40(v84, a2, v10, v11, v12, v13);
    if ( v14 <= 7 )
      goto LABEL_50;
    if ( (v14 & (v14 - 1)) == 0 )
    {
      v15 = v14 >> 1;
      if ( v15 == 32 )
      {
        v16 = 5;
        goto LABEL_14;
      }
      if ( v15 > 0x20 )
      {
        if ( v15 == 64 )
        {
          v16 = 6;
          goto LABEL_14;
        }
        if ( v15 == 128 )
        {
          v16 = 7;
          goto LABEL_14;
        }
      }
      else
      {
        if ( v15 == 8 )
        {
          v16 = 3;
          goto LABEL_14;
        }
        if ( v15 == 16 )
        {
          v16 = 4;
LABEL_14:
          v17 = 0;
LABEL_15:
          *(_BYTE *)a1 = 2;
          *(_BYTE *)(a1 + 8) = v16;
          *(_QWORD *)(a1 + 16) = v17;
          return a1;
        }
      }
      v16 = sub_1F58CC0(a3);
      goto LABEL_15;
    }
    if ( v14 == 8 )
    {
LABEL_50:
      v24 = 0;
      v22 = 3;
    }
    else
    {
      _BitScanReverse(&v19, v14 - 1);
      v20 = v19 ^ 0x1F;
      v21 = 32 - v20;
      if ( v20 == 27 )
      {
        v22 = 5;
      }
      else if ( 1 << (32 - v20) > 32 )
      {
        if ( v21 == 6 )
        {
          v22 = 6;
        }
        else
        {
          if ( v21 != 7 )
            goto LABEL_22;
          v22 = 7;
        }
      }
      else if ( v21 == 3 )
      {
        v22 = 3;
      }
      else
      {
        v22 = 4;
        if ( v21 != 4 )
        {
LABEL_22:
          v22 = sub_1F58CC0(a3);
          v24 = v23;
          goto LABEL_23;
        }
      }
      v24 = 0;
    }
LABEL_23:
    v25 = v22;
    sub_1F40D10(&v88, a2, a3, v22, v24);
    if ( v88.m128i_i8[0] == 1 )
    {
      v45 = v89;
      *(__m128i *)a1 = _mm_loadu_si128(&v88);
      *(_QWORD *)(a1 + 16) = v45;
    }
    else
    {
      *(_BYTE *)a1 = 1;
      *(_QWORD *)(a1 + 8) = v25;
      *(_QWORD *)(a1 + 16) = v24;
    }
    return a1;
  }
  v29 = (unsigned int)sub_1F58D30(v84);
  v30 = sub_1F596B0(v84);
  v85.m128i_i8[0] = v30;
  v31 = v30;
  v85.m128i_i64[1] = v32;
  if ( (_DWORD)v29 == 1 )
  {
    v46 = _mm_loadu_si128(&v85);
    *(_BYTE *)a1 = 5;
    *(__m128i *)(a1 + 8) = v46;
    return a1;
  }
  if ( v30 )
    v33 = (unsigned __int8)(v30 - 2) <= 5u || (unsigned __int8)(v30 - 14) <= 0x47u;
  else
    v33 = sub_1F58CF0(&v85);
  if ( !v33 )
  {
LABEL_46:
    while ( 1 )
    {
      v42 = (((((((((v29 | (v29 >> 1)) >> 2) | v29 | (v29 >> 1)) >> 4) | ((v29 | (v29 >> 1)) >> 2) | v29 | (v29 >> 1)) >> 8)
             | ((((v29 | (v29 >> 1)) >> 2) | v29 | (v29 >> 1)) >> 4)
             | ((v29 | (v29 >> 1)) >> 2)
             | v29
             | (v29 >> 1)) >> 16)
           | ((((((v29 | (v29 >> 1)) >> 2) | v29 | (v29 >> 1)) >> 4) | ((v29 | (v29 >> 1)) >> 2) | v29 | (v29 >> 1)) >> 8)
           | ((((v29 | (v29 >> 1)) >> 2) | v29 | (v29 >> 1)) >> 4)
           | ((v29 | (v29 >> 1)) >> 2)
           | v29
           | (v29 >> 1))
          + 1;
      v29 = ((unsigned int)((((((((v29 | (v29 >> 1)) >> 2) | v29 | (v29 >> 1)) >> 4)
                             | ((v29 | (v29 >> 1)) >> 2)
                             | v29
                             | (v29 >> 1)) >> 8)
                           | ((((v29 | (v29 >> 1)) >> 2) | v29 | (v29 >> 1)) >> 4)
                           | ((v29 | (v29 >> 1)) >> 2)
                           | v29
                           | (v29 >> 1)) >> 16)
           | (unsigned int)((((((v29 | (v29 >> 1)) >> 2) | v29 | (v29 >> 1)) >> 4)
                           | ((v29 | (v29 >> 1)) >> 2)
                           | v29
                           | (v29 >> 1)) >> 8)
           | (unsigned int)((((v29 | (v29 >> 1)) >> 2) | v29 | (v29 >> 1)) >> 4)
           | (unsigned int)((v29 | (v29 >> 1)) >> 2)
           | (unsigned int)v29
           | (unsigned int)(v29 >> 1))
          + 1;
      if ( !v31 )
        break;
      v41 = sub_1D15020(v31, v42);
      if ( !v41 )
        break;
      if ( !*(_BYTE *)(v5 + v41 + 73900) )
      {
        *(_BYTE *)a1 = 7;
        *(_BYTE *)(a1 + 8) = v41;
        *(_QWORD *)(a1 + 16) = 0;
        return a1;
      }
      v31 = v85.m128i_i8[0];
    }
    if ( LOBYTE(v84[0]) )
    {
      v43 = word_42F2F80[(unsigned __int8)(LOBYTE(v84[0]) - 14)];
      v44 = v43 - 1;
      if ( (v43 & (v43 - 1)) != 0 )
      {
        switch ( LOBYTE(v84[0]) )
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
            v47 = 2;
            goto LABEL_76;
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
            v47 = 3;
            goto LABEL_76;
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
            v47 = 4;
            goto LABEL_76;
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
            v47 = 5;
            goto LABEL_76;
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
            v47 = 6;
            goto LABEL_76;
          case 0x37:
            _BitScanReverse(&v53, v44);
            v49 = 7;
            v50 = 0;
            v51 = 7;
            v52 = 1 << (32 - (v53 ^ 0x1F));
            goto LABEL_78;
          case 0x56:
          case 0x57:
          case 0x58:
          case 0x62:
          case 0x63:
          case 0x64:
            v47 = 8;
            goto LABEL_76;
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
            v47 = 9;
            goto LABEL_76;
          case 0x5E:
          case 0x5F:
          case 0x60:
          case 0x61:
          case 0x6A:
          case 0x6B:
          case 0x6C:
          case 0x6D:
            v47 = 10;
LABEL_76:
            _BitScanReverse(&v48, v44);
            v49 = v47;
            v50 = 0;
            v51 = v47;
            v52 = 1 << (32 - (v48 ^ 0x1F));
            if ( (unsigned __int8)(LOBYTE(v84[0]) - 56) > 0x1Du && (unsigned __int8)(LOBYTE(v84[0]) - 98) > 0xBu )
              goto LABEL_78;
            v39 = sub_1D154A0(v47, v52);
            break;
          default:
            ++*(_DWORD *)(a1 + 16);
            BUG();
        }
        goto LABEL_79;
      }
    }
    else
    {
      v43 = sub_1F58D30(v84);
      if ( ((v43 - 1) & v43) != 0 )
      {
        _BitScanReverse(&v73, v43 - 1);
        v52 = 1 << (32 - (v73 ^ 0x1F));
        v74 = sub_1F596B0(v84);
        v49 = v74;
        v50 = v75;
        v51 = v74;
LABEL_78:
        v39 = sub_1D15020(v51, v52);
LABEL_79:
        v40 = 0;
        if ( !v39 )
          v39 = sub_1F593D0(a3, v49, v50, v52);
        goto LABEL_42;
      }
    }
    v71 = v85;
    v72 = v43 >> 1;
    v27 = sub_1D15020(v85.m128i_i8[0], v43 >> 1);
    v28 = 0;
    if ( !v27 )
      v27 = sub_1F593D0(a3, v71.m128i_u32[0], v71.m128i_i64[1], v72);
    goto LABEL_32;
  }
  if ( LOBYTE(v84[0]) )
    v34 = word_42F2F80[(unsigned __int8)(LOBYTE(v84[0]) - 14)];
  else
    v34 = sub_1F58D30(v84);
  if ( (v34 & (v34 - 1)) != 0 )
  {
    v35 = v85.m128i_i64[1];
    v36 = ((v29 | (v29 >> 1)) >> 2) | v29 | (v29 >> 1);
    v37 = v85.m128i_i32[0];
    v38 = ((((((v36 >> 4) | v36) >> 8) | (v36 >> 4) | v36) >> 16) | (((v36 >> 4) | v36) >> 8) | (v36 >> 4) | v36) + 1;
    v39 = sub_1D15020(v85.m128i_i8[0], v38);
    v40 = 0;
    if ( !v39 )
      v39 = sub_1F593D0(a3, v37, v35, v38);
LABEL_42:
    *(_BYTE *)a1 = 7;
    *(_BYTE *)(a1 + 8) = v39;
    *(_QWORD *)(a1 + 16) = v40;
    return a1;
  }
  sub_1F40D10(&v88, a2, a3, v85.m128i_u32[0], v85.m128i_i64[1]);
  if ( v88.m128i_i8[0] == 2 )
  {
    v79 = v85.m128i_i32[0];
    v80 = (unsigned int)v29 >> 1;
    v81 = v85.m128i_i64[1];
    v27 = sub_1D15020(v85.m128i_i8[0], v80);
    v28 = 0;
    if ( !v27 )
      v27 = sub_1F593D0(a3, v79, v81, v80);
    goto LABEL_32;
  }
  v31 = v85.m128i_i8[0];
  v82 = v85.m128i_i64[1];
  for ( i = v85.m128i_i8[0]; ; i = v85.m128i_i8[0] )
  {
    if ( i )
      v59 = sub_1F3E310(&v85);
    else
      v59 = sub_1F58D40(&v85, a2, v54, v55, v56, v57);
    v60 = (unsigned int)(v59 + 1);
    if ( v59 == 31 )
    {
      v86[0] = 5;
      v87 = 0;
      goto LABEL_101;
    }
    if ( (unsigned int)v60 > 0x20 )
    {
      if ( v59 == 63 )
      {
        v86[0] = 6;
        v87 = 0;
        goto LABEL_101;
      }
      if ( v59 == 127 )
      {
        v86[0] = 7;
        v87 = 0;
        goto LABEL_101;
      }
    }
    else
    {
      switch ( v59 )
      {
        case 7:
          v86[0] = 3;
          v87 = 0;
          goto LABEL_101;
        case 15:
          v86[0] = 4;
          v87 = 0;
LABEL_101:
          v61 = sub_1F3E310(v86);
          goto LABEL_102;
        case 0:
          v86[0] = 2;
          v87 = 0;
          goto LABEL_101;
      }
    }
    v86[0] = sub_1F58CC0(a3);
    v87 = v64;
    if ( v86[0] )
      goto LABEL_101;
    v61 = sub_1F58D40(v86, v60, v64, v65, v66, v67);
LABEL_102:
    if ( v61 <= 8 )
      goto LABEL_103;
    _BitScanReverse(&v61, v61 - 1);
    v68 = v61 ^ 0x1F;
    v69 = 32 - v68;
    if ( v68 == 27 )
    {
      v85.m128i_i8[0] = 5;
      v62 = 5;
      v85.m128i_i64[1] = 0;
      goto LABEL_104;
    }
    if ( 1 << (32 - v68) > 32 )
    {
      if ( v69 == 6 )
      {
        v85.m128i_i8[0] = 6;
        v62 = 6;
        v85.m128i_i64[1] = 0;
        goto LABEL_104;
      }
      if ( v69 == 7 )
      {
        v85.m128i_i8[0] = 7;
        v62 = 7;
        v85.m128i_i64[1] = 0;
        goto LABEL_104;
      }
    }
    else
    {
      if ( v69 == 3 )
      {
LABEL_103:
        v85.m128i_i8[0] = 3;
        v62 = 3;
        v85.m128i_i64[1] = 0;
        goto LABEL_104;
      }
      if ( v69 == 4 )
      {
        v85.m128i_i8[0] = 4;
        v62 = 4;
        v85.m128i_i64[1] = 0;
        goto LABEL_104;
      }
    }
    v85.m128i_i8[0] = sub_1F58CC0(a3);
    v62 = v85.m128i_i8[0];
    v85.m128i_i64[1] = v70;
    if ( !v85.m128i_i8[0] )
    {
      v85.m128i_i8[0] = v31;
      v85.m128i_i64[1] = v82;
      goto LABEL_46;
    }
LABEL_104:
    a2 = (unsigned int)v29;
    v63 = sub_1D15020(v62, v29);
    if ( v63 )
    {
      if ( !*(_BYTE *)(v5 + v63 + 73900) )
        break;
    }
  }
  v76 = v85;
  v77 = sub_1D15020(v85.m128i_i8[0], v29);
  v78 = 0;
  if ( !v77 )
    v77 = sub_1F593D0(a3, v76.m128i_u32[0], v76.m128i_i64[1], (unsigned int)v29);
  *(_BYTE *)a1 = 1;
  *(_BYTE *)(a1 + 8) = v77;
  *(_QWORD *)(a1 + 16) = v78;
  return a1;
}
