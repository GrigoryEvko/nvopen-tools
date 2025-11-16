// Function: sub_2136F40
// Address: 0x2136f40
//
__int64 *__fastcall sub_2136F40(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v5; // ebx
  unsigned int v6; // r14d
  __int64 v7; // r15
  __int64 v8; // rax
  char v9; // dl
  char *v10; // rax
  __int64 v11; // rsi
  char v12; // dl
  __int64 v13; // rax
  int v14; // r8d
  int v15; // r9d
  char v16; // r12
  unsigned int v17; // eax
  const void **v18; // rdx
  unsigned int v19; // ecx
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rsi
  __m128i v23; // xmm0
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // r15d
  unsigned int v27; // r14d
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r9
  __int64 v33; // rdx
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // r11
  __int64 *v37; // r8
  __int64 v38; // r9
  __int64 v39; // rcx
  unsigned int v40; // eax
  const void **v41; // rdx
  __int128 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  __int64 *v47; // rax
  __int64 v48; // rax
  _QWORD *v49; // rdx
  __int64 *v50; // r14
  __int128 v52; // [rsp-20h] [rbp-1B0h]
  __int128 v53; // [rsp-10h] [rbp-1A0h]
  __int128 v54; // [rsp-10h] [rbp-1A0h]
  __int64 *v55; // [rsp+20h] [rbp-170h]
  __int64 v56; // [rsp+28h] [rbp-168h]
  __int64 v57; // [rsp+30h] [rbp-160h]
  __int64 v58; // [rsp+38h] [rbp-158h]
  const void **v59; // [rsp+40h] [rbp-150h]
  unsigned int v60; // [rsp+48h] [rbp-148h]
  unsigned int v61; // [rsp+48h] [rbp-148h]
  __int64 *v62; // [rsp+50h] [rbp-140h]
  __int64 *v63; // [rsp+50h] [rbp-140h]
  __int64 v64; // [rsp+50h] [rbp-140h]
  __int64 v65; // [rsp+58h] [rbp-138h]
  __int64 v67; // [rsp+68h] [rbp-128h]
  unsigned int v68; // [rsp+78h] [rbp-118h]
  unsigned int v69; // [rsp+7Ch] [rbp-114h]
  unsigned int v70; // [rsp+7Ch] [rbp-114h]
  unsigned __int64 v71; // [rsp+88h] [rbp-108h]
  char v72[8]; // [rsp+90h] [rbp-100h] BYREF
  __int64 v73; // [rsp+98h] [rbp-F8h]
  __int64 v74; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v75; // [rsp+A8h] [rbp-E8h]
  __int64 v76; // [rsp+B0h] [rbp-E0h] BYREF
  const void **v77; // [rsp+B8h] [rbp-D8h]
  __int64 v78; // [rsp+C0h] [rbp-D0h] BYREF
  int v79; // [rsp+C8h] [rbp-C8h]
  _QWORD *v80; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v81; // [rsp+D8h] [rbp-B8h]
  _QWORD v82[22]; // [rsp+E0h] [rbp-B0h] BYREF

  v8 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v9 = *(_BYTE *)v8;
  v73 = *(_QWORD *)(v8 + 8);
  v10 = *(char **)(a2 + 40);
  v72[0] = v9;
  v11 = *(_QWORD *)a1;
  v12 = *v10;
  v75 = *((_QWORD *)v10 + 1);
  v13 = *(_QWORD *)(a1 + 8);
  LOBYTE(v74) = v12;
  sub_1F40D10((__int64)&v80, v11, *(_QWORD *)(v13 + 48), v74, v75);
  v16 = v81;
  v77 = (const void **)v82[0];
  LOBYTE(v76) = v81;
  if ( !(_BYTE)v74 )
  {
    v68 = sub_1F58D30((__int64)&v74);
    if ( v16 )
      goto LABEL_3;
LABEL_5:
    LOBYTE(v17) = sub_1F596B0((__int64)&v76);
    v60 = v17;
    v59 = v18;
    goto LABEL_6;
  }
  v68 = word_4310720[(unsigned __int8)(v74 - 14)];
  if ( !(_BYTE)v81 )
    goto LABEL_5;
LABEL_3:
  switch ( v16 )
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
      LOBYTE(v17) = 2;
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
      LOBYTE(v17) = 3;
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
      LOBYTE(v17) = 4;
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
      LOBYTE(v17) = 5;
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
      LOBYTE(v17) = 6;
      break;
    case 55:
      LOBYTE(v17) = 7;
      break;
    case 86:
    case 87:
    case 88:
    case 98:
    case 99:
    case 100:
      LOBYTE(v17) = 8;
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
      LOBYTE(v17) = 9;
      break;
    case 94:
    case 95:
    case 96:
    case 97:
    case 106:
    case 107:
    case 108:
    case 109:
      LOBYTE(v17) = 10;
      break;
  }
  v59 = 0;
LABEL_6:
  v19 = v60;
  LOBYTE(v19) = v17;
  v61 = v19;
  v20 = *(_QWORD *)(a2 + 72);
  v78 = v20;
  if ( v20 )
    sub_1623A60((__int64)&v78, v20, 2);
  v79 = *(_DWORD *)(a2 + 64);
  v21 = *(_QWORD *)(a2 + 32);
  v22 = *(_QWORD *)(v21 + 40);
  v23 = _mm_loadu_si128((const __m128i *)(v21 + 40));
  v24 = *(unsigned int *)(v21 + 48);
  v80 = v82;
  v81 = 0x800000000LL;
  v71 = v23.m128i_u64[1];
  if ( v68 > 8 )
  {
    v70 = v24;
    sub_16CD150((__int64)&v80, v82, v68, 16, v14, v15);
    v24 = v70;
  }
  else if ( !v68 )
  {
    v49 = v82;
    v48 = 0;
    goto LABEL_19;
  }
  v58 = v24;
  v67 = 16 * v24;
  v69 = 0;
  v25 = v7;
  v26 = v6;
  v27 = v5;
  v28 = v25;
  do
  {
    v62 = *(__int64 **)(a1 + 8);
    v29 = *(_QWORD *)(v22 + 40) + v67;
    LOBYTE(v28) = *(_BYTE *)v29;
    v30 = sub_1D38BB0(
            (__int64)v62,
            v69,
            (__int64)&v78,
            (unsigned int)v28,
            *(const void ***)(v29 + 8),
            0,
            v23,
            a4,
            a5,
            0);
    v32 = v31;
    v33 = *(_QWORD *)(v22 + 40) + v67;
    LOBYTE(v26) = *(_BYTE *)v33;
    *((_QWORD *)&v52 + 1) = v32;
    *(_QWORD *)&v52 = v30;
    v71 = v58 | v71 & 0xFFFFFFFF00000000LL;
    v34 = sub_1D332F0(
            v62,
            52,
            (__int64)&v78,
            v26,
            *(const void ***)(v33 + 8),
            0,
            *(double *)v23.m128i_i64,
            a4,
            a5,
            v22,
            v71,
            v52);
    v36 = *(__int64 **)(a1 + 8);
    v37 = v34;
    v38 = v35;
    v39 = *(_QWORD *)(a2 + 32);
    if ( v72[0] )
    {
      switch ( v72[0] )
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
          LOBYTE(v40) = 2;
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
          LOBYTE(v40) = 3;
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
          LOBYTE(v40) = 4;
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
          LOBYTE(v40) = 5;
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
          LOBYTE(v40) = 6;
          break;
        case 0x37:
          LOBYTE(v40) = 7;
          break;
        case 0x56:
        case 0x57:
        case 0x58:
        case 0x62:
        case 0x63:
        case 0x64:
          LOBYTE(v40) = 8;
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
          LOBYTE(v40) = 9;
          break;
        case 0x5E:
        case 0x5F:
        case 0x60:
        case 0x61:
        case 0x6A:
        case 0x6B:
        case 0x6C:
        case 0x6D:
          LOBYTE(v40) = 10;
          break;
        default:
          *(_DWORD *)(v28 + 8) = (2 * (*(_DWORD *)(v28 + 8) >> 1) + 2) | *(_DWORD *)(v28 + 8) & 1;
          BUG();
      }
      v41 = 0;
    }
    else
    {
      v55 = v34;
      v56 = v35;
      v57 = *(_QWORD *)(a2 + 32);
      v63 = *(__int64 **)(a1 + 8);
      LOBYTE(v40) = sub_1F596B0((__int64)v72);
      v37 = v55;
      v38 = v56;
      v39 = v57;
      v36 = v63;
      v27 = v40;
    }
    *((_QWORD *)&v53 + 1) = v38;
    LOBYTE(v27) = v40;
    *(_QWORD *)&v53 = v37;
    *(_QWORD *)&v42 = sub_1D332F0(
                        v36,
                        106,
                        (__int64)&v78,
                        v27,
                        v41,
                        0,
                        *(double *)v23.m128i_i64,
                        a4,
                        a5,
                        *(_QWORD *)v39,
                        *(_QWORD *)(v39 + 8),
                        v53);
    v44 = sub_1D309E0(
            *(__int64 **)(a1 + 8),
            144,
            (__int64)&v78,
            v61,
            v59,
            0,
            *(double *)v23.m128i_i64,
            a4,
            *(double *)a5.m128i_i64,
            v42);
    v45 = v43;
    v46 = (unsigned int)v81;
    if ( (unsigned int)v81 >= HIDWORD(v81) )
    {
      v65 = v43;
      v64 = v44;
      sub_16CD150((__int64)&v80, v82, 0, 16, v44, v43);
      v46 = (unsigned int)v81;
      v44 = v64;
      v45 = v65;
    }
    v47 = &v80[2 * v46];
    ++v69;
    *v47 = v44;
    v47[1] = v45;
    v48 = (unsigned int)(v81 + 1);
    LODWORD(v81) = v81 + 1;
  }
  while ( v69 != v68 );
  v49 = v80;
LABEL_19:
  *((_QWORD *)&v54 + 1) = v48;
  *(_QWORD *)&v54 = v49;
  v50 = sub_1D359D0(*(__int64 **)(a1 + 8), 104, (__int64)&v78, v76, v77, 0, *(double *)v23.m128i_i64, a4, a5, v54);
  if ( v80 != v82 )
    _libc_free((unsigned __int64)v80);
  if ( v78 )
    sub_161E7C0((__int64)&v78, v78);
  return v50;
}
