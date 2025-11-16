// Function: sub_1FEC5F0
// Address: 0x1fec5f0
//
_QWORD *__fastcall sub_1FEC5F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int64 a10,
        __int64 a11)
{
  __int64 v11; // r15
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // r14
  __int64 v15; // r12
  __int64 v16; // rbx
  int v17; // eax
  unsigned __int8 *v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r9
  __int64 v21; // r14
  char v22; // al
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  _QWORD *v26; // rax
  _QWORD *v27; // r13
  unsigned int v28; // r10d
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r12
  __int64 v33; // r13
  __int64 v34; // rax
  _QWORD *v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r13
  _QWORD *v39; // r14
  unsigned int v40; // edx
  unsigned __int64 v41; // r12
  _QWORD *result; // rax
  __int8 v43; // al
  unsigned __int64 v44; // rcx
  char v45; // al
  __int64 v46; // rdx
  __int64 v47; // rsi
  __int8 v48; // r9
  unsigned __int8 *v49; // rsi
  __int64 v50; // rsi
  bool v51; // al
  __int64 v52; // rax
  unsigned __int8 *v53; // r14
  __int8 v54; // al
  __int64 v55; // rdx
  unsigned __int64 v56; // rdx
  unsigned int v57; // r9d
  _QWORD *v58; // r15
  _QWORD *v59; // r8
  _QWORD *v60; // r14
  __int64 v61; // rax
  unsigned int v62; // ebx
  unsigned int v63; // esi
  _QWORD *v64; // r12
  __int64 v65; // rcx
  unsigned int v66; // r13d
  _QWORD *v67; // rdx
  _QWORD *v68; // rdx
  unsigned __int8 *v69; // rax
  unsigned int v70; // eax
  __int64 v71; // rcx
  char v72; // r9
  __int64 v73; // r10
  __int64 v74; // r11
  unsigned __int8 *v75; // rdx
  unsigned int v76; // eax
  unsigned int v77; // eax
  __int128 v78; // [rsp-10h] [rbp-170h]
  __int64 v79; // [rsp+8h] [rbp-158h]
  __int64 v80; // [rsp+10h] [rbp-150h]
  __int64 v81; // [rsp+18h] [rbp-148h]
  unsigned __int8 *v82; // [rsp+20h] [rbp-140h]
  __int64 v83; // [rsp+20h] [rbp-140h]
  __int64 v84; // [rsp+20h] [rbp-140h]
  __int64 v85; // [rsp+28h] [rbp-138h]
  __int64 v86; // [rsp+30h] [rbp-130h]
  _QWORD *v87; // [rsp+30h] [rbp-130h]
  unsigned __int8 *v88; // [rsp+30h] [rbp-130h]
  __int64 v89; // [rsp+30h] [rbp-130h]
  __int64 v90; // [rsp+38h] [rbp-128h]
  __int64 v91; // [rsp+38h] [rbp-128h]
  unsigned __int8 v92; // [rsp+40h] [rbp-120h]
  __int64 v93; // [rsp+40h] [rbp-120h]
  __int64 v94; // [rsp+48h] [rbp-118h]
  unsigned int v95; // [rsp+48h] [rbp-118h]
  __int8 v96; // [rsp+48h] [rbp-118h]
  unsigned __int8 *v97; // [rsp+48h] [rbp-118h]
  unsigned __int8 *v98; // [rsp+48h] [rbp-118h]
  char v99; // [rsp+48h] [rbp-118h]
  unsigned int v100; // [rsp+50h] [rbp-110h]
  __int64 v101; // [rsp+50h] [rbp-110h]
  __int64 v102; // [rsp+50h] [rbp-110h]
  unsigned int v103; // [rsp+50h] [rbp-110h]
  __int64 v106; // [rsp+60h] [rbp-100h]
  __int64 v107; // [rsp+68h] [rbp-F8h]
  __int64 v108; // [rsp+70h] [rbp-F0h]
  __int64 v109; // [rsp+70h] [rbp-F0h]
  __int64 v110; // [rsp+78h] [rbp-E8h]
  __int64 v111; // [rsp+78h] [rbp-E8h]
  __int64 v112; // [rsp+80h] [rbp-E0h]
  _QWORD *v114; // [rsp+90h] [rbp-D0h]
  unsigned int v115; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v116; // [rsp+B8h] [rbp-A8h]
  __m128i v117; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v118; // [rsp+D0h] [rbp-90h]
  __int128 v119; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v120; // [rsp+F0h] [rbp-70h]
  __m128 v121; // [rsp+100h] [rbp-60h] BYREF
  _QWORD v122[10]; // [rsp+110h] [rbp-50h] BYREF

  v12 = a4;
  v13 = a5;
  v14 = (unsigned int)a3;
  v15 = a2;
  v16 = a1;
  v17 = *(unsigned __int16 *)(a10 + 24);
  v18 = (unsigned __int8 *)(16LL * (unsigned int)a3 + *(_QWORD *)(a2 + 40));
  v112 = 16 * v14;
  if ( v17 != 32 )
  {
    v19 = (unsigned int)a5;
    if ( v17 != 10 )
    {
LABEL_3:
      v20 = v14;
      v21 = v19;
      v22 = *v18;
      v23 = *((_QWORD *)v18 + 1);
      LOBYTE(v115) = v22;
      v116 = v23;
      if ( v22 )
      {
        switch ( v22 )
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
            LOBYTE(v24) = 2;
            v25 = 0;
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
            LOBYTE(v24) = 3;
            v25 = 0;
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
            LOBYTE(v24) = 4;
            v25 = 0;
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
            LOBYTE(v24) = 5;
            v25 = 0;
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
            LOBYTE(v24) = 6;
            v25 = 0;
            break;
          case 55:
            LOBYTE(v24) = 7;
            v25 = 0;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            LOBYTE(v24) = 8;
            v25 = 0;
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
            LOBYTE(v24) = 9;
            v25 = 0;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            LOBYTE(v24) = 10;
            v25 = 0;
            break;
          default:
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            BUG();
        }
      }
      else
      {
        v108 = v20;
        LOBYTE(v24) = sub_1F596B0((__int64)&v115);
        v20 = v108;
        v11 = v24;
      }
      v110 = v25;
      LOBYTE(v11) = v24;
      v94 = v20;
      v26 = sub_1D29C20(*(_QWORD **)(a1 + 16), v115, v116, 1, a5, v20);
      v27 = *(_QWORD **)(a1 + 16);
      v28 = *((_DWORD *)v26 + 21);
      v121 = 0u;
      v122[0] = 0;
      v107 = v29;
      v100 = v28;
      v106 = (__int64)v26;
      sub_1E341E0((__int64)&v119, v27[4], v28, 0);
      v30 = sub_1D2BF40(v27, *(_QWORD *)(a1 + 16) + 88LL, 0, a6, v15, v94, v106, v107, v119, v120, 0, 0, (__int64)&v121);
      v32 = v31;
      v33 = v30;
      v34 = sub_20BD400(*(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 16), v106, v107, v115, v116, a10, (unsigned int)a11);
      v35 = *(_QWORD **)(a1 + 16);
      v118 = 0;
      v121 = 0u;
      v122[0] = 0;
      v117 = 0u;
      v37 = sub_1D2C750(v35, v33, v32, a6, a4, v21, v34, v36, 0, 0, v11, v110, 0, 0, (__int64)&v121);
      v121 = 0u;
      v38 = v37;
      v39 = *(_QWORD **)(v16 + 16);
      v122[0] = 0;
      v41 = v40 | v32 & 0xFFFFFFFF00000000LL;
      sub_1E341E0((__int64)&v117, v39[4], v100, 0);
      return (_QWORD *)sub_1D2B730(
                         v39,
                         v115,
                         v116,
                         a6,
                         v38,
                         v41,
                         v106,
                         v107,
                         *(_OWORD *)&v117,
                         v118,
                         0,
                         0,
                         (__int64)&v121,
                         0);
    }
  }
  v43 = *v18;
  v44 = *((_QWORD *)v18 + 1);
  v121.m128_i8[0] = v43;
  v121.m128_u64[1] = v44;
  if ( v43 )
  {
    switch ( v43 )
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
        v48 = 2;
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
        v48 = 3;
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
        v48 = 4;
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
        v48 = 5;
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
        v48 = 6;
        break;
      case 55:
        v48 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v48 = 8;
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
        v48 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v48 = 10;
        break;
    }
    v19 = (unsigned int)a5;
    v117.m128i_i8[0] = v48;
    v117.m128i_i64[1] = 0;
    v101 = 0;
    v69 = (unsigned __int8 *)(*(_QWORD *)(a4 + 40) + 16LL * (unsigned int)a5);
    a5 = *v69;
    v50 = *((_QWORD *)v69 + 1);
    if ( (_BYTE)a5 == v48 )
      goto LABEL_33;
  }
  else
  {
    v86 = v12;
    v90 = a5;
    v95 = a5;
    v45 = sub_1F596B0((__int64)&v121);
    v19 = v95;
    v47 = v46;
    v48 = v45;
    v101 = v46;
    v18 = (unsigned __int8 *)(*(_QWORD *)(v15 + 40) + v112);
    v117.m128i_i64[1] = v47;
    v117.m128i_i8[0] = v45;
    v12 = v86;
    v49 = (unsigned __int8 *)(*(_QWORD *)(a4 + 40) + 16LL * v95);
    v13 = v90;
    a5 = *v49;
    v50 = *((_QWORD *)v49 + 1);
    if ( v45 == (_BYTE)a5 )
    {
LABEL_33:
      if ( v101 == v50 || (_BYTE)a5 )
        goto LABEL_16;
      LOBYTE(a5) = 0;
      goto LABEL_12;
    }
    if ( !v45 )
    {
LABEL_12:
      v79 = v19;
      v80 = v12;
      v81 = v13;
      v92 = a5;
      v82 = v18;
      v96 = v48;
      v51 = sub_1F58CF0((__int64)&v117);
      v19 = v79;
      v12 = v80;
      v13 = v81;
      a5 = v92;
      v18 = v82;
      v48 = v96;
      goto LABEL_13;
    }
  }
  v51 = (unsigned __int8)(v48 - 14) <= 0x47u || (unsigned __int8)(v48 - 2) <= 5u;
LABEL_13:
  if ( !v51 )
    goto LABEL_3;
  a7 = _mm_loadu_si128(&v117);
  LOBYTE(v119) = a5;
  *((_QWORD *)&v119 + 1) = v50;
  v121 = (__m128)a7;
  if ( (_BYTE)a5 == v48 )
  {
    if ( v48 || v101 == v50 )
      goto LABEL_16;
  }
  else if ( (_BYTE)a5 )
  {
    v97 = v18;
    v70 = sub_1FEB8F0(a5);
    v75 = v97;
    v103 = v70;
    goto LABEL_42;
  }
  v93 = v19;
  v83 = v12;
  v85 = v13;
  v88 = v18;
  v99 = v48;
  v77 = sub_1F58D40((__int64)&v119);
  v71 = v93;
  v73 = v83;
  v103 = v77;
  v74 = v85;
  v75 = v88;
  v72 = v99;
LABEL_42:
  if ( v72 )
  {
    v98 = v75;
    v76 = sub_1FEB8F0(v72);
  }
  else
  {
    v84 = v71;
    v89 = v73;
    v91 = v74;
    v98 = v75;
    v76 = sub_1F58D40((__int64)&v121);
    v19 = v84;
    v12 = v89;
    v13 = v91;
  }
  v18 = v98;
  if ( v76 > v103 )
    goto LABEL_3;
LABEL_16:
  *((_QWORD *)&v78 + 1) = v13;
  *(_QWORD *)&v78 = v12;
  v52 = sub_1D309E0(
          *(__int64 **)(a1 + 16),
          111,
          a6,
          *v18,
          *((const void ***)v18 + 1),
          0,
          *(double *)a7.m128i_i64,
          a8,
          *(double *)a9.m128i_i64,
          v78);
  v53 = (unsigned __int8 *)(*(_QWORD *)(v15 + 40) + v112);
  v109 = v52;
  v54 = *v53;
  v111 = v55;
  v56 = *((_QWORD *)v53 + 1);
  v121.m128_i8[0] = v54;
  v121.m128_u64[1] = v56;
  if ( v54 )
    v57 = word_42FEB00[(unsigned __int8)(v54 - 14)];
  else
    v57 = sub_1F58D30((__int64)&v121);
  v58 = v122;
  v121.m128_u64[0] = (unsigned __int64)v122;
  v121.m128_u64[1] = 0x800000000LL;
  if ( v57 )
  {
    v59 = v122;
    v60 = 0;
    v102 = v15;
    v61 = 0;
    v62 = v57;
    v63 = 8;
    v64 = (_QWORD *)v57;
    while ( 1 )
    {
      v65 = *(_QWORD *)(a10 + 88);
      v66 = (unsigned int)v60;
      v67 = *(_QWORD **)(v65 + 24);
      if ( *(_DWORD *)(v65 + 32) > 0x40u )
        v67 = (_QWORD *)*v67;
      if ( v60 == v67 )
        v66 = v62;
      if ( (unsigned int)v61 >= v63 )
      {
        v87 = v59;
        sub_16CD150((__int64)&v121, v59, 0, 4, (int)v59, v57);
        v61 = v121.m128_u32[2];
        v59 = v87;
      }
      v60 = (_QWORD *)((char *)v60 + 1);
      *(_DWORD *)(v121.m128_u64[0] + 4 * v61) = v66;
      v61 = (unsigned int)++v121.m128_i32[2];
      if ( v60 == v64 )
        break;
      v63 = v121.m128_u32[3];
    }
    v15 = v102;
    v68 = (_QWORD *)v121.m128_u64[0];
    v58 = v59;
    v16 = a1;
    v53 = (unsigned __int8 *)(*(_QWORD *)(v102 + 40) + v112);
  }
  else
  {
    v61 = 0;
    v68 = v122;
  }
  result = sub_1D41320(
             *(_QWORD *)(v16 + 16),
             *v53,
             *((const void ***)v53 + 1),
             a6,
             v15,
             a3,
             *(double *)a7.m128i_i64,
             a8,
             a9,
             v109,
             v111,
             v68,
             v61);
  if ( (_QWORD *)v121.m128_u64[0] != v58 )
  {
    v114 = result;
    _libc_free(v121.m128_u64[0]);
    return v114;
  }
  return result;
}
