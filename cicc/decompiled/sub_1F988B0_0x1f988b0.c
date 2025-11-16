// Function: sub_1F988B0
// Address: 0x1f988b0
//
__int64 __fastcall sub_1F988B0(
        __int64 **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9,
        __int64 a10)
{
  __int64 v12; // rax
  unsigned __int8 v13; // cl
  char v14; // al
  __int64 v15; // rdx
  unsigned int v16; // ebx
  __int64 v17; // r14
  __int64 v18; // rax
  char v19; // r14
  __int64 *v20; // rbx
  __int64 v21; // rax
  int v22; // eax
  __int64 result; // rax
  __m128i v24; // xmm0
  _BOOL8 v25; // rdx
  __int64 (*v26)(); // rax
  __int64 v27; // rax
  __int64 v28; // rsi
  __m128i v29; // xmm1
  unsigned __int8 *v30; // rax
  const void **v31; // rbx
  unsigned int v32; // r14d
  __int64 *v33; // r11
  int v34; // eax
  __int64 v35; // rax
  __int64 *v36; // rbx
  unsigned int v37; // edx
  int v38; // eax
  __int128 v39; // rax
  unsigned int v40; // edx
  __int64 v41; // rax
  unsigned int v42; // eax
  __int64 v43; // rax
  _QWORD *v44; // rbx
  int v45; // eax
  __int64 v46; // r11
  unsigned int v47; // ebx
  unsigned int v48; // edx
  __int64 *v49; // rdx
  __int64 v50; // rax
  unsigned __int64 v51; // rcx
  int v52; // eax
  __int64 v53; // rbx
  unsigned __int64 v54; // rdx
  __m128i v55; // xmm2
  unsigned int v56; // edx
  char v57; // r13
  __int64 v58; // rsi
  __int64 *v59; // r10
  __int64 v60; // rax
  unsigned __int16 v61; // bx
  __int64 *v62; // r13
  __int64 v63; // rbx
  unsigned int v64; // edx
  char v65; // r13
  __int64 v66; // r13
  __int32 v67; // edx
  __int32 v68; // ecx
  __int64 *v69; // rax
  __int64 v70; // rdx
  __int64 *v71; // rdi
  __int64 i; // rbx
  unsigned int v73; // ebx
  unsigned int v74; // eax
  unsigned int v75; // r10d
  __int64 v76; // rsi
  __int64 *v77; // r13
  __int64 v78; // rax
  unsigned __int16 v79; // cx
  __int128 *v80; // rbx
  __int32 v81; // edx
  char v82; // si
  __int64 v83; // rax
  unsigned int v84; // eax
  __int64 *v85; // r13
  __int64 v86; // rsi
  __int32 v87; // edx
  __int128 v88; // [rsp-10h] [rbp-1B0h]
  const void **v89; // [rsp+8h] [rbp-198h]
  unsigned __int64 v90; // [rsp+18h] [rbp-188h]
  const void **v92; // [rsp+30h] [rbp-170h]
  char v93; // [rsp+3Bh] [rbp-165h]
  unsigned int v94; // [rsp+3Ch] [rbp-164h]
  __int64 v95; // [rsp+40h] [rbp-160h]
  __int128 v96; // [rsp+40h] [rbp-160h]
  unsigned int v97; // [rsp+40h] [rbp-160h]
  __int64 *v98; // [rsp+40h] [rbp-160h]
  unsigned int v99; // [rsp+40h] [rbp-160h]
  unsigned __int64 v100; // [rsp+48h] [rbp-158h]
  __int64 v101; // [rsp+48h] [rbp-158h]
  __int64 *v103; // [rsp+50h] [rbp-150h]
  unsigned __int8 v104; // [rsp+58h] [rbp-148h]
  unsigned __int16 v105; // [rsp+58h] [rbp-148h]
  __int32 v106; // [rsp+58h] [rbp-148h]
  unsigned int v107; // [rsp+58h] [rbp-148h]
  __int32 v108; // [rsp+58h] [rbp-148h]
  __int64 *v109; // [rsp+70h] [rbp-130h]
  _QWORD v110[2]; // [rsp+B0h] [rbp-F0h] BYREF
  unsigned int v111; // [rsp+C0h] [rbp-E0h] BYREF
  const void **v112; // [rsp+C8h] [rbp-D8h]
  __m128i v113; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v114; // [rsp+E0h] [rbp-C0h] BYREF
  int v115; // [rsp+E8h] [rbp-B8h]
  __int128 v116; // [rsp+F0h] [rbp-B0h]
  __int64 v117; // [rsp+100h] [rbp-A0h]
  __int64 (__fastcall **v118)(); // [rsp+110h] [rbp-90h] BYREF
  __int64 v119; // [rsp+118h] [rbp-88h]
  __int64 *v120; // [rsp+120h] [rbp-80h]
  __int64 **v121; // [rsp+128h] [rbp-78h]
  __int64 v122; // [rsp+130h] [rbp-70h] BYREF
  int v123; // [rsp+138h] [rbp-68h]
  __int64 v124; // [rsp+140h] [rbp-60h]
  int v125; // [rsp+148h] [rbp-58h]
  __m128i v126; // [rsp+150h] [rbp-50h] BYREF
  __int64 v127; // [rsp+160h] [rbp-40h]
  int v128; // [rsp+168h] [rbp-38h]

  v12 = *(_QWORD *)(a2 + 40);
  v110[0] = a3;
  v110[1] = a4;
  v13 = *(_BYTE *)v12;
  v92 = *(const void ***)(v12 + 8);
  v112 = v92;
  v104 = v13;
  LOBYTE(v111) = v13;
  if ( (_BYTE)a3 )
  {
    switch ( (char)a3 )
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
        v14 = 2;
        v15 = 0;
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
        v14 = 3;
        v15 = 0;
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
        v14 = 4;
        v15 = 0;
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
        v14 = 5;
        v15 = 0;
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
        v14 = 6;
        v15 = 0;
        break;
      case 55:
        v14 = 7;
        v15 = 0;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v14 = 8;
        v15 = 0;
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
        v14 = 9;
        v15 = 0;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v14 = 10;
        v15 = 0;
        break;
    }
  }
  else
  {
    v14 = sub_1F596B0((__int64)v110);
  }
  v113.m128i_i8[0] = v14;
  v113.m128i_i64[1] = v15;
  v16 = sub_1E34390(*(_QWORD *)(a10 + 104));
  v17 = sub_1E0A0C0((*a1)[4]);
  v18 = sub_1F58E60((__int64)&v113, (_QWORD *)(*a1)[6]);
  v94 = sub_15A9FE0(v17, v18);
  if ( v94 > v16 )
    return 0;
  v19 = v113.m128i_i8[0];
  v20 = a1[1];
  v21 = 1;
  if ( v113.m128i_i8[0] != 1 )
  {
    if ( !v113.m128i_i8[0] )
      return 0;
    v21 = v113.m128i_u8[0];
    if ( !v20[v113.m128i_u8[0] + 15] )
      return 0;
  }
  v22 = *((_BYTE *)v20 + 259 * v21 + 2607) & 0xFB;
  v93 = v22;
  if ( v22 )
    return 0;
  v24 = _mm_loadu_si128(&v113);
  v126 = v24;
  if ( v104 == v113.m128i_i8[0] )
  {
    if ( v104 || v92 == (const void **)v126.m128i_i64[1] )
    {
      v25 = 1;
      goto LABEL_15;
    }
    goto LABEL_39;
  }
  if ( !v104 )
  {
LABEL_39:
    v97 = sub_1F58D40((__int64)&v111);
    goto LABEL_25;
  }
  v97 = sub_1F6C8D0(v104);
LABEL_25:
  if ( v19 )
    v42 = sub_1F6C8D0(v19);
  else
    v42 = sub_1F58D40((__int64)&v126);
  v25 = v42 >= v97;
LABEL_15:
  v26 = *(__int64 (**)())(*v20 + 416);
  if ( v26 != sub_1F3CAB0
    && !((unsigned __int8 (__fastcall *)(__int64 *, __int64, _BOOL8, _QWORD, __int64))v26)(
          v20,
          a10,
          v25,
          v113.m128i_u32[0],
          v113.m128i_i64[1]) )
  {
    return 0;
  }
  v27 = *(_QWORD *)(a10 + 32);
  v28 = *(_QWORD *)(a2 + 72);
  v29 = _mm_loadu_si128((const __m128i *)(v27 + 40));
  v30 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v27 + 40) + 40LL) + 16LL * *(unsigned int *)(v27 + 48));
  v31 = (const void **)*((_QWORD *)v30 + 1);
  v32 = *v30;
  v114 = v28;
  v89 = v31;
  if ( v28 )
    sub_1623A60((__int64)&v114, v28, 2);
  v33 = *a1;
  v115 = *(_DWORD *)(a2 + 64);
  v34 = *(unsigned __int16 *)(a5 + 24);
  if ( v34 == 10 || v34 == 32 )
  {
    v43 = *(_QWORD *)(a5 + 88);
    v44 = *(_QWORD **)(v43 + 24);
    if ( *(_DWORD *)(v43 + 32) > 0x40u )
      v44 = (_QWORD *)*v44;
    if ( v113.m128i_i8[0] )
    {
      v45 = sub_1F6C8D0(v113.m128i_i8[0]);
    }
    else
    {
      v103 = v33;
      v45 = sub_1F58D40((__int64)&v113);
      v46 = (__int64)v103;
    }
    v47 = (unsigned int)(v45 * (_DWORD)v44) >> 3;
    *(_QWORD *)&v96 = sub_1D38BB0(v46, v47, (__int64)&v114, v32, v89, 0, v24, *(double *)v29.m128i_i64, a9, 0);
    *((_QWORD *)&v96 + 1) = v48;
    v49 = *(__int64 **)(a10 + 104);
    v50 = *v49;
    v51 = *v49 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v51 )
    {
      v82 = *((_BYTE *)v49 + 16);
      v53 = v49[1] + v47;
      v54 = *v49 & 0xFFFFFFFFFFFFFFF8LL;
      v93 = v82;
      if ( (v50 & 4) != 0 )
      {
        v52 = *(_DWORD *)(v51 + 12);
        v54 = v51 | 4;
      }
      else
      {
        v83 = *(_QWORD *)v51;
        if ( *(_BYTE *)(*(_QWORD *)v51 + 8LL) == 16 )
          v83 = **(_QWORD **)(v83 + 16);
        v52 = *(_DWORD *)(v83 + 8) >> 8;
      }
    }
    else
    {
      v52 = *((_DWORD *)v49 + 5);
      v53 = 0;
      v54 = 0;
    }
    *((_QWORD *)&v116 + 1) = v53;
    *(_QWORD *)&v116 = v54;
    LOBYTE(v117) = v93;
    HIDWORD(v117) = v52;
  }
  else
  {
    v35 = sub_1D323C0(
            v33,
            a5,
            a6,
            (__int64)&v114,
            v32,
            v31,
            *(double *)v24.m128i_i64,
            *(double *)v29.m128i_i64,
            *(double *)a9.m128i_i64);
    v36 = *a1;
    v95 = v35;
    v100 = v37;
    if ( v113.m128i_i8[0] )
      v38 = sub_1F6C8D0(v113.m128i_i8[0]);
    else
      v38 = sub_1F58D40((__int64)&v113);
    *(_QWORD *)&v39 = sub_1D38BB0(
                        (__int64)v36,
                        (unsigned int)(v38 + 7) >> 3,
                        (__int64)&v114,
                        v32,
                        v89,
                        0,
                        v24,
                        *(double *)v29.m128i_i64,
                        a9,
                        0);
    *(_QWORD *)&v96 = sub_1D332F0(
                        v36,
                        54,
                        (__int64)&v114,
                        v32,
                        v89,
                        0,
                        *(double *)v24.m128i_i64,
                        *(double *)v29.m128i_i64,
                        a9,
                        v95,
                        v100,
                        v39);
    *((_QWORD *)&v96 + 1) = v40 | v100 & 0xFFFFFFFF00000000LL;
    v41 = *(_QWORD *)(*(_QWORD *)(a10 + 104) + 16LL);
    v116 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a10 + 104));
    v117 = v41;
  }
  v109 = sub_1D332F0(
           *a1,
           52,
           (__int64)&v114,
           v32,
           v89,
           0,
           *(double *)v24.m128i_i64,
           *(double *)v29.m128i_i64,
           a9,
           v29.m128i_i64[0],
           v29.m128i_u64[1],
           v96);
  v55 = _mm_loadu_si128(&v113);
  v57 = v113.m128i_i8[0];
  v126 = v55;
  v90 = v56 | v29.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  if ( v104 == v113.m128i_i8[0] )
  {
    if ( v104 || v92 == (const void **)v126.m128i_i64[1] )
      goto LABEL_49;
  }
  else if ( v104 )
  {
    v73 = sub_1F6C8D0(v104);
    goto LABEL_64;
  }
  v73 = sub_1F58D40((__int64)&v111);
LABEL_64:
  if ( v57 )
    v74 = sub_1F6C8D0(v57);
  else
    v74 = sub_1F58D40((__int64)&v126);
  if ( v74 < v73 )
  {
    if ( v113.m128i_i8[0] && v104 )
      v75 = (int)*((unsigned __int16 *)a1[1] + 115 * v104 + v113.m128i_u8[0] + 16104) >> 12 == 0 ? 3 : 1;
    else
      v75 = 1;
    v76 = *(_QWORD *)(a2 + 72);
    v77 = *a1;
    v78 = *(_QWORD *)(a10 + 104);
    v126 = _mm_loadu_si128((const __m128i *)(v78 + 40));
    v127 = *(_QWORD *)(v78 + 56);
    v79 = *(_WORD *)(v78 + 32);
    v122 = v76;
    v80 = *(__int128 **)(a10 + 32);
    if ( v76 )
    {
      v99 = v75;
      v105 = v79;
      sub_1623A60((__int64)&v122, v76, 2);
      v75 = v99;
      v79 = v105;
    }
    v123 = *(_DWORD *)(a2 + 64);
    v63 = sub_1D2B810(
            v77,
            v75,
            (__int64)&v122,
            v111,
            (__int64)v112,
            v94,
            *v80,
            (__int64)v109,
            v90,
            v116,
            v117,
            v113.m128i_i64[0],
            v113.m128i_i64[1],
            v79,
            (__int64)&v126);
    v66 = v63;
    v68 = v81;
    if ( v122 )
    {
      v106 = v81;
      sub_161E7C0((__int64)&v122, v122);
      v68 = v106;
    }
    goto LABEL_56;
  }
LABEL_49:
  v58 = *(_QWORD *)(a2 + 72);
  v59 = *a1;
  v60 = *(_QWORD *)(a10 + 104);
  v126 = _mm_loadu_si128((const __m128i *)(v60 + 40));
  v127 = *(_QWORD *)(v60 + 56);
  v61 = *(_WORD *)(v60 + 32);
  v122 = v58;
  v62 = *(__int64 **)(a10 + 32);
  if ( v58 )
  {
    v98 = v59;
    sub_1623A60((__int64)&v122, v58, 2);
    v59 = v98;
  }
  v123 = *(_DWORD *)(a2 + 64);
  v63 = sub_1D2B730(
          v59,
          v113.m128i_u32[0],
          v113.m128i_i64[1],
          (__int64)&v122,
          *v62,
          v62[1],
          (__int64)v109,
          v90,
          v116,
          v117,
          v94,
          v61,
          (__int64)&v126,
          0);
  v101 = v64;
  if ( v122 )
    sub_161E7C0((__int64)&v122, v122);
  v65 = v113.m128i_i8[0];
  v126 = _mm_loadu_si128(&v113);
  if ( v104 != v113.m128i_i8[0] )
  {
    if ( v104 )
    {
      v107 = sub_1F6C8D0(v104);
      goto LABEL_85;
    }
LABEL_95:
    v107 = sub_1F58D40((__int64)&v111);
LABEL_85:
    if ( v65 )
      v84 = sub_1F6C8D0(v65);
    else
      v84 = sub_1F58D40((__int64)&v126);
    v85 = *a1;
    if ( v84 > v107 )
    {
      v86 = *(_QWORD *)(a2 + 72);
      v126.m128i_i64[0] = v86;
      if ( v86 )
        sub_1623A60((__int64)&v126, v86, 2);
      *((_QWORD *)&v88 + 1) = v101;
      *(_QWORD *)&v88 = v63;
      v126.m128i_i32[2] = *(_DWORD *)(a2 + 64);
      v66 = sub_1D309E0(
              v85,
              145,
              (__int64)&v126,
              v111,
              v112,
              0,
              *(double *)v24.m128i_i64,
              *(double *)v29.m128i_i64,
              *(double *)v55.m128i_i64,
              v88);
      v68 = v87;
      if ( v126.m128i_i64[0] )
      {
        v108 = v87;
        sub_161E7C0((__int64)&v126, v126.m128i_i64[0]);
        v68 = v108;
      }
      goto LABEL_56;
    }
    goto LABEL_55;
  }
  if ( !v113.m128i_i8[0] && v92 != (const void **)v126.m128i_i64[1] )
    goto LABEL_95;
LABEL_55:
  v66 = sub_1D32840(
          *a1,
          v111,
          v112,
          v63,
          v101,
          *(double *)v24.m128i_i64,
          *(double *)v29.m128i_i64,
          *(double *)v55.m128i_i64);
  v68 = v67;
LABEL_56:
  v69 = *a1;
  v121 = a1;
  v70 = v69[83];
  v120 = v69;
  v119 = v70;
  v69[83] = (__int64)&v118;
  v71 = *a1;
  v118 = off_49FFF30;
  v126.m128i_i32[2] = v68;
  v124 = a10;
  v127 = v63;
  v122 = a2;
  v123 = 0;
  v125 = 1;
  v126.m128i_i64[0] = v66;
  v128 = 1;
  sub_1D451D0((__int64)v71, &v122, v126.m128i_i64, 2);
  sub_1F81BC0((__int64)a1, v66);
  for ( i = *(_QWORD *)(v66 + 48); i; i = *(_QWORD *)(i + 32) )
    sub_1F81BC0((__int64)a1, *(_QWORD *)(i + 16));
  sub_1F81BC0((__int64)a1, a2);
  result = a2;
  v120[83] = v119;
  if ( v114 )
  {
    sub_161E7C0((__int64)&v114, v114);
    return a2;
  }
  return result;
}
