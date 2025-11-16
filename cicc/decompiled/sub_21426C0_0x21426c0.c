// Function: sub_21426C0
// Address: 0x21426c0
//
__int64 __fastcall sub_21426C0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  const __m128i *v7; // rax
  __m128i v8; // xmm1
  unsigned __int8 *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rsi
  char *v12; // rax
  unsigned __int8 v13; // di
  __int64 v14; // r8
  __int64 v15; // rsi
  __int64 v16; // rsi
  char v17; // bl
  __int64 *v18; // r12
  __int128 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 *v23; // r12
  __int128 v24; // rax
  char v25; // r15
  __int64 *v26; // r12
  __int128 v27; // rax
  __int64 *v28; // r15
  int v29; // eax
  char v30; // cl
  __int64 v31; // rdx
  int v32; // esi
  int v33; // r11d
  unsigned int v34; // edi
  int *v35; // rbx
  int v36; // r8d
  __int64 v37; // rax
  char v38; // al
  unsigned int v39; // esi
  __int64 v40; // r9
  int v41; // esi
  int v42; // edx
  unsigned int v43; // ecx
  __int64 v44; // rdi
  int v45; // r11d
  __int128 v46; // rax
  unsigned __int64 *v47; // rax
  __int64 v48; // rdx
  __int32 v49; // edx
  __int32 v50; // edx
  __int64 *v51; // r12
  __int64 v52; // rdx
  unsigned int v53; // eax
  const void **v54; // r8
  unsigned int v55; // eax
  unsigned int v56; // edx
  _DWORD *v57; // rax
  __int64 v58; // rax
  char v59; // bl
  const void **v60; // rdx
  unsigned int v61; // esi
  int v62; // ebx
  int v63; // eax
  int v64; // r15d
  int v65; // eax
  __int64 v66; // rbx
  unsigned int v67; // edx
  __int64 v68; // r8
  int v69; // ecx
  unsigned int v70; // r9d
  unsigned int v71; // [rsp+0h] [rbp-120h]
  const void **v72; // [rsp+8h] [rbp-118h]
  __int64 v73; // [rsp+10h] [rbp-110h]
  __int64 v74; // [rsp+18h] [rbp-108h]
  __int64 v75; // [rsp+20h] [rbp-100h]
  char v76; // [rsp+2Eh] [rbp-F2h]
  unsigned __int8 v77; // [rsp+30h] [rbp-F0h]
  __int128 v78; // [rsp+30h] [rbp-F0h]
  int *v79; // [rsp+30h] [rbp-F0h]
  __int64 v80; // [rsp+40h] [rbp-E0h]
  int v81; // [rsp+40h] [rbp-E0h]
  __m128i v82; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v83; // [rsp+A0h] [rbp-80h] BYREF
  const void **v84; // [rsp+A8h] [rbp-78h]
  __int64 v85; // [rsp+B0h] [rbp-70h] BYREF
  int v86; // [rsp+B8h] [rbp-68h]
  __m128i v87; // [rsp+C0h] [rbp-60h] BYREF
  __m128i v88; // [rsp+D0h] [rbp-50h] BYREF
  const void **v89; // [rsp+E0h] [rbp-40h]

  v7 = *(const __m128i **)(a2 + 32);
  v8 = _mm_loadu_si128(v7);
  v74 = v7->m128i_i64[0];
  v73 = v7->m128i_u32[2];
  v9 = (unsigned __int8 *)(*(_QWORD *)(v7->m128i_i64[0] + 40) + 16 * v73);
  v10 = *((_QWORD *)v9 + 1);
  v77 = *v9;
  sub_1F40D10((__int64)&v88, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v9, v10);
  v11 = *(_QWORD *)a1;
  v76 = v88.m128i_i8[8];
  v82.m128i_i8[0] = v88.m128i_i8[8];
  v12 = *(char **)(a2 + 40);
  v72 = v89;
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v82.m128i_i64[1] = (__int64)v89;
  v75 = v14;
  sub_1F40D10((__int64)&v88, v11, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v13, v14);
  v15 = *(_QWORD *)(a2 + 72);
  LOBYTE(v83) = v88.m128i_i8[8];
  v85 = v15;
  v84 = v89;
  if ( v15 )
    sub_1623A60((__int64)&v85, v15, 2);
  v16 = *(_QWORD *)a1;
  v86 = *(_DWORD *)(a2 + 64);
  sub_1F40D10((__int64)&v88, v16, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v77, v10);
  v17 = v88.m128i_i8[0];
  switch ( v88.m128i_i8[0] )
  {
    case 1:
      a5 = _mm_loadu_si128(&v82);
      v25 = v83;
      v88 = a5;
      if ( v76 == (_BYTE)v83 )
      {
        if ( v76 )
          goto LABEL_15;
        if ( v72 == v84 )
        {
          if ( sub_1F58D20((__int64)&v83) )
            goto LABEL_5;
LABEL_82:
          if ( !sub_1F58D20((__int64)&v82) )
          {
LABEL_18:
            v26 = *(__int64 **)(a1 + 8);
            *(_QWORD *)&v27 = sub_2138AD0(a1, v8.m128i_u64[0], v8.m128i_i64[1]);
            goto LABEL_49;
          }
          goto LABEL_5;
        }
      }
      else if ( (_BYTE)v83 )
      {
        v62 = sub_2127930(v83);
        goto LABEL_62;
      }
      v62 = sub_1F58D40((__int64)&v83);
LABEL_62:
      if ( v76 )
        v63 = sub_2127930(v76);
      else
        v63 = sub_1F58D40((__int64)&v88);
      if ( v63 != v62 )
        goto LABEL_5;
      if ( !v25 )
      {
        if ( sub_1F58D20((__int64)&v83) )
          goto LABEL_5;
        goto LABEL_16;
      }
LABEL_15:
      if ( (unsigned __int8)(v25 - 14) <= 0x5Fu )
        goto LABEL_5;
LABEL_16:
      if ( v76 )
      {
        if ( (unsigned __int8)(v76 - 14) <= 0x5Fu )
          goto LABEL_5;
        goto LABEL_18;
      }
      goto LABEL_82;
    case 3:
      v28 = *(__int64 **)(a1 + 8);
      v29 = sub_200F8F0(a1, v8.m128i_u64[0], v8.m128i_i64[1]);
      v30 = *(_BYTE *)(a1 + 752) & 1;
      if ( v30 )
      {
        v31 = a1 + 760;
        v32 = 7;
      }
      else
      {
        v61 = *(_DWORD *)(a1 + 768);
        v31 = *(_QWORD *)(a1 + 760);
        if ( !v61 )
          goto LABEL_94;
        v32 = v61 - 1;
      }
      v33 = 1;
      v34 = v32 & (37 * v29);
      v35 = (int *)(v31 + 8LL * v34);
      v36 = *v35;
      if ( v29 == *v35 )
        goto LABEL_22;
      while ( v36 != -1 )
      {
        v34 = v32 & (v33 + v34);
        v35 = (int *)(v31 + 8LL * v34);
        v36 = *v35;
        if ( v29 == *v35 )
          goto LABEL_22;
        ++v33;
      }
      if ( v30 )
      {
        v66 = 64;
        goto LABEL_89;
      }
      v61 = *(_DWORD *)(a1 + 768);
LABEL_94:
      v66 = 8LL * v61;
LABEL_89:
      v35 = (int *)(v31 + v66);
LABEL_22:
      v37 = 64;
      if ( !v30 )
        v37 = 8LL * *(unsigned int *)(a1 + 768);
      if ( v35 == (int *)(v37 + v31) )
        goto LABEL_30;
      sub_200D1B0(a1, v35 + 1);
      v38 = *(_BYTE *)(a1 + 352) & 1;
      if ( v38 )
      {
        v40 = a1 + 360;
        v41 = 7;
      }
      else
      {
        v39 = *(_DWORD *)(a1 + 368);
        v40 = *(_QWORD *)(a1 + 360);
        if ( !v39 )
        {
          v67 = *(_DWORD *)(a1 + 352);
          ++*(_QWORD *)(a1 + 344);
          v68 = 0;
          v69 = (v67 >> 1) + 1;
LABEL_97:
          v70 = 3 * v39;
          goto LABEL_98;
        }
        v41 = v39 - 1;
      }
      v42 = v35[1];
      v43 = v41 & (37 * v42);
      v44 = v40 + 24LL * v43;
      v45 = *(_DWORD *)v44;
      if ( v42 == *(_DWORD *)v44 )
      {
LABEL_29:
        v74 = *(_QWORD *)(v44 + 8);
        v73 = *(unsigned int *)(v44 + 16);
        goto LABEL_30;
      }
      v81 = 1;
      v68 = 0;
      while ( v45 != -1 )
      {
        if ( v45 == -2 && !v68 )
          v68 = v44;
        v43 = v41 & (v81 + v43);
        v44 = v40 + 24LL * v43;
        v45 = *(_DWORD *)v44;
        if ( v42 == *(_DWORD *)v44 )
          goto LABEL_29;
        ++v81;
      }
      v67 = *(_DWORD *)(a1 + 352);
      v70 = 24;
      v39 = 8;
      if ( !v68 )
        v68 = v44;
      ++*(_QWORD *)(a1 + 344);
      v69 = (v67 >> 1) + 1;
      if ( !v38 )
      {
        v39 = *(_DWORD *)(a1 + 368);
        goto LABEL_97;
      }
LABEL_98:
      if ( 4 * v69 >= v70 )
      {
        v79 = v35 + 1;
        v39 *= 2;
      }
      else
      {
        if ( v39 - *(_DWORD *)(a1 + 356) - v69 > v39 >> 3 )
          goto LABEL_100;
        v79 = v35 + 1;
      }
      sub_200F500(a1 + 344, v39);
      sub_2032230(a1 + 344, v79, &v88);
      v68 = v88.m128i_i64[0];
      v67 = *(_DWORD *)(a1 + 352);
LABEL_100:
      *(_DWORD *)(a1 + 352) = (2 * (v67 >> 1) + 2) | v67 & 1;
      if ( *(_DWORD *)v68 != -1 )
        --*(_DWORD *)(a1 + 356);
      v74 = 0;
      v73 = 0;
      *(_DWORD *)v68 = v35[1];
      *(_QWORD *)(v68 + 8) = 0;
      *(_DWORD *)(v68 + 16) = 0;
LABEL_30:
      *(_QWORD *)&v46 = v74;
      *((_QWORD *)&v46 + 1) = v73;
LABEL_31:
      v21 = sub_1D309E0(
              v28,
              144,
              (__int64)&v85,
              v83,
              v84,
              0,
              *(double *)a3.m128i_i64,
              *(double *)v8.m128i_i64,
              *(double *)a5.m128i_i64,
              v46);
      goto LABEL_7;
    case 5:
      if ( (_BYTE)v83 )
      {
        if ( (unsigned __int8)(v83 - 14) <= 0x5Fu )
          goto LABEL_5;
      }
      else if ( sub_1F58D20((__int64)&v83) )
      {
        goto LABEL_5;
      }
      v28 = *(__int64 **)(a1 + 8);
      v88.m128i_i32[0] = sub_200F8F0(a1, v8.m128i_u64[0], v8.m128i_i64[1]);
      v57 = sub_20322E0(a1 + 1016, &v88);
      v58 = sub_21387B0(a1, v57 + 1);
      *(_QWORD *)&v46 = sub_200D2A0(
                          a1,
                          *(_QWORD *)v58,
                          *(unsigned int *)(v58 + 8),
                          *(double *)a3.m128i_i64,
                          *(double *)v8.m128i_i64,
                          *(double *)a5.m128i_i64);
      goto LABEL_31;
    case 6:
      v47 = *(unsigned __int64 **)(a2 + 32);
      v87.m128i_i32[2] = 0;
      v88.m128i_i64[0] = 0;
      v88.m128i_i32[2] = 0;
      v48 = v47[1];
      v87.m128i_i64[0] = 0;
      sub_2017DE0(a1, *v47, v48, &v87, &v88);
      v87.m128i_i64[0] = sub_200D2A0(
                           a1,
                           v87.m128i_i64[0],
                           v87.m128i_i64[1],
                           *(double *)a3.m128i_i64,
                           *(double *)v8.m128i_i64,
                           *(double *)a5.m128i_i64);
      v87.m128i_i32[2] = v49;
      v88.m128i_i64[0] = sub_200D2A0(
                           a1,
                           v88.m128i_i64[0],
                           v88.m128i_i64[1],
                           *(double *)a3.m128i_i64,
                           *(double *)v8.m128i_i64,
                           *(double *)a5.m128i_i64);
      v88.m128i_i32[2] = v50;
      if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL)) )
      {
        a3 = _mm_loadu_si128(&v87);
        v87.m128i_i64[0] = v88.m128i_i64[0];
        v87.m128i_i32[2] = v88.m128i_i32[2];
        v88.m128i_i64[0] = a3.m128i_i64[0];
        v88.m128i_i32[2] = a3.m128i_i32[2];
      }
      v51 = *(__int64 **)(a1 + 8);
      *(_QWORD *)&v78 = sub_200DAC0(
                          a1,
                          v87.m128i_i64[0],
                          v87.m128i_i64[1],
                          v88.m128i_i64[0],
                          v88.m128i_i64[1],
                          a3,
                          *(double *)v8.m128i_i64,
                          a5);
      *((_QWORD *)&v78 + 1) = v52;
      if ( (_BYTE)v83 )
        v53 = sub_2127930(v83);
      else
        v53 = sub_1F58D40((__int64)&v83);
      if ( v53 == 32 )
      {
        v17 = 5;
        goto LABEL_40;
      }
      if ( v53 > 0x20 )
      {
        if ( v53 != 64 )
        {
          if ( v53 != 128 )
            goto LABEL_51;
          v17 = 7;
        }
LABEL_40:
        v54 = 0;
        goto LABEL_41;
      }
      if ( v53 == 8 )
      {
        v17 = 3;
        goto LABEL_40;
      }
      v17 = 4;
      if ( v53 == 16 )
        goto LABEL_40;
      v17 = 2;
      if ( v53 == 1 )
        goto LABEL_40;
LABEL_51:
      v71 = sub_1F58CC0(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL), v53);
      v17 = v71;
      v54 = v60;
LABEL_41:
      v55 = v71;
      LOBYTE(v55) = v17;
      v80 = sub_1D309E0(
              v51,
              144,
              (__int64)&v85,
              v55,
              v54,
              0,
              *(double *)a3.m128i_i64,
              *(double *)v8.m128i_i64,
              *(double *)a5.m128i_i64,
              v78);
      v21 = sub_1D309E0(
              *(__int64 **)(a1 + 8),
              158,
              (__int64)&v85,
              v83,
              v84,
              0,
              *(double *)a3.m128i_i64,
              *(double *)v8.m128i_i64,
              *(double *)a5.m128i_i64,
              __PAIR128__(v56 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL, v80));
LABEL_7:
      if ( v85 )
        sub_161E7C0((__int64)&v85, v85);
      return v21;
    case 7:
      v59 = v83;
      v88 = _mm_loadu_si128(&v82);
      if ( v76 == (_BYTE)v83 )
      {
        if ( v76 )
        {
LABEL_47:
          if ( (unsigned __int8)(v59 - 14) <= 0x5Fu )
            goto LABEL_5;
          goto LABEL_48;
        }
        if ( v72 == v84 )
        {
LABEL_74:
          if ( sub_1F58D20((__int64)&v83) )
            goto LABEL_5;
LABEL_48:
          v26 = *(__int64 **)(a1 + 8);
          *(_QWORD *)&v27 = sub_20363F0(a1, v8.m128i_u64[0], v8.m128i_i64[1]);
LABEL_49:
          v20 = sub_1D309E0(
                  v26,
                  158,
                  (__int64)&v85,
                  v83,
                  v84,
                  0,
                  *(double *)a3.m128i_i64,
                  *(double *)v8.m128i_i64,
                  *(double *)a5.m128i_i64,
                  v27);
          goto LABEL_6;
        }
      }
      else if ( (_BYTE)v83 )
      {
        v64 = sub_2127930(v83);
        goto LABEL_70;
      }
      v64 = sub_1F58D40((__int64)&v83);
LABEL_70:
      if ( v76 )
        v65 = sub_2127930(v76);
      else
        v65 = sub_1F58D40((__int64)&v88);
      if ( v65 != v64 )
        goto LABEL_5;
      if ( v59 )
        goto LABEL_47;
      goto LABEL_74;
    case 8:
      if ( (_BYTE)v83 )
      {
        if ( (unsigned __int8)(v83 - 14) <= 0x5Fu )
          goto LABEL_5;
      }
      else if ( sub_1F58D20((__int64)&v83) )
      {
LABEL_5:
        v18 = *(__int64 **)(a1 + 8);
        *(_QWORD *)&v19 = sub_200D7B0(a1, v8.m128i_i64[0], v8.m128i_i64[1], v13, v75);
        v20 = sub_1D309E0(
                v18,
                144,
                (__int64)&v85,
                v83,
                v84,
                0,
                *(double *)a3.m128i_i64,
                *(double *)v8.m128i_i64,
                *(double *)a5.m128i_i64,
                v19);
LABEL_6:
        v21 = v20;
        goto LABEL_7;
      }
      v23 = *(__int64 **)(a1 + 8);
      *(_QWORD *)&v24 = sub_2125740(a1, v8.m128i_u64[0], v8.m128i_i64[1]);
      v20 = sub_1D309E0(
              v23,
              161,
              (__int64)&v85,
              v83,
              v84,
              0,
              *(double *)a3.m128i_i64,
              *(double *)v8.m128i_i64,
              *(double *)a5.m128i_i64,
              v24);
      goto LABEL_6;
    default:
      goto LABEL_5;
  }
}
