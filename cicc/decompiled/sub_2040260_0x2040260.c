// Function: sub_2040260
// Address: 0x2040260
//
__int64 __fastcall sub_2040260(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned __int64 *v7; // rax
  __int64 v8; // rbx
  unsigned __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int8 v13; // dl
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rsi
  int v18; // r13d
  char v19; // r8
  unsigned int v20; // eax
  char v21; // cl
  int v22; // eax
  int v23; // edx
  __int64 v24; // rcx
  int v25; // r9d
  unsigned __int8 v26; // r13
  const void **v27; // rdx
  const void **v28; // r8
  unsigned int v29; // eax
  int v30; // eax
  int v31; // edx
  __int64 v32; // r14
  _DWORD *v34; // r13
  char v35; // al
  __int64 v36; // r9
  __int64 v37; // rdx
  unsigned int v38; // ecx
  __int64 v39; // rdi
  int v40; // r11d
  __int64 v41; // rax
  unsigned __int8 *v42; // rax
  const void **v43; // rax
  __int64 *v44; // rdi
  __int64 v45; // r14
  unsigned __int64 v46; // r15
  unsigned int v47; // edx
  unsigned __int8 *v48; // rax
  const void **v49; // rax
  char v50; // r8
  int v51; // eax
  int v52; // eax
  __int64 v53; // rdx
  __int64 v54; // r13
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  unsigned int v58; // eax
  const void **v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  _QWORD *v63; // rax
  int v64; // edx
  const void **v65; // r8
  _QWORD *v66; // rsi
  unsigned __int64 v67; // rax
  __int64 v68; // rcx
  __int64 v69; // rax
  unsigned __int64 v70; // r9
  __int64 *v71; // r14
  __int64 v72; // rdx
  bool v73; // al
  int v74; // eax
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  int v79; // eax
  unsigned int v80; // edx
  unsigned __int64 v81; // r8
  int v82; // ecx
  unsigned int v83; // edi
  __int128 v84; // [rsp-10h] [rbp-1F0h]
  __int128 v85; // [rsp-10h] [rbp-1F0h]
  __int128 v86; // [rsp-10h] [rbp-1F0h]
  __int128 v87; // [rsp-10h] [rbp-1F0h]
  int v88; // [rsp+0h] [rbp-1E0h]
  int v89; // [rsp+0h] [rbp-1E0h]
  __int64 v90; // [rsp+8h] [rbp-1D8h]
  __int64 v91; // [rsp+8h] [rbp-1D8h]
  int v92; // [rsp+10h] [rbp-1D0h]
  int v93; // [rsp+10h] [rbp-1D0h]
  unsigned int v94; // [rsp+10h] [rbp-1D0h]
  __int64 v95; // [rsp+10h] [rbp-1D0h]
  int v96; // [rsp+10h] [rbp-1D0h]
  __int64 v97; // [rsp+18h] [rbp-1C8h]
  __int64 v98; // [rsp+18h] [rbp-1C8h]
  unsigned int v99; // [rsp+20h] [rbp-1C0h]
  __int64 v100; // [rsp+20h] [rbp-1C0h]
  const void **v101; // [rsp+20h] [rbp-1C0h]
  int v102; // [rsp+20h] [rbp-1C0h]
  unsigned int v103; // [rsp+28h] [rbp-1B8h]
  int v104; // [rsp+28h] [rbp-1B8h]
  int *v105; // [rsp+28h] [rbp-1B8h]
  __int64 v106; // [rsp+30h] [rbp-1B0h]
  __m128i v107; // [rsp+60h] [rbp-180h] BYREF
  unsigned int v108; // [rsp+70h] [rbp-170h] BYREF
  const void **v109; // [rsp+78h] [rbp-168h]
  __int64 v110; // [rsp+80h] [rbp-160h] BYREF
  int v111; // [rsp+88h] [rbp-158h]
  __m128i v112; // [rsp+90h] [rbp-150h] BYREF
  __m128 v113; // [rsp+A0h] [rbp-140h] BYREF
  _QWORD v114[38]; // [rsp+B0h] [rbp-130h] BYREF

  v7 = *(unsigned __int64 **)(a2 + 32);
  v8 = *v7;
  v9 = *v7;
  v10 = v7[1];
  v11 = *((unsigned int *)v7 + 2);
  v106 = v8;
  v103 = v11;
  v12 = *(_QWORD *)(v8 + 40) + 16 * v11;
  v13 = *(_BYTE *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v107.m128i_i8[0] = v13;
  v15 = *(_QWORD *)(a1 + 8);
  v107.m128i_i64[1] = v14;
  sub_1F40D10(
    (__int64)&v113,
    *(_QWORD *)a1,
    *(_QWORD *)(v15 + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v16 = *(_QWORD *)(a2 + 72);
  LOBYTE(v108) = v113.m128_i8[8];
  v110 = v16;
  v109 = (const void **)v114[0];
  if ( v16 )
    sub_1623A60((__int64)&v110, v16, 2);
  v17 = *(_QWORD *)a1;
  v111 = *(_DWORD *)(a2 + 64);
  sub_1F40D10((__int64)&v113, v17, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v107.m128i_u8[0], v107.m128i_i64[1]);
  if ( v113.m128_i8[0] == 1 )
  {
    v18 = v107.m128i_u8[0];
    if ( v107.m128i_i8[0] )
    {
      if ( (unsigned __int8)(v107.m128i_i8[0] - 14) <= 0x5Fu )
        goto LABEL_6;
    }
    else if ( sub_1F58D20((__int64)&v107) )
    {
      goto LABEL_6;
    }
    v113.m128_i32[0] = sub_200F8F0(a1, v9, v10);
    v34 = sub_20322E0(a1 + 552, &v113);
    sub_200D1B0(a1, v34 + 1);
    v35 = *(_BYTE *)(a1 + 352) & 1;
    if ( v35 )
    {
      v36 = a1 + 360;
      v17 = 7;
    }
    else
    {
      v17 = *(unsigned int *)(a1 + 368);
      v36 = *(_QWORD *)(a1 + 360);
      if ( !(_DWORD)v17 )
      {
        v80 = *(_DWORD *)(a1 + 352);
        v81 = 0;
        ++*(_QWORD *)(a1 + 344);
        v82 = (v80 >> 1) + 1;
        goto LABEL_62;
      }
      v17 = (unsigned int)(v17 - 1);
    }
    v37 = (unsigned int)v34[1];
    v38 = v17 & (37 * v37);
    v39 = v36 + 24LL * v38;
    v40 = *(_DWORD *)v39;
    if ( *(_DWORD *)v39 == (_DWORD)v37 )
    {
LABEL_24:
      v106 = *(_QWORD *)(v39 + 8);
      v103 = *(_DWORD *)(v39 + 16);
      v41 = 16LL * v103;
      goto LABEL_25;
    }
    v104 = 1;
    v81 = 0;
    while ( v40 != -1 )
    {
      if ( v40 == -2 && !v81 )
        v81 = v39;
      v38 = v17 & (v104 + v38);
      v39 = v36 + 24LL * v38;
      v40 = *(_DWORD *)v39;
      if ( (_DWORD)v37 == *(_DWORD *)v39 )
        goto LABEL_24;
      ++v104;
    }
    v80 = *(_DWORD *)(a1 + 352);
    v17 = 8;
    if ( !v81 )
      v81 = v39;
    v83 = 24;
    ++*(_QWORD *)(a1 + 344);
    v82 = (v80 >> 1) + 1;
    if ( v35 )
    {
LABEL_63:
      v36 = a1 + 344;
      if ( v83 <= 4 * v82 )
      {
        v105 = v34 + 1;
        LODWORD(v17) = 2 * v17;
      }
      else
      {
        if ( (int)v17 - *(_DWORD *)(a1 + 356) - v82 > (unsigned int)v17 >> 3 )
        {
LABEL_65:
          v37 = (2 * (v80 >> 1) + 2) | v80 & 1;
          *(_DWORD *)(a1 + 352) = v37;
          if ( *(_DWORD *)v81 != -1 )
            --*(_DWORD *)(a1 + 356);
          v103 = 0;
          v106 = 0;
          *(_DWORD *)v81 = v34[1];
          v41 = 0;
          *(_QWORD *)(v81 + 8) = 0;
          *(_DWORD *)(v81 + 16) = 0;
LABEL_25:
          v42 = (unsigned __int8 *)(*(_QWORD *)(v106 + 40) + v41);
          v18 = *v42;
          v43 = (const void **)*((_QWORD *)v42 + 1);
          v107.m128i_i8[0] = v18;
          v107.m128i_i64[1] = (__int64)v43;
          a4 = _mm_loadu_si128(&v107);
          v113 = (__m128)a4;
          if ( (_BYTE)v18 == (_BYTE)v108 && ((_BYTE)v18 || v43 == v109)
            || (LOBYTE(v97) = v108,
                v102 = sub_1D159A0((char *)&v108, v17, v37, v106, (unsigned __int8)v108, v36, v88, v90, v92, v97),
                v79 = sub_1D159A0((char *)&v113, v17, v75, v76, v77, v78, v89, v91, v96, v98),
                v19 = v97,
                v102 == v79) )
          {
            v44 = *(__int64 **)(a1 + 8);
            v45 = v106;
            v46 = v103 | v10 & 0xFFFFFFFF00000000LL;
LABEL_37:
            *((_QWORD *)&v84 + 1) = v46;
            *(_QWORD *)&v84 = v45;
            v32 = sub_1D309E0(
                    v44,
                    158,
                    (__int64)&v110,
                    v108,
                    v109,
                    0,
                    *(double *)a3.m128i_i64,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128i_i64,
                    v84);
            goto LABEL_16;
          }
          goto LABEL_7;
        }
        v105 = v34 + 1;
      }
      sub_200F500(a1 + 344, v17);
      v17 = (__int64)v105;
      sub_2032230(a1 + 344, v105, &v113);
      v81 = v113.m128_u64[0];
      v80 = *(_DWORD *)(a1 + 352);
      goto LABEL_65;
    }
    v17 = *(unsigned int *)(a1 + 368);
LABEL_62:
    v83 = 3 * v17;
    goto LABEL_63;
  }
  if ( v113.m128_i8[0] != 7 )
  {
    v18 = v107.m128i_u8[0];
LABEL_6:
    v19 = v108;
    goto LABEL_7;
  }
  v17 = v9;
  v100 = sub_20363F0(a1, v9, v10);
  v106 = v100;
  v97 = v47;
  v48 = (unsigned __int8 *)(*(_QWORD *)(v100 + 40) + 16LL * v47);
  v18 = *v48;
  v49 = (const void **)*((_QWORD *)v48 + 1);
  v103 = v47;
  v107.m128i_i8[0] = v18;
  v107.m128i_i64[1] = (__int64)v49;
  a3 = _mm_loadu_si128(&v107);
  v113 = (__m128)a3;
  if ( (_BYTE)v18 == (_BYTE)v108 )
  {
    if ( (_BYTE)v18 || v49 == v109 )
    {
LABEL_36:
      v44 = *(__int64 **)(a1 + 8);
      v45 = v100;
      v46 = v10 & 0xFFFFFFFF00000000LL | v97;
      goto LABEL_37;
    }
  }
  else if ( (_BYTE)v108 )
  {
    v93 = sub_2021900(v108);
    goto LABEL_31;
  }
  LOBYTE(v90) = v108;
  v74 = sub_1F58D40((__int64)&v108);
  v50 = v90;
  v93 = v74;
LABEL_31:
  if ( (_BYTE)v18 )
  {
    v51 = sub_2021900(v18);
  }
  else
  {
    LOBYTE(v90) = v50;
    v51 = sub_1F58D40((__int64)&v113);
    v19 = v90;
  }
  if ( v51 == v93 )
    goto LABEL_36;
LABEL_7:
  if ( v19 )
  {
    v99 = sub_2021900(v19);
    if ( (_BYTE)v18 )
      goto LABEL_9;
LABEL_14:
    v97 = (__int64)&v107;
    v29 = sub_1F58D40((__int64)&v107);
    v31 = v99 % v29;
    v30 = v99 / v29;
    if ( v31 )
    {
LABEL_15:
      v32 = sub_200D7B0(a1, v106, v103 | v10 & 0xFFFFFFFF00000000LL, v108, (__int64)v109);
      goto LABEL_16;
    }
    LODWORD(v97) = v30;
    if ( !sub_1F58D20((__int64)&v107) )
      goto LABEL_12;
LABEL_41:
    LOBYTE(v52) = sub_1F7E0F0((__int64)&v107);
    v54 = v53;
    v113.m128_u64[1] = v53;
    v113.m128_i32[0] = v52;
    v58 = sub_1D159A0((char *)&v113, v17, v53, v55, v56, v57, v88, v90, v52, v97);
    v24 = sub_1F7DEB0(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL), v94, v54, v99 / v58, 0);
    v26 = v24;
    v28 = v59;
    goto LABEL_42;
  }
  v99 = sub_1F58D40((__int64)&v108);
  if ( !(_BYTE)v18 )
    goto LABEL_14;
LABEL_9:
  v20 = sub_2021900(v18);
  v23 = v99 % v20;
  v22 = v99 / v20;
  if ( v23 || !v21 )
    goto LABEL_15;
  v17 = (unsigned int)(v18 - 14);
  LODWORD(v97) = v22;
  if ( (unsigned __int8)(v18 - 14) <= 0x5Fu )
    goto LABEL_41;
LABEL_12:
  v24 = sub_1F7DEB0(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL), v107.m128i_u32[0], v107.m128i_i64[1], v97, 0);
  v26 = v24;
  v28 = v27;
LABEL_42:
  v95 = v24;
  v101 = v28;
  if ( !v26 || !*(_QWORD *)(*(_QWORD *)a1 + 8LL * v26 + 120) )
    goto LABEL_15;
  v113.m128_u64[0] = (unsigned __int64)v114;
  v113.m128_u64[1] = 0x1000000000LL;
  v112.m128i_i64[0] = 0;
  v112.m128i_i32[2] = 0;
  sub_202F910((__int64)&v113, (unsigned int)v97, &v112, v24, (int)v28, v25);
  v63 = sub_1D2B530(*(_QWORD **)(a1 + 8), v107.m128i_u32[0], v107.m128i_i64[1], v60, v61, v62);
  v65 = v101;
  v66 = v63;
  v67 = v113.m128_u64[0];
  v68 = v95;
  *(_QWORD *)v113.m128_u64[0] = v106;
  *(_DWORD *)(v67 + 8) = v103;
  v69 = 16;
  if ( (unsigned int)v97 > 1 )
  {
    do
    {
      v70 = v113.m128_u64[0];
      *(_QWORD *)(v113.m128_u64[0] + v69) = v66;
      *(_DWORD *)(v70 + v69 + 8) = v64;
      v69 += 16;
    }
    while ( 16LL * (unsigned int)v97 != v69 );
  }
  if ( v107.m128i_i8[0] )
  {
    if ( (unsigned __int8)(v107.m128i_i8[0] - 14) > 0x5Fu )
    {
LABEL_48:
      LOBYTE(v68) = v26;
      *((_QWORD *)&v85 + 1) = v113.m128_u32[2];
      *(_QWORD *)&v85 = v113.m128_u64[0];
      v71 = sub_1D359D0(
              *(__int64 **)(a1 + 8),
              104,
              (__int64)&v110,
              v68,
              v65,
              0,
              *(double *)a3.m128i_i64,
              *(double *)a4.m128i_i64,
              a5,
              v85);
      v72 = (unsigned int)v72;
      goto LABEL_49;
    }
  }
  else
  {
    v73 = sub_1F58D20((__int64)&v107);
    v65 = v101;
    v68 = v95;
    if ( !v73 )
      goto LABEL_48;
  }
  LOBYTE(v68) = v26;
  *((_QWORD *)&v87 + 1) = v113.m128_u32[2];
  *(_QWORD *)&v87 = v113.m128_u64[0];
  v71 = sub_1D359D0(
          *(__int64 **)(a1 + 8),
          107,
          (__int64)&v110,
          v68,
          v65,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          v87);
  v72 = (unsigned int)v72;
LABEL_49:
  *((_QWORD *)&v86 + 1) = v72;
  *(_QWORD *)&v86 = v71;
  v32 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          158,
          (__int64)&v110,
          v108,
          v109,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          v86);
  if ( (_QWORD *)v113.m128_u64[0] != v114 )
    _libc_free(v113.m128_u64[0]);
LABEL_16:
  if ( v110 )
    sub_161E7C0((__int64)&v110, v110);
  return v32;
}
