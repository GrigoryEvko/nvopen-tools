// Function: sub_1243EC0
// Address: 0x1243ec0
//
__int64 __fastcall sub_1243EC0(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        unsigned __int64 a4,
        int a5,
        int a6,
        int a7,
        char a8,
        char a9,
        char a10)
{
  __int64 v12; // rbx
  int v13; // eax
  __int64 v14; // r14
  unsigned int v15; // r15d
  unsigned int v16; // eax
  unsigned __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned int v21; // esi
  __int64 v22; // r12
  __int16 v23; // dx
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rcx
  unsigned int v31; // r10d
  const char *v32; // rax
  __int64 v34; // rdx
  void *v35; // r12
  void **v36; // rbx
  void *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rdi
  __int64 v41; // rax
  void **v42; // rcx
  void *v43; // r14
  void **v44; // rbx
  __int64 v45; // rbx
  __int64 v46; // rcx
  __int64 v47; // rax
  unsigned __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rcx
  __m128i *v51; // rax
  __int64 v52; // rcx
  __int64 v53; // rax
  unsigned __int64 v54; // [rsp+10h] [rbp-180h]
  __int64 v55; // [rsp+20h] [rbp-170h]
  unsigned int v56; // [rsp+28h] [rbp-168h]
  __int64 v57; // [rsp+28h] [rbp-168h]
  _DWORD *v58; // [rsp+28h] [rbp-168h]
  unsigned __int64 v59; // [rsp+30h] [rbp-160h]
  _QWORD *v60; // [rsp+30h] [rbp-160h]
  __int64 v61; // [rsp+30h] [rbp-160h]
  unsigned __int64 v62; // [rsp+38h] [rbp-158h]
  __int64 *v66; // [rsp+60h] [rbp-130h] BYREF
  __int64 v67; // [rsp+68h] [rbp-128h] BYREF
  _QWORD v68[2]; // [rsp+70h] [rbp-120h] BYREF
  __int64 v69; // [rsp+80h] [rbp-110h] BYREF
  __int64 v70[2]; // [rsp+90h] [rbp-100h] BYREF
  __m128i v71; // [rsp+A0h] [rbp-F0h] BYREF
  char v72; // [rsp+B0h] [rbp-E0h]
  char v73; // [rsp+B1h] [rbp-DFh]
  _QWORD v74[4]; // [rsp+C0h] [rbp-D0h] BYREF
  _QWORD *v75; // [rsp+E0h] [rbp-B0h]
  __int64 v76; // [rsp+E8h] [rbp-A8h]
  _QWORD v77[2]; // [rsp+F0h] [rbp-A0h] BYREF
  _QWORD *v78; // [rsp+100h] [rbp-90h]
  __int64 v79; // [rsp+108h] [rbp-88h]
  _QWORD v80[2]; // [rsp+110h] [rbp-80h] BYREF
  __int64 v81; // [rsp+120h] [rbp-70h]
  unsigned int v82; // [rsp+128h] [rbp-68h]
  char v83; // [rsp+12Ch] [rbp-64h]
  void *v84; // [rsp+130h] [rbp-60h] BYREF
  void **v85; // [rsp+138h] [rbp-58h]
  __int64 v86; // [rsp+148h] [rbp-48h]
  __int64 v87; // [rsp+150h] [rbp-40h]
  char v88; // [rsp+158h] [rbp-38h]

  v12 = a1;
  v13 = *(_DWORD *)(a1 + 240);
  if ( v13 == 98 )
  {
    v14 = a1 + 176;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    if ( !a5 )
    {
      v15 = 1;
      goto LABEL_5;
    }
    v16 = a5 - 7;
    LOBYTE(v31) = (unsigned int)(a5 - 1) <= 4 || (unsigned int)(a5 - 7) <= 1;
    v15 = v31;
    if ( !(_BYTE)v31 )
    {
      BYTE1(v75) = 1;
      v32 = "invalid linkage type for alias";
      goto LABEL_36;
    }
  }
  else
  {
    if ( v13 != 99 )
      BUG();
    v14 = a1 + 176;
    v15 = 0;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    v16 = a5 - 7;
  }
  if ( v16 <= 1 )
  {
    if ( a6 )
    {
      BYTE1(v75) = 1;
      v32 = "symbol with local linkage must have default visibility";
    }
    else
    {
      if ( !a7 )
        goto LABEL_5;
      BYTE1(v75) = 1;
      v32 = "symbol with local linkage cannot have a DLL storage class";
    }
LABEL_36:
    v74[0] = v32;
    LOBYTE(v75) = 3;
    sub_11FD800(v14, a4, (__int64)v74, 1);
    return 1;
  }
LABEL_5:
  v62 = *(_QWORD *)(a1 + 232);
  v74[0] = "expected type";
  LOWORD(v75) = 259;
  if ( (unsigned __int8)sub_12190A0(a1, &v66, (int *)v74, 0)
    || (unsigned __int8)sub_120AFE0(a1, 4, "expected comma after alias or ifunc's type") )
  {
    return 1;
  }
  v54 = *(_QWORD *)(a1 + 232);
  v17 = (unsigned int)(*(_DWORD *)(a1 + 240) - 364);
  if ( (unsigned int)v17 > 0x1E || (v34 = 1073741837, !_bittest64(&v34, v17)) )
  {
    if ( !(unsigned __int8)sub_1224A40((__int64 **)a1, &v67) )
      goto LABEL_9;
    return 1;
  }
  LOBYTE(v80[0]) = 0;
  v75 = v77;
  LODWORD(v74[0]) = 0;
  v74[1] = 0;
  v74[3] = 0;
  v76 = 0;
  LOBYTE(v77[0]) = 0;
  v78 = v80;
  v79 = 0;
  v82 = 1;
  v81 = 0;
  v83 = 0;
  v58 = sub_C33320();
  sub_C3B1B0((__int64)v70, 0.0);
  sub_C407B0(&v84, v70, v58);
  sub_C338F0((__int64)v70);
  v87 = 0;
  v88 = 0;
  if ( (unsigned __int8)sub_1221570((_QWORD **)a1, (__int64)v74, 0, 0) )
  {
LABEL_47:
    if ( v87 )
      j_j___libc_free_0_0(v87);
    v35 = sub_C33340();
    if ( v84 == v35 )
    {
      if ( v85 )
      {
        v36 = &v85[3 * (_QWORD)*(v85 - 1)];
        while ( v85 != v36 )
        {
          v36 -= 3;
          if ( v35 == *v36 )
            sub_969EE0((__int64)v36);
          else
            sub_C338F0((__int64)v36);
        }
        j_j_j___libc_free_0_0(v36 - 1);
      }
    }
    else
    {
      sub_C338F0((__int64)&v84);
    }
    if ( v82 > 0x40 && v81 )
      j_j___libc_free_0_0(v81);
    if ( v78 != v80 )
      j_j___libc_free_0(v78, v80[0] + 1LL);
    if ( v75 != v77 )
      j_j___libc_free_0(v75, v77[0] + 1LL);
    return 1;
  }
  if ( LODWORD(v74[0]) != 12 )
  {
    v73 = 1;
    v70[0] = (__int64)"invalid aliasee";
    v72 = 3;
    sub_11FD800(v14, v54, (__int64)v70, 1);
    goto LABEL_47;
  }
  v67 = v86;
  if ( v87 )
    j_j___libc_free_0_0(v87);
  v37 = sub_C33340();
  if ( v84 == v37 )
  {
    if ( v85 )
    {
      v42 = &v85[3 * (_QWORD)*(v85 - 1)];
      if ( v85 != v42 )
      {
        v61 = v14;
        v43 = v37;
        v44 = &v85[3 * (_QWORD)*(v85 - 1)];
        do
        {
          v44 -= 3;
          if ( v43 == *v44 )
            sub_969EE0((__int64)v44);
          else
            sub_C338F0((__int64)v44);
        }
        while ( v85 != v44 );
        v42 = v44;
        v14 = v61;
        v12 = a1;
      }
      j_j_j___libc_free_0_0(v42 - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v84);
  }
  if ( v82 > 0x40 && v81 )
    j_j___libc_free_0_0(v81);
  if ( v78 != v80 )
    j_j___libc_free_0(v78, v80[0] + 1LL);
  if ( v75 != v77 )
    j_j___libc_free_0(v75, v77[0] + 1LL);
LABEL_9:
  v18 = v67;
  v19 = *(_QWORD *)(v67 + 8);
  if ( *(_BYTE *)(v19 + 8) != 14 )
  {
    v15 = 1;
    v74[0] = "An alias or ifunc must have pointer type";
    LOWORD(v75) = 259;
    sub_11FD800(v14, v54, (__int64)v74, 1);
    return v15;
  }
  v56 = *(_DWORD *)(v19 + 8) >> 8;
  v59 = a2[1];
  if ( !v59 )
  {
    v38 = *(_QWORD *)(v12 + 1160);
    v39 = v12 + 1152;
    v60 = (_QWORD *)v38;
    if ( !v38 )
      goto LABEL_13;
    v40 = v12 + 1152;
    do
    {
      if ( *(_DWORD *)(v38 + 32) < a3 )
      {
        v38 = *(_QWORD *)(v38 + 24);
      }
      else
      {
        v40 = v38;
        v38 = *(_QWORD *)(v38 + 16);
      }
    }
    while ( v38 );
    v60 = 0;
    if ( v40 == v39 || *(_DWORD *)(v40 + 32) > a3 )
      goto LABEL_13;
    v60 = *(_QWORD **)(v40 + 40);
    v41 = sub_220F330(v40, v39);
    j_j___libc_free_0(v41, 56);
    --*(_QWORD *)(v12 + 1184);
    goto LABEL_88;
  }
  v20 = sub_1212F00(v12 + 1096, (__int64)a2);
  if ( v20 == v12 + 1104 )
  {
    v60 = (_QWORD *)sub_BA8B30(*(_QWORD *)(v12 + 344), *a2, v59);
    if ( v60 )
    {
      sub_8FD6D0((__int64)v68, "redefinition of global '@", a2);
      if ( v68[1] == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
      v51 = (__m128i *)sub_2241490(v68, "'", 1, v50);
      v70[0] = (__int64)&v71;
      if ( (__m128i *)v51->m128i_i64[0] == &v51[1] )
      {
        v71 = _mm_loadu_si128(v51 + 1);
      }
      else
      {
        v70[0] = v51->m128i_i64[0];
        v71.m128i_i64[0] = v51[1].m128i_i64[0];
      }
      v70[1] = v51->m128i_i64[1];
      v51->m128i_i64[0] = (__int64)v51[1].m128i_i64;
      v51->m128i_i64[1] = 0;
      v51[1].m128i_i8[0] = 0;
      LOWORD(v75) = 260;
      v74[0] = v70;
      sub_11FD800(v14, a4, (__int64)v74, 1);
      if ( (__m128i *)v70[0] != &v71 )
        j_j___libc_free_0(v70[0], v71.m128i_i64[0] + 1);
      if ( (__int64 *)v68[0] != &v69 )
        j_j___libc_free_0(v68[0], v69 + 1);
      return 1;
    }
LABEL_88:
    v18 = v67;
    goto LABEL_13;
  }
  v60 = *(_QWORD **)(v20 + 64);
  sub_1214E50(v12 + 1096, (__int64)a2);
  v18 = v67;
LABEL_13:
  LOWORD(v75) = 260;
  v74[0] = a2;
  v21 = v56;
  if ( (_BYTE)v15 )
  {
    v55 = 0;
    v57 = sub_B30500(v66, v56, a5, (__int64)v74, v18, 0);
    v22 = v57;
  }
  else
  {
    v57 = 0;
    v55 = sub_B30730(v66, v21, a5, (__int64)v74, v18, 0);
    v22 = v55;
  }
  v23 = *(_WORD *)(v22 + 32) & 0xE3CF | (16 * (a6 & 3)) | ((a9 & 7) << 10);
  *(_WORD *)(v22 + 32) = v23;
  if ( (v23 & 0xFu) - 7 <= 1 || (v23 & 0x30) != 0 && (v23 & 0xF) != 9 )
    *(_BYTE *)(v22 + 33) |= 0x40u;
  *(_WORD *)(v22 + 32) = *(_WORD *)(v22 + 32) & 0xFC3F | ((a10 & 3) << 6) | ((a7 & 3) << 8);
  if ( a8 )
    *(_BYTE *)(v22 + 33) |= 0x40u;
  while ( *(_DWORD *)(v12 + 240) == 4 )
  {
    v24 = sub_1205200(v14);
    *(_DWORD *)(v12 + 240) = v24;
    if ( v24 != 96 )
    {
      v28 = *(_QWORD *)(v12 + 232);
      v74[0] = "unknown alias or ifunc property!";
      LOWORD(v75) = 259;
      sub_11FD800(v14, v28, (__int64)v74, 1);
      goto LABEL_28;
    }
    v25 = sub_1205200(v14);
    v26 = *(_QWORD *)(v12 + 248);
    v27 = *(_QWORD *)(v12 + 256);
    *(_DWORD *)(v12 + 240) = v25;
    sub_B30D10(v22, v26, v27);
    v28 = 512;
    if ( (unsigned __int8)sub_120AFE0(v12, 512, "expected partition string") )
      goto LABEL_28;
  }
  if ( !a2[1] )
    sub_1243C70(v12 + 1192, a3, v22);
  if ( v60 )
  {
    if ( *(_QWORD *)(v22 + 8) != v60[1] )
    {
      v28 = v62;
      v74[0] = "forward reference and definition of alias have different types";
      LOWORD(v75) = 259;
      sub_11FD800(v14, v62, (__int64)v74, 1);
LABEL_28:
      v15 = 1;
      if ( v55 )
      {
        sub_B2F9E0(v55, v28, v29, v30);
        sub_BD2DD0(v55);
      }
LABEL_30:
      if ( v57 )
      {
        sub_AD0030(v57);
        sub_BD7260(v57, v28);
        sub_BD2DD0(v57);
      }
      return v15;
    }
    sub_BD84D0((__int64)v60, v22);
    sub_B30810(v60);
  }
  v45 = *(_QWORD *)(v12 + 344);
  if ( !(_BYTE)v15 )
  {
    v28 = v55;
    sub_BA86C0(v45 + 56, v55);
    v52 = *(_QWORD *)(v45 + 56);
    v53 = *(_QWORD *)(v55 + 56);
    *(_QWORD *)(v55 + 64) = v45 + 56;
    v52 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v55 + 56) = v52 | v53 & 7;
    *(_QWORD *)(v52 + 8) = v55 + 56;
    *(_QWORD *)(v45 + 56) = *(_QWORD *)(v45 + 56) & 7LL | (v55 + 56);
    goto LABEL_30;
  }
  sub_BA8640(v45 + 40, v57);
  v46 = *(_QWORD *)(v45 + 40);
  v47 = *(_QWORD *)(v57 + 48);
  *(_QWORD *)(v57 + 56) = v45 + 40;
  v48 = v46 & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v57 + 48) = v48 | v47 & 7;
  v15 = 0;
  *(_QWORD *)(v48 + 8) = v57 + 48;
  v49 = *(_QWORD *)(v45 + 40) & 7LL | (v57 + 48);
  *(_QWORD *)(v45 + 40) = v49;
  if ( v55 )
  {
    sub_B2F9E0(v55, v57, v49, v48);
    sub_BD2DD0(v55);
  }
  return v15;
}
