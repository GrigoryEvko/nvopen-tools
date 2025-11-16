// Function: sub_23A7740
// Address: 0x23a7740
//
unsigned __int64 *__fastcall sub_23A7740(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, int a4)
{
  __int8 v5; // al
  char v6; // bl
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rax
  bool v12; // bl
  __int64 v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  int v21; // r9d
  bool v22; // dl
  char v23; // al
  __int64 v24; // r8
  bool v25; // bl
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 i; // r14
  unsigned __int64 *v30; // rsi
  unsigned __int64 v31; // rbx
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rbx
  unsigned __int64 v34; // rdi
  __int64 v35; // rbx
  unsigned __int64 v36; // rdi
  _QWORD *v37; // rax
  _QWORD *v38; // rax
  int v40; // edx
  _QWORD *v41; // rbx
  _QWORD *v42; // r14
  _QWORD *v43; // rbx
  _QWORD *v44; // r14
  _QWORD *v45; // rbx
  _QWORD *v46; // r14
  _QWORD *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 *v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  _QWORD *v54; // rax
  _QWORD *v55; // rax
  _QWORD *v56; // rax
  int v57; // ebx
  __int64 v58; // rax
  __int64 v59; // rbx
  _QWORD *v60; // rax
  _QWORD *v61; // rax
  __int64 v62; // rax
  char v63; // dl
  unsigned int v64; // [rsp+Ch] [rbp-9E4h]
  unsigned __int64 v65; // [rsp+18h] [rbp-9D8h]
  char v67; // [rsp+20h] [rbp-9D0h]
  char v68; // [rsp+28h] [rbp-9C8h]
  char v69; // [rsp+29h] [rbp-9C7h]
  char v70; // [rsp+29h] [rbp-9C7h]
  char v71; // [rsp+2Ah] [rbp-9C6h]
  bool v72; // [rsp+2Bh] [rbp-9C5h]
  __int64 v74; // [rsp+30h] [rbp-9C0h] BYREF
  unsigned __int64 v75; // [rsp+38h] [rbp-9B8h] BYREF
  __int64 v76; // [rsp+44h] [rbp-9ACh]
  int v77; // [rsp+4Ch] [rbp-9A4h]
  __int64 v78[4]; // [rsp+50h] [rbp-9A0h] BYREF
  unsigned __int64 v79[6]; // [rsp+70h] [rbp-980h] BYREF
  __int64 v80; // [rsp+A0h] [rbp-950h] BYREF
  __int64 v81; // [rsp+A8h] [rbp-948h]
  __m128i v82; // [rsp+B0h] [rbp-940h] BYREF
  __int64 v83; // [rsp+C0h] [rbp-930h]
  __m128i v84; // [rsp+D0h] [rbp-920h] BYREF
  __m128i v85; // [rsp+E0h] [rbp-910h] BYREF
  __int64 *v86; // [rsp+F0h] [rbp-900h]
  __int64 v87; // [rsp+F8h] [rbp-8F8h]
  int v88; // [rsp+108h] [rbp-8E8h] BYREF
  unsigned __int64 v89; // [rsp+110h] [rbp-8E0h]
  int *v90; // [rsp+118h] [rbp-8D8h]
  int *v91; // [rsp+120h] [rbp-8D0h]
  __int64 v92; // [rsp+128h] [rbp-8C8h]
  _QWORD *v93; // [rsp+138h] [rbp-8B8h] BYREF
  _QWORD *v94; // [rsp+140h] [rbp-8B0h]
  _QWORD *v95; // [rsp+148h] [rbp-8A8h]
  _QWORD *v96; // [rsp+150h] [rbp-8A0h]
  __int64 v97; // [rsp+158h] [rbp-898h]
  _QWORD *v98; // [rsp+160h] [rbp-890h]
  _QWORD *v99; // [rsp+168h] [rbp-888h]
  _QWORD *v100; // [rsp+188h] [rbp-868h]
  _QWORD *v101; // [rsp+190h] [rbp-860h]

  v65 = HIDWORD(a3);
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  v5 = *(_BYTE *)(a2 + 192);
  v64 = a3;
  if ( !v5 )
    goto LABEL_8;
  if ( *(_BYTE *)(a2 + 181) )
  {
    if ( a4 == 2 )
    {
      if ( *(_DWORD *)(a2 + 168) != 3 )
      {
LABEL_5:
        v5 = 0;
LABEL_6:
        v84.m128i_i8[1] = v5;
        v84.m128i_i8[0] = 1;
        sub_23A2390(a1, v84.m128i_i16);
        goto LABEL_24;
      }
      v5 = *(_BYTE *)(a2 + 181);
      if ( byte_4FDCC08 == 1 )
        goto LABEL_6;
      goto LABEL_20;
    }
    v59 = *(_QWORD *)a2;
    v60 = (_QWORD *)sub_22077B0(0x10u);
    if ( v60 )
    {
      v60[1] = v59;
      *v60 = &unk_4A0DF38;
    }
    v84.m128i_i64[0] = (__int64)v60;
    sub_23A2230(a1, (unsigned __int64 *)&v84);
    sub_23501E0(v84.m128i_i64);
    v6 = *(_BYTE *)(a2 + 192);
    if ( v6 && *(_DWORD *)(a2 + 168) == 3 )
      goto LABEL_10;
LABEL_9:
    v6 = 0;
    goto LABEL_10;
  }
  if ( *(_DWORD *)(a2 + 168) != 3 )
  {
LABEL_8:
    if ( a4 == 2 )
      goto LABEL_5;
    goto LABEL_9;
  }
  v6 = (a4 != 2) | byte_4FDCC08 ^ 1;
  if ( !v6 )
    goto LABEL_6;
  if ( a4 == 2 )
    goto LABEL_20;
LABEL_10:
  v7 = (_QWORD *)sub_22077B0(0x10u);
  if ( v7 )
    *v7 = &unk_4A0D478;
  v84.m128i_i64[0] = (__int64)v7;
  sub_23A2230(a1, (unsigned __int64 *)&v84);
  sub_23501E0(v84.m128i_i64);
  v8 = (_QWORD *)sub_22077B0(0x10u);
  if ( v8 )
    *v8 = &unk_4A0CFF8;
  v84.m128i_i64[0] = (__int64)v8;
  sub_23A2230(a1, (unsigned __int64 *)&v84);
  sub_23501E0(v84.m128i_i64);
  v84 = 0u;
  v85 = 0u;
  v86 = 0;
  v9 = sub_22077B0(0x10u);
  if ( v9 )
  {
    *(_BYTE *)(v9 + 8) = 0;
    *(_QWORD *)v9 = &unk_4A118F8;
  }
  v80 = v9;
  sub_23A1F40((unsigned __int64 *)&v84, (unsigned __int64 *)&v80);
  sub_233EFE0(&v80);
  v10 = (_QWORD *)sub_22077B0(0x10u);
  if ( v10 )
    *v10 = &unk_4A0FEB8;
  v80 = (__int64)v10;
  sub_23A1F40((unsigned __int64 *)&v84, (unsigned __int64 *)&v80);
  sub_233EFE0(&v80);
  sub_29744A0(&v80);
  sub_23A1F80((unsigned __int64 *)&v84, &v80);
  sub_291E720(&v80, 0);
  sub_23A2000((unsigned __int64 *)&v84, (char *)&v80);
  LOBYTE(v80) = 0;
  sub_23A2060((unsigned __int64 *)&v84, (char *)&v80);
  if ( dword_5033EF0[1] == (_DWORD)v65 && dword_5033EF0[0] == v64 )
  {
    v61 = (_QWORD *)sub_22077B0(0x10u);
    if ( v61 )
      *v61 = &unk_4A0EFF8;
    v80 = (__int64)v61;
    sub_23A1F40((unsigned __int64 *)&v84, (unsigned __int64 *)&v80);
    sub_233EFE0(&v80);
  }
  sub_234AAB0((__int64)&v80, v84.m128i_i64, *(_BYTE *)(a2 + 32));
  sub_23571D0(a1, &v80);
  sub_233EFE0(&v80);
  sub_233F7F0((__int64)&v84);
  if ( v6 )
  {
LABEL_20:
    v78[0] = 0;
    sub_2241BD0(&v80, a2 + 104);
    sub_2241BD0((__int64 *)v79, a2 + 40);
    sub_26C1D00((unsigned int)&v84, (unsigned int)v79, (unsigned int)&v80, a4, (unsigned int)v78, 0, 0);
    sub_2357AD0(a1, &v84);
    sub_233AA80((unsigned __int64 *)&v84);
    sub_2240A30(v79);
    sub_2240A30((unsigned __int64 *)&v80);
    if ( v78[0] )
      sub_23569D0((volatile signed __int32 *)(v78[0] + 8));
    sub_23A2700(a1);
    if ( (a4 & 0xFFFFFFFD) != 1 )
    {
      v84.m128i_i16[0] = 257;
      sub_23A2390(a1, v84.m128i_i16);
    }
  }
LABEL_24:
  v11 = sub_22077B0(0x10u);
  if ( v11 )
  {
    *(_DWORD *)(v11 + 8) = a4;
    *(_QWORD *)v11 = &unk_4A0DAB8;
  }
  v84.m128i_i64[0] = v11;
  sub_23A2230(a1, (unsigned __int64 *)&v84);
  sub_23501E0(v84.m128i_i64);
  if ( (byte_4FDC708 & 1) != 0 )
  {
    v47 = (_QWORD *)sub_22077B0(0x10u);
    if ( v47 )
      *v47 = &unk_4A0CE78;
    v84.m128i_i64[0] = (__int64)v47;
    sub_23A2230(a1, (unsigned __int64 *)&v84);
    sub_23501E0(v84.m128i_i64);
    if ( a4 != 2 )
      goto LABEL_28;
    goto LABEL_107;
  }
  if ( a4 == 2 )
  {
LABEL_107:
    v84 = 0u;
    v85.m128i_i64[0] = 0;
    v85.m128i_i32[2] = 1;
    sub_23A23F0(a1, v84.m128i_i8);
  }
LABEL_28:
  sub_23A12F0(a2, (__int64)a1, a3, a4);
  if ( unk_5033EEC != (_DWORD)v65 || (v12 = 0, unk_5033EE8 != v64) )
  {
    if ( HIDWORD(qword_5033EE0) != (_DWORD)v65 || (v12 = 0, (_DWORD)qword_5033EE0 != v64) )
      v12 = (a4 & 0xFFFFFFFD) != 1;
  }
  v13 = sub_22077B0(0x10u);
  if ( v13 )
  {
    *(_BYTE *)(v13 + 8) = v12;
    *(_QWORD *)v13 = &unk_4A0E8B8;
  }
  v84.m128i_i64[0] = v13;
  sub_23A2230(a1, (unsigned __int64 *)&v84);
  sub_23501E0(v84.m128i_i64);
  v14 = (_QWORD *)sub_22077B0(0x10u);
  if ( v14 )
    *v14 = &unk_4A0CEF8;
  v84.m128i_i64[0] = (__int64)v14;
  sub_23A2230(a1, (unsigned __int64 *)&v84);
  sub_23501E0(v84.m128i_i64);
  v15 = (_QWORD *)sub_22077B0(0x10u);
  if ( v15 )
    *v15 = &unk_4A0D3B8;
  v84.m128i_i64[0] = (__int64)v15;
  sub_23A2230(a1, (unsigned __int64 *)&v84);
  sub_23501E0(v84.m128i_i64);
  memset(v79, 0, 40);
  v16 = (_QWORD *)sub_22077B0(0x10u);
  if ( v16 )
    *v16 = &unk_4A0FFF8;
  v84.m128i_i64[0] = (__int64)v16;
  sub_23A1F40(v79, (unsigned __int64 *)&v84);
  sub_233EFE0(v84.m128i_i64);
  LOBYTE(v76) = 0;
  HIDWORD(v76) = 1;
  LOBYTE(v77) = 0;
  sub_F10C20((__int64)&v84, v76, v77);
  sub_2353C90(v79, (__int64)&v84, v17, v18, v19, v20);
  sub_233BCC0((__int64)&v84);
  sub_23A0D70(a2, (__int64)v79, a3);
  v80 = 0x100010000000001LL;
  v81 = 0x1000101000000LL;
  v82.m128i_i64[0] = 0;
  sub_29744D0(&v84, &v80);
  sub_23A1F80(v79, v84.m128i_i64);
  sub_234AAB0((__int64)&v84, (__int64 *)v79, *(_BYTE *)(a2 + 32));
  sub_23571D0(a1, v84.m128i_i64);
  sub_233EFE0(v84.m128i_i64);
  v22 = a4 == 1;
  if ( a4 == 2 )
    goto LABEL_40;
  v23 = sub_249DE50();
  v22 = a4 == 1;
  if ( v23 )
  {
LABEL_41:
    v72 = 0;
    v71 = 0;
    v24 = 0;
    v25 = v22 && qword_502E468[9] != 0;
    if ( v23 )
      goto LABEL_74;
    goto LABEL_42;
  }
  v71 = *(_BYTE *)(a2 + 192);
  v25 = v22 && qword_502E468[9] != 0;
  if ( !v71 )
  {
LABEL_40:
    v23 = 0;
    goto LABEL_41;
  }
  v40 = *(_DWORD *)(a2 + 168);
  v69 = v40 == 1;
  v24 = LOBYTE(qword_4FEA8E0[17]);
  v72 = *(_QWORD *)(a2 + 144) != 0;
  if ( (_BYTE)v24 )
    LOBYTE(v24) = qword_4FDC450 != 0;
  if ( (unsigned int)(v40 - 1) <= 1 )
  {
    sub_23A4250(a2, a1, a3, a4, v24, v21);
    v51 = *(_QWORD *)(a2 + 184);
    v78[0] = v51;
    if ( v51 )
      _InterlockedAdd((volatile signed __int32 *)(v51 + 8), 1u);
    sub_2241BD0(v84.m128i_i64, a2 + 104);
    sub_2241BD0(&v80, a2 + 40);
    sub_23A2D30(a2, a1, a3, v69, 0, *(_BYTE *)(a2 + 182), (unsigned __int64 *)&v80, (__int64)&v84, v78);
    sub_2240A30((unsigned __int64 *)&v80);
    sub_2240A30((unsigned __int64 *)&v84);
    if ( v78[0] )
      sub_23569D0((volatile signed __int32 *)(v78[0] + 8));
    goto LABEL_128;
  }
  if ( !*(_QWORD *)(a2 + 144) )
  {
LABEL_42:
    if ( !v25 && !(_BYTE)v24 )
    {
LABEL_44:
      if ( v71 && *(_DWORD *)(a2 + 172) == 1 )
      {
        v67 = byte_4FDC628;
        sub_2241BD0(v78, a2 + 72);
        sub_2241BD0(&v80, (__int64)v78);
        v84.m128i_i64[0] = (__int64)&v85;
        LOBYTE(v83) = v67;
        if ( (__m128i *)v80 == &v82 )
        {
          v85 = _mm_load_si128(&v82);
        }
        else
        {
          v84.m128i_i64[0] = v80;
          v85.m128i_i64[0] = v82.m128i_i64[0];
        }
        v80 = (__int64)&v82;
        v84.m128i_i64[1] = v81;
        v81 = 0;
        v82.m128i_i8[0] = 0;
        LOBYTE(v86) = v67;
        v62 = sub_22077B0(0x30u);
        if ( v62 )
        {
          *(_QWORD *)v62 = &unk_4A15CD0;
          *(_QWORD *)(v62 + 8) = v62 + 24;
          if ( (__m128i *)v84.m128i_i64[0] == &v85 )
          {
            *(__m128i *)(v62 + 24) = _mm_load_si128(&v85);
          }
          else
          {
            *(_QWORD *)(v62 + 8) = v84.m128i_i64[0];
            *(_QWORD *)(v62 + 24) = v85.m128i_i64[0];
          }
          v84.m128i_i64[0] = (__int64)&v85;
          v63 = (char)v86;
          v85.m128i_i8[0] = 0;
          *(_QWORD *)(v62 + 16) = v84.m128i_i64[1];
          v84.m128i_i64[1] = 0;
          *(_BYTE *)(v62 + 40) = v63;
        }
        v75 = v62;
        sub_23A2230(a1, &v75);
        sub_23501E0((__int64 *)&v75);
        sub_2240A30((unsigned __int64 *)&v84);
        sub_2240A30((unsigned __int64 *)&v80);
        sub_2240A30((unsigned __int64 *)v78);
      }
      if ( v72 )
      {
        v48 = *(_QWORD *)(a2 + 184);
        v74 = v48;
        if ( v48 )
          _InterlockedAdd((volatile signed __int32 *)(v48 + 8), 1u);
        sub_2241BD0(v78, a2 + 136);
        sub_2487D40(&v80, v78, &v74);
        v84.m128i_i64[0] = (__int64)&v85;
        if ( (__m128i *)v80 == &v82 )
        {
          v85 = _mm_load_si128(&v82);
        }
        else
        {
          v84.m128i_i64[0] = v80;
          v85.m128i_i64[0] = v82.m128i_i64[0];
        }
        v80 = (__int64)&v82;
        v84.m128i_i64[1] = v81;
        v81 = 0;
        v82.m128i_i8[0] = 0;
        v86 = (__int64 *)v83;
        v83 = 0;
        v49 = sub_22077B0(0x30u);
        if ( v49 )
        {
          *(_QWORD *)v49 = &unk_4A0E938;
          *(_QWORD *)(v49 + 8) = v49 + 24;
          if ( (__m128i *)v84.m128i_i64[0] == &v85 )
          {
            *(__m128i *)(v49 + 24) = _mm_load_si128(&v85);
          }
          else
          {
            *(_QWORD *)(v49 + 8) = v84.m128i_i64[0];
            *(_QWORD *)(v49 + 24) = v85.m128i_i64[0];
          }
          v84.m128i_i64[0] = (__int64)&v85;
          v50 = v86;
          v85.m128i_i8[0] = 0;
          *(_QWORD *)(v49 + 16) = v84.m128i_i64[1];
          v84.m128i_i64[1] = 0;
          *(_QWORD *)(v49 + 40) = v50;
          v86 = 0;
        }
        v75 = v49;
        sub_23A2230(a1, &v75);
        sub_23501E0((__int64 *)&v75);
        if ( v86 )
          sub_23569D0((volatile signed __int32 *)v86 + 2);
        sub_2240A30((unsigned __int64 *)&v84);
        if ( v83 )
          sub_23569D0((volatile signed __int32 *)(v83 + 8));
        sub_2240A30((unsigned __int64 *)&v80);
        sub_2240A30((unsigned __int64 *)v78);
        if ( v74 )
          sub_23569D0((volatile signed __int32 *)(v74 + 8));
      }
      if ( *(_BYTE *)(a2 + 192) && (unsigned int)(*(_DWORD *)(a2 + 168) - 2) <= 1 )
      {
        v57 = *(_DWORD *)(a2 + 176);
        v58 = sub_22077B0(0x10u);
        if ( v58 )
        {
          *(_DWORD *)(v58 + 8) = v57;
          *(_QWORD *)v58 = &unk_4A0D838;
        }
        v84.m128i_i64[0] = v58;
        sub_23A2230(a1, (unsigned __int64 *)&v84);
        sub_23501E0(v84.m128i_i64);
      }
      v26 = sub_22077B0(0x10u);
      if ( v26 )
      {
        *(_BYTE *)(v26 + 8) = 1;
        *(_QWORD *)v26 = &unk_4A0CDB8;
      }
      v84.m128i_i64[0] = v26;
      sub_23A2230(a1, (unsigned __int64 *)&v84);
      sub_23501E0(v84.m128i_i64);
      v27 = (v65 << 32) | v64;
      if ( byte_4FDDF48 )
      {
        sub_23A7230((unsigned __int64 *)&v84, a2, v27, a4);
        v28 = v84.m128i_i64[1];
        for ( i = v84.m128i_i64[0]; v28 != i; i += 8 )
        {
          v30 = (unsigned __int64 *)i;
          sub_23A2230(a1, v30);
        }
        sub_234A900((__int64)&v84);
      }
      else
      {
        sub_23A6B00((unsigned __int64 *)&v84, a2, v27, a4);
        sub_2357600(a1, (__int64)&v84);
        v41 = v101;
        v42 = v100;
        if ( v101 != v100 )
        {
          do
          {
            if ( *v42 )
              (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v42 + 8LL))(*v42);
            ++v42;
          }
          while ( v41 != v42 );
          v42 = v100;
        }
        if ( v42 )
          j_j___libc_free_0((unsigned __int64)v42);
        v43 = v99;
        v44 = v98;
        if ( v99 != v98 )
        {
          do
          {
            if ( *v44 )
              (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v44 + 8LL))(*v44);
            ++v44;
          }
          while ( v43 != v44 );
          v44 = v98;
        }
        if ( v44 )
          j_j___libc_free_0((unsigned __int64)v44);
        v45 = v94;
        v46 = v93;
        if ( v94 != v93 )
        {
          do
          {
            if ( *v46 )
              (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v46 + 8LL))(*v46);
            ++v46;
          }
          while ( v45 != v46 );
          v46 = v93;
        }
        if ( v46 )
          j_j___libc_free_0((unsigned __int64)v46);
      }
      v84.m128i_i32[2] = 0;
      v85.m128i_i64[1] = (__int64)&v84.m128i_i64[1];
      v86 = &v84.m128i_i64[1];
      v90 = &v88;
      v91 = &v88;
      v85.m128i_i64[0] = 0;
      v87 = 0;
      v88 = 0;
      v89 = 0;
      v92 = 0;
      LODWORD(v93) = 0;
      v94 = 0;
      v95 = &v93;
      v96 = &v93;
      v97 = 0;
      LOBYTE(v98) = 0;
      sub_2358990(a1, (__int64)&v84);
      v31 = (unsigned __int64)v94;
      if ( v94 )
      {
        do
        {
          sub_239F400(*(_QWORD *)(v31 + 24));
          v32 = v31;
          v31 = *(_QWORD *)(v31 + 16);
          j_j___libc_free_0(v32);
        }
        while ( v31 );
      }
      v33 = v89;
      while ( v33 )
      {
        sub_239F230(*(_QWORD *)(v33 + 24));
        v34 = v33;
        v33 = *(_QWORD *)(v33 + 16);
        j_j___libc_free_0(v34);
      }
      v35 = v85.m128i_i64[0];
      while ( v35 )
      {
        sub_239F5D0(*(_QWORD *)(v35 + 24));
        v36 = v35;
        v35 = *(_QWORD *)(v35 + 16);
        j_j___libc_free_0(v36);
      }
      if ( a4 != 1 )
      {
        v37 = (_QWORD *)sub_22077B0(0x10u);
        if ( v37 )
          *v37 = &unk_4A0CFB8;
        v84.m128i_i64[0] = (__int64)v37;
        sub_23A2230(a1, (unsigned __int64 *)&v84);
        sub_23501E0(v84.m128i_i64);
      }
      v38 = (_QWORD *)sub_22077B0(0x10u);
      if ( v38 )
        *v38 = &unk_4A0D3B8;
      v84.m128i_i64[0] = (__int64)v38;
      sub_23A2230(a1, (unsigned __int64 *)&v84);
      sub_23501E0(v84.m128i_i64);
      sub_23A0BA0((__int64)&v84, 0);
      sub_23A2670(a1, (__int64)&v84);
      sub_233AAF0((__int64)&v84);
      goto LABEL_68;
    }
  }
LABEL_74:
  v68 = v23;
  v70 = v24;
  sub_23A4250(a2, a1, a3, a4, v24, v21);
  if ( !v25 && !v68 )
  {
    if ( v70 )
    {
      v78[0] = 0;
      v84.m128i_i64[0] = (__int64)&v85;
      sub_239F180(v84.m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
      sub_2241BD0(&v80, (__int64)&qword_4FDC448);
      sub_23A2D30(a2, a1, a3, 1, 0, 0, (unsigned __int64 *)&v80, (__int64)&v84, v78);
      sub_2240A30((unsigned __int64 *)&v80);
      sub_2240A30((unsigned __int64 *)&v84);
      if ( v78[0] )
        sub_23569D0((volatile signed __int32 *)(v78[0] + 8));
    }
    goto LABEL_44;
  }
  v53 = sub_22077B0(0x10u);
  if ( v53 )
  {
    *(_DWORD *)(v53 + 8) = 3;
    *(_QWORD *)v53 = &unk_4A0D078;
  }
  v84.m128i_i64[0] = v53;
  sub_23A2230(a1, (unsigned __int64 *)&v84);
  if ( v84.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v84.m128i_i64[0] + 8LL))(v84.m128i_i64[0]);
  v54 = (_QWORD *)sub_22077B0(0x10u);
  if ( v54 )
    *v54 = &unk_4A0CE38;
  v84.m128i_i64[0] = (__int64)v54;
  sub_23A2230(a1, (unsigned __int64 *)&v84);
  sub_23501E0(v84.m128i_i64);
  if ( !v25 )
  {
    v55 = (_QWORD *)sub_22077B0(0x10u);
    if ( v55 )
      *v55 = &unk_4A0D0F8;
    v84.m128i_i64[0] = (__int64)v55;
    sub_23A2230(a1, (unsigned __int64 *)&v84);
    sub_23501E0(v84.m128i_i64);
    sub_23A2470(a2, a1, a3);
    v56 = (_QWORD *)sub_22077B0(0x10u);
    if ( v56 )
      *v56 = &unk_4A0D5B8;
    v84.m128i_i64[0] = (__int64)v56;
    sub_23A2230(a1, (unsigned __int64 *)&v84);
    sub_23501E0(v84.m128i_i64);
LABEL_128:
    v52 = sub_22077B0(0x10u);
    if ( v52 )
    {
      *(_QWORD *)v52 = &unk_4A0DB38;
      *(_WORD *)(v52 + 8) = 0;
    }
    v84.m128i_i64[0] = v52;
    sub_23A2230(a1, (unsigned __int64 *)&v84);
    if ( v84.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v84.m128i_i64[0] + 8LL))(v84.m128i_i64[0]);
    goto LABEL_44;
  }
LABEL_68:
  sub_233F7F0((__int64)v79);
  return a1;
}
