// Function: sub_1855B10
// Address: 0x1855b10
//
__int64 __fastcall sub_1855B10(
        __int64 a1,
        __m128i a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  unsigned int v10; // r14d
  __int64 v11; // r12
  size_t v12; // rsi
  const void *v13; // rdi
  __int64 i; // r8
  __int64 v15; // rdi
  __int64 j; // rax
  double v17; // xmm4_8
  double v18; // xmm5_8
  char v19; // dl
  __int64 *v20; // rdi
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // r13
  _QWORD *v23; // r15
  _QWORD *v24; // rdi
  __int64 v25; // rdi
  unsigned __int64 *v26; // rbx
  unsigned __int64 *v27; // r13
  unsigned __int64 v28; // rdi
  unsigned __int64 *v29; // rbx
  unsigned __int64 v30; // r13
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  unsigned __int64 v34; // r8
  __int64 v35; // r13
  __int64 v36; // rbx
  unsigned __int64 v37; // rdi
  __int64 v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // r12
  unsigned __int64 *v41; // rbx
  unsigned __int64 *v42; // r13
  unsigned __int64 v43; // rdi
  unsigned __int64 *v44; // rbx
  unsigned __int64 v45; // r13
  unsigned __int64 v46; // rdi
  unsigned __int64 v47; // rdi
  __int64 v48; // rax
  unsigned __int64 v49; // r8
  __int64 v50; // r13
  __int64 v51; // rbx
  unsigned __int64 v52; // rdi
  _QWORD *v54; // rax
  __m128i *v55; // rdx
  __m128i si128; // xmm0
  __int64 v57; // rax
  __int64 v58; // rcx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  unsigned __int8 v62; // al
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rcx
  __int64 *v66; // rsi
  __int64 v67; // r8
  __int64 v68; // r9
  unsigned __int8 v69; // al
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // [rsp+8h] [rbp-D8h]
  __int64 v74; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v75; // [rsp+20h] [rbp-C0h] BYREF
  unsigned __int8 v76; // [rsp+28h] [rbp-B8h]
  __int64 v77; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int8 v78; // [rsp+38h] [rbp-A8h]
  const char *v79; // [rsp+40h] [rbp-A0h]
  __int64 v80; // [rsp+50h] [rbp-90h]
  __int64 *v81; // [rsp+60h] [rbp-80h] BYREF
  __int64 v82; // [rsp+68h] [rbp-78h]
  _QWORD v83[2]; // [rsp+70h] [rbp-70h] BYREF
  __m128i *v84; // [rsp+80h] [rbp-60h] BYREF
  __int64 v85; // [rsp+88h] [rbp-58h] BYREF
  __m128i v86; // [rsp+90h] [rbp-50h] BYREF
  __int64 (__fastcall *v87)(__int64, __int64 **, __int64, __m128i); // [rsp+A0h] [rbp-40h]

  if ( !qword_4FAA908 )
    sub_16BD130("error: -function-import requires -summary-file\n", 1u);
  sub_150F150((__int64)&v75, qword_4FAA900, qword_4FAA908, 0, a2);
  v10 = v76 & 1;
  v76 = (2 * v10) | v76 & 0xFD;
  if ( (_BYTE)v10 )
  {
    v81 = v83;
    v82 = 0;
    LOBYTE(v83[0]) = 0;
    sub_2240E30(&v81, qword_4FAA908 + 20);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v82) <= 0x13
      || (sub_2241490(&v81, "Error loading file '", 20),
          sub_2241490(&v81, (const char *)qword_4FAA900, qword_4FAA908),
          (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v82) <= 2) )
    {
      sub_4262D8((__int64)"basic_string::append");
    }
    v57 = sub_2241490(&v81, "': ", 3);
    v84 = &v86;
    if ( *(_QWORD *)v57 == v57 + 16 )
    {
      v86 = _mm_loadu_si128((const __m128i *)(v57 + 16));
    }
    else
    {
      v84 = *(__m128i **)v57;
      v86.m128i_i64[0] = *(_QWORD *)(v57 + 16);
    }
    v58 = *(_QWORD *)(v57 + 8);
    LOWORD(v80) = 260;
    v85 = v58;
    *(_QWORD *)v57 = v57 + 16;
    *(_QWORD *)(v57 + 8) = 0;
    *(_BYTE *)(v57 + 16) = 0;
    v79 = (const char *)&v84;
    v38 = (__int64)sub_16E8CB0();
    v62 = v76;
    v63 = v76 & 0xFD;
    v76 &= ~2u;
    if ( (v62 & 1) != 0 )
    {
      v64 = v75;
      v75 = 0;
      v77 = v64 | 1;
    }
    else
    {
      v77 = 1;
    }
    sub_16BCD30((unsigned __int64 *)&v77, (__int64 *)v38, v63, v59, v60, v61, (char)v79);
    if ( (v77 & 1) != 0 || (v77 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_16BCAE0(&v77, v38, v39);
    if ( v84 != &v86 )
    {
      v38 = v86.m128i_i64[0] + 1;
      j_j___libc_free_0(v84, v86.m128i_i64[0] + 1);
    }
    if ( v81 != v83 )
    {
      v38 = v83[0] + 1LL;
      j_j___libc_free_0(v81, v83[0] + 1LL);
    }
    v10 = 0;
  }
  else
  {
    v11 = v75;
    v83[0] = 0x4000000000LL;
    v12 = *(_QWORD *)(a1 + 184);
    v75 = 0;
    v13 = *(const void **)(a1 + 176);
    v81 = 0;
    v82 = 0;
    if ( byte_4FAA820 )
      sub_1852BE0(v13, v12, v11, (__int64)&v81);
    else
      sub_1854750(v13, v12, v11, (__int64)&v81);
    for ( i = *(_QWORD *)(v11 + 24); v11 + 8 != i; i = sub_220EEE0(i) )
    {
      v15 = *(_QWORD *)(i + 64);
      for ( j = *(_QWORD *)(i + 56); v15 != j; j += 8 )
      {
        if ( (*(_BYTE *)(*(_QWORD *)j + 12LL) & 0xFu) - 7 <= 1 )
          *(_BYTE *)(*(_QWORD *)j + 12LL) &= 0xF0u;
      }
    }
    if ( (unsigned __int8)sub_1ACEF80(a1, v11, 0) )
    {
      v54 = sub_16E8CB0();
      v55 = (__m128i *)v54[3];
      if ( v54[2] - (_QWORD)v55 <= 0x15u )
      {
        sub_16E7EE0((__int64)v54, "Error renaming module\n", 0x16u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_42BA3A0);
        v55[1].m128i_i32[0] = 1819632751;
        v55[1].m128i_i16[2] = 2661;
        *v55 = si128;
        v54[3] += 22LL;
      }
    }
    else
    {
      v84 = (__m128i *)v11;
      v85 = a1;
      v86.m128i_i64[1] = (__int64)sub_1851490;
      v87 = sub_1852070;
      sub_1854A20((__int64)&v77, (char **)&v84, a1, &v81, *(double *)a2.m128i_i64, a3, a4, a5, v17, v18, a8, a9);
      v10 = (unsigned __int8)v77;
      v19 = v78 & 1;
      v78 = (2 * (v78 & 1)) | v78 & 0xFD;
      if ( v19 )
      {
        v79 = "Error importing module: ";
        LOWORD(v80) = 259;
        v66 = (__int64 *)sub_16E8CB0();
        v69 = v78;
        v70 = v78 & 0xFD;
        v78 &= ~2u;
        if ( (v69 & 1) != 0 )
        {
          v71 = v77;
          v77 = 0;
          v74 = v71 | 1;
        }
        else
        {
          v74 = 1;
        }
        sub_16BCD30((unsigned __int64 *)&v74, v66, v70, v65, v67, v68, (char)v79);
        if ( (v74 & 1) != 0 || (v74 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_16BCAE0(&v74, (__int64)v66, v72);
        v10 = (v78 & 2) != 0;
        if ( (v78 & 2) != 0 )
          sub_1517600(&v77, (__int64)v66, v72);
        if ( (v78 & 1) != 0 && v77 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v77 + 8LL))(v77);
      }
      if ( v86.m128i_i64[1] )
        ((void (__fastcall *)(__int64 *, __int64 *, __int64))v86.m128i_i64[1])(&v85, &v85, 3);
    }
    v20 = v81;
    if ( HIDWORD(v82) && (_DWORD)v82 )
    {
      v21 = 0;
      v73 = 8LL * (unsigned int)v82;
      do
      {
        v22 = v20[v21 / 8];
        if ( v22 != -8 && v22 )
        {
          v23 = *(_QWORD **)(v22 + 24);
          while ( v23 )
          {
            v24 = v23;
            v23 = (_QWORD *)*v23;
            j_j___libc_free_0(v24, 16);
          }
          memset(*(void **)(v22 + 8), 0, 8LL * *(_QWORD *)(v22 + 16));
          v25 = *(_QWORD *)(v22 + 8);
          *(_QWORD *)(v22 + 32) = 0;
          *(_QWORD *)(v22 + 24) = 0;
          if ( v25 != v22 + 56 )
            j_j___libc_free_0(v25, 8LL * *(_QWORD *)(v22 + 16));
          _libc_free(v22);
          v20 = v81;
        }
        v21 += 8LL;
      }
      while ( v73 != v21 );
    }
    _libc_free((unsigned __int64)v20);
    v26 = *(unsigned __int64 **)(v11 + 296);
    v27 = &v26[*(unsigned int *)(v11 + 304)];
    while ( v27 != v26 )
    {
      v28 = *v26++;
      _libc_free(v28);
    }
    v29 = *(unsigned __int64 **)(v11 + 344);
    v30 = (unsigned __int64)&v29[2 * *(unsigned int *)(v11 + 352)];
    if ( v29 != (unsigned __int64 *)v30 )
    {
      do
      {
        v31 = *v29;
        v29 += 2;
        _libc_free(v31);
      }
      while ( (unsigned __int64 *)v30 != v29 );
      v30 = *(_QWORD *)(v11 + 344);
    }
    if ( v30 != v11 + 360 )
      _libc_free(v30);
    v32 = *(_QWORD *)(v11 + 296);
    if ( v32 != v11 + 312 )
      _libc_free(v32);
    sub_18525A0(*(_QWORD **)(v11 + 248));
    sub_18525A0(*(_QWORD **)(v11 + 200));
    sub_1851A90(*(_QWORD *)(v11 + 144));
    sub_18524A0(*(_QWORD **)(v11 + 96));
    if ( *(_DWORD *)(v11 + 60) )
    {
      v33 = *(unsigned int *)(v11 + 56);
      v34 = *(_QWORD *)(v11 + 48);
      if ( (_DWORD)v33 )
      {
        v35 = 8 * v33;
        v36 = 0;
        do
        {
          v37 = *(_QWORD *)(v34 + v36);
          if ( v37 && v37 != -8 )
          {
            _libc_free(v37);
            v34 = *(_QWORD *)(v11 + 48);
          }
          v36 += 8;
        }
        while ( v35 != v36 );
      }
    }
    else
    {
      v34 = *(_QWORD *)(v11 + 48);
    }
    _libc_free(v34);
    sub_1851910(*(_QWORD **)(v11 + 16));
    v38 = 392;
    j_j___libc_free_0(v11, 392);
  }
  if ( (v76 & 2) != 0 )
    sub_1852F40(&v75, v38, v39);
  v40 = v75;
  if ( (v76 & 1) != 0 )
  {
    if ( v75 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v75 + 8LL))(v75);
  }
  else if ( v75 )
  {
    v41 = *(unsigned __int64 **)(v75 + 296);
    v42 = &v41[*(unsigned int *)(v75 + 304)];
    while ( v42 != v41 )
    {
      v43 = *v41++;
      _libc_free(v43);
    }
    v44 = *(unsigned __int64 **)(v40 + 344);
    v45 = (unsigned __int64)&v44[2 * *(unsigned int *)(v40 + 352)];
    if ( v44 != (unsigned __int64 *)v45 )
    {
      do
      {
        v46 = *v44;
        v44 += 2;
        _libc_free(v46);
      }
      while ( (unsigned __int64 *)v45 != v44 );
      v45 = *(_QWORD *)(v40 + 344);
    }
    if ( v45 != v40 + 360 )
      _libc_free(v45);
    v47 = *(_QWORD *)(v40 + 296);
    if ( v47 != v40 + 312 )
      _libc_free(v47);
    sub_18525A0(*(_QWORD **)(v40 + 248));
    sub_18525A0(*(_QWORD **)(v40 + 200));
    sub_1851A90(*(_QWORD *)(v40 + 144));
    sub_18524A0(*(_QWORD **)(v40 + 96));
    if ( *(_DWORD *)(v40 + 60) )
    {
      v48 = *(unsigned int *)(v40 + 56);
      v49 = *(_QWORD *)(v40 + 48);
      if ( (_DWORD)v48 )
      {
        v50 = 8 * v48;
        v51 = 0;
        do
        {
          v52 = *(_QWORD *)(v49 + v51);
          if ( v52 != -8 && v52 )
          {
            _libc_free(v52);
            v49 = *(_QWORD *)(v40 + 48);
          }
          v51 += 8;
        }
        while ( v51 != v50 );
      }
    }
    else
    {
      v49 = *(_QWORD *)(v40 + 48);
    }
    _libc_free(v49);
    sub_1851910(*(_QWORD **)(v40 + 16));
    j_j___libc_free_0(v40, 392);
  }
  return v10;
}
