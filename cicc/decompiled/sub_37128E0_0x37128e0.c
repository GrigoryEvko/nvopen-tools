// Function: sub_37128E0
// Address: 0x37128e0
//
__m128i *__fastcall sub_37128E0(__int64 *a1, _QWORD *a2, char a3, unsigned __int8 a4, unsigned __int16 a5)
{
  __m128i *v5; // r12
  __int64 *v9; // rax
  __int64 v10; // rdx
  char *v11; // rax
  __int64 v12; // rdx
  unsigned __int8 v13; // r9
  unsigned __int16 v14; // r8
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // rax
  __int64 v19; // rdx
  unsigned __int16 v20; // r8
  size_t v21; // r10
  __int64 *v22; // rcx
  __int8 *v23; // rcx
  __m128i *v24; // rdi
  __int64 v25; // r8
  __m128i *v26; // rdi
  unsigned int v27; // ecx
  char **v28; // r9
  __int64 *v29; // rax
  const __m128i *v30; // r12
  const __m128i *v31; // r13
  __int16 v32; // dx
  __int64 v33; // rdx
  const __m128i *v34; // rbx
  unsigned __int64 v35; // r11
  __m128i *v36; // rdx
  char *v37; // r15
  unsigned __int64 v38; // rax
  __m128i *v39; // rsi
  _BYTE *v40; // rsi
  __int64 v41; // rdx
  __m128i *v42; // rax
  size_t v43; // rcx
  __m128i *v44; // r9
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rdi
  __m128i *v47; // rax
  unsigned __int64 v48; // rcx
  __m128i *v49; // rdx
  unsigned __int64 *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rbx
  __m128i *v53; // r15
  unsigned __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __m128i *v58; // rbx
  const __m128i *v59; // rdi
  signed __int64 v60; // rbx
  char *v61; // [rsp+30h] [rbp-310h]
  size_t n; // [rsp+48h] [rbp-2F8h]
  unsigned __int16 src; // [rsp+50h] [rbp-2F0h]
  __int8 *srca; // [rsp+50h] [rbp-2F0h]
  unsigned __int8 v66; // [rsp+58h] [rbp-2E8h]
  unsigned __int16 v68; // [rsp+60h] [rbp-2E0h]
  unsigned __int16 v69; // [rsp+60h] [rbp-2E0h]
  __m128i *v70; // [rsp+60h] [rbp-2E0h]
  _BYTE *v71[2]; // [rsp+70h] [rbp-2D0h] BYREF
  char v72; // [rsp+80h] [rbp-2C0h] BYREF
  __m128i *v73; // [rsp+90h] [rbp-2B0h] BYREF
  __int64 v74; // [rsp+98h] [rbp-2A8h]
  __m128i v75; // [rsp+A0h] [rbp-2A0h] BYREF
  __m128i *v76; // [rsp+B0h] [rbp-290h] BYREF
  __int64 v77; // [rsp+B8h] [rbp-288h]
  __m128i v78; // [rsp+C0h] [rbp-280h] BYREF
  _QWORD *v79; // [rsp+D0h] [rbp-270h] BYREF
  __int64 v80; // [rsp+D8h] [rbp-268h]
  _BYTE v81[16]; // [rsp+E0h] [rbp-260h] BYREF
  __m128i *v82; // [rsp+F0h] [rbp-250h] BYREF
  size_t v83; // [rsp+F8h] [rbp-248h]
  __m128i v84; // [rsp+100h] [rbp-240h] BYREF
  char *v85; // [rsp+110h] [rbp-230h] BYREF
  size_t v86; // [rsp+118h] [rbp-228h]
  _QWORD v87[2]; // [rsp+120h] [rbp-220h] BYREF
  __m128i *v88; // [rsp+130h] [rbp-210h] BYREF
  size_t v89; // [rsp+138h] [rbp-208h]
  __m128i v90; // [rsp+140h] [rbp-200h] BYREF
  __m128i *v91; // [rsp+150h] [rbp-1F0h] BYREF
  size_t v92; // [rsp+158h] [rbp-1E8h]
  __m128i v93; // [rsp+160h] [rbp-1E0h] BYREF
  __m128i *v94; // [rsp+170h] [rbp-1D0h] BYREF
  size_t v95; // [rsp+178h] [rbp-1C8h]
  _BYTE v96[448]; // [rsp+180h] [rbp-1C0h] BYREF

  v5 = (__m128i *)a1;
  if ( !a2[7] || a2[5] || a2[6] )
  {
    *a1 = (__int64)(a1 + 2);
    sub_370CD40(a1, byte_3F871B3, (__int64)byte_3F871B3);
    return v5;
  }
  v9 = sub_3707A10();
  v11 = (char *)sub_370C9C0(a2, a3, v9, v10);
  v71[0] = &v72;
  sub_370CD40((__int64 *)v71, v11, (__int64)&v11[v12]);
  v73 = &v75;
  sub_370CBD0((__int64 *)&v73, v71[0], (__int64)&v71[0][(unsigned __int64)v71[1]]);
  v13 = a4;
  v14 = a5;
  if ( !a4 )
    goto LABEL_7;
  v68 = a5;
  v66 = v13;
  v18 = sub_3707A30();
  v20 = v68;
  if ( !a2[7] || a2[5] || a2[6] )
  {
    v21 = 0;
    v91 = &v93;
  }
  else
  {
    v22 = &v18[5 * v19];
    if ( v22 == v18 )
    {
LABEL_92:
      v21 = 0;
      v94 = 0;
      v91 = &v93;
      goto LABEL_18;
    }
    while ( v66 != *((unsigned __int16 *)v18 + 16) )
    {
      v18 += 5;
      if ( v22 == v18 )
        goto LABEL_92;
    }
    v23 = (__int8 *)*v18;
    v21 = v18[1];
    v91 = &v93;
    v24 = &v93;
    if ( &v23[v21] && !v23 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v94 = (__m128i *)v21;
    if ( v21 <= 0xF )
    {
      if ( v21 == 1 )
      {
        v93.m128i_i8[0] = *v23;
        goto LABEL_18;
      }
      if ( !v21 )
        goto LABEL_18;
    }
    else
    {
      n = v21;
      srca = v23;
      v91 = (__m128i *)sub_22409D0((__int64)&v91, (unsigned __int64 *)&v94, 0);
      v24 = v91;
      v23 = srca;
      v21 = n;
      v93.m128i_i64[0] = (__int64)v94;
      v20 = v68;
    }
    v69 = v20;
    memcpy(v24, v23, v21);
    v21 = (size_t)v94;
    v20 = v69;
  }
LABEL_18:
  v92 = v21;
  src = v20;
  v91->m128i_i8[v21] = 0;
  sub_8FD6D0((__int64)&v94, ", ", &v91);
  sub_2241490((unsigned __int64 *)&v73, v94->m128i_i8, v95);
  sub_2240A30((unsigned __int64 *)&v94);
  sub_2240A30((unsigned __int64 *)&v91);
  v14 = src;
LABEL_7:
  if ( !v14 )
    goto LABEL_8;
  v16 = sub_3707A20();
  if ( !a2[7] || a2[5] || a2[6] )
  {
    v91 = &v93;
    sub_370CD40((__int64 *)&v91, byte_3F871B3, (__int64)byte_3F871B3);
    goto LABEL_14;
  }
  v25 = (__int64)v16;
  v26 = (__m128i *)v96;
  v27 = 0;
  v95 = 0xA00000000LL;
  v28 = (char **)&v94;
  v29 = &v16[5 * v17];
  v94 = (__m128i *)v96;
  if ( (__int64 *)v25 != v29 )
  {
    v70 = v5;
    v30 = (const __m128i *)v29;
    v31 = (const __m128i *)v25;
    do
    {
      v32 = v31[2].m128i_i16[0];
      if ( v32 && v32 == ((unsigned __int16)v32 & a5) )
      {
        v33 = v27;
        v34 = v31;
        v35 = v27 + 1LL;
        if ( v35 > HIDWORD(v95) )
        {
          if ( v26 > v31 || (__m128i *)((char *)v26 + 40 * v27) <= v31 )
          {
            v34 = v31;
            sub_C8D5F0((__int64)&v94, v96, v35, 0x28u, v25, (__int64)v28);
            v26 = v94;
            v33 = (unsigned int)v95;
          }
          else
          {
            v60 = (char *)v31 - (char *)v26;
            sub_C8D5F0((__int64)&v94, v96, v35, 0x28u, v25, (__int64)v28);
            v26 = v94;
            v33 = (unsigned int)v95;
            v34 = (__m128i *)((char *)v94 + v60);
          }
        }
        v36 = (__m128i *)((char *)v26 + 40 * v33);
        *v36 = _mm_loadu_si128(v34);
        v36[1] = _mm_loadu_si128(v34 + 1);
        v36[2].m128i_i64[0] = v34[2].m128i_i64[0];
        v26 = v94;
        v27 = v95 + 1;
        LODWORD(v95) = v95 + 1;
      }
      v31 = (const __m128i *)((char *)v31 + 40);
    }
    while ( v30 != v31 );
    v5 = v70;
    v52 = 40LL * v27;
    v53 = (__m128i *)((char *)v26 + v52);
    if ( v26 != (__m128i *)&v26->m128i_i8[v52] )
    {
      _BitScanReverse64(&v54, 0xCCCCCCCCCCCCCCCDLL * (v52 >> 3));
      sub_3712700(
        (__int64)v26,
        (__m128i *)((char *)v26 + v52),
        2LL * (int)(63 - (v54 ^ 0x3F)),
        (unsigned __int8 (__fastcall *)(__m128i *, __m128i *))sub_370CDF0,
        v25,
        (__int64)v28);
      if ( (unsigned __int64)v52 <= 0x280 )
      {
        sub_3711F50(v26, v53, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_370CDF0, v55, v56, v57);
      }
      else
      {
        v58 = v26 + 40;
        sub_3711F50(v26, v26 + 40, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_370CDF0, v55, v56, v57);
        if ( &v26[40] != v53 )
        {
          do
          {
            v59 = v58;
            v58 = (__m128i *)((char *)v58 + 40);
            sub_3711EC0(v59, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_370CDF0);
          }
          while ( v58 != v53 );
        }
      }
    }
  }
  v37 = (char *)v94;
  v77 = 0;
  v76 = &v78;
  v78.m128i_i8[0] = 0;
  v61 = &v94->m128i_i8[40 * (unsigned int)v95];
  if ( v94 == (__m128i *)v61 )
  {
LABEL_82:
    v91 = &v93;
    if ( v76 == &v78 )
    {
      v93 = _mm_load_si128(&v78);
    }
    else
    {
      v91 = v76;
      v93.m128i_i64[0] = v78.m128i_i64[0];
    }
    v78.m128i_i8[0] = 0;
    v92 = 0;
    v76 = &v78;
    v77 = 0;
    goto LABEL_85;
  }
  v38 = v94[2].m128i_u16[0];
  if ( v94[2].m128i_i16[0] )
    goto LABEL_72;
LABEL_43:
  v93.m128i_i8[0] = 48;
  v39 = &v93;
  while ( 1 )
  {
    v85 = (char *)v87;
    sub_370CBD0((__int64 *)&v85, v39, (__int64)v93.m128i_i64 + 1);
    v40 = *(_BYTE **)v37;
    if ( *(_QWORD *)v37 )
    {
      v41 = (__int64)&v40[*((_QWORD *)v37 + 1)];
      v79 = v81;
      sub_370CD40((__int64 *)&v79, v40, v41);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v80) <= 3 )
        goto LABEL_105;
    }
    else
    {
      v81[0] = 0;
      v80 = 0;
      v79 = v81;
    }
    v42 = (__m128i *)sub_2241490((unsigned __int64 *)&v79, " (0x", 4u);
    v82 = &v84;
    if ( (__m128i *)v42->m128i_i64[0] == &v42[1] )
    {
      v84 = _mm_loadu_si128(v42 + 1);
    }
    else
    {
      v82 = (__m128i *)v42->m128i_i64[0];
      v84.m128i_i64[0] = v42[1].m128i_i64[0];
    }
    v43 = v42->m128i_u64[1];
    v42[1].m128i_i8[0] = 0;
    v83 = v43;
    v42->m128i_i64[0] = (__int64)v42[1].m128i_i64;
    v44 = v82;
    v42->m128i_i64[1] = 0;
    v45 = 15;
    v46 = 15;
    if ( v44 != &v84 )
      v46 = v84.m128i_i64[0];
    if ( v83 + v86 <= v46 )
      goto LABEL_54;
    if ( v85 != (char *)v87 )
      v45 = v87[0];
    if ( v83 + v86 <= v45 )
    {
      v47 = (__m128i *)sub_2241130((unsigned __int64 *)&v85, 0, 0, v44, v83);
      v88 = &v90;
      v48 = v47->m128i_i64[0];
      v49 = v47 + 1;
      if ( (__m128i *)v47->m128i_i64[0] != &v47[1] )
      {
LABEL_55:
        v88 = (__m128i *)v48;
        v90.m128i_i64[0] = v47[1].m128i_i64[0];
        goto LABEL_56;
      }
    }
    else
    {
LABEL_54:
      v47 = (__m128i *)sub_2241490((unsigned __int64 *)&v82, v85, v86);
      v88 = &v90;
      v48 = v47->m128i_i64[0];
      v49 = v47 + 1;
      if ( (__m128i *)v47->m128i_i64[0] != &v47[1] )
        goto LABEL_55;
    }
    v90 = _mm_loadu_si128(v47 + 1);
LABEL_56:
    v89 = v47->m128i_u64[1];
    v47->m128i_i64[0] = (__int64)v49;
    v47->m128i_i64[1] = 0;
    v47[1].m128i_i8[0] = 0;
    if ( v89 == 0x3FFFFFFFFFFFFFFFLL )
      goto LABEL_105;
    v50 = sub_2241490((unsigned __int64 *)&v88, ")", 1u);
    v91 = &v93;
    if ( (unsigned __int64 *)*v50 == v50 + 2 )
    {
      v93 = _mm_loadu_si128((const __m128i *)v50 + 1);
    }
    else
    {
      v91 = (__m128i *)*v50;
      v93.m128i_i64[0] = v50[2];
    }
    v92 = v50[1];
    *v50 = (unsigned __int64)(v50 + 2);
    v50[1] = 0;
    *((_BYTE *)v50 + 16) = 0;
    sub_2241490((unsigned __int64 *)&v76, v91->m128i_i8, v92);
    if ( v91 != &v93 )
      j_j___libc_free_0((unsigned __int64)v91);
    if ( v88 != &v90 )
      j_j___libc_free_0((unsigned __int64)v88);
    if ( v82 != &v84 )
      j_j___libc_free_0((unsigned __int64)v82);
    if ( v79 != (_QWORD *)v81 )
      j_j___libc_free_0((unsigned __int64)v79);
    if ( v85 != (char *)v87 )
      j_j___libc_free_0((unsigned __int64)v85);
    v37 += 40;
    if ( v61 == v37 )
      break;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v77) <= 2 )
LABEL_105:
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)&v76, " | ", 3u);
    v38 = *((unsigned __int16 *)v37 + 16);
    if ( !*((_WORD *)v37 + 16) )
      goto LABEL_43;
LABEL_72:
    v39 = (__m128i *)&v93.m128i_i8[1];
    do
    {
      v39 = (__m128i *)((char *)v39 - 1);
      v51 = v38 & 0xF;
      v38 >>= 4;
      v39->m128i_i8[0] = a0123456789abcd_10[v51];
    }
    while ( v38 );
  }
  if ( !v77 )
    goto LABEL_82;
  v91 = &v93;
  v88 = &v90;
  v90.m128i_i32[0] = 2107424;
  v89 = 3;
  sub_370CBD0((__int64 *)&v91, v76, (__int64)v76->m128i_i64 + v77);
  sub_2241520((unsigned __int64 *)&v91, " )");
  sub_2241490((unsigned __int64 *)&v88, v91->m128i_i8, v92);
  sub_2240A30((unsigned __int64 *)&v91);
  v91 = &v93;
  if ( v88 == &v90 )
  {
    v93 = _mm_load_si128(&v90);
  }
  else
  {
    v91 = v88;
    v93.m128i_i64[0] = v90.m128i_i64[0];
  }
  v88 = &v90;
  v92 = v89;
  v89 = 0;
  v90.m128i_i8[0] = 0;
  sub_2240A30((unsigned __int64 *)&v88);
LABEL_85:
  sub_2240A30((unsigned __int64 *)&v76);
  if ( v94 != (__m128i *)v96 )
    _libc_free((unsigned __int64)v94);
LABEL_14:
  sub_8FD6D0((__int64)&v94, ", ", &v91);
  sub_2241490((unsigned __int64 *)&v73, v94->m128i_i8, v95);
  sub_2240A30((unsigned __int64 *)&v94);
  sub_2240A30((unsigned __int64 *)&v91);
LABEL_8:
  v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
  if ( v73 == &v75 )
  {
    v5[1] = _mm_load_si128(&v75);
  }
  else
  {
    v5->m128i_i64[0] = (__int64)v73;
    v5[1].m128i_i64[0] = v75.m128i_i64[0];
  }
  v15 = v74;
  v74 = 0;
  v75.m128i_i8[0] = 0;
  v5->m128i_i64[1] = v15;
  v73 = &v75;
  sub_2240A30((unsigned __int64 *)&v73);
  sub_2240A30((unsigned __int64 *)v71);
  return v5;
}
