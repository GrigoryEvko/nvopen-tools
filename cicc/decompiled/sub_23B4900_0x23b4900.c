// Function: sub_23B4900
// Address: 0x23b4900
//
__m128i *__fastcall sub_23B4900(__m128i *a1, __int64 a2)
{
  size_t v3; // rdx
  unsigned __int64 v4; // r9
  unsigned __int64 v5; // r10
  __int64 v6; // rax
  __int64 v7; // r9
  unsigned __int64 v8; // r10
  __int64 v9; // rdx
  char *v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r9
  unsigned __int64 v14; // r10
  size_t v15; // rdx
  _BYTE *v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // r12
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rbx
  __m128i *v23; // rax
  _BYTE *v24; // rax
  _BYTE *v25; // rax
  __int64 v26; // rax
  __m128i *v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // r12
  void **v30; // rbx
  void *v31; // rax
  _BYTE *v32; // rsi
  unsigned __int64 v33; // rcx
  char *v34; // rdx
  _BYTE *v35; // rax
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rcx
  __int64 v38; // rax
  __int64 *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // r14
  __m128i *v43; // rax
  size_t v44; // rdx
  __m128i *v45; // rcx
  int v46; // eax
  size_t v47; // r14
  __int64 v48; // rdx
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  __m128i v52; // [rsp+0h] [rbp-2F0h]
  const char *v53; // [rsp+20h] [rbp-2D0h]
  __int64 v54; // [rsp+28h] [rbp-2C8h]
  __m128i v55; // [rsp+40h] [rbp-2B0h]
  size_t v56; // [rsp+40h] [rbp-2B0h]
  __m128i *v57; // [rsp+50h] [rbp-2A0h]
  unsigned __int64 v58; // [rsp+60h] [rbp-290h]
  __int64 v59; // [rsp+60h] [rbp-290h]
  __int128 v60; // [rsp+70h] [rbp-280h] BYREF
  __int128 v61; // [rsp+80h] [rbp-270h]
  void *s2; // [rsp+90h] [rbp-260h] BYREF
  size_t n; // [rsp+98h] [rbp-258h]
  __m128i v64; // [rsp+A0h] [rbp-250h] BYREF
  _BYTE *v65; // [rsp+B0h] [rbp-240h] BYREF
  unsigned __int64 v66; // [rsp+B8h] [rbp-238h]
  __m128i v67; // [rsp+D0h] [rbp-220h] BYREF
  __m128i v68; // [rsp+E0h] [rbp-210h] BYREF
  __m128i v69; // [rsp+F0h] [rbp-200h] BYREF
  __m128i v70; // [rsp+100h] [rbp-1F0h] BYREF
  __int64 v71; // [rsp+110h] [rbp-1E0h]
  __int64 v72; // [rsp+118h] [rbp-1D8h]
  __m128i *v73; // [rsp+120h] [rbp-1D0h]
  __m128i s1; // [rsp+130h] [rbp-1C0h] BYREF
  __m128i v75; // [rsp+140h] [rbp-1B0h] BYREF
  __int64 v76; // [rsp+150h] [rbp-1A0h]
  __int64 v77; // [rsp+158h] [rbp-198h] BYREF
  __int64 *v78; // [rsp+160h] [rbp-190h]
  __int64 *v79; // [rsp+168h] [rbp-188h] BYREF
  __m128i v80; // [rsp+170h] [rbp-180h] BYREF
  __int64 v81; // [rsp+180h] [rbp-170h]
  _BYTE v82[88]; // [rsp+188h] [rbp-168h] BYREF
  const char *v83; // [rsp+1E0h] [rbp-110h] BYREF
  __int64 v84; // [rsp+1E8h] [rbp-108h]
  __int64 v85; // [rsp+1F0h] [rbp-100h]
  _QWORD v86[11]; // [rsp+1F8h] [rbp-F8h] BYREF
  __m128i v87; // [rsp+250h] [rbp-A0h] BYREF
  __int64 v88; // [rsp+260h] [rbp-90h] BYREF
  __int64 v89; // [rsp+268h] [rbp-88h] BYREF
  char v90; // [rsp+270h] [rbp-80h]
  void *v91; // [rsp+278h] [rbp-78h] BYREF
  __int64 *v92; // [rsp+280h] [rbp-70h]
  _QWORD v93[13]; // [rsp+288h] [rbp-68h] BYREF

  v3 = *(_QWORD *)(a2 + 40);
  if ( qword_4FDEE10 != v3 || v3 && memcmp(*(const void **)(a2 + 32), qword_4FDEE08, v3) )
  {
    sub_23AF980((__int64)&v65, *(_BYTE **)(*(_QWORD *)(a2 + 16) + 32LL), *(_QWORD *)(*(_QWORD *)(a2 + 16) + 40LL));
    v4 = (unsigned __int64)v65;
    v5 = v66;
    if ( *v65 == 10 )
    {
      v4 = sub_23AEA50((__int64)v65, v66, 1u);
      v5 = v51;
    }
    v6 = sub_23AEA50(v4, v5, 0);
    if ( v9 )
    {
      v10 = (char *)v6;
      do
      {
        if ( sub_23AE2B0((__int64)&v87, *v10) )
          break;
        v10 = (char *)(v11 + 1);
      }
      while ( v12 != 1 );
    }
    s2 = (void *)sub_23AEA50(v7, v8, 0);
    n = v15;
    if ( v14 )
    {
      v16 = (_BYTE *)v13;
      while ( *v16 != 10 )
      {
        if ( ++v16 == (_BYTE *)(v13 + v14) )
          goto LABEL_70;
      }
      v17 = (unsigned __int64)&v16[-v13];
    }
    else
    {
LABEL_70:
      v17 = -1;
    }
    v18 = sub_23AEA50(v13, v14, v17);
    v20 = sub_23AEA50(v18, v19, 1u);
    v58 = v21;
    v22 = v21;
    sub_95CA80((__int64 *)&v83, (__int64)&s2);
    sub_95CA80(v69.m128i_i64, a2 + 32);
    v23 = (__m128i *)sub_2241130((unsigned __int64 *)&v69, 0, 0, "<FONT COLOR=\"", 0xDu);
    s1.m128i_i64[0] = (__int64)&v75;
    if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
    {
      v75 = _mm_loadu_si128(v23 + 1);
    }
    else
    {
      s1.m128i_i64[0] = v23->m128i_i64[0];
      v75.m128i_i64[0] = v23[1].m128i_i64[0];
    }
    s1.m128i_i64[1] = v23->m128i_i64[1];
    v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
    v23->m128i_i64[1] = 0;
    v23[1].m128i_i8[0] = 0;
    sub_94F930(&v80, (__int64)&s1, "\">");
    sub_8FD5D0(&v87, (__int64)&v80, &v83);
    sub_94F930(&v67, (__int64)&v87, ":");
    sub_2240A30((unsigned __int64 *)&v87);
    sub_2240A30((unsigned __int64 *)&v80);
    sub_2240A30((unsigned __int64 *)&s1);
    sub_2240A30((unsigned __int64 *)&v69);
    sub_2240A30((unsigned __int64 *)&v83);
    if ( v58 )
    {
      while ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v67.m128i_i64[1]) > 0x11 )
      {
        sub_2241490((unsigned __int64 *)&v67, "<BR align=\"left\"/>", 0x12u);
        v24 = (_BYTE *)v20;
        while ( *v24 != 10 )
        {
          if ( (_BYTE *)(v20 + v22) == ++v24 )
          {
            v25 = (_BYTE *)v22;
            goto LABEL_22;
          }
        }
        v25 = &v24[-v20];
        if ( (unsigned __int64)v25 > v22 )
          v25 = (_BYTE *)v22;
LABEL_22:
        v84 = (__int64)v25;
        v83 = (const char *)v20;
        sub_95CA80(v87.m128i_i64, (__int64)&v83);
        sub_2241490((unsigned __int64 *)&v67, (char *)v87.m128i_i64[0], v87.m128i_u64[1]);
        if ( (__int64 *)v87.m128i_i64[0] != &v88 )
          j_j___libc_free_0(v87.m128i_u64[0]);
        v26 = v84 + 1;
        if ( v84 + 1 <= v22 )
        {
          v20 += v26;
          v22 -= v26;
          if ( v22 )
            continue;
        }
        goto LABEL_26;
      }
    }
    else
    {
LABEL_26:
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v67.m128i_i64[1]) > 0x18 )
      {
        sub_2241490((unsigned __int64 *)&v67, "<BR align=\"left\"/></FONT>", 0x19u);
        v27 = (__m128i *)v67.m128i_i64[0];
        a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
        if ( v27 == &v68 )
        {
          a1[1] = _mm_load_si128(&v68);
        }
        else
        {
          a1->m128i_i64[0] = (__int64)v27;
          a1[1].m128i_i64[0] = v68.m128i_i64[0];
        }
        v28 = v67.m128i_i64[1];
        v67.m128i_i64[0] = (__int64)&v68;
        v67.m128i_i64[1] = 0;
        a1->m128i_i64[1] = v28;
        v68.m128i_i8[0] = 0;
        sub_2240A30((unsigned __int64 *)&v67);
        sub_2240A30((unsigned __int64 *)&v65);
        return a1;
      }
    }
    sub_4262D8((__int64)"basic_string::append");
  }
  v29 = a2 + 16;
  v30 = (void **)&v60;
  v60 = 0;
  v61 = 0;
  do
  {
    v31 = *(void **)(*(_QWORD *)v29 + 40LL);
    *v30 = *(void **)(*(_QWORD *)v29 + 32LL);
    v30[1] = v31;
    sub_95CB50((const void **)v30, "\n", 1u);
    v32 = *v30;
    v33 = (unsigned __int64)v30[1];
    v34 = (char *)*v30;
    if ( v33 )
    {
      v34 = &v32[v33];
      v35 = *v30;
      while ( *v35 != 10 )
      {
        if ( v34 == ++v35 )
        {
          v36 = -1;
          if ( v33 != -1 )
            goto LABEL_52;
LABEL_38:
          v37 = v33 - v36;
          v34 = &v32[v36];
          if ( v37 == -1 )
          {
            v38 = -2;
          }
          else
          {
            if ( !v37 )
              goto LABEL_52;
            v38 = v37 - 1;
          }
          ++v34;
          goto LABEL_41;
        }
      }
      v36 = v35 - v32;
      if ( v36 > v33 )
        goto LABEL_52;
      goto LABEL_38;
    }
LABEL_52:
    v38 = 0;
LABEL_41:
    *v30 = v34;
    v29 += 8;
    v30 += 2;
    *(v30 - 1) = (void *)v38;
  }
  while ( v30 != &s2 );
  v92 = &qword_4FDEF80;
  v80.m128i_i64[0] = (__int64)v82;
  v87.m128i_i64[0] = (__int64)"<FONT COLOR=\"{0}\">%l</FONT><BR align=\"left\"/>";
  v88 = (__int64)v93;
  v93[0] = &v91;
  v83 = (const char *)&unk_49DD288;
  v91 = &unk_4A16028;
  v86[2] = 0x100000000LL;
  v86[3] = &v80;
  v87.m128i_i64[1] = 45;
  v89 = 1;
  v90 = 1;
  v80.m128i_i64[1] = 0;
  v81 = 80;
  v84 = 2;
  v85 = 0;
  v86[0] = 0;
  v86[1] = 0;
  sub_CB5980((__int64)&v83, 0, 0, 0);
  sub_CB6840((__int64)&v83, (__int64)&v87);
  v83 = (const char *)&unk_49DD388;
  sub_CB5840((__int64)&v83);
  v92 = &qword_4FDEE80;
  v78 = (__int64 *)&v83;
  v83 = (const char *)v86;
  v88 = (__int64)v93;
  v93[0] = &v91;
  v87.m128i_i64[0] = (__int64)"<FONT COLOR=\"{0}\">%l</FONT><BR align=\"left\"/>";
  s1.m128i_i64[0] = (__int64)&unk_49DD288;
  v91 = &unk_4A16028;
  v77 = 0x100000000LL;
  v87.m128i_i64[1] = 45;
  v89 = 1;
  v90 = 1;
  v84 = 0;
  v85 = 80;
  s1.m128i_i64[1] = 2;
  v75 = 0u;
  v76 = 0;
  sub_CB5980((__int64)&s1, 0, 0, 0);
  sub_CB6840((__int64)&s1, (__int64)&v87);
  s1.m128i_i64[0] = (__int64)&unk_49DD388;
  sub_CB5840((__int64)&s1);
  v78 = &qword_4FDED80;
  v72 = 0x100000000LL;
  v75.m128i_i64[0] = (__int64)&v79;
  v87.m128i_i64[0] = (__int64)&v89;
  s1.m128i_i64[0] = (__int64)"<FONT COLOR=\"{0}\">%l</FONT><BR align=\"left\"/>";
  v69.m128i_i64[0] = (__int64)&unk_49DD288;
  v77 = (__int64)&unk_4A16028;
  v79 = &v77;
  v73 = &v87;
  s1.m128i_i64[1] = 45;
  v75.m128i_i64[1] = 1;
  LOBYTE(v76) = 1;
  v87.m128i_i64[1] = 0;
  v88 = 80;
  v69.m128i_i64[1] = 2;
  v70 = 0u;
  v71 = 0;
  sub_CB5980((__int64)&v69, 0, 0, 0);
  sub_CB6840((__int64)&v69, (__int64)&s1);
  v69.m128i_i64[0] = (__int64)&unk_49DD388;
  sub_CB5840((__int64)&v69);
  v39 = *(__int64 **)(a2 + 16);
  v40 = *v39;
  v41 = v39[1];
  s1.m128i_i64[0] = v40;
  s1.m128i_i64[1] = v41;
  sub_95CA80((__int64 *)&s2, (__int64)&s1);
  v52 = v87;
  v53 = v83;
  v54 = v84;
  v55 = v80;
  sub_23AF980((__int64)&v67, (_BYTE *)v61, *((unsigned __int64 *)&v61 + 1));
  v42 = v67.m128i_i64[1];
  v59 = v67.m128i_i64[0];
  sub_23AF980((__int64)&v65, (_BYTE *)v60, *((unsigned __int64 *)&v60 + 1));
  sub_BC7A80(
    &v69,
    (__int64)v65,
    v66,
    v59,
    v42,
    v54,
    (const char *)v55.m128i_i64[0],
    v55.m128i_i64[1],
    v53,
    v54,
    (const char *)v52.m128i_i64[0],
    v52.m128i_i64[1]);
  v43 = (__m128i *)sub_2241130((unsigned __int64 *)&v69, 0, 0, ":\n<BR align=\"left\"/>", 0x14u);
  s1.m128i_i64[0] = (__int64)&v75;
  if ( (__m128i *)v43->m128i_i64[0] == &v43[1] )
  {
    v75 = _mm_loadu_si128(v43 + 1);
  }
  else
  {
    s1.m128i_i64[0] = v43->m128i_i64[0];
    v75.m128i_i64[0] = v43[1].m128i_i64[0];
  }
  s1.m128i_i64[1] = v43->m128i_i64[1];
  v43->m128i_i64[0] = (__int64)v43[1].m128i_i64;
  v43->m128i_i64[1] = 0;
  v43[1].m128i_i8[0] = 0;
  sub_2241490((unsigned __int64 *)&s2, (char *)s1.m128i_i64[0], s1.m128i_u64[1]);
  sub_2240A30((unsigned __int64 *)&s1);
  sub_2240A30((unsigned __int64 *)&v69);
  sub_2240A30((unsigned __int64 *)&v65);
  sub_2240A30((unsigned __int64 *)&v67);
  sub_C88F40((__int64)&v67, (__int64)"<FONT COLOR=\"\\w+\"></FONT>", 25, 0);
  while ( 1 )
  {
    v69.m128i_i64[0] = (__int64)&v70;
    v69.m128i_i64[1] = 0;
    v70.m128i_i8[0] = 0;
    sub_C894D0(&s1, &v67, (char *)byte_3F871B3, 0, (char *)s2, n, &v69);
    if ( sub_2241AC0((__int64)&v69, byte_3F871B3) )
      break;
    v44 = n;
    if ( s1.m128i_i64[1] == n )
    {
      v45 = (__m128i *)s2;
      if ( !n
        || (v56 = n, v57 = (__m128i *)s2, v46 = memcmp((const void *)s1.m128i_i64[0], s2, n), v45 = v57, v44 = v56, !v46) )
      {
        v47 = v44;
        a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
        if ( v45 == &v64 )
        {
          a1[1] = _mm_load_si128(&v64);
        }
        else
        {
          v48 = v64.m128i_i64[0];
          a1->m128i_i64[0] = (__int64)v45;
          a1[1].m128i_i64[0] = v48;
        }
        s2 = &v64;
        n = 0;
        a1->m128i_i64[1] = v47;
        v64.m128i_i8[0] = 0;
        goto LABEL_59;
      }
    }
    sub_2240AE0((unsigned __int64 *)&s2, (unsigned __int64 *)&s1);
    if ( (__m128i *)s1.m128i_i64[0] != &v75 )
      j_j___libc_free_0(s1.m128i_u64[0]);
    if ( (__m128i *)v69.m128i_i64[0] != &v70 )
      j_j___libc_free_0(v69.m128i_u64[0]);
  }
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v69.m128i_i64[0] == &v70 )
  {
    a1[1] = _mm_load_si128(&v70);
  }
  else
  {
    a1->m128i_i64[0] = v69.m128i_i64[0];
    a1[1].m128i_i64[0] = v70.m128i_i64[0];
  }
  v50 = v69.m128i_i64[1];
  v69.m128i_i64[1] = 0;
  v70.m128i_i8[0] = 0;
  a1->m128i_i64[1] = v50;
  v69.m128i_i64[0] = (__int64)&v70;
LABEL_59:
  sub_2240A30((unsigned __int64 *)&s1);
  sub_2240A30((unsigned __int64 *)&v69);
  sub_C88FF0(&v67);
  sub_2240A30((unsigned __int64 *)&s2);
  if ( (__int64 *)v87.m128i_i64[0] != &v89 )
    _libc_free(v87.m128i_u64[0]);
  if ( v83 != (const char *)v86 )
    _libc_free((unsigned __int64)v83);
  if ( (_BYTE *)v80.m128i_i64[0] != v82 )
    _libc_free(v80.m128i_u64[0]);
  return a1;
}
