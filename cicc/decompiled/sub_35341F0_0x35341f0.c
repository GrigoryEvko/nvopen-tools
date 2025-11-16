// Function: sub_35341F0
// Address: 0x35341f0
//
void __fastcall sub_35341F0(__int64 a1, __int64 *a2, int a3, _QWORD *a4, __int64 a5, unsigned __int64 a6)
{
  __int64 *v6; // r13
  __int64 v7; // r14
  __int64 *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // rdx
  unsigned int v14; // ecx
  unsigned __int64 v15; // rsi
  unsigned int v16; // eax
  unsigned __int64 v17; // rsi
  char *v18; // rsi
  int v19; // ecx
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  char v22; // r10
  __int64 v23; // r9
  char *v24; // rax
  __int64 v25; // rdx
  __m128i *v26; // rax
  size_t v27; // rcx
  __m128i *v28; // rcx
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // r9
  __m128i *v31; // rax
  unsigned __int64 v32; // rcx
  __m128i *v33; // rdx
  unsigned __int8 *v34; // rdi
  unsigned __int64 v35; // r14
  __int64 *v39; // [rsp+20h] [rbp-130h]
  __int64 v40; // [rsp+28h] [rbp-128h]
  unsigned __int64 v41[2]; // [rsp+30h] [rbp-120h] BYREF
  __m128i v42; // [rsp+40h] [rbp-110h] BYREF
  _QWORD *v43; // [rsp+50h] [rbp-100h] BYREF
  __int64 v44; // [rsp+58h] [rbp-F8h]
  _BYTE v45[16]; // [rsp+60h] [rbp-F0h] BYREF
  __m128i *v46; // [rsp+70h] [rbp-E0h] BYREF
  size_t v47; // [rsp+78h] [rbp-D8h]
  __m128i v48; // [rsp+80h] [rbp-D0h] BYREF
  _QWORD *v49; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+98h] [rbp-B8h]
  _BYTE v51[48]; // [rsp+A0h] [rbp-B0h] BYREF
  char *v52; // [rsp+D0h] [rbp-80h] BYREF
  size_t v53; // [rsp+D8h] [rbp-78h]
  _QWORD v54[2]; // [rsp+E0h] [rbp-70h] BYREF
  __int16 v55; // [rsp+F0h] [rbp-60h]
  int v56; // [rsp+110h] [rbp-40h]

  v6 = (__int64 *)a2[41];
  v49 = v51;
  v50 = 0x600000000LL;
  v39 = a2 + 40;
  if ( a2 + 40 == v6 )
  {
    if ( (_BYTE)qword_503D828 )
      goto LABEL_14;
    goto LABEL_55;
  }
  do
  {
    while ( 1 )
    {
      v7 = v6[7];
      v8 = v6 + 6;
      if ( v6 + 6 != (__int64 *)v7 )
        break;
LABEL_11:
      v6 = (__int64 *)v6[1];
      if ( v39 == v6 )
        goto LABEL_12;
    }
    while ( 1 )
    {
      v9 = sub_3572DD0(v7, 0, 0, 0);
      if ( !v9 )
        break;
      v10 = (unsigned int)v50;
      a6 = (unsigned int)v50 + 1LL;
      if ( a6 > HIDWORD(v50) )
      {
        v40 = v9;
        sub_C8D5F0((__int64)&v49, v51, (unsigned int)v50 + 1LL, 8u, a5, a6);
        v10 = (unsigned int)v50;
        v9 = v40;
      }
      a4 = v49;
      v49[v10] = v9;
      LODWORD(v50) = v50 + 1;
      if ( !v7 )
        BUG();
      if ( (*(_BYTE *)v7 & 4) != 0 )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 == (__int64 *)v7 )
          goto LABEL_11;
      }
      else
      {
        while ( (*(_BYTE *)(v7 + 44) & 8) != 0 )
          v7 = *(_QWORD *)(v7 + 8);
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 == (__int64 *)v7 )
          goto LABEL_11;
      }
    }
    LODWORD(v50) = 0;
    v6 = (__int64 *)v6[1];
  }
  while ( v39 != v6 );
LABEL_12:
  if ( (_BYTE)qword_503D828 )
  {
    if ( !(_DWORD)v50 )
      goto LABEL_14;
    v11 = sub_CBF760(v49, 8LL * (unsigned int)v50);
    v12 = v11;
    if ( v11 <= 9 )
    {
      v52 = (char *)v54;
      sub_2240A50((__int64 *)&v52, 1u, 0);
      v18 = v52;
LABEL_34:
      *v18 = v12 + 48;
      goto LABEL_35;
    }
    if ( v11 <= 0x63 )
    {
      v52 = (char *)v54;
      sub_2240A50((__int64 *)&v52, 2u, 0);
      v18 = v52;
      goto LABEL_67;
    }
    if ( v11 <= 0x3E7 )
    {
      v17 = 3;
    }
    else
    {
      if ( v11 > 0x270F )
      {
        v13 = v11;
        v14 = 1;
        while ( 1 )
        {
          v15 = v13;
          v16 = v14;
          v14 += 4;
          v13 /= 0x2710u;
          if ( v15 <= 0x1869F )
          {
            v17 = v14;
            goto LABEL_30;
          }
          if ( v15 <= 0xF423F )
            break;
          if ( v15 <= (unsigned __int64)&loc_98967F )
          {
            v17 = v16 + 6;
            goto LABEL_30;
          }
          if ( v15 <= 0x5F5E0FF )
          {
            v17 = v16 + 7;
            goto LABEL_30;
          }
        }
        v52 = (char *)v54;
        v17 = v16 + 5;
LABEL_31:
        sub_2240A50((__int64 *)&v52, v17, 0);
        v18 = v52;
        v19 = v53 - 1;
        do
        {
          v20 = v12
              - 20
              * ((((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v12 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL) + v12 / 0x64);
          v21 = v12;
          v12 /= 0x64u;
          v22 = a00010203040506_0[2 * v20 + 1];
          LOBYTE(v20) = a00010203040506_0[2 * v20];
          v18[v19] = v22;
          v23 = (unsigned int)(v19 - 1);
          v19 -= 2;
          v18[v23] = v20;
        }
        while ( v21 > 0x270F );
        if ( v21 <= 0x3E7 )
          goto LABEL_34;
LABEL_67:
        v18[1] = a00010203040506_0[2 * v12 + 1];
        *v18 = a00010203040506_0[2 * v12];
LABEL_35:
        v24 = (char *)sub_2E791E0(a2);
        if ( v24 )
        {
          v43 = v45;
          sub_35323D0((__int64 *)&v43, v24, (__int64)&v24[v25]);
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v44) <= 8 )
            sub_4262D8((__int64)"basic_string::append");
        }
        else
        {
          v45[0] = 0;
          v43 = v45;
          v44 = 0;
        }
        v26 = (__m128i *)sub_2241490((unsigned __int64 *)&v43, ".content.", 9u);
        v46 = &v48;
        if ( (__m128i *)v26->m128i_i64[0] == &v26[1] )
        {
          v48 = _mm_loadu_si128(v26 + 1);
        }
        else
        {
          v46 = (__m128i *)v26->m128i_i64[0];
          v48.m128i_i64[0] = v26[1].m128i_i64[0];
        }
        v27 = v26->m128i_u64[1];
        v26[1].m128i_i8[0] = 0;
        v47 = v27;
        v26->m128i_i64[0] = (__int64)v26[1].m128i_i64;
        v28 = v46;
        v26->m128i_i64[1] = 0;
        v29 = 15;
        v30 = 15;
        if ( v28 != &v48 )
          v30 = v48.m128i_i64[0];
        if ( v47 + v53 <= v30 )
          goto LABEL_45;
        if ( v52 != (char *)v54 )
          v29 = v54[0];
        if ( v47 + v53 <= v29 )
        {
          v31 = (__m128i *)sub_2241130((unsigned __int64 *)&v52, 0, 0, v28, v47);
          v41[0] = (unsigned __int64)&v42;
          v32 = v31->m128i_i64[0];
          v33 = v31 + 1;
          if ( (__m128i *)v31->m128i_i64[0] != &v31[1] )
            goto LABEL_46;
        }
        else
        {
LABEL_45:
          v31 = (__m128i *)sub_2241490((unsigned __int64 *)&v46, v52, v53);
          v41[0] = (unsigned __int64)&v42;
          v32 = v31->m128i_i64[0];
          v33 = v31 + 1;
          if ( (__m128i *)v31->m128i_i64[0] != &v31[1] )
          {
LABEL_46:
            v41[0] = v32;
            v42.m128i_i64[0] = v31[1].m128i_i64[0];
LABEL_47:
            v41[1] = v31->m128i_u64[1];
            v31->m128i_i64[0] = (__int64)v33;
            v31->m128i_i64[1] = 0;
            v31[1].m128i_i8[0] = 0;
            if ( v46 != &v48 )
              j_j___libc_free_0((unsigned __int64)v46);
            if ( v43 != (_QWORD *)v45 )
              j_j___libc_free_0((unsigned __int64)v43);
            if ( v52 != (char *)v54 )
              j_j___libc_free_0((unsigned __int64)v52);
            v55 = 260;
            v34 = (unsigned __int8 *)*a2;
            v52 = (char *)v41;
            sub_BD6B50(v34, (const char **)&v52);
            if ( (__m128i *)v41[0] != &v42 )
              j_j___libc_free_0(v41[0]);
            goto LABEL_55;
          }
        }
        v42 = _mm_loadu_si128(v31 + 1);
        goto LABEL_47;
      }
      v17 = 4;
    }
LABEL_30:
    v52 = (char *)v54;
    goto LABEL_31;
  }
LABEL_55:
  if ( *(_DWORD *)(a1 + 208) == 2 )
  {
    if ( (_DWORD)v50 )
    {
      v35 = *(_QWORD *)(a1 + 200);
      v53 = 0x600000000LL;
      v52 = (char *)v54;
      sub_3532480((__int64)&v52, (__int64)&v49, (unsigned int)v50, (__int64)a4, a5, a6);
      v56 = a3;
      sub_3114990(v35, (__int64)&v52);
      if ( v52 != (char *)v54 )
        _libc_free((unsigned __int64)v52);
    }
  }
LABEL_14:
  if ( v49 != (_QWORD *)v51 )
    _libc_free((unsigned __int64)v49);
}
