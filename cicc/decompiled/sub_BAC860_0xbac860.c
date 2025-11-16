// Function: sub_BAC860
// Address: 0xbac860
//
__m128i *__fastcall sub_BAC860(__m128i *a1, unsigned __int64 a2, unsigned __int64 a3, unsigned __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // r13
  unsigned __int64 v7; // r14
  __int64 v8; // rsi
  int v9; // eax
  _BYTE *v10; // rdi
  __int64 v11; // r8
  const char *v12; // rcx
  int v13; // esi
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  char v16; // r10
  __int64 v17; // r9
  unsigned __int64 v18; // rdx
  __int64 v19; // rsi
  int v20; // eax
  _BYTE *v21; // r8
  const char *v22; // rcx
  int v23; // esi
  __int64 v24; // r10
  unsigned __int64 v25; // rdx
  char v26; // r11
  __int64 v27; // rax
  __m128i *v28; // rax
  __m128i *v29; // rcx
  __m128i *v30; // rdx
  __int64 v31; // rcx
  __m128i *v32; // rax
  __int64 v33; // rcx
  __m128i *v34; // r10
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rcx
  __m128i *v38; // rax
  __int64 v39; // rcx
  __m128i *v40; // rdx
  __m128i *v42; // r8
  __int64 v43; // rsi
  int v44; // eax
  _BYTE *v45; // rdi
  unsigned int v46; // esi
  __int64 v47; // rdx
  unsigned __int64 v48; // rax
  char v49; // r10
  __int64 v50; // r9
  _QWORD v51[2]; // [rsp+10h] [rbp-D0h] BYREF
  _QWORD v52[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD *v53; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v54; // [rsp+38h] [rbp-A8h]
  _QWORD v55[2]; // [rsp+40h] [rbp-A0h] BYREF
  __m128i *v56; // [rsp+50h] [rbp-90h] BYREF
  __int64 v57; // [rsp+58h] [rbp-88h]
  __m128i v58; // [rsp+60h] [rbp-80h] BYREF
  __m128i *v59; // [rsp+70h] [rbp-70h] BYREF
  __int64 v60; // [rsp+78h] [rbp-68h]
  __m128i v61; // [rsp+80h] [rbp-60h] BYREF
  _QWORD *v62; // [rsp+90h] [rbp-50h] BYREF
  __int64 v63; // [rsp+98h] [rbp-48h]
  _QWORD v64[8]; // [rsp+A0h] [rbp-40h] BYREF

  v5 = a3;
  if ( a2 == -1 )
  {
    v42 = a1 + 1;
    if ( a3 > 9 )
    {
      if ( a3 <= 0x63 )
      {
        a1->m128i_i64[0] = (__int64)v42;
        sub_2240A50(a1, 2, 0, a4, v42);
        v45 = (_BYTE *)a1->m128i_i64[0];
      }
      else
      {
        if ( a3 <= 0x3E7 )
        {
          v43 = 3;
        }
        else if ( a3 <= 0x270F )
        {
          v43 = 4;
        }
        else
        {
          LODWORD(v43) = 1;
          while ( 1 )
          {
            a4 = a3;
            v44 = v43;
            v43 = (unsigned int)(v43 + 4);
            a3 /= 0x2710u;
            if ( a4 <= 0x1869F )
              break;
            if ( a4 <= 0xF423F )
            {
              a1->m128i_i64[0] = (__int64)v42;
              v43 = (unsigned int)(v44 + 5);
              goto LABEL_67;
            }
            if ( a4 <= (unsigned __int64)&loc_98967F )
            {
              v43 = (unsigned int)(v44 + 6);
              break;
            }
            if ( a4 <= 0x5F5E0FF )
            {
              v43 = (unsigned int)(v44 + 7);
              break;
            }
          }
        }
        a1->m128i_i64[0] = (__int64)v42;
LABEL_67:
        sub_2240A50(a1, v43, 0, a4, v42);
        v45 = (_BYTE *)a1->m128i_i64[0];
        v46 = a1->m128i_i32[2] - 1;
        do
        {
          v47 = v5
              - 20
              * (v5 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v5 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
          v48 = v5;
          v5 /= 0x64u;
          v49 = a00010203040506_0[2 * v47 + 1];
          LOBYTE(v47) = a00010203040506_0[2 * v47];
          v45[v46] = v49;
          v50 = v46 - 1;
          v46 -= 2;
          v45[v50] = v47;
        }
        while ( v48 > 0x270F );
        if ( v48 <= 0x3E7 )
          goto LABEL_70;
      }
      v45[1] = a00010203040506_0[2 * v5 + 1];
      *v45 = a00010203040506_0[2 * v5];
      return a1;
    }
    a1->m128i_i64[0] = (__int64)v42;
    sub_2240A50(a1, 1, 0, a4, v42);
    v45 = (_BYTE *)a1->m128i_i64[0];
LABEL_70:
    *v45 = v5 + 48;
    return a1;
  }
  v7 = a2;
  if ( a3 <= 9 )
  {
    v62 = v64;
    sub_2240A50(&v62, 1, 0, a4, a5);
    v10 = v62;
LABEL_15:
    *v10 = v5 + 48;
    goto LABEL_16;
  }
  if ( a3 <= 0x63 )
  {
    v62 = v64;
    sub_2240A50(&v62, 2, 0, a4, a5);
    v10 = v62;
    v12 = "00010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354"
          "555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
  }
  else
  {
    if ( a3 <= 0x3E7 )
    {
      v8 = 3;
    }
    else if ( a3 <= 0x270F )
    {
      v8 = 4;
    }
    else
    {
      LODWORD(v8) = 1;
      while ( 1 )
      {
        a4 = a3;
        v9 = v8;
        v8 = (unsigned int)(v8 + 4);
        a3 /= 0x2710u;
        if ( a4 <= 0x1869F )
          break;
        if ( a4 <= 0xF423F )
        {
          v62 = v64;
          v8 = (unsigned int)(v9 + 5);
          goto LABEL_12;
        }
        if ( a4 <= (unsigned __int64)&loc_98967F )
        {
          v8 = (unsigned int)(v9 + 6);
          break;
        }
        if ( a4 <= 0x5F5E0FF )
        {
          v8 = (unsigned int)(v9 + 7);
          break;
        }
      }
    }
    v62 = v64;
LABEL_12:
    sub_2240A50(&v62, v8, 0, a4, a5);
    v10 = v62;
    v11 = 0x28F5C28F5C28F5C3LL;
    v12 = "00010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354"
          "555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
    v13 = v63 - 1;
    do
    {
      v14 = v5
          - 20 * (v5 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v5 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
      v15 = v5;
      v5 /= 0x64u;
      v16 = a00010203040506_0[2 * v14 + 1];
      LOBYTE(v14) = a00010203040506_0[2 * v14];
      v10[v13] = v16;
      v17 = (unsigned int)(v13 - 1);
      v13 -= 2;
      v10[v17] = v14;
    }
    while ( v15 > 0x270F );
    if ( v15 <= 0x3E7 )
      goto LABEL_15;
  }
  v10[1] = a00010203040506_0[2 * v5 + 1];
  *v10 = a00010203040506_0[2 * v5];
LABEL_16:
  if ( v7 > 9 )
  {
    if ( v7 <= 0x63 )
    {
      v53 = v55;
      sub_2240A50(&v53, 2, 0, v12, v11);
      v21 = v53;
      v22 = "000102030405060708091011121314151617181920212223242526272829303132333435363738394041424344454647484950515253"
            "54555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
    }
    else
    {
      if ( v7 <= 0x3E7 )
      {
        v19 = 3;
      }
      else if ( v7 <= 0x270F )
      {
        v19 = 4;
      }
      else
      {
        v18 = v7;
        LODWORD(v19) = 1;
        while ( 1 )
        {
          v12 = (const char *)v18;
          v20 = v19;
          v19 = (unsigned int)(v19 + 4);
          v18 /= 0x2710u;
          if ( (unsigned __int64)v12 <= 0x1869F )
            break;
          if ( (unsigned __int64)v12 <= 0xF423F )
          {
            v53 = v55;
            v19 = (unsigned int)(v20 + 5);
            goto LABEL_26;
          }
          if ( v12 <= (const char *)&loc_98967F )
          {
            v19 = (unsigned int)(v20 + 6);
            break;
          }
          if ( (unsigned __int64)v12 <= 0x5F5E0FF )
          {
            v19 = (unsigned int)(v20 + 7);
            break;
          }
        }
      }
      v53 = v55;
LABEL_26:
      sub_2240A50(&v53, v19, 0, v12, v11);
      v21 = v53;
      v22 = "000102030405060708091011121314151617181920212223242526272829303132333435363738394041424344454647484950515253"
            "54555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
      v23 = v54 - 1;
      do
      {
        v24 = v7
            - 20 * (v7 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v7 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
        v25 = v7;
        v7 /= 0x64u;
        v26 = a00010203040506_0[2 * v24 + 1];
        LOBYTE(v24) = a00010203040506_0[2 * v24];
        v21[v23] = v26;
        v27 = (unsigned int)(v23 - 1);
        v23 -= 2;
        v21[v27] = v24;
      }
      while ( v25 > 0x270F );
      if ( v25 <= 0x3E7 )
        goto LABEL_29;
    }
    v21[1] = a00010203040506_0[2 * v7 + 1];
    *v21 = a00010203040506_0[2 * v7];
    goto LABEL_30;
  }
  v53 = v55;
  sub_2240A50(&v53, 1, 0, v12, v11);
  v21 = v53;
LABEL_29:
  *v21 = v7 + 48;
LABEL_30:
  v51[1] = 1;
  LOWORD(v52[0]) = 77;
  v51[0] = v52;
  if ( (unsigned __int64)(v54 + 1) <= 0xF || v53 == v55 || (unsigned __int64)(v54 + 1) > v55[0] )
  {
    v28 = (__m128i *)sub_2241490(v51, v53, v54, v22);
    v56 = &v58;
    v29 = (__m128i *)v28->m128i_i64[0];
    v30 = v28 + 1;
    if ( (__m128i *)v28->m128i_i64[0] != &v28[1] )
    {
LABEL_34:
      v56 = v29;
      v58.m128i_i64[0] = v28[1].m128i_i64[0];
      goto LABEL_35;
    }
  }
  else
  {
    v28 = (__m128i *)sub_2241130(&v53, 0, 0, v52, 1);
    v56 = &v58;
    v29 = (__m128i *)v28->m128i_i64[0];
    v30 = v28 + 1;
    if ( (__m128i *)v28->m128i_i64[0] != &v28[1] )
      goto LABEL_34;
  }
  v58 = _mm_loadu_si128(v28 + 1);
LABEL_35:
  v31 = v28->m128i_i64[1];
  v57 = v31;
  v28->m128i_i64[0] = (__int64)v30;
  v28->m128i_i64[1] = 0;
  v28[1].m128i_i8[0] = 0;
  if ( v57 == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v32 = (__m128i *)sub_2241490(&v56, "_", 1, v31);
  v59 = &v61;
  if ( (__m128i *)v32->m128i_i64[0] == &v32[1] )
  {
    v61 = _mm_loadu_si128(v32 + 1);
  }
  else
  {
    v59 = (__m128i *)v32->m128i_i64[0];
    v61.m128i_i64[0] = v32[1].m128i_i64[0];
  }
  v33 = v32->m128i_i64[1];
  v32[1].m128i_i8[0] = 0;
  v60 = v33;
  v32->m128i_i64[0] = (__int64)v32[1].m128i_i64;
  v34 = v59;
  v32->m128i_i64[1] = 0;
  v35 = 15;
  v36 = 15;
  if ( v34 != &v61 )
    v36 = v61.m128i_i64[0];
  v37 = v60 + v63;
  if ( v60 + v63 > v36 )
  {
    if ( v62 != v64 )
      v35 = v64[0];
    if ( v37 <= v35 )
    {
      v38 = (__m128i *)sub_2241130(&v62, 0, 0, v34, v60);
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v39 = v38->m128i_i64[0];
      v40 = v38 + 1;
      if ( (__m128i *)v38->m128i_i64[0] != &v38[1] )
        goto LABEL_45;
LABEL_87:
      a1[1] = _mm_loadu_si128(v38 + 1);
      goto LABEL_46;
    }
  }
  v38 = (__m128i *)sub_2241490(&v59, v62, v63, v37);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  v39 = v38->m128i_i64[0];
  v40 = v38 + 1;
  if ( (__m128i *)v38->m128i_i64[0] == &v38[1] )
    goto LABEL_87;
LABEL_45:
  a1->m128i_i64[0] = v39;
  a1[1].m128i_i64[0] = v38[1].m128i_i64[0];
LABEL_46:
  a1->m128i_i64[1] = v38->m128i_i64[1];
  v38->m128i_i64[0] = (__int64)v40;
  v38->m128i_i64[1] = 0;
  v38[1].m128i_i8[0] = 0;
  if ( v59 != &v61 )
    j_j___libc_free_0(v59, v61.m128i_i64[0] + 1);
  if ( v56 != &v58 )
    j_j___libc_free_0(v56, v58.m128i_i64[0] + 1);
  if ( (_QWORD *)v51[0] != v52 )
    j_j___libc_free_0(v51[0], v52[0] + 1LL);
  if ( v53 != v55 )
    j_j___libc_free_0(v53, v55[0] + 1LL);
  if ( v62 != v64 )
    j_j___libc_free_0(v62, v64[0] + 1LL);
  return a1;
}
