// Function: sub_BAC460
// Address: 0xbac460
//
__m128i *__fastcall sub_BAC460(__m128i *a1, _QWORD *a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  unsigned __int64 *v6; // rax
  __int8 *v7; // r14
  size_t v8; // r12
  _BYTE *m128i_i8; // rdi
  __int64 v10; // rdx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rdx
  __int64 v13; // rsi
  int v14; // eax
  _BYTE *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  char v19; // r10
  __int64 v20; // r9
  __m128i *v21; // r15
  __m128i *v22; // rax
  __int64 v23; // rcx
  __m128i *v24; // rdx
  size_t v26; // rdx
  __int64 v27; // rax
  _QWORD v28[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v29[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD *v30; // [rsp+20h] [rbp-50h] BYREF
  __int64 v31; // [rsp+28h] [rbp-48h]
  _QWORD v32[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = (unsigned __int64 *)(*a2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*a2 & 1) != 0 )
  {
    sub_BD5D20(v6[1]);
    a5 = v10;
    v6 = (unsigned __int64 *)(*a2 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v10 )
    {
      v7 = (__int8 *)v6[1];
      if ( (*a2 & 1) != 0 )
      {
        v7 = (__int8 *)sub_BD5D20(v6[1]);
        v8 = v26;
LABEL_5:
        m128i_i8 = a1[1].m128i_i8;
        a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
        if ( !v7 )
        {
          a1->m128i_i64[1] = 0;
          a1[1].m128i_i8[0] = 0;
          return a1;
        }
        v30 = (_QWORD *)v8;
        if ( v8 > 0xF )
        {
          v27 = sub_22409D0(a1, &v30, 0);
          a1->m128i_i64[0] = v27;
          m128i_i8 = (_BYTE *)v27;
          a1[1].m128i_i64[0] = (__int64)v30;
        }
        else
        {
          if ( v8 == 1 )
          {
            a1[1].m128i_i8[0] = *v7;
LABEL_9:
            a1->m128i_i64[1] = v8;
            m128i_i8[v8] = 0;
            return a1;
          }
          if ( !v8 )
            goto LABEL_9;
        }
        memcpy(m128i_i8, v7, v8);
        v8 = (size_t)v30;
        m128i_i8 = (_BYTE *)a1->m128i_i64[0];
        goto LABEL_9;
      }
LABEL_4:
      v8 = v6[2];
      goto LABEL_5;
    }
  }
  else if ( v6[2] )
  {
    v7 = (__int8 *)v6[1];
    goto LABEL_4;
  }
  v11 = *v6;
  if ( *v6 > 9 )
  {
    if ( v11 <= 0x63 )
    {
      v30 = v32;
      sub_2240A50(&v30, 2, 0, a4, a5);
      v15 = v30;
    }
    else
    {
      if ( v11 <= 0x3E7 )
      {
        v13 = 3;
      }
      else if ( v11 <= 0x270F )
      {
        v13 = 4;
      }
      else
      {
        v12 = *v6;
        LODWORD(v13) = 1;
        while ( 1 )
        {
          a4 = v12;
          v14 = v13;
          v13 = (unsigned int)(v13 + 4);
          v12 /= 0x2710u;
          if ( a4 <= 0x1869F )
            break;
          if ( a4 <= 0xF423F )
          {
            v30 = v32;
            v13 = (unsigned int)(v14 + 5);
            goto LABEL_21;
          }
          if ( a4 <= (unsigned __int64)&loc_98967F )
          {
            v13 = (unsigned int)(v14 + 6);
            break;
          }
          if ( a4 <= 0x5F5E0FF )
          {
            v13 = (unsigned int)(v14 + 7);
            break;
          }
        }
      }
      v30 = v32;
LABEL_21:
      sub_2240A50(&v30, v13, 0, a4, a5);
      v15 = v30;
      LODWORD(v16) = v31 - 1;
      do
      {
        v17 = v11
            - 20
            * (v11 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v11 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
        v18 = v11;
        v11 /= 0x64u;
        v19 = a00010203040506_0[2 * v17 + 1];
        LOBYTE(v17) = a00010203040506_0[2 * v17];
        v15[(unsigned int)v16] = v19;
        v20 = (unsigned int)(v16 - 1);
        v16 = (unsigned int)(v16 - 2);
        v15[v20] = v17;
      }
      while ( v18 > 0x270F );
      if ( v18 <= 0x3E7 )
        goto LABEL_24;
    }
    v15[1] = a00010203040506_0[2 * v11 + 1];
    *v15 = a00010203040506_0[2 * v11];
    goto LABEL_25;
  }
  v30 = v32;
  sub_2240A50(&v30, 1, 0, a4, a5);
  v15 = v30;
LABEL_24:
  *v15 = v11 + 48;
LABEL_25:
  v21 = a1 + 1;
  v28[0] = v29;
  LOWORD(v29[0]) = 64;
  v28[1] = 1;
  if ( (unsigned __int64)(v31 + 1) > 0xF && v30 != v32 && (unsigned __int64)(v31 + 1) <= v32[0] )
  {
    v22 = (__m128i *)sub_2241130(&v30, 0, 0, v29, 1);
    a1->m128i_i64[0] = (__int64)v21;
    v23 = v22->m128i_i64[0];
    v24 = v22 + 1;
    if ( (__m128i *)v22->m128i_i64[0] != &v22[1] )
      goto LABEL_29;
LABEL_41:
    a1[1] = _mm_loadu_si128(v22 + 1);
    goto LABEL_30;
  }
  v22 = (__m128i *)sub_2241490(v28, v30, v31, v16);
  a1->m128i_i64[0] = (__int64)v21;
  v23 = v22->m128i_i64[0];
  v24 = v22 + 1;
  if ( (__m128i *)v22->m128i_i64[0] == &v22[1] )
    goto LABEL_41;
LABEL_29:
  a1->m128i_i64[0] = v23;
  a1[1].m128i_i64[0] = v22[1].m128i_i64[0];
LABEL_30:
  a1->m128i_i64[1] = v22->m128i_i64[1];
  v22->m128i_i64[0] = (__int64)v24;
  v22->m128i_i64[1] = 0;
  v22[1].m128i_i8[0] = 0;
  if ( (_QWORD *)v28[0] != v29 )
    j_j___libc_free_0(v28[0], v29[0] + 1LL);
  if ( v30 != v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  return a1;
}
