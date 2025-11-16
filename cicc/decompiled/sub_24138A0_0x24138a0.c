// Function: sub_24138A0
// Address: 0x24138a0
//
void __fastcall sub_24138A0(__int64 a1)
{
  char *v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rax
  _BYTE *v4; // rsi
  __int64 v5; // rdx
  unsigned __int64 *v6; // rax
  size_t v7; // rcx
  __m128i *v8; // rax
  unsigned __int64 *v9; // rax
  size_t v10; // rcx
  unsigned __int64 v11; // rdx
  char *v12; // r15
  unsigned __int64 *v13; // rbx
  __m128i *v14; // r13
  __m128i *v15; // rdi
  size_t v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // r14
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  size_t v24; // rdx
  size_t v25; // [rsp+8h] [rbp-178h]
  _BYTE *v26[2]; // [rsp+40h] [rbp-140h] BYREF
  _QWORD v27[2]; // [rsp+50h] [rbp-130h] BYREF
  char *v28; // [rsp+60h] [rbp-120h] BYREF
  size_t v29; // [rsp+68h] [rbp-118h]
  _QWORD v30[2]; // [rsp+70h] [rbp-110h] BYREF
  __int128 v31; // [rsp+80h] [rbp-100h] BYREF
  _QWORD v32[2]; // [rsp+90h] [rbp-F0h] BYREF
  __m128i *v33; // [rsp+A0h] [rbp-E0h]
  size_t v34; // [rsp+A8h] [rbp-D8h]
  __m128i v35; // [rsp+B0h] [rbp-D0h] BYREF
  _BYTE *v36; // [rsp+C0h] [rbp-C0h] BYREF
  size_t v37; // [rsp+C8h] [rbp-B8h]
  _QWORD v38[2]; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned __int64 v39[2]; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+F0h] [rbp-90h] BYREF
  __m128i *v41; // [rsp+100h] [rbp-80h] BYREF
  __int64 v42; // [rsp+108h] [rbp-78h]
  __m128i v43; // [rsp+110h] [rbp-70h] BYREF
  __m128i *v44; // [rsp+120h] [rbp-60h] BYREF
  size_t n; // [rsp+128h] [rbp-58h]
  __m128i v46; // [rsp+130h] [rbp-50h] BYREF
  __int16 v47; // [rsp+140h] [rbp-40h]

  v1 = (char *)sub_BD5D20(a1);
  v26[0] = v27;
  sub_240D760((__int64 *)v26, v1, (__int64)&v1[v2]);
  v28 = (char *)v30;
  sub_240D760((__int64 *)&v28, ".dfsan", (__int64)"");
  v41 = &v43;
  sub_240DB00((__int64 *)&v41, v26[0], (__int64)&v26[0][(unsigned __int64)v26[1]]);
  sub_2241490((unsigned __int64 *)&v41, v28, v29);
  v47 = 260;
  v44 = (__m128i *)&v41;
  sub_BD6B50((unsigned __int8 *)a1, (const char **)&v44);
  if ( v41 != &v43 )
    j_j___libc_free_0((unsigned __int64)v41);
  v3 = *(_QWORD *)(a1 + 40);
  v4 = *(_BYTE **)(v3 + 88);
  v5 = *(_QWORD *)(v3 + 96);
  *(_QWORD *)&v31 = v32;
  sub_240DB00((__int64 *)&v31, v4, (__int64)&v4[v5]);
  sub_8FD6D0((__int64)&v44, ".symver ", v26);
  if ( n == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_62;
  v6 = sub_2241490((unsigned __int64 *)&v44, ",", 1u);
  v33 = &v35;
  if ( (unsigned __int64 *)*v6 == v6 + 2 )
  {
    v35 = _mm_loadu_si128((const __m128i *)v6 + 1);
  }
  else
  {
    v33 = (__m128i *)*v6;
    v35.m128i_i64[0] = v6[2];
  }
  v7 = v6[1];
  *((_BYTE *)v6 + 16) = 0;
  v34 = v7;
  *v6 = (unsigned __int64)(v6 + 2);
  v6[1] = 0;
  if ( v44 != &v46 )
    j_j___libc_free_0((unsigned __int64)v44);
  v25 = sub_22416F0((__int64 *)&v31, v33->m128i_i8, 0, v34);
  if ( v25 != -1 )
  {
    sub_8FD6D0((__int64)v39, ".symver ", v26);
    v8 = (__m128i *)sub_2241490(v39, v28, v29);
    v41 = &v43;
    if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
    {
      v43 = _mm_loadu_si128(v8 + 1);
    }
    else
    {
      v41 = (__m128i *)v8->m128i_i64[0];
      v43.m128i_i64[0] = v8[1].m128i_i64[0];
    }
    v42 = v8->m128i_i64[1];
    v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
    v8->m128i_i64[1] = 0;
    v8[1].m128i_i8[0] = 0;
    if ( v42 != 0x3FFFFFFFFFFFFFFFLL )
    {
      v9 = sub_2241490((unsigned __int64 *)&v41, ",", 1u);
      v44 = &v46;
      if ( (unsigned __int64 *)*v9 == v9 + 2 )
      {
        v46 = _mm_loadu_si128((const __m128i *)v9 + 1);
      }
      else
      {
        v44 = (__m128i *)*v9;
        v46.m128i_i64[0] = v9[2];
      }
      n = v9[1];
      *v9 = (unsigned __int64)(v9 + 2);
      v9[1] = 0;
      v10 = *((_QWORD *)&v31 + 1);
      *((_BYTE *)v9 + 16) = 0;
      v11 = v10 - v25;
      if ( v34 <= v10 - v25 )
        v11 = v34;
      if ( v25 > v10 )
        sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::replace", v25, v10);
      sub_2241130((unsigned __int64 *)&v31, v25, v11, v44, n);
      if ( v44 != &v46 )
        j_j___libc_free_0((unsigned __int64)v44);
      if ( v41 != &v43 )
        j_j___libc_free_0((unsigned __int64)v41);
      if ( (__int64 *)v39[0] != &v40 )
        j_j___libc_free_0(v39[0]);
      v12 = sub_22417D0((__int64 *)&v31, 64, 0);
      if ( v12 == (char *)-1LL )
      {
        v47 = 1283;
        v44 = (__m128i *)"unsupported .symver: ";
        v46 = (__m128i)v31;
        sub_C64D30((__int64)&v44, 1u);
      }
      v36 = v38;
      sub_240DB00((__int64 *)&v36, v28, (__int64)&v28[v29]);
      if ( v37 != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)&v36, "@", 1u);
        if ( (unsigned __int64)v12 > *((_QWORD *)&v31 + 1) )
          sub_222CF80(
            "%s: __pos (which is %zu) > this->size() (which is %zu)",
            "basic_string::replace",
            (size_t)v12,
            *((size_t *)&v31 + 1));
        sub_2241130((unsigned __int64 *)&v31, (size_t)v12, *((_QWORD *)&v31 + 1) != (_QWORD)v12, v36, v37);
        if ( v36 != (_BYTE *)v38 )
          j_j___libc_free_0((unsigned __int64)v36);
        v13 = *(unsigned __int64 **)(a1 + 40);
        v44 = &v46;
        v14 = (__m128i *)(v13 + 13);
        sub_240D760((__int64 *)&v44, (_BYTE *)v31, v31 + *((_QWORD *)&v31 + 1));
        v15 = (__m128i *)v13[11];
        if ( v44 == &v46 )
        {
          v24 = n;
          if ( n )
          {
            if ( n == 1 )
              v15->m128i_i8[0] = v46.m128i_i8[0];
            else
              memcpy(v15, &v46, n);
            v24 = n;
            v15 = (__m128i *)v13[11];
          }
          v13[12] = v24;
          v15->m128i_i8[v24] = 0;
          v15 = v44;
          goto LABEL_32;
        }
        v16 = n;
        v17 = v46.m128i_i64[0];
        if ( v15 == v14 )
        {
          v13[11] = (unsigned __int64)v44;
          v13[12] = v16;
          v13[13] = v17;
        }
        else
        {
          v18 = v13[13];
          v13[11] = (unsigned __int64)v44;
          v13[12] = v16;
          v13[13] = v17;
          if ( v15 )
          {
            v44 = v15;
            v46.m128i_i64[0] = v18;
LABEL_32:
            n = 0;
            v15->m128i_i8[0] = 0;
            if ( v44 != &v46 )
              j_j___libc_free_0((unsigned __int64)v44);
            v19 = v13[12];
            if ( v19 )
            {
              v20 = v13[11];
              if ( *(_BYTE *)(v20 + v19 - 1) != 10 )
              {
                v21 = v19 + 1;
                if ( (__m128i *)v20 == v14 )
                  v22 = 15;
                else
                  v22 = v13[13];
                if ( v21 > v22 )
                {
                  sub_2240BB0(v13 + 11, v13[12], 0, 0, 1u);
                  v20 = v13[11];
                }
                *(_BYTE *)(v20 + v19) = 10;
                v23 = v13[11];
                v13[12] = v21;
                *(_BYTE *)(v23 + v19 + 1) = 0;
              }
            }
            goto LABEL_36;
          }
        }
        v44 = &v46;
        v15 = &v46;
        goto LABEL_32;
      }
    }
LABEL_62:
    sub_4262D8((__int64)"basic_string::append");
  }
LABEL_36:
  if ( v33 != &v35 )
    j_j___libc_free_0((unsigned __int64)v33);
  if ( (_QWORD *)v31 != v32 )
    j_j___libc_free_0(v31);
  if ( v28 != (char *)v30 )
    j_j___libc_free_0((unsigned __int64)v28);
  if ( (_QWORD *)v26[0] != v27 )
    j_j___libc_free_0((unsigned __int64)v26[0]);
}
