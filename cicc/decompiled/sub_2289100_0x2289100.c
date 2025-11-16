// Function: sub_2289100
// Address: 0x2289100
//
void __fastcall sub_2289100(__int64 a1, __int64 a2, __int64 a3)
{
  __m128i *v4; // rax
  __m128i *v5; // rax
  _BYTE *v6; // rsi
  __int64 v7; // rdx
  __m128i *v8; // rax
  __m128i *v9; // rax
  __m128i *v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rax
  _DWORD *v15; // rdx
  __int64 v16; // r9
  _QWORD *v17; // rdi
  _BYTE *v18; // rax
  __int64 v19; // rsi
  _QWORD *v20; // rax
  __m128i *v21; // rdx
  __m128i si128; // xmm0
  size_t v23; // rdx
  size_t v24; // rdx
  _OWORD *v25; // rdi
  _OWORD *v26; // rdi
  unsigned __int8 *dest; // [rsp+30h] [rbp-1D0h]
  size_t v30; // [rsp+38h] [rbp-1C8h]
  _QWORD v31[2]; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 v32[2]; // [rsp+50h] [rbp-1B0h] BYREF
  _QWORD v33[2]; // [rsp+60h] [rbp-1A0h] BYREF
  __m128i *v34; // [rsp+70h] [rbp-190h] BYREF
  __int64 (__fastcall **v35)(); // [rsp+78h] [rbp-188h]
  __m128i v36; // [rsp+80h] [rbp-180h] BYREF
  void *v37[2]; // [rsp+90h] [rbp-170h] BYREF
  _QWORD v38[2]; // [rsp+A0h] [rbp-160h] BYREF
  __int16 v39; // [rsp+B0h] [rbp-150h]
  unsigned __int64 v40[2]; // [rsp+C0h] [rbp-140h] BYREF
  _OWORD v41[4]; // [rsp+D0h] [rbp-130h] BYREF
  __m128i *v42; // [rsp+110h] [rbp-F0h] BYREF
  __int64 v43; // [rsp+118h] [rbp-E8h]
  __m128i v44; // [rsp+120h] [rbp-E0h] BYREF
  unsigned int v45; // [rsp+138h] [rbp-C8h]
  _BYTE v46[16]; // [rsp+148h] [rbp-B8h] BYREF
  void (__fastcall *v47)(_BYTE *, _BYTE *, __int64); // [rsp+158h] [rbp-A8h]
  _OWORD *v48; // [rsp+170h] [rbp-90h] BYREF
  size_t n; // [rsp+178h] [rbp-88h]
  _OWORD src[8]; // [rsp+180h] [rbp-80h] BYREF

  dest = (unsigned __int8 *)v31;
  LOBYTE(v31[0]) = 0;
  if ( qword_4FDB050 )
  {
    v32[0] = (__int64)v33;
    sub_11F4570(v32, (_BYTE *)qword_4FDB048, qword_4FDB048 + qword_4FDB050);
    if ( v32[1] != 0x3FFFFFFFFFFFFFFFLL )
    {
      sub_2241490((unsigned __int64 *)v32, ".", 1u);
      v4 = (__m128i *)sub_2241490((unsigned __int64 *)v32, (char *)qword_4FDAF48, qword_4FDAF50);
      v34 = &v36;
      if ( (__m128i *)v4->m128i_i64[0] == &v4[1] )
      {
        v36 = _mm_loadu_si128(v4 + 1);
      }
      else
      {
        v34 = (__m128i *)v4->m128i_i64[0];
        v36.m128i_i64[0] = v4[1].m128i_i64[0];
      }
      v35 = (__int64 (__fastcall **)())v4->m128i_i64[1];
      v4->m128i_i64[0] = (__int64)v4[1].m128i_i64;
      v4->m128i_i64[1] = 0;
      v4[1].m128i_i8[0] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)v35) > 3 )
      {
        v5 = (__m128i *)sub_2241490((unsigned __int64 *)&v34, ".dot", 4u);
        v48 = src;
        if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
        {
          src[0] = _mm_loadu_si128(v5 + 1);
        }
        else
        {
          v48 = (_OWORD *)v5->m128i_i64[0];
          *(_QWORD *)&src[0] = v5[1].m128i_i64[0];
        }
        n = v5->m128i_u64[1];
        v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
        v5->m128i_i64[1] = 0;
        v5[1].m128i_i8[0] = 0;
        if ( v48 == src )
        {
          v23 = n;
          if ( n )
          {
            if ( n == 1 )
            {
              LOBYTE(v31[0]) = src[0];
              v23 = 1;
            }
            else
            {
              memcpy(v31, src, n);
              v23 = n;
            }
          }
          v30 = v23;
          *((_BYTE *)v31 + v23) = 0;
          v26 = v48;
        }
        else
        {
          dest = (unsigned __int8 *)v48;
          v30 = n;
          v31[0] = *(_QWORD *)&src[0];
          v48 = src;
          v26 = src;
        }
        n = 0;
        *(_BYTE *)v26 = 0;
        if ( v48 != src )
          j_j___libc_free_0((unsigned __int64)v48);
        if ( v34 != &v36 )
          j_j___libc_free_0((unsigned __int64)v34);
        if ( (_QWORD *)v32[0] != v33 )
          j_j___libc_free_0(v32[0]);
        goto LABEL_35;
      }
    }
LABEL_69:
    sub_4262D8((__int64)"basic_string::append");
  }
  v6 = *(_BYTE **)(a1 + 168);
  v7 = *(_QWORD *)(a1 + 176);
  v37[0] = v38;
  sub_11F4570((__int64 *)v37, v6, (__int64)&v6[v7]);
  if ( v37[1] == (void *)0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_69;
  v8 = (__m128i *)sub_2241490((unsigned __int64 *)v37, ".", 1u);
  v40[0] = (unsigned __int64)v41;
  if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
  {
    v41[0] = _mm_loadu_si128(v8 + 1);
  }
  else
  {
    v40[0] = v8->m128i_i64[0];
    *(_QWORD *)&v41[0] = v8[1].m128i_i64[0];
  }
  v40[1] = v8->m128i_u64[1];
  v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
  v8->m128i_i64[1] = 0;
  v8[1].m128i_i8[0] = 0;
  v9 = (__m128i *)sub_2241490(v40, (char *)qword_4FDAF48, qword_4FDAF50);
  v42 = &v44;
  if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
  {
    v44 = _mm_loadu_si128(v9 + 1);
  }
  else
  {
    v42 = (__m128i *)v9->m128i_i64[0];
    v44.m128i_i64[0] = v9[1].m128i_i64[0];
  }
  v43 = v9->m128i_i64[1];
  v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
  v9->m128i_i64[1] = 0;
  v9[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v43) <= 3 )
    goto LABEL_69;
  v10 = (__m128i *)sub_2241490((unsigned __int64 *)&v42, ".dot", 4u);
  v48 = src;
  if ( (__m128i *)v10->m128i_i64[0] == &v10[1] )
  {
    src[0] = _mm_loadu_si128(v10 + 1);
  }
  else
  {
    v48 = (_OWORD *)v10->m128i_i64[0];
    *(_QWORD *)&src[0] = v10[1].m128i_i64[0];
  }
  n = v10->m128i_u64[1];
  v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
  v10->m128i_i64[1] = 0;
  v10[1].m128i_i8[0] = 0;
  if ( v48 == src )
  {
    v24 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        LOBYTE(v31[0]) = src[0];
        v24 = 1;
      }
      else
      {
        memcpy(v31, src, n);
        v24 = n;
      }
    }
    v30 = v24;
    *((_BYTE *)v31 + v24) = 0;
    v25 = v48;
  }
  else
  {
    dest = (unsigned __int8 *)v48;
    v30 = n;
    v31[0] = *(_QWORD *)&src[0];
    v48 = src;
    v25 = src;
  }
  n = 0;
  *(_BYTE *)v25 = 0;
  if ( v48 != src )
    j_j___libc_free_0((unsigned __int64)v48);
  if ( v42 != &v44 )
    j_j___libc_free_0((unsigned __int64)v42);
  if ( (_OWORD *)v40[0] != v41 )
    j_j___libc_free_0(v40[0]);
  if ( v37[0] != v38 )
    j_j___libc_free_0((unsigned __int64)v37[0]);
LABEL_35:
  v11 = sub_CB72A0();
  v12 = v11[4];
  v13 = (__int64)v11;
  if ( (unsigned __int64)(v11[3] - v12) <= 8 )
  {
    v13 = sub_CB6200((__int64)v11, "Writing '", 9u);
  }
  else
  {
    *(_BYTE *)(v12 + 8) = 39;
    *(_QWORD *)v12 = 0x20676E6974697257LL;
    v11[4] += 9LL;
  }
  v14 = sub_CB6200(v13, dest, v30);
  v15 = *(_DWORD **)(v14 + 32);
  if ( *(_QWORD *)(v14 + 24) - (_QWORD)v15 <= 3u )
  {
    sub_CB6200(v14, "'...", 4u);
  }
  else
  {
    *v15 = 774778407;
    *(_QWORD *)(v14 + 32) += 4LL;
  }
  LODWORD(v34) = 0;
  v35 = sub_2241E40();
  sub_CB7060((__int64)&v48, dest, v30, (__int64)&v34, 1u);
  sub_D12090((__int64)v40, a1);
  sub_2286A70((__int64)&v42, a1, (__int64)v40, a2, a3, v16);
  if ( (_DWORD)v34 )
  {
    v20 = sub_CB72A0();
    v21 = (__m128i *)v20[4];
    if ( v20[3] - (_QWORD)v21 <= 0x20u )
    {
      sub_CB6200((__int64)v20, "  error opening file for writing!", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
      v21[2].m128i_i8[0] = 33;
      *v21 = si128;
      v21[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
      v20[4] += 33LL;
    }
  }
  else
  {
    v39 = 257;
    v32[0] = (__int64)&v42;
    sub_2289010((__int64)&v48, (__int64)v32, 0, v37);
  }
  v17 = sub_CB72A0();
  v18 = (_BYTE *)v17[4];
  if ( (_BYTE *)v17[3] == v18 )
  {
    sub_CB6200((__int64)v17, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v18 = 10;
    ++v17[4];
  }
  if ( v47 )
    v47(v46, v46, 3);
  v19 = 16LL * v45;
  sub_C7D6A0(v44.m128i_i64[1], v19, 8);
  sub_D0FA70((__int64)v40);
  sub_CB5B00((int *)&v48, v19);
  if ( dest != (unsigned __int8 *)v31 )
    j_j___libc_free_0((unsigned __int64)dest);
}
