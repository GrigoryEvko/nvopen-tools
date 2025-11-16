// Function: sub_370EDF0
// Address: 0x370edf0
//
__int64 *__fastcall sub_370EDF0(__int64 *a1, __int64 a2, unsigned __int16 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r15
  unsigned __int16 v9; // ax
  bool v10; // zf
  unsigned __int64 v12; // rdx
  char *v13; // r8
  _QWORD *v14; // rax
  size_t v15; // rbx
  __int64 v16; // rdx
  char *v17; // rsi
  __m128i *v18; // rax
  __m128i *v19; // rax
  unsigned int v20; // r8d
  unsigned __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 v23; // rax
  char *v24; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v25[2]; // [rsp+10h] [rbp-C0h] BYREF
  _QWORD dest[2]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v27[2]; // [rsp+30h] [rbp-A0h] BYREF
  _BYTE v28[16]; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v29; // [rsp+50h] [rbp-80h] BYREF
  __int64 v30; // [rsp+58h] [rbp-78h]
  __m128i v31; // [rsp+60h] [rbp-70h] BYREF
  __m128i v32; // [rsp+70h] [rbp-60h] BYREF
  __m128i v33; // [rsp+80h] [rbp-50h] BYREF
  __int16 v34; // [rsp+90h] [rbp-40h]

  v6 = (_QWORD *)(a2 + 16);
  v32.m128i_i64[0] = 0x10000FEF4LL;
  sub_3700D50(&v29, a2 + 16, 0x10000FEF4LL, a4, a5, a6);
  if ( (v29 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v29 & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  v9 = *a3;
  v10 = *(_QWORD *)(a2 + 72) == 0;
  *(_BYTE *)(a2 + 14) = 1;
  *(_WORD *)(a2 + 12) = v9;
  if ( !v10 && !*(_QWORD *)(a2 + 56) && !*(_QWORD *)(a2 + 64) )
  {
    v13 = sub_370C640(*a3);
    v14 = dest;
    v15 = v12;
    v25[0] = (unsigned __int64)dest;
    if ( &v13[v12] && !v13 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v32.m128i_i64[0] = v12;
    if ( v12 > 0xF )
    {
      v24 = v13;
      v23 = sub_22409D0((__int64)v25, (unsigned __int64 *)&v32, 0);
      v13 = v24;
      v25[0] = v23;
      v22 = (_QWORD *)v23;
      dest[0] = v32.m128i_i64[0];
    }
    else
    {
      if ( v12 == 1 )
      {
        LOBYTE(dest[0]) = *v13;
LABEL_13:
        v25[1] = v12;
        *((_BYTE *)v14 + v12) = 0;
        v17 = (char *)sub_370CB70(v6, *a3);
        if ( v17 )
        {
          v27[0] = (__int64)v28;
          sub_370CD40(v27, v17, (__int64)&v17[v16]);
        }
        else
        {
          v28[0] = 0;
          v27[0] = (__int64)v28;
          v27[1] = 0;
        }
        v18 = (__m128i *)sub_2241130((unsigned __int64 *)v27, 0, 0, " ( ", 3u);
        v29 = (unsigned __int64)&v31;
        if ( (__m128i *)v18->m128i_i64[0] == &v18[1] )
        {
          v31 = _mm_loadu_si128(v18 + 1);
        }
        else
        {
          v29 = v18->m128i_i64[0];
          v31.m128i_i64[0] = v18[1].m128i_i64[0];
        }
        v30 = v18->m128i_i64[1];
        v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
        v18->m128i_i64[1] = 0;
        v18[1].m128i_i8[0] = 0;
        if ( v30 == 0x3FFFFFFFFFFFFFFFLL || v30 == 4611686018427387902LL )
          sub_4262D8((__int64)"basic_string::append");
        v19 = (__m128i *)sub_2241490(&v29, " )", 2u);
        v32.m128i_i64[0] = (__int64)&v33;
        if ( (__m128i *)v19->m128i_i64[0] == &v19[1] )
        {
          v33 = _mm_loadu_si128(v19 + 1);
        }
        else
        {
          v32.m128i_i64[0] = v19->m128i_i64[0];
          v33.m128i_i64[0] = v19[1].m128i_i64[0];
        }
        v32.m128i_i64[1] = v19->m128i_i64[1];
        v19->m128i_i64[0] = (__int64)v19[1].m128i_i64;
        v19->m128i_i64[1] = 0;
        v19[1].m128i_i8[0] = 0;
        sub_2241490(v25, (char *)v32.m128i_i64[0], v32.m128i_u64[1]);
        sub_2240A30((unsigned __int64 *)&v32);
        sub_2240A30(&v29);
        sub_2240A30((unsigned __int64 *)v27);
        sub_8FD6D0((__int64)&v29, "Member kind: ", v25);
        v34 = 260;
        v32.m128i_i64[0] = (__int64)&v29;
        sub_370E990((unsigned __int64 *)v27, v6, a3, &v32, v20);
        sub_2240A30(&v29);
        v21 = v27[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (v27[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          *a1 = 0;
          v27[0] = v21 | 1;
          sub_9C6670(a1, v27);
          sub_9C66B0(v27);
          sub_2240A30(v25);
          return a1;
        }
        v27[0] = 0;
        sub_9C66B0(v27);
        sub_2240A30(v25);
        goto LABEL_4;
      }
      if ( !v12 )
        goto LABEL_13;
      v22 = dest;
    }
    memcpy(v22, v13, v15);
    v12 = v32.m128i_i64[0];
    v14 = (_QWORD *)v25[0];
    goto LABEL_13;
  }
LABEL_4:
  *a1 = 1;
  return a1;
}
