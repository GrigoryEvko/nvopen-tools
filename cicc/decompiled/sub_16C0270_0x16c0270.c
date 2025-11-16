// Function: sub_16C0270
// Address: 0x16c0270
//
__int64 *__fastcall sub_16C0270(__int64 *a1)
{
  __m128i *v2; // rax
  __m128i si128; // xmm0
  __m128i **v4; // rsi
  size_t v5; // r13
  _BYTE *v6; // r15
  void *v7; // rdi
  _BYTE *v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __m128i *v13; // [rsp+0h] [rbp-E0h] BYREF
  __int16 v14; // [rsp+10h] [rbp-D0h]
  __m128i v15; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+30h] [rbp-B0h] BYREF
  __m128i src; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD v18[6]; // [rsp+50h] [rbp-90h] BYREF
  __m128i v19; // [rsp+80h] [rbp-60h] BYREF
  _QWORD v20[10]; // [rsp+90h] [rbp-50h] BYREF

  v19.m128i_i64[0] = (__int64)v20;
  src.m128i_i64[0] = 28;
  v2 = (__m128i *)sub_22409D0(&v19, &src, 0);
  si128 = _mm_load_si128((const __m128i *)&xmmword_42AEE90);
  v19.m128i_i64[0] = (__int64)v2;
  v20[0] = src.m128i_i64[0];
  qmemcpy(&v2[1], "inux-unknown", 12);
  *v2 = si128;
  v19.m128i_i64[1] = src.m128i_i64[0];
  *(_BYTE *)(v19.m128i_i64[0] + src.m128i_i64[0]) = 0;
  sub_16C00D0(&v15, &v19);
  if ( (_QWORD *)v19.m128i_i64[0] != v20 )
    j_j___libc_free_0(v19.m128i_i64[0], v20[0] + 1LL);
  sub_16E1150(&v19, v15.m128i_i64[0], v15.m128i_i64[1]);
  v4 = &v13;
  v14 = 260;
  v13 = &v19;
  sub_16E1010(&src);
  if ( (_QWORD *)v19.m128i_i64[0] != v20 )
  {
    v4 = (__m128i **)(v20[0] + 1LL);
    j_j___libc_free_0(v19.m128i_i64[0], v20[0] + 1LL);
  }
  if ( (unsigned __int8)sub_16E2920(&src, v4) )
  {
    sub_16E2940(&v19, &src);
    v9 = (_BYTE *)src.m128i_i64[0];
    if ( (_QWORD *)v19.m128i_i64[0] == v20 )
    {
      v12 = v19.m128i_i64[1];
      if ( v19.m128i_i64[1] )
      {
        if ( v19.m128i_i64[1] == 1 )
          *(_BYTE *)src.m128i_i64[0] = v20[0];
        else
          memcpy((void *)src.m128i_i64[0], v20, v19.m128i_u64[1]);
        v12 = v19.m128i_i64[1];
        v9 = (_BYTE *)src.m128i_i64[0];
      }
      src.m128i_i64[1] = v12;
      v9[v12] = 0;
      v9 = (_BYTE *)v19.m128i_i64[0];
      goto LABEL_22;
    }
    if ( (_QWORD *)src.m128i_i64[0] == v18 )
    {
      src = v19;
      v18[0] = v20[0];
    }
    else
    {
      v10 = v18[0];
      src = v19;
      v18[0] = v20[0];
      if ( v9 )
      {
        v19.m128i_i64[0] = (__int64)v9;
        v20[0] = v10;
        goto LABEL_22;
      }
    }
    v19.m128i_i64[0] = (__int64)v20;
    v9 = v20;
LABEL_22:
    v19.m128i_i64[1] = 0;
    *v9 = 0;
    v18[2] = v20[2];
    v18[3] = v20[3];
    v18[4] = v20[4];
    if ( (_QWORD *)v19.m128i_i64[0] != v20 )
      j_j___libc_free_0(v19.m128i_i64[0], v20[0] + 1LL);
  }
  v5 = src.m128i_u64[1];
  v6 = (_BYTE *)src.m128i_i64[0];
  v7 = a1 + 2;
  *a1 = (__int64)(a1 + 2);
  if ( &v6[v5] && !v6 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v19.m128i_i64[0] = v5;
  if ( v5 > 0xF )
  {
    v11 = sub_22409D0(a1, &v19, 0);
    *a1 = v11;
    v7 = (void *)v11;
    a1[2] = v19.m128i_i64[0];
LABEL_25:
    memcpy(v7, v6, v5);
    v5 = v19.m128i_i64[0];
    v7 = (void *)*a1;
    goto LABEL_11;
  }
  if ( v5 == 1 )
  {
    *((_BYTE *)a1 + 16) = *v6;
    goto LABEL_11;
  }
  if ( v5 )
    goto LABEL_25;
LABEL_11:
  a1[1] = v5;
  *((_BYTE *)v7 + v5) = 0;
  if ( (_QWORD *)src.m128i_i64[0] != v18 )
    j_j___libc_free_0(src.m128i_i64[0], v18[0] + 1LL);
  if ( (__int64 *)v15.m128i_i64[0] != &v16 )
    j_j___libc_free_0(v15.m128i_i64[0], v16 + 1);
  return a1;
}
