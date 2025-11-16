// Function: sub_38B5120
// Address: 0x38b5120
//
__int64 __fastcall sub_38B5120(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned __int8 v4; // al
  __m128i *v5; // rsi
  __int64 v6; // rsi
  const __m128i *v7; // r12
  const __m128i *v8; // rdx
  __m128i *v9; // rax
  __m128i *v10; // r12
  unsigned int *v11; // r15
  __int64 v12; // rbx
  __int64 v13; // r13
  unsigned __int64 *v14; // rax
  __int64 v15; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v16; // [rsp+18h] [rbp-B8h]
  __int64 i; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v18; // [rsp+28h] [rbp-A8h]
  unsigned __int64 *v19; // [rsp+30h] [rbp-A0h]
  unsigned __int8 v21; // [rsp+4Bh] [rbp-85h]
  __int32 v22; // [rsp+4Ch] [rbp-84h]
  __m128i v23; // [rsp+50h] [rbp-80h] BYREF
  __int64 v24; // [rsp+60h] [rbp-70h]
  __int64 v25; // [rsp+68h] [rbp-68h]
  __int64 v26; // [rsp+70h] [rbp-60h] BYREF
  int v27; // [rsp+78h] [rbp-58h] BYREF
  _QWORD *v28; // [rsp+80h] [rbp-50h]
  int *v29; // [rsp+88h] [rbp-48h]
  int *v30; // [rsp+90h] [rbp-40h]
  __int64 v31; // [rsp+98h] [rbp-38h]

  v15 = a1;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  else
  {
    v27 = 0;
    v28 = 0;
    v29 = &v27;
    v30 = &v27;
    v31 = 0;
    while ( 1 )
    {
      v4 = sub_38B4EF0(a1, v23.m128i_i64, &v26, (__int64)(a3[1] - *a3) >> 4);
      if ( v4 )
      {
        v21 = v4;
        goto LABEL_33;
      }
      v5 = (__m128i *)a3[1];
      if ( v5 == (__m128i *)a3[2] )
      {
        sub_142E240((__int64)a3, v5, &v23);
      }
      else
      {
        if ( v5 )
        {
          *v5 = _mm_loadu_si128(&v23);
          v5 = (__m128i *)a3[1];
        }
        a3[1] = v5 + 1;
      }
      if ( *(_DWORD *)(a1 + 64) != 4 )
        break;
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    }
    v6 = 13;
    v21 = sub_388AF10(a1, 13, "expected ')' here");
    if ( !v21 )
    {
      for ( i = (__int64)v29; (int *)i != &v27; i = sub_220EEE0(i) )
      {
        v7 = *(const __m128i **)(i + 48);
        v8 = *(const __m128i **)(i + 40);
        v22 = *(_DWORD *)(i + 32);
        v16 = (char *)v7 - (char *)v8;
        if ( v7 == v8 )
        {
          v18 = 0;
          if ( v7 == v8 )
            goto LABEL_32;
        }
        else
        {
          if ( v16 > 0x7FFFFFFFFFFFFFF0LL )
            sub_4261EA(a1, v6, v8);
          v18 = sub_22077B0(v16);
          v7 = *(const __m128i **)(i + 48);
          v8 = *(const __m128i **)(i + 40);
          if ( v8 == v7 )
            goto LABEL_30;
        }
        v9 = (__m128i *)v18;
        v10 = (__m128i *)(v18 + (char *)v7 - (char *)v8);
        do
        {
          if ( v9 )
            *v9 = _mm_loadu_si128(v8);
          ++v9;
          ++v8;
        }
        while ( v9 != v10 );
        v11 = (unsigned int *)v18;
        if ( (__m128i *)v18 == v10 )
        {
LABEL_31:
          v6 = v16;
          j_j___libc_free_0(v18);
          goto LABEL_32;
        }
        do
        {
          while ( 1 )
          {
            v12 = *v11;
            v23.m128i_i64[1] = 0;
            v13 = *((_QWORD *)v11 + 1);
            v23.m128i_i32[0] = v22;
            v24 = 0;
            v25 = 0;
            v14 = (unsigned __int64 *)sub_38917E0((_QWORD *)(v15 + 1344), v23.m128i_i32);
            if ( v23.m128i_i64[1] )
            {
              v19 = v14;
              j_j___libc_free_0(v23.m128i_u64[1]);
              v14 = v19;
            }
            v23.m128i_i64[1] = v13;
            v23.m128i_i64[0] = *a3 + 16 * v12;
            v6 = v14[6];
            if ( v6 != v14[7] )
              break;
            v11 += 4;
            sub_38952E0(v14 + 5, (const __m128i *)v6, &v23);
            if ( v10 == (__m128i *)v11 )
              goto LABEL_30;
          }
          if ( v6 )
          {
            *(__m128i *)v6 = _mm_loadu_si128(&v23);
            v6 = v14[6];
          }
          v6 += 16;
          v11 += 4;
          v14[6] = v6;
        }
        while ( v10 != (__m128i *)v11 );
LABEL_30:
        if ( v18 )
          goto LABEL_31;
LABEL_32:
        a1 = i;
      }
    }
LABEL_33:
    sub_3889030(v28);
  }
  return v21;
}
