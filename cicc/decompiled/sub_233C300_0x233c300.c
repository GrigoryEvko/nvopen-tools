// Function: sub_233C300
// Address: 0x233c300
//
__m128i *__fastcall sub_233C300(__m128i *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __m128i *v4; // r15
  __int64 v7; // rbx
  __int64 v8; // r14
  __m128i v9; // xmm0
  __int64 v10; // rdx
  __int64 v11; // rax
  __m128i v12; // xmm1
  __int64 v14; // [rsp+8h] [rbp-68h]
  _QWORD v15[2]; // [rsp+10h] [rbp-60h] BYREF
  __m128i v16; // [rsp+20h] [rbp-50h] BYREF
  __int64 v17; // [rsp+30h] [rbp-40h]
  __int64 v18; // [rsp+38h] [rbp-38h]

  v4 = a1;
  if ( a4 == 3 && *(_WORD *)a3 == 27745 && *(_BYTE *)(a3 + 2) == 108 )
  {
    a1[1].m128i_i64[0] = 0;
    a1[2].m128i_i8[0] = 1;
  }
  else
  {
    v7 = *(_QWORD *)(a2 + 2208);
    v8 = 32LL * *(unsigned int *)(a2 + 2216);
    v14 = v7 + v8;
    if ( v7 + v8 == v7 )
    {
LABEL_10:
      v4[2].m128i_i8[0] = 0;
    }
    else
    {
      while ( 1 )
      {
        v15[0] = a3;
        v15[1] = a4;
        if ( !*(_QWORD *)(v7 + 16) )
          sub_4263D6(a1, a2, a3);
        a2 = v7;
        a1 = &v16;
        (*(void (__fastcall **)(__m128i *, __int64, _QWORD *))(v7 + 24))(&v16, v7, v15);
        if ( v17 )
          break;
        v7 += 32;
        if ( v7 == v14 )
          goto LABEL_10;
      }
      v9 = _mm_loadu_si128(&v16);
      v10 = v4[1].m128i_i64[1];
      v4[1].m128i_i64[0] = v17;
      v11 = v18;
      v4[2].m128i_i8[0] = 1;
      v12 = _mm_loadu_si128(v4);
      *v4 = v9;
      v4[1].m128i_i64[1] = v11;
      v17 = 0;
      v18 = v10;
      v16 = v12;
      sub_A17130((__int64)&v16);
    }
  }
  return v4;
}
