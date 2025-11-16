// Function: sub_1E360F0
// Address: 0x1e360f0
//
__int8 *__fastcall sub_1E360F0(__m128i *dest, _QWORD *a2, char *a3, unsigned __int64 a4, char a5)
{
  __int8 *v5; // r14
  unsigned __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r10
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // r10
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rdi
  __int64 v22; // rsi
  unsigned __int64 v23; // rdi
  __int64 v24; // r11
  unsigned __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __m128i v29; // xmm1
  __m128i v30; // xmm2
  __int64 v31; // rax
  __m128i v32; // [rsp+10h] [rbp-80h] BYREF
  __m128i v33; // [rsp+20h] [rbp-70h] BYREF
  __m128i v34; // [rsp+30h] [rbp-60h] BYREF
  __int64 v35; // [rsp+40h] [rbp-50h]
  _BYTE src[52]; // [rsp+5Ch] [rbp-34h] BYREF

  v5 = a3 + 1;
  src[0] = a5;
  if ( a4 >= (unsigned __int64)(a3 + 1) )
  {
    *a3 = a5;
  }
  else
  {
    v9 = a4 - (_QWORD)a3;
    memcpy(a3, src, a4 - (_QWORD)a3);
    if ( *a2 )
    {
      v10 = dest[4].m128i_i64[1];
      v11 = dest[6].m128i_i64[0];
      v12 = dest[5].m128i_i64[0];
      v13 = v10 + dest[4].m128i_i64[0];
      v14 = __ROL8__(dest[3].m128i_i64[0] + v11 + v10, 22);
      v15 = dest[7].m128i_i64[0];
      v16 = dest->m128i_i64[0] - 0x4B6D499041670D8DLL * v11;
      v17 = dest[2].m128i_i64[1] + dest[5].m128i_i64[1] - 0x4B6D499041670D8DLL * v14;
      v18 = dest[6].m128i_i64[1];
      v19 = v15 ^ (0xB492B66FBE98F273LL * __ROL8__(dest->m128i_i64[1] + dest[5].m128i_i64[1] + v13, 27));
      v20 = dest[2].m128i_i64[0] + v15;
      dest[4].m128i_i64[1] = v17;
      v21 = v18 + v12;
      v22 = dest[1].m128i_i64[1] + v18;
      dest[5].m128i_i64[0] = v19;
      v23 = 0xB492B66FBE98F273LL * __ROL8__(v21, 31);
      v24 = v16 + dest[1].m128i_i64[0] + dest->m128i_i64[1];
      v25 = v23 + v20;
      dest[5].m128i_i64[1] = dest[1].m128i_i64[1] + v24;
      v26 = dest[2].m128i_i64[1] + dest[3].m128i_i64[0];
      dest[4].m128i_i64[0] = v23;
      dest[6].m128i_i64[0] = __ROL8__(v24, 20) + v16 + __ROR8__(v19 + v16 + v22, 21);
      v27 = __ROR8__(v25 + dest[1].m128i_i64[0] + dest[3].m128i_i64[1] + v17, 21);
      dest[6].m128i_i64[1] = dest[3].m128i_i64[1] + v25 + v26;
      dest[7].m128i_i64[0] = __ROL8__(v25 + v26, 20) + v25 + v27;
      *a2 += 64LL;
    }
    else
    {
      sub_15938B0((unsigned __int64 *)&v32, dest->m128i_i64, dest[7].m128i_u64[1]);
      v29 = _mm_loadu_si128(&v33);
      v30 = _mm_loadu_si128(&v34);
      v31 = v35;
      dest[4] = _mm_loadu_si128(&v32);
      dest[7].m128i_i64[0] = v31;
      dest[5] = v29;
      dest[6] = v30;
      *a2 = 64;
    }
    v5 = &dest->m128i_i8[1 - v9];
    if ( a4 < (unsigned __int64)v5 )
      abort();
    memcpy(dest, &src[v9], 1 - v9);
  }
  return v5;
}
