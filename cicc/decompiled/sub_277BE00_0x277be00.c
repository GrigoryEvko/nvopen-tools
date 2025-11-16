// Function: sub_277BE00
// Address: 0x277be00
//
unsigned __int64 __fastcall sub_277BE00(int *a1, __int64 *a2, __int64 *a3)
{
  int v3; // r8d
  __int8 *v4; // rax
  __int64 v5; // rbx
  __int8 *v6; // rdi
  __int64 v7; // rax
  __m128i *v8; // r15
  char *v9; // r14
  char *v10; // r14
  __int64 v11; // rax
  unsigned __int64 v12; // r15
  __m128i v14; // xmm4
  __m128i v15; // xmm5
  __m128i v16; // xmm6
  char *v17; // r8
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __m128i v23; // [rsp+10h] [rbp-110h] BYREF
  __m128i v24; // [rsp+20h] [rbp-100h] BYREF
  __m128i v25; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v26; // [rsp+40h] [rbp-E0h]
  __int64 v27; // [rsp+58h] [rbp-C8h] BYREF
  __int64 src; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v29; // [rsp+68h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v31; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v32; // [rsp+C0h] [rbp-60h]
  __m128i v33; // [rsp+D0h] [rbp-50h]
  __int64 v34; // [rsp+E0h] [rbp-40h]
  void (__fastcall *v35)(__int64, __int64); // [rsp+E8h] [rbp-38h]

  v3 = *a1;
  v31 = 0u;
  v32 = 0u;
  v33 = 0u;
  v34 = 0;
  v35 = sub_C64CA0;
  v27 = 0;
  memset(dest, 0, sizeof(dest));
  v4 = sub_AF6D70(dest, &v27, dest[0].m128i_i8, (unsigned __int64)&v31, v3);
  v5 = v27;
  v6 = v4;
  v7 = *a2;
  v8 = (__m128i *)(v6 + 8);
  src = *a2;
  if ( v6 + 8 <= (__int8 *)&v31 )
  {
    *(_QWORD *)v6 = v7;
  }
  else
  {
    v9 = (char *)((char *)&v31 - v6);
    memcpy(v6, &src, (char *)&v31 - v6);
    if ( v5 )
    {
      v5 += 64;
      sub_AC2A10((unsigned __int64 *)&v31, dest);
    }
    else
    {
      v5 = 64;
      sub_AC28A0((unsigned __int64 *)&v23, dest[0].m128i_i64, (unsigned __int64)v35);
      v20 = _mm_loadu_si128(&v24);
      v21 = _mm_loadu_si128(&v25);
      v31 = _mm_loadu_si128(&v23);
      v34 = v26;
      v32 = v20;
      v33 = v21;
    }
    v8 = (__m128i *)((char *)dest + 8LL - (_QWORD)v9);
    if ( v8 > &v31 )
LABEL_5:
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v9, 8LL - (_QWORD)v9);
  }
  v10 = &v8->m128i_i8[8];
  v11 = *a3;
  v29 = *a3;
  if ( &v8->m128i_u64[1] > (unsigned __int64 *)&v31 )
  {
    memcpy(v8, &v29, (char *)&v31 - (char *)v8);
    if ( v5 )
    {
      v5 += 64;
      sub_AC2A10((unsigned __int64 *)&v31, dest);
      v17 = (char *)((char *)&v31 - (char *)v8);
    }
    else
    {
      v5 = 64;
      sub_AC28A0((unsigned __int64 *)&v23, dest[0].m128i_i64, (unsigned __int64)v35);
      v14 = _mm_loadu_si128(&v23);
      v15 = _mm_loadu_si128(&v24);
      v16 = _mm_loadu_si128(&v25);
      v34 = v26;
      v17 = (char *)((char *)&v31 - (char *)v8);
      v31 = v14;
      v32 = v15;
      v33 = v16;
    }
    v10 = &dest[0].m128i_i8[8LL - (_QWORD)v17];
    if ( v10 > (char *)&v31 )
      goto LABEL_5;
    v12 = 8LL - (_QWORD)v17;
    memcpy(dest, (char *)&v29 + (_QWORD)v17, 8LL - (_QWORD)v17);
    if ( !v5 )
      return sub_AC25F0(dest, v12, (__int64)v35);
  }
  else
  {
    v8->m128i_i64[0] = v11;
    v12 = v10 - (char *)dest;
    if ( !v5 )
      return sub_AC25F0(&dest[0], v12, (__int64)v35);
  }
  sub_2778790(dest[0].m128i_i8, v10, v31.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v31, dest);
  v18 = v31.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v34 ^ v33.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v34 ^ v33.m128i_i64[0]))
         ^ v34))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v34 ^ v33.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v34 ^ v33.m128i_i64[0]))
          ^ v34)) >> 47))
      - 0x4B6D499041670D8DLL * (((v5 + v12) >> 47) ^ (v5 + v12));
  v19 = 0x9DDFEA08EB382D69LL
      * (v18
       ^ (v32.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v31.m128i_i64[1] ^ ((unsigned __int64)v31.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v33.m128i_i64[1] ^ v32.m128i_i64[1]))
            ^ v33.m128i_i64[1]
            ^ ((0x9DDFEA08EB382D69LL * (v33.m128i_i64[1] ^ v32.m128i_i64[1])) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v33.m128i_i64[1] ^ v32.m128i_i64[1]))
           ^ v33.m128i_i64[1]
           ^ ((0x9DDFEA08EB382D69LL * (v33.m128i_i64[1] ^ v32.m128i_i64[1])) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * ((v19 >> 47) ^ v19 ^ v18)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v19 >> 47) ^ v19 ^ v18)));
}
