// Function: sub_AFA150
// Address: 0xafa150
//
unsigned __int64 __fastcall sub_AFA150(__int64 *a1, __int64 *a2, __int8 *a3)
{
  __int64 v4; // r8
  __int8 *v5; // rax
  __int64 v6; // r8
  __int8 *v7; // rax
  __int64 v8; // r15
  __int8 *v9; // rdi
  __int8 v10; // al
  char *v11; // r14
  char *v12; // rbx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __m128i v18; // [rsp+10h] [rbp-110h] BYREF
  __m128i v19; // [rsp+20h] [rbp-100h] BYREF
  __m128i v20; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v21; // [rsp+40h] [rbp-E0h]
  __int8 src; // [rsp+5Fh] [rbp-C1h] BYREF
  __int64 v23; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v24; // [rsp+68h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v26; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v27; // [rsp+C0h] [rbp-60h]
  __m128i v28; // [rsp+D0h] [rbp-50h]
  __int64 v29; // [rsp+E0h] [rbp-40h]
  __int64 (__fastcall *v30)(); // [rsp+E8h] [rbp-38h]

  v4 = *a1;
  memset(dest, 0, sizeof(dest));
  v26 = 0u;
  v27 = 0u;
  v28 = 0u;
  v29 = 0;
  v30 = sub_C64CA0;
  v23 = 0;
  v5 = sub_AF8740(dest, &v23, dest[0].m128i_i8, (unsigned __int64)&v26, v4);
  v6 = *a2;
  v24 = v23;
  v7 = sub_AF70F0(dest, &v24, v5, (unsigned __int64)&v26, v6);
  v8 = v24;
  v9 = v7;
  v10 = *a3;
  v11 = v9 + 1;
  src = *a3;
  if ( v9 + 1 <= (__int8 *)&v26 )
  {
    *v9 = v10;
  }
  else
  {
    v12 = (char *)((char *)&v26 - v9);
    memcpy(v9, &src, (char *)&v26 - v9);
    if ( v8 )
    {
      v8 += 64;
      sub_AC2A10((unsigned __int64 *)&v26, dest);
    }
    else
    {
      v8 = 64;
      sub_AC28A0((unsigned __int64 *)&v18, dest[0].m128i_i64, (unsigned __int64)v30);
      v16 = _mm_loadu_si128(&v19);
      v17 = _mm_loadu_si128(&v20);
      v26 = _mm_loadu_si128(&v18);
      v29 = v21;
      v27 = v16;
      v28 = v17;
    }
    v11 = &dest[0].m128i_i8[1LL - (_QWORD)v12];
    if ( v11 > (char *)&v26 )
      BUG();
    memcpy(dest, &src + (_QWORD)v12, 1LL - (_QWORD)v12);
  }
  if ( !v8 )
    return sub_AC25F0(dest, v11 - (char *)dest, (__int64)v30);
  sub_AF1140(dest[0].m128i_i8, v11, v26.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v26, dest);
  v14 = v26.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v29 ^ v28.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v29 ^ v28.m128i_i64[0]))
         ^ v29))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v29 ^ v28.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v29 ^ v28.m128i_i64[0]))
          ^ v29)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v8 + v11 - (char *)dest) >> 47) ^ (v8 + v11 - (char *)dest));
  v15 = 0x9DDFEA08EB382D69LL
      * (v14
       ^ (v27.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v26.m128i_i64[1] ^ ((unsigned __int64)v26.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1])) >> 47)
            ^ (0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1]))
            ^ v28.m128i_i64[1])) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1])) >> 47)
           ^ (0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1]))
           ^ v28.m128i_i64[1])))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * ((v15 >> 47) ^ v15 ^ v14)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v15 >> 47) ^ v15 ^ v14)));
}
