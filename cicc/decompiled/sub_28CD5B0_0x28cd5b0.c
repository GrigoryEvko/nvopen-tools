// Function: sub_28CD5B0
// Address: 0x28cd5b0
//
unsigned __int64 __fastcall sub_28CD5B0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r8
  char *v3; // rax
  __int64 v4; // r15
  char *v5; // rdi
  __int64 v6; // rax
  char *v7; // r14
  char *v8; // rbx
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // rsi
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __m128i v14; // [rsp+10h] [rbp-100h] BYREF
  __m128i v15; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v16; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v17; // [rsp+40h] [rbp-D0h]
  __int64 v18; // [rsp+50h] [rbp-C0h] BYREF
  __int64 src; // [rsp+58h] [rbp-B8h] BYREF
  _OWORD dest[4]; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v21; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v22; // [rsp+B0h] [rbp-60h]
  __m128i v23; // [rsp+C0h] [rbp-50h]
  __int64 v24; // [rsp+D0h] [rbp-40h]
  void (__fastcall *v25)(__int64, __int64); // [rsp+D8h] [rbp-38h]

  v2 = *a1;
  v21 = 0u;
  v22 = 0u;
  v23 = 0u;
  v24 = 0;
  v25 = sub_C64CA0;
  v18 = 0;
  memset(dest, 0, sizeof(dest));
  v3 = (char *)sub_CA5190((unsigned __int64 *)dest, &v18, dest, (unsigned __int64)&v21, v2);
  v4 = v18;
  v5 = v3;
  v6 = *a2;
  v7 = v5 + 8;
  src = *a2;
  if ( v5 + 8 <= (char *)&v21 )
  {
    *(_QWORD *)v5 = v6;
  }
  else
  {
    v8 = (char *)((char *)&v21 - v5);
    memcpy(v5, &src, (char *)&v21 - v5);
    if ( v4 )
    {
      v4 += 64;
      sub_AC2A10((unsigned __int64 *)&v21, dest);
    }
    else
    {
      v4 = 64;
      sub_AC28A0((unsigned __int64 *)&v14, (__int64 *)dest, (unsigned __int64)v25);
      v12 = _mm_loadu_si128(&v15);
      v13 = _mm_loadu_si128(&v16);
      v21 = _mm_loadu_si128(&v14);
      v24 = v17;
      v22 = v12;
      v23 = v13;
    }
    v7 = (char *)dest + 8LL - (_QWORD)v8;
    if ( v7 > (char *)&v21 )
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v8, 8LL - (_QWORD)v8);
  }
  if ( !v4 )
    return sub_AC25F0(dest, v7 - (char *)dest, (__int64)v25);
  sub_28C7830((char *)dest, v7, v21.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v21, dest);
  v10 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v4 + v7 - (char *)dest) >> 47) ^ (v4 + v7 - (char *)dest))
      + v21.m128i_i64[0]
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v24 ^ v23.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v24 ^ v23.m128i_i64[0]))
          ^ v24)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v24 ^ v23.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v24 ^ v23.m128i_i64[0]))
         ^ v24)));
  v11 = v22.m128i_i64[0] - 0x4B6D499041670D8DLL * (v21.m128i_i64[1] ^ ((unsigned __int64)v21.m128i_i64[1] >> 47));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v10
           ^ (0x9DDFEA08EB382D69LL
            * (v10
             ^ (v11
              - 0x622015F714C7D297LL
              * (((0x9DDFEA08EB382D69LL
                 * (((0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1])) >> 47)
                  ^ (0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1]))
                  ^ v23.m128i_i64[1])) >> 47)
               ^ (0x9DDFEA08EB382D69LL
                * (((0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1])) >> 47)
                 ^ (0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1]))
                 ^ v23.m128i_i64[1]))))))
           ^ ((0x9DDFEA08EB382D69LL
             * (v10
              ^ (v11
               - 0x622015F714C7D297LL
               * (((0x9DDFEA08EB382D69LL
                  * (((0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1])) >> 47)
                   ^ (0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1]))
                   ^ v23.m128i_i64[1])) >> 47)
                ^ (0x9DDFEA08EB382D69LL
                 * (((0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1])) >> 47)
                  ^ (0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1]))
                  ^ v23.m128i_i64[1])))))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v10
          ^ (0x9DDFEA08EB382D69LL
           * (v10
            ^ (v11
             - 0x622015F714C7D297LL
             * (((0x9DDFEA08EB382D69LL
                * (((0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1])) >> 47)
                 ^ (0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1]))
                 ^ v23.m128i_i64[1])) >> 47)
              ^ (0x9DDFEA08EB382D69LL
               * (((0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1])) >> 47)
                ^ (0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1]))
                ^ v23.m128i_i64[1]))))))
          ^ ((0x9DDFEA08EB382D69LL
            * (v10
             ^ (v11
              - 0x622015F714C7D297LL
              * (((0x9DDFEA08EB382D69LL
                 * (((0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1])) >> 47)
                  ^ (0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1]))
                  ^ v23.m128i_i64[1])) >> 47)
               ^ (0x9DDFEA08EB382D69LL
                * (((0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1])) >> 47)
                 ^ (0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1]))
                 ^ v23.m128i_i64[1])))))) >> 47))));
}
