// Function: sub_15B2420
// Address: 0x15b2420
//
unsigned __int64 __fastcall sub_15B2420(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r8
  __int8 *v3; // rax
  __int64 v4; // rcx
  __int8 *v5; // rdi
  __int64 v6; // rax
  char *v7; // r15
  char *v8; // r14
  __int64 v9; // rcx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rsi
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __int64 v16; // [rsp+8h] [rbp-108h]
  __int64 v17; // [rsp+8h] [rbp-108h]
  __int64 v18; // [rsp+8h] [rbp-108h]
  __m128i v19; // [rsp+10h] [rbp-100h] BYREF
  __m128i v20; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v21; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v22; // [rsp+40h] [rbp-D0h]
  __int64 v23; // [rsp+50h] [rbp-C0h] BYREF
  __int64 src; // [rsp+58h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v26; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v27; // [rsp+B0h] [rbp-60h]
  __m128i v28; // [rsp+C0h] [rbp-50h]
  __int64 v29; // [rsp+D0h] [rbp-40h]
  __int64 v30; // [rsp+D8h] [rbp-38h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v11 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v11 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v11;
    sub_2207640(byte_4F99930);
  }
  v2 = *a1;
  v30 = qword_4F99938;
  v23 = 0;
  v3 = sub_15B2320(dest, &v23, dest[0].m128i_i8, (unsigned __int64)&v26, v2);
  v4 = v23;
  v5 = v3;
  v6 = *a2;
  v7 = v5 + 8;
  src = *a2;
  if ( v5 + 8 <= (__int8 *)&v26 )
  {
    *(_QWORD *)v5 = v6;
  }
  else
  {
    v16 = v23;
    v8 = (char *)((char *)&v26 - v5);
    memcpy(v5, &src, (char *)&v26 - v5);
    if ( v16 )
    {
      sub_1593A20((unsigned __int64 *)&v26, dest);
      v9 = v16 + 64;
    }
    else
    {
      sub_15938B0((unsigned __int64 *)&v19, dest[0].m128i_i64, v30);
      v14 = _mm_loadu_si128(&v20);
      v9 = 64;
      v15 = _mm_loadu_si128(&v21);
      v26 = _mm_loadu_si128(&v19);
      v29 = v22;
      v27 = v14;
      v28 = v15;
    }
    v17 = v9;
    v7 = &dest[0].m128i_i8[8LL - (_QWORD)v8];
    if ( v7 > (char *)&v26 )
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v8, 8LL - (_QWORD)v8);
    v4 = v17;
  }
  if ( !v4 )
    return sub_1593600(dest, v7 - (char *)dest, v30);
  v18 = v4;
  sub_15AF6E0(dest[0].m128i_i8, v7, v26.m128i_i8);
  sub_1593A20((unsigned __int64 *)&v26, dest);
  v12 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v18 + v7 - (char *)dest) >> 47) ^ (v18 + v7 - (char *)dest))
      + v26.m128i_i64[0]
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v29 ^ v28.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v29 ^ v28.m128i_i64[0]))
          ^ v29)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v29 ^ v28.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v29 ^ v28.m128i_i64[0]))
         ^ v29)));
  v13 = v27.m128i_i64[0] - 0x4B6D499041670D8DLL * (v26.m128i_i64[1] ^ ((unsigned __int64)v26.m128i_i64[1] >> 47));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v12
           ^ (0x9DDFEA08EB382D69LL
            * (v12
             ^ (v13
              - 0x622015F714C7D297LL
              * (((0x9DDFEA08EB382D69LL
                 * ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1]))
                  ^ v28.m128i_i64[1]
                  ^ ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1])) >> 47))) >> 47)
               ^ (0x9DDFEA08EB382D69LL
                * ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1]))
                 ^ v28.m128i_i64[1]
                 ^ ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1])) >> 47)))))))
           ^ ((0x9DDFEA08EB382D69LL
             * (v12
              ^ (v13
               - 0x622015F714C7D297LL
               * (((0x9DDFEA08EB382D69LL
                  * ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1]))
                   ^ v28.m128i_i64[1]
                   ^ ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1])) >> 47))) >> 47)
                ^ (0x9DDFEA08EB382D69LL
                 * ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1]))
                  ^ v28.m128i_i64[1]
                  ^ ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1])) >> 47))))))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v12
          ^ (0x9DDFEA08EB382D69LL
           * (v12
            ^ (v13
             - 0x622015F714C7D297LL
             * (((0x9DDFEA08EB382D69LL
                * ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1]))
                 ^ v28.m128i_i64[1]
                 ^ ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1])) >> 47))) >> 47)
              ^ (0x9DDFEA08EB382D69LL
               * ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1]))
                ^ v28.m128i_i64[1]
                ^ ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1])) >> 47)))))))
          ^ ((0x9DDFEA08EB382D69LL
            * (v12
             ^ (v13
              - 0x622015F714C7D297LL
              * (((0x9DDFEA08EB382D69LL
                 * ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1]))
                  ^ v28.m128i_i64[1]
                  ^ ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1])) >> 47))) >> 47)
               ^ (0x9DDFEA08EB382D69LL
                * ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1]))
                 ^ v28.m128i_i64[1]
                 ^ ((0x9DDFEA08EB382D69LL * (v28.m128i_i64[1] ^ v27.m128i_i64[1])) >> 47))))))) >> 47))));
}
