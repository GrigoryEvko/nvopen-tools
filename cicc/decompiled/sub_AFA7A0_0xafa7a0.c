// Function: sub_AFA7A0
// Address: 0xafa7a0
//
unsigned __int64 __fastcall sub_AFA7A0(__int64 *a1, _QWORD *a2)
{
  __int64 v2; // r8
  __int8 *v3; // r13
  __int64 v4; // r15
  __int64 v5; // rax
  char *v6; // r8
  char *v7; // rbx
  signed __int64 v8; // r13
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __m128i v14; // [rsp+10h] [rbp-100h] BYREF
  __m128i v15; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v16; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v17; // [rsp+40h] [rbp-D0h]
  __int64 v18; // [rsp+50h] [rbp-C0h] BYREF
  __int64 src; // [rsp+58h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v21; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v22; // [rsp+B0h] [rbp-60h]
  __m128i v23; // [rsp+C0h] [rbp-50h]
  __int64 v24; // [rsp+D0h] [rbp-40h]
  __int64 (__fastcall *v25)(); // [rsp+D8h] [rbp-38h]

  v2 = *a1;
  memset(dest, 0, sizeof(dest));
  v21 = 0u;
  v22 = 0u;
  v23 = 0u;
  v24 = 0;
  v25 = sub_C64CA0;
  v18 = 0;
  v3 = sub_AF8740(dest, &v18, dest[0].m128i_i8, (unsigned __int64)&v21, v2);
  v4 = v18;
  v5 = sub_C94880(*a2, a2[1]);
  v6 = v3 + 8;
  src = v5;
  if ( v3 + 8 <= (__int8 *)&v21 )
  {
    *(_QWORD *)v3 = v5;
  }
  else
  {
    v7 = (char *)((char *)&v21 - v3);
    memcpy(v3, &src, (char *)&v21 - v3);
    if ( v4 )
    {
      v4 += 64;
      sub_AC2A10((unsigned __int64 *)&v21, dest);
    }
    else
    {
      v4 = 64;
      sub_AC28A0((unsigned __int64 *)&v14, dest[0].m128i_i64, (unsigned __int64)v25);
      v12 = _mm_loadu_si128(&v15);
      v13 = _mm_loadu_si128(&v16);
      v21 = _mm_loadu_si128(&v14);
      v24 = v17;
      v22 = v12;
      v23 = v13;
    }
    if ( (__m128i *)((char *)dest + 8LL - (_QWORD)v7) > &v21 )
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v7, 8LL - (_QWORD)v7);
    v6 = &dest[0].m128i_i8[8LL - (_QWORD)v7];
  }
  v8 = v6 - (char *)dest;
  if ( !v4 )
    return sub_AC25F0(dest, v6 - (char *)dest, (__int64)v25);
  sub_AF1140(dest[0].m128i_i8, v6, v21.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v21, dest);
  v10 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v4 + v8) >> 47) ^ (v4 + v8))
      + v21.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v24 ^ v23.m128i_i64[0]))
         ^ v24
         ^ ((0x9DDFEA08EB382D69LL * (v24 ^ v23.m128i_i64[0])) >> 47)))
       ^ ((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v24 ^ v23.m128i_i64[0]))
          ^ v24
          ^ ((0x9DDFEA08EB382D69LL * (v24 ^ v23.m128i_i64[0])) >> 47))) >> 47));
  v11 = 0x9DDFEA08EB382D69LL
      * (v10
       ^ (v22.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v21.m128i_i64[1] ^ ((unsigned __int64)v21.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1])) >> 47)
            ^ (0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1]))
            ^ v23.m128i_i64[1])) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1])) >> 47)
           ^ (0x9DDFEA08EB382D69LL * (v23.m128i_i64[1] ^ v22.m128i_i64[1]))
           ^ v23.m128i_i64[1])))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * ((v11 >> 47) ^ v11 ^ v10)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v11 >> 47) ^ v11 ^ v10)));
}
