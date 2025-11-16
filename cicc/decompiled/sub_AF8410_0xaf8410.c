// Function: sub_AF8410
// Address: 0xaf8410
//
unsigned __int64 __fastcall sub_AF8410(int *a1, __int8 *a2, __int64 *a3)
{
  int v4; // r8d
  __int8 *v5; // rax
  __int64 v6; // r15
  __int8 *v7; // rdi
  __int8 v8; // al
  char *v9; // r14
  char *v10; // rcx
  __int64 v11; // r8
  __int8 *v12; // rax
  __int64 v13; // r14
  signed __int64 v15; // rbx
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rax
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __m128i v20; // [rsp+10h] [rbp-100h] BYREF
  __m128i v21; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v22; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v23; // [rsp+40h] [rbp-D0h]
  __int64 v24; // [rsp+50h] [rbp-C0h] BYREF
  __int64 src; // [rsp+58h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v27; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v28; // [rsp+B0h] [rbp-60h]
  __m128i v29; // [rsp+C0h] [rbp-50h]
  __int64 v30; // [rsp+D0h] [rbp-40h]
  __int64 (__fastcall *v31)(); // [rsp+D8h] [rbp-38h]

  v4 = *a1;
  v27 = 0u;
  v28 = 0u;
  v29 = 0u;
  v30 = 0;
  v31 = sub_C64CA0;
  v24 = 0;
  memset(dest, 0, sizeof(dest));
  v5 = sub_AF6D70(dest, &v24, dest[0].m128i_i8, (unsigned __int64)&v27, v4);
  v6 = v24;
  v7 = v5;
  v8 = *a2;
  v9 = v7 + 1;
  LOBYTE(src) = *a2;
  if ( v7 + 1 <= (__int8 *)&v27 )
  {
    *v7 = v8;
  }
  else
  {
    memcpy(v7, &src, (char *)&v27 - v7);
    if ( v6 )
    {
      v6 += 64;
      sub_AC2A10((unsigned __int64 *)&v27, dest);
      v10 = (char *)((char *)&v27 - v7);
    }
    else
    {
      v6 = 64;
      sub_AC28A0((unsigned __int64 *)&v20, dest[0].m128i_i64, (unsigned __int64)v31);
      v18 = _mm_loadu_si128(&v21);
      v19 = _mm_loadu_si128(&v22);
      v10 = (char *)((char *)&v27 - v7);
      v27 = _mm_loadu_si128(&v20);
      v30 = v23;
      v28 = v18;
      v29 = v19;
    }
    v9 = &dest[0].m128i_i8[1LL - (_QWORD)v10];
    if ( v9 > (char *)&v27 )
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v10, 1LL - (_QWORD)v10);
  }
  v11 = *a3;
  src = v6;
  v12 = sub_AF70F0(dest, &src, v9, (unsigned __int64)&v27, v11);
  v13 = src;
  if ( !src )
    return sub_AC25F0(dest, v12 - (__int8 *)dest, (__int64)v31);
  v15 = v12 - (__int8 *)dest;
  sub_AF1140(dest[0].m128i_i8, v12, v27.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v27, dest);
  v16 = v27.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v30 ^ v29.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v30 ^ v29.m128i_i64[0]))
         ^ v30))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v30 ^ v29.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v30 ^ v29.m128i_i64[0]))
          ^ v30)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v13 + v15) >> 47) ^ (v13 + v15));
  v17 = 0x9DDFEA08EB382D69LL
      * (v16
       ^ (v28.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v27.m128i_i64[1] ^ ((unsigned __int64)v27.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v29.m128i_i64[1] ^ v28.m128i_i64[1])) >> 47)
            ^ (0x9DDFEA08EB382D69LL * (v29.m128i_i64[1] ^ v28.m128i_i64[1]))
            ^ v29.m128i_i64[1])) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v29.m128i_i64[1] ^ v28.m128i_i64[1])) >> 47)
           ^ (0x9DDFEA08EB382D69LL * (v29.m128i_i64[1] ^ v28.m128i_i64[1]))
           ^ v29.m128i_i64[1])))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v17 ^ v16 ^ (v17 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v17 ^ v16 ^ (v17 >> 47))));
}
