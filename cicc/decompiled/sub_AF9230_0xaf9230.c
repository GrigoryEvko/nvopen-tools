// Function: sub_AF9230
// Address: 0xaf9230
//
unsigned __int64 __fastcall sub_AF9230(int *a1, __int64 *a2, __int64 *a3, __int8 *a4, __int64 *a5)
{
  int v8; // r8d
  __int8 *v9; // rax
  __int64 v10; // r8
  __int8 *v11; // rax
  __int64 v12; // r8
  __int8 *v13; // rax
  __int64 v14; // r15
  __int8 *v15; // rdi
  __int8 v16; // al
  char *v17; // r14
  char *v18; // rcx
  __int64 v19; // r8
  __int8 *v20; // rax
  __int64 v21; // r14
  signed __int64 v23; // rbx
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // rax
  __m128i v26; // xmm2
  __m128i v27; // xmm3
  __m128i v28; // [rsp+10h] [rbp-110h] BYREF
  __m128i v29; // [rsp+20h] [rbp-100h] BYREF
  __m128i v30; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v31; // [rsp+40h] [rbp-E0h]
  __int64 v32; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v33; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v34; // [rsp+60h] [rbp-C0h] BYREF
  __int64 src; // [rsp+68h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v37; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v38; // [rsp+C0h] [rbp-60h]
  __m128i v39; // [rsp+D0h] [rbp-50h]
  __int64 v40; // [rsp+E0h] [rbp-40h]
  __int64 (__fastcall *v41)(); // [rsp+E8h] [rbp-38h]

  v8 = *a1;
  memset(dest, 0, sizeof(dest));
  v37 = 0u;
  v38 = 0u;
  v39 = 0u;
  v40 = 0;
  v41 = sub_C64CA0;
  v32 = 0;
  v9 = sub_AF6D70(dest, &v32, dest[0].m128i_i8, (unsigned __int64)&v37, v8);
  v10 = *a2;
  v33 = v32;
  v11 = sub_AF8740(dest, &v33, v9, (unsigned __int64)&v37, v10);
  v12 = *a3;
  v34 = v33;
  v13 = sub_AF70F0(dest, &v34, v11, (unsigned __int64)&v37, v12);
  v14 = v34;
  v15 = v13;
  v16 = *a4;
  v17 = v15 + 1;
  LOBYTE(src) = v16;
  if ( v15 + 1 <= (__int8 *)&v37 )
  {
    *v15 = v16;
  }
  else
  {
    memcpy(v15, &src, (char *)&v37 - v15);
    if ( v14 )
    {
      v14 += 64;
      sub_AC2A10((unsigned __int64 *)&v37, dest);
      v18 = (char *)((char *)&v37 - v15);
    }
    else
    {
      v14 = 64;
      sub_AC28A0((unsigned __int64 *)&v28, dest[0].m128i_i64, (unsigned __int64)v41);
      v26 = _mm_loadu_si128(&v29);
      v27 = _mm_loadu_si128(&v30);
      v18 = (char *)((char *)&v37 - v15);
      v37 = _mm_loadu_si128(&v28);
      v40 = v31;
      v38 = v26;
      v39 = v27;
    }
    v17 = &dest[0].m128i_i8[1LL - (_QWORD)v18];
    if ( v17 > (char *)&v37 )
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v18, 1LL - (_QWORD)v18);
  }
  v19 = *a5;
  src = v14;
  v20 = sub_AF70F0(dest, &src, v17, (unsigned __int64)&v37, v19);
  v21 = src;
  if ( !src )
    return sub_AC25F0(dest, v20 - (__int8 *)dest, (__int64)v41);
  v23 = v20 - (__int8 *)dest;
  sub_AF1140(dest[0].m128i_i8, v20, v37.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v37, dest);
  v24 = v37.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v40 ^ v39.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v40 ^ v39.m128i_i64[0]))
         ^ v40))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v40 ^ v39.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v40 ^ v39.m128i_i64[0]))
          ^ v40)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v21 + v23) >> 47) ^ (v21 + v23));
  v25 = 0x9DDFEA08EB382D69LL
      * (v24
       ^ (v38.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v37.m128i_i64[1] ^ ((unsigned __int64)v37.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v39.m128i_i64[1] ^ v38.m128i_i64[1]))
            ^ v39.m128i_i64[1]
            ^ ((0x9DDFEA08EB382D69LL * (v39.m128i_i64[1] ^ v38.m128i_i64[1])) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v39.m128i_i64[1] ^ v38.m128i_i64[1]))
           ^ v39.m128i_i64[1]
           ^ ((0x9DDFEA08EB382D69LL * (v39.m128i_i64[1] ^ v38.m128i_i64[1])) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v25 ^ v24 ^ (v25 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v25 ^ v24 ^ (v25 >> 47))));
}
