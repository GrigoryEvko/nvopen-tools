// Function: sub_AFB7E0
// Address: 0xafb7e0
//
unsigned __int64 __fastcall sub_AFB7E0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  __int8 *v4; // rax
  __int64 v5; // r14
  signed __int64 v7; // rbx
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // [rsp+8h] [rbp-A8h] BYREF
  __m128i src; // [rsp+10h] [rbp-A0h] BYREF
  unsigned __int64 v13; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v14; // [rsp+58h] [rbp-58h]
  __int64 v15; // [rsp+60h] [rbp-50h]
  __int64 v16; // [rsp+68h] [rbp-48h]
  __int64 v17; // [rsp+70h] [rbp-40h]
  __int64 v18; // [rsp+78h] [rbp-38h]
  __int64 v19; // [rsp+80h] [rbp-30h]
  __int64 (__fastcall *v20)(); // [rsp+88h] [rbp-28h]

  memset(&src, 0, 64);
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = sub_C64CA0;
  v2 = sub_C4F0E0();
  v3 = *a2;
  src.m128i_i64[0] = v2;
  v11 = 0;
  v4 = sub_AF8740(&src, &v11, &src.m128i_i8[8], (unsigned __int64)&v13, v3);
  v5 = v11;
  if ( !v11 )
    return sub_AC25F0(&src, v4 - (__int8 *)&src, (__int64)v20);
  v7 = v4 - (__int8 *)&src;
  sub_AF1140(src.m128i_i8, v4, (char *)&v13);
  sub_AC2A10(&v13, &src);
  v8 = 0x9DDFEA08EB382D69LL
     * ((0x9DDFEA08EB382D69LL * (v19 ^ v17)) ^ v19 ^ ((0x9DDFEA08EB382D69LL * (v19 ^ v17)) >> 47));
  v9 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v5 + v7) >> 47) ^ (v5 + v7))
     + v13
     - 0x622015F714C7D297LL * (v8 ^ (v8 >> 47));
  v10 = 0x9DDFEA08EB382D69LL
      * (v9
       ^ (0xB492B66FBE98F273LL * (v14 ^ (v14 >> 47))
        + v15
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v18 ^ v16)) ^ v18 ^ ((0x9DDFEA08EB382D69LL * (v18 ^ v16)) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v18 ^ v16)) ^ v18 ^ ((0x9DDFEA08EB382D69LL * (v18 ^ v16)) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v10 ^ v9 ^ (v10 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v10 ^ v9 ^ (v10 >> 47))));
}
