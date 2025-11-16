// Function: sub_AF7D50
// Address: 0xaf7d50
//
unsigned __int64 __fastcall sub_AF7D50(__int64 *a1, __int64 *a2, __int64 *a3, __int64 *a4)
{
  __int64 v6; // r8
  __int64 v7; // rax
  __int8 *v8; // rax
  __int64 v9; // r8
  __int8 *v10; // rax
  __int64 v11; // r8
  __int8 *v12; // rax
  __int64 v13; // r14
  signed __int64 v15; // rbx
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-B8h] BYREF
  __int64 v20; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v21; // [rsp+18h] [rbp-A8h] BYREF
  _BYTE src[56]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v23; // [rsp+58h] [rbp-68h]
  unsigned __int64 v24; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int64 v25; // [rsp+68h] [rbp-58h]
  __int64 v26; // [rsp+70h] [rbp-50h]
  __int64 v27; // [rsp+78h] [rbp-48h]
  __int64 v28; // [rsp+80h] [rbp-40h]
  __int64 v29; // [rsp+88h] [rbp-38h]
  __int64 v30; // [rsp+90h] [rbp-30h]
  __int64 (__fastcall *v31)(); // [rsp+98h] [rbp-28h]

  v6 = *a2;
  memset(&src[8], 0, 48);
  v31 = sub_C64CA0;
  v7 = *a1;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  *(_QWORD *)src = v7;
  v19 = 0;
  v8 = sub_AF70F0((__m128i *)src, &v19, &src[8], (unsigned __int64)&v24, v6);
  v9 = *a3;
  v20 = v19;
  v10 = sub_AF70F0((__m128i *)src, &v20, v8, (unsigned __int64)&v24, v9);
  v11 = *a4;
  v21 = v20;
  v12 = sub_AF70F0((__m128i *)src, &v21, v10, (unsigned __int64)&v24, v11);
  v13 = v21;
  if ( !v21 )
    return sub_AC25F0(src, v12 - src, (__int64)v31);
  v15 = v12 - src;
  sub_AF1140(src, v12, (char *)&v24);
  sub_AC2A10(&v24, src);
  v16 = 0x9DDFEA08EB382D69LL
      * ((0x9DDFEA08EB382D69LL * (v30 ^ v28)) ^ v30 ^ ((0x9DDFEA08EB382D69LL * (v30 ^ v28)) >> 47));
  v17 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v13 + v15) >> 47) ^ (v13 + v15))
      + v24
      - 0x622015F714C7D297LL * (v16 ^ (v16 >> 47));
  v18 = 0x9DDFEA08EB382D69LL
      * (v17
       ^ (0xB492B66FBE98F273LL * (v25 ^ (v25 >> 47))
        + v26
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v29 ^ v27)) ^ v29 ^ ((0x9DDFEA08EB382D69LL * (v29 ^ v27)) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v29 ^ v27)) ^ v29 ^ ((0x9DDFEA08EB382D69LL * (v29 ^ v27)) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v18 ^ v17 ^ (v18 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v18 ^ v17 ^ (v18 >> 47))));
}
