// Function: sub_AF9E80
// Address: 0xaf9e80
//
unsigned __int64 __fastcall sub_AF9E80(
        __int64 *a1,
        __int64 *a2,
        int *a3,
        __int64 *a4,
        __int64 *a5,
        int *a6,
        __int64 *a7)
{
  __int64 v10; // r8
  __int8 *v11; // rax
  __int64 v12; // r8
  __int8 *v13; // rax
  int v14; // r8d
  __int8 *v15; // rax
  __int64 v16; // r8
  __int8 *v17; // rax
  __int64 v18; // r8
  __int8 *v19; // rax
  int v20; // r8d
  __int8 *v21; // rax
  __int8 *v22; // rax
  __int64 v23; // r14
  signed __int64 v25; // rbx
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // rax
  __int64 v29; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v30; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v32; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v33; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v34; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v35; // [rsp+48h] [rbp-B8h] BYREF
  __m128i src[4]; // [rsp+50h] [rbp-B0h] BYREF
  unsigned __int64 v37; // [rsp+90h] [rbp-70h] BYREF
  unsigned __int64 v38; // [rsp+98h] [rbp-68h]
  __int64 v39; // [rsp+A0h] [rbp-60h]
  __int64 v40; // [rsp+A8h] [rbp-58h]
  __int64 v41; // [rsp+B0h] [rbp-50h]
  __int64 v42; // [rsp+B8h] [rbp-48h]
  __int64 v43; // [rsp+C0h] [rbp-40h]
  __int64 (__fastcall *v44)(); // [rsp+C8h] [rbp-38h]

  v10 = *a1;
  memset(src, 0, sizeof(src));
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = sub_C64CA0;
  v29 = 0;
  v11 = sub_AF8740(src, &v29, src[0].m128i_i8, (unsigned __int64)&v37, v10);
  v12 = *a2;
  v30 = v29;
  v13 = sub_AF70F0(src, &v30, v11, (unsigned __int64)&v37, v12);
  v14 = *a3;
  v31 = v30;
  v15 = sub_AF6D70(src, &v31, v13, (unsigned __int64)&v37, v14);
  v16 = *a4;
  v32 = v31;
  v17 = sub_AF8740(src, &v32, v15, (unsigned __int64)&v37, v16);
  v18 = *a5;
  v33 = v32;
  v19 = sub_AF8740(src, &v33, v17, (unsigned __int64)&v37, v18);
  v20 = *a6;
  v34 = v33;
  v21 = sub_AF6D70(src, &v34, v19, (unsigned __int64)&v37, v20);
  v35 = v34;
  v22 = sub_AF70F0(src, &v35, v21, (unsigned __int64)&v37, *a7);
  v23 = v35;
  if ( !v35 )
    return sub_AC25F0(src, v22 - (__int8 *)src, (__int64)v44);
  v25 = v22 - (__int8 *)src;
  sub_AF1140(src[0].m128i_i8, v22, (char *)&v37);
  sub_AC2A10(&v37, src);
  v26 = v37
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v43 ^ v41)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v43 ^ v41)) ^ v43))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v43 ^ v41)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v43 ^ v41)) ^ v43)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v23 + v25) >> 47) ^ (v23 + v25));
  v27 = 0x9DDFEA08EB382D69LL
      * (v26
       ^ (v39
        - 0x4B6D499041670D8DLL * (v38 ^ (v38 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v42 ^ v40)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v42 ^ v40)) ^ v42)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v42 ^ v40)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v42 ^ v40)) ^ v42)))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v27 ^ v26 ^ (v27 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v27 ^ v26 ^ (v27 >> 47))));
}
