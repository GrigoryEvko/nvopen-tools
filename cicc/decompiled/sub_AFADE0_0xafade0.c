// Function: sub_AFADE0
// Address: 0xafade0
//
unsigned __int64 __fastcall sub_AFADE0(
        __int64 *a1,
        __int64 *a2,
        int *a3,
        __int64 *a4,
        __int64 *a5,
        __int64 *a6,
        __int64 *a7,
        __int64 *a8)
{
  __int64 v11; // r8
  __int8 *v12; // rax
  __int64 v13; // r8
  __int8 *v14; // rax
  int v15; // r8d
  __int8 *v16; // rax
  __int64 v17; // r8
  __int8 *v18; // rax
  __int64 v19; // r8
  __int8 *v20; // rax
  __int64 v21; // r8
  __int8 *v22; // rax
  __int8 *v23; // rax
  __int8 *v24; // rax
  __int64 v25; // r14
  signed __int64 v27; // rbx
  unsigned __int64 v28; // rbx
  unsigned __int64 v29; // rax
  __int64 v31; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v32; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v33; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v34; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v35; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v36; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v37; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+48h] [rbp-B8h] BYREF
  __m128i src[4]; // [rsp+50h] [rbp-B0h] BYREF
  unsigned __int64 v40; // [rsp+90h] [rbp-70h] BYREF
  unsigned __int64 v41; // [rsp+98h] [rbp-68h]
  __int64 v42; // [rsp+A0h] [rbp-60h]
  __int64 v43; // [rsp+A8h] [rbp-58h]
  __int64 v44; // [rsp+B0h] [rbp-50h]
  __int64 v45; // [rsp+B8h] [rbp-48h]
  __int64 v46; // [rsp+C0h] [rbp-40h]
  __int64 (__fastcall *v47)(); // [rsp+C8h] [rbp-38h]

  v11 = *a1;
  memset(src, 0, sizeof(src));
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = sub_C64CA0;
  v31 = 0;
  v12 = sub_AF8740(src, &v31, src[0].m128i_i8, (unsigned __int64)&v40, v11);
  v13 = *a2;
  v32 = v31;
  v14 = sub_AF70F0(src, &v32, v12, (unsigned __int64)&v40, v13);
  v15 = *a3;
  v33 = v32;
  v16 = sub_AF6D70(src, &v33, v14, (unsigned __int64)&v40, v15);
  v17 = *a4;
  v34 = v33;
  v18 = sub_AF70F0(src, &v34, v16, (unsigned __int64)&v40, v17);
  v19 = *a5;
  v35 = v34;
  v20 = sub_AF70F0(src, &v35, v18, (unsigned __int64)&v40, v19);
  v21 = *a6;
  v36 = v35;
  v22 = sub_AF70F0(src, &v36, v20, (unsigned __int64)&v40, v21);
  v37 = v36;
  v23 = sub_AF70F0(src, &v37, v22, (unsigned __int64)&v40, *a7);
  v38 = v37;
  v24 = sub_AF70F0(src, &v38, v23, (unsigned __int64)&v40, *a8);
  v25 = v38;
  if ( !v38 )
    return sub_AC25F0(src, v24 - (__int8 *)src, (__int64)v47);
  v27 = v24 - (__int8 *)src;
  sub_AF1140(src[0].m128i_i8, v24, (char *)&v40);
  sub_AC2A10(&v40, src);
  v28 = v40
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v46 ^ v44)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v46 ^ v44)) ^ v46))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v46 ^ v44)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v46 ^ v44)) ^ v46)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v25 + v27) >> 47) ^ (v25 + v27));
  v29 = 0x9DDFEA08EB382D69LL
      * (v28
       ^ (v42
        - 0x4B6D499041670D8DLL * (v41 ^ (v41 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v45 ^ v43)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v45 ^ v43)) ^ v45)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v45 ^ v43)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v45 ^ v43)) ^ v45)))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v29 ^ v28 ^ (v29 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v29 ^ v28 ^ (v29 >> 47))));
}
