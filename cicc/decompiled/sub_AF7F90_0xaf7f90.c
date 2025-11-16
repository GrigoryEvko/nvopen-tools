// Function: sub_AF7F90
// Address: 0xaf7f90
//
unsigned __int64 __fastcall sub_AF7F90(int *a1, int *a2, __int64 *a3, __int64 *a4)
{
  int v6; // r8d
  __int8 *v7; // rax
  int v8; // r8d
  __int8 *v9; // rax
  __int64 v10; // r8
  __int8 *v11; // rax
  __int64 v12; // r8
  __int8 *v13; // rax
  __int64 v14; // r14
  signed __int64 v16; // rbx
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rax
  __int64 v19; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v20; // [rsp+8h] [rbp-C8h] BYREF
  __int64 v21; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v22; // [rsp+18h] [rbp-B8h] BYREF
  __m128i src[4]; // [rsp+20h] [rbp-B0h] BYREF
  unsigned __int64 v24; // [rsp+60h] [rbp-70h] BYREF
  unsigned __int64 v25; // [rsp+68h] [rbp-68h]
  __int64 v26; // [rsp+70h] [rbp-60h]
  __int64 v27; // [rsp+78h] [rbp-58h]
  __int64 v28; // [rsp+80h] [rbp-50h]
  __int64 v29; // [rsp+88h] [rbp-48h]
  __int64 v30; // [rsp+90h] [rbp-40h]
  __int64 (__fastcall *v31)(); // [rsp+98h] [rbp-38h]

  v6 = *a1;
  memset(src, 0, sizeof(src));
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = sub_C64CA0;
  v19 = 0;
  v7 = sub_AF6D70(src, &v19, src[0].m128i_i8, (unsigned __int64)&v24, v6);
  v8 = *a2;
  v20 = v19;
  v9 = sub_AF6D70(src, &v20, v7, (unsigned __int64)&v24, v8);
  v10 = *a3;
  v21 = v20;
  v11 = sub_AF70F0(src, &v21, v9, (unsigned __int64)&v24, v10);
  v12 = *a4;
  v22 = v21;
  v13 = sub_AF70F0(src, &v22, v11, (unsigned __int64)&v24, v12);
  v14 = v22;
  if ( !v22 )
    return sub_AC25F0(src, v13 - (__int8 *)src, (__int64)v31);
  v16 = v13 - (__int8 *)src;
  sub_AF1140(src[0].m128i_i8, v13, (char *)&v24);
  sub_AC2A10(&v24, src);
  v17 = v24
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v30 ^ v28)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v30 ^ v28)) ^ v30))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v30 ^ v28)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v30 ^ v28)) ^ v30)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v14 + v16) >> 47) ^ (v14 + v16));
  v18 = 0x9DDFEA08EB382D69LL
      * (v17
       ^ (v26
        - 0x4B6D499041670D8DLL * (v25 ^ (v25 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v29 ^ v27)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v29 ^ v27)) ^ v29)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v29 ^ v27)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v29 ^ v27)) ^ v29)))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v18 ^ v17 ^ (v18 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v18 ^ v17 ^ (v18 >> 47))));
}
