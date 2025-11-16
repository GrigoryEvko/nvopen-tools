// Function: sub_AF7B60
// Address: 0xaf7b60
//
unsigned __int64 __fastcall sub_AF7B60(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r8
  __int8 *v3; // rax
  __int64 v4; // r8
  __int8 *v5; // rax
  __int64 v6; // r14
  signed __int64 v8; // rbx
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v12; // [rsp+8h] [rbp-A8h] BYREF
  __m128i src[4]; // [rsp+10h] [rbp-A0h] BYREF
  unsigned __int64 v14; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v15; // [rsp+58h] [rbp-58h]
  __int64 v16; // [rsp+60h] [rbp-50h]
  __int64 v17; // [rsp+68h] [rbp-48h]
  __int64 v18; // [rsp+70h] [rbp-40h]
  __int64 v19; // [rsp+78h] [rbp-38h]
  __int64 v20; // [rsp+80h] [rbp-30h]
  __int64 (__fastcall *v21)(); // [rsp+88h] [rbp-28h]

  v2 = *a1;
  memset(src, 0, sizeof(src));
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = sub_C64CA0;
  v11 = 0;
  v3 = sub_AF70F0(src, &v11, src[0].m128i_i8, (unsigned __int64)&v14, v2);
  v4 = *a2;
  v12 = v11;
  v5 = sub_AF70F0(src, &v12, v3, (unsigned __int64)&v14, v4);
  v6 = v12;
  if ( !v12 )
    return sub_AC25F0(src, v5 - (__int8 *)src, (__int64)v21);
  v8 = v5 - (__int8 *)src;
  sub_AF1140(src[0].m128i_i8, v5, (char *)&v14);
  sub_AC2A10(&v14, src);
  v9 = v14
     - 0x622015F714C7D297LL
     * ((0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v20 ^ v18)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v20 ^ v18)) ^ v20))
      ^ ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v20 ^ v18)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v20 ^ v18)) ^ v20)) >> 47))
     - 0x4B6D499041670D8DLL * (((unsigned __int64)(v6 + v8) >> 47) ^ (v6 + v8));
  v10 = 0x9DDFEA08EB382D69LL
      * (v9
       ^ (v16
        - 0x4B6D499041670D8DLL * (v15 ^ (v15 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v19 ^ v17)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v19 ^ v17)) ^ v19)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v19 ^ v17)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v19 ^ v17)) ^ v19)))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v10 ^ v9 ^ (v10 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v10 ^ v9 ^ (v10 >> 47))));
}
