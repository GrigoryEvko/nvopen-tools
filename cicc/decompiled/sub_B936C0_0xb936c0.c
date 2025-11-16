// Function: sub_B936C0
// Address: 0xb936c0
//
unsigned __int64 __fastcall sub_B936C0(__int64 a1)
{
  __int64 v2; // rbx
  _BYTE *v3; // rdi
  __int8 *v4; // rax
  __int64 v5; // r8
  __int8 *v6; // rax
  __int64 v7; // r12
  signed __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v13; // [rsp+8h] [rbp-A8h] BYREF
  __m128i dest[4]; // [rsp+10h] [rbp-A0h] BYREF
  unsigned __int64 v15; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v16; // [rsp+58h] [rbp-58h]
  __int64 v17; // [rsp+60h] [rbp-50h]
  __int64 v18; // [rsp+68h] [rbp-48h]
  __int64 v19; // [rsp+70h] [rbp-40h]
  __int64 v20; // [rsp+78h] [rbp-38h]
  __int64 v21; // [rsp+80h] [rbp-30h]
  __int64 (__fastcall *v22)(); // [rsp+88h] [rbp-28h]

  if ( *(_DWORD *)a1 != 13 )
    return sub_AF95C0(
             (int *)a1,
             (__int64 *)(a1 + 8),
             (__int64 *)(a1 + 16),
             (int *)(a1 + 24),
             (__int64 *)(a1 + 32),
             (__int64 *)(a1 + 40),
             (int *)(a1 + 84));
  v2 = *(_QWORD *)(a1 + 8);
  if ( !v2 )
    return sub_AF95C0(
             (int *)a1,
             (__int64 *)(a1 + 8),
             (__int64 *)(a1 + 16),
             (int *)(a1 + 24),
             (__int64 *)(a1 + 32),
             (__int64 *)(a1 + 40),
             (int *)(a1 + 84));
  v3 = *(_BYTE **)(a1 + 32);
  if ( !v3 || *v3 != 14 || !sub_AF5140((__int64)v3, 7u) )
    return sub_AF95C0(
             (int *)a1,
             (__int64 *)(a1 + 8),
             (__int64 *)(a1 + 16),
             (int *)(a1 + 24),
             (__int64 *)(a1 + 32),
             (__int64 *)(a1 + 40),
             (int *)(a1 + 84));
  memset(dest, 0, sizeof(dest));
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = sub_C64CA0;
  v12 = 0;
  v4 = sub_AF8740(dest, &v12, dest[0].m128i_i8, (unsigned __int64)&v15, v2);
  v5 = *(_QWORD *)(a1 + 32);
  v13 = v12;
  v6 = sub_AF70F0(dest, &v13, v4, (unsigned __int64)&v15, v5);
  v7 = v13;
  if ( !v13 )
    return sub_AC25F0(dest, v6 - (__int8 *)dest, (__int64)v22);
  v9 = v6 - (__int8 *)dest;
  sub_B8EB50(dest[0].m128i_i8, v6, (char *)&v15);
  sub_AC2A10(&v15, dest);
  v10 = v15
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v21 ^ v19)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v21 ^ v19)) ^ v21)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v21 ^ v19)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v21 ^ v19)) ^ v21)))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v7 + v9) >> 47) ^ (v7 + v9));
  v11 = 0x9DDFEA08EB382D69LL
      * (v10
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v20 ^ v18)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v20 ^ v18)) ^ v20)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v20 ^ v18)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v20 ^ v18)) ^ v20)))
        + v17
        - 0x4B6D499041670D8DLL * (v16 ^ (v16 >> 47))));
  return 0x9DDFEA08EB382D69LL
       * ((0x9DDFEA08EB382D69LL * ((v11 >> 47) ^ v11 ^ v10)) ^ ((0x9DDFEA08EB382D69LL * ((v11 >> 47) ^ v11 ^ v10)) >> 47));
}
