// Function: sub_277F3A0
// Address: 0x277f3a0
//
unsigned __int64 __fastcall sub_277F3A0(int *a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // r8
  int v4; // eax
  __int64 v5; // rax
  __int8 *v6; // rax
  __int64 v7; // r14
  signed __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-A8h] BYREF
  int src; // [rsp+10h] [rbp-A0h] BYREF
  _BYTE v15[20]; // [rsp+14h] [rbp-9Ch] BYREF
  __int128 v16; // [rsp+28h] [rbp-88h]
  __int128 v17; // [rsp+38h] [rbp-78h]
  __int64 v18; // [rsp+48h] [rbp-68h]
  unsigned __int64 v19; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v20; // [rsp+58h] [rbp-58h]
  __int64 v21; // [rsp+60h] [rbp-50h]
  __int64 v22; // [rsp+68h] [rbp-48h]
  __int64 v23; // [rsp+70h] [rbp-40h]
  __int64 v24; // [rsp+78h] [rbp-38h]
  __int64 v25; // [rsp+80h] [rbp-30h]
  void (__fastcall *v26)(__int64, __int64); // [rsp+88h] [rbp-28h]

  v3 = *a3;
  *(_OWORD *)&v15[4] = 0;
  v18 = 0;
  v26 = sub_C64CA0;
  v4 = *a1;
  v19 = 0;
  src = v4;
  v5 = *a2;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  *(_QWORD *)v15 = v5;
  v13 = 0;
  v16 = 0;
  v17 = 0;
  v6 = sub_277DD80((__m128i *)&src, &v13, &v15[8], (unsigned __int64)&v19, v3);
  v7 = v13;
  if ( !v13 )
    return sub_AC25F0(&src, v6 - (__int8 *)&src, (__int64)v26);
  v9 = v6 - (__int8 *)&src;
  sub_2778790((char *)&src, v6, (char *)&v19);
  sub_AC2A10(&v19, &src);
  v10 = 0x9DDFEA08EB382D69LL
      * ((0x9DDFEA08EB382D69LL * (v25 ^ v23)) ^ v25 ^ ((0x9DDFEA08EB382D69LL * (v25 ^ v23)) >> 47));
  v11 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v7 + v9) >> 47) ^ (v7 + v9))
      + v19
      - 0x622015F714C7D297LL * (v10 ^ (v10 >> 47));
  v12 = 0x9DDFEA08EB382D69LL
      * (v11
       ^ (0xB492B66FBE98F273LL * (v20 ^ (v20 >> 47))
        + v21
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v24 ^ v22)) ^ v24 ^ ((0x9DDFEA08EB382D69LL * (v24 ^ v22)) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v24 ^ v22)) ^ v24 ^ ((0x9DDFEA08EB382D69LL * (v24 ^ v22)) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v12 ^ v11 ^ (v12 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v12 ^ v11 ^ (v12 >> 47))));
}
