// Function: sub_15B49E0
// Address: 0x15b49e0
//
unsigned __int64 __fastcall sub_15B49E0(__int32 *a1, __int64 *a2, __int64 *a3, __int64 *a4)
{
  __int64 v6; // r8
  __int8 *v7; // rax
  __int64 v8; // r8
  __int8 *v9; // rax
  __int64 v10; // r8
  __int8 *v11; // rax
  __int64 v12; // r14
  unsigned __int64 v14; // rdx
  signed __int64 v15; // rbx
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v20; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v21; // [rsp+28h] [rbp-B8h] BYREF
  __m128i src[4]; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 v23; // [rsp+70h] [rbp-70h] BYREF
  unsigned __int64 v24; // [rsp+78h] [rbp-68h]
  __int64 v25; // [rsp+80h] [rbp-60h]
  __int64 v26; // [rsp+88h] [rbp-58h]
  __int64 v27; // [rsp+90h] [rbp-50h]
  __int64 v28; // [rsp+98h] [rbp-48h]
  __int64 v29; // [rsp+A0h] [rbp-40h]
  __int64 v30; // [rsp+A8h] [rbp-38h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v14 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v14 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v14;
    sub_2207640(byte_4F99930);
  }
  v6 = *a2;
  v19 = 0;
  v30 = qword_4F99938;
  src[0].m128i_i32[0] = *a1;
  v7 = sub_15B3A60(src, &v19, &src[0].m128i_i8[4], (unsigned __int64)&v23, v6);
  v8 = *a3;
  v20 = v19;
  v9 = sub_15B2320(src, &v20, v7, (unsigned __int64)&v23, v8);
  v10 = *a4;
  v21 = v20;
  v11 = sub_15B2320(src, &v21, v9, (unsigned __int64)&v23, v10);
  v12 = v21;
  if ( !v21 )
    return sub_1593600(src, v11 - (__int8 *)src, v30);
  v15 = v11 - (__int8 *)src;
  sub_15AF6E0(src[0].m128i_i8, v11, (char *)&v23);
  sub_1593A20(&v23, src);
  v16 = 0x9DDFEA08EB382D69LL
      * ((0x9DDFEA08EB382D69LL * (v29 ^ v27)) ^ v29 ^ ((0x9DDFEA08EB382D69LL * (v29 ^ v27)) >> 47));
  v17 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v12 + v15) >> 47) ^ (v12 + v15))
      + v23
      - 0x622015F714C7D297LL * (v16 ^ (v16 >> 47));
  v18 = 0x9DDFEA08EB382D69LL
      * (v17
       ^ (0xB492B66FBE98F273LL * (v24 ^ (v24 >> 47))
        + v25
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v28 ^ v26)) ^ v28 ^ ((0x9DDFEA08EB382D69LL * (v28 ^ v26)) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v28 ^ v26)) ^ v28 ^ ((0x9DDFEA08EB382D69LL * (v28 ^ v26)) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v18 ^ v17 ^ (v18 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v18 ^ v17 ^ (v18 >> 47))));
}
