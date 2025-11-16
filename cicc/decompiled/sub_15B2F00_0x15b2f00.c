// Function: sub_15B2F00
// Address: 0x15b2f00
//
unsigned __int64 __fastcall sub_15B2F00(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r8
  __int8 *v3; // rax
  __int64 v4; // r14
  unsigned __int64 v6; // rax
  signed __int64 v7; // rbx
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // [rsp+8h] [rbp-A8h] BYREF
  __m128i src[4]; // [rsp+10h] [rbp-A0h] BYREF
  unsigned __int64 v13; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v14; // [rsp+58h] [rbp-58h]
  __int64 v15; // [rsp+60h] [rbp-50h]
  __int64 v16; // [rsp+68h] [rbp-48h]
  __int64 v17; // [rsp+70h] [rbp-40h]
  __int64 v18; // [rsp+78h] [rbp-38h]
  __int64 v19; // [rsp+80h] [rbp-30h]
  __int64 v20; // [rsp+88h] [rbp-28h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v6 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v6 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v6;
    sub_2207640(byte_4F99930);
  }
  v2 = *a2;
  v11 = 0;
  v20 = qword_4F99938;
  src[0].m128i_i64[0] = *a1;
  v3 = sub_15B2320(src, &v11, &src[0].m128i_i8[8], (unsigned __int64)&v13, v2);
  v4 = v11;
  if ( !v11 )
    return sub_1593600(src, v3 - (__int8 *)src, v20);
  v7 = v3 - (__int8 *)src;
  sub_15AF6E0(src[0].m128i_i8, v3, (char *)&v13);
  sub_1593A20(&v13, src);
  v8 = 0x9DDFEA08EB382D69LL
     * ((0x9DDFEA08EB382D69LL * (v19 ^ v17)) ^ v19 ^ ((0x9DDFEA08EB382D69LL * (v19 ^ v17)) >> 47));
  v9 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v4 + v7) >> 47) ^ (v4 + v7))
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
