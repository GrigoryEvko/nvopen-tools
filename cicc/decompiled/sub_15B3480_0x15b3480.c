// Function: sub_15B3480
// Address: 0x15b3480
//
unsigned __int64 __fastcall sub_15B3480(
        __int64 *a1,
        __int64 *a2,
        char *a3,
        __int64 *a4,
        __int64 *a5,
        __int64 *a6,
        __int64 *a7)
{
  __int64 *v7; // r10
  __int64 v12; // r8
  __int8 *v13; // rax
  __int64 v14; // r8
  __int8 *v15; // rax
  __int64 v16; // r8
  __int8 *v17; // rax
  __int8 *v18; // rax
  __int64 v19; // r14
  int v21; // eax
  unsigned __int64 v22; // rdx
  signed __int64 v23; // rbx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rsi
  unsigned __int64 v26; // rax
  __int64 v27; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v28; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v29; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+28h] [rbp-B8h] BYREF
  __m128i src; // [rsp+30h] [rbp-B0h] BYREF
  char v32; // [rsp+40h] [rbp-A0h]
  char v33[47]; // [rsp+41h] [rbp-9Fh] BYREF
  unsigned __int64 v34; // [rsp+70h] [rbp-70h] BYREF
  unsigned __int64 v35; // [rsp+78h] [rbp-68h]
  __int64 v36; // [rsp+80h] [rbp-60h]
  __int64 v37; // [rsp+88h] [rbp-58h]
  __int64 v38; // [rsp+90h] [rbp-50h]
  __int64 v39; // [rsp+98h] [rbp-48h]
  __int64 v40; // [rsp+A0h] [rbp-40h]
  __int64 v41; // [rsp+A8h] [rbp-38h]

  v7 = a1;
  if ( !byte_4F99930[0] )
  {
    v21 = sub_2207590(byte_4F99930);
    v7 = a1;
    if ( v21 )
    {
      v22 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v22 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v22;
      sub_2207640(byte_4F99930);
      v7 = a1;
    }
  }
  v12 = *a4;
  v27 = 0;
  v41 = qword_4F99938;
  src.m128i_i64[0] = *v7;
  src.m128i_i64[1] = *a2;
  v32 = *a3;
  v13 = sub_15B2320(&src, &v27, v33, (unsigned __int64)&v34, v12);
  v14 = *a5;
  v28 = v27;
  v15 = sub_15B2320(&src, &v28, v13, (unsigned __int64)&v34, v14);
  v16 = *a6;
  v29 = v28;
  v17 = sub_15B2320(&src, &v29, v15, (unsigned __int64)&v34, v16);
  v30 = v29;
  v18 = sub_15B2320(&src, &v30, v17, (unsigned __int64)&v34, *a7);
  v19 = v30;
  if ( !v30 )
    return sub_1593600(&src, v18 - (__int8 *)&src, v41);
  v23 = v18 - (__int8 *)&src;
  sub_15AF6E0(src.m128i_i8, v18, (char *)&v34);
  sub_1593A20(&v34, &src);
  v24 = 0x9DDFEA08EB382D69LL
      * ((0x9DDFEA08EB382D69LL * (v40 ^ v38)) ^ v40 ^ ((0x9DDFEA08EB382D69LL * (v40 ^ v38)) >> 47));
  v25 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v19 + v23) >> 47) ^ (v19 + v23))
      + v34
      - 0x622015F714C7D297LL * (v24 ^ (v24 >> 47));
  v26 = 0x9DDFEA08EB382D69LL
      * (v25
       ^ (0xB492B66FBE98F273LL * (v35 ^ (v35 >> 47))
        + v36
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v39 ^ v37)) ^ v39 ^ ((0x9DDFEA08EB382D69LL * (v39 ^ v37)) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v39 ^ v37)) ^ v39 ^ ((0x9DDFEA08EB382D69LL * (v39 ^ v37)) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v26 ^ v25 ^ (v26 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v26 ^ v25 ^ (v26 >> 47))));
}
