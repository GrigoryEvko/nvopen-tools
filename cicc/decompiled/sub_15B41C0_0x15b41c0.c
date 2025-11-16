// Function: sub_15B41C0
// Address: 0x15b41c0
//
unsigned __int64 __fastcall sub_15B41C0(__int64 *a1, __int64 *a2, __int64 *a3, int *a4, __int64 *a5, int *a6, int *a7)
{
  __int64 *v7; // r10
  __int64 v10; // r8
  __int8 *v11; // rax
  __int64 v12; // r8
  __int8 *v13; // rax
  __int8 *v14; // rax
  int v15; // r8d
  __int8 *v16; // rax
  __int64 v17; // r8
  __int8 *v18; // rax
  __int8 *v19; // rax
  __int8 *v20; // rax
  __int64 v21; // r13
  int v23; // eax
  unsigned __int64 v24; // rdx
  signed __int64 v25; // rbx
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // rdx
  __int64 v30; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v31; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v32; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v33; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v34; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v35; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v36; // [rsp+58h] [rbp-B8h] BYREF
  __m128i src[4]; // [rsp+60h] [rbp-B0h] BYREF
  unsigned __int64 v38; // [rsp+A0h] [rbp-70h] BYREF
  unsigned __int64 v39; // [rsp+A8h] [rbp-68h]
  __int64 v40; // [rsp+B0h] [rbp-60h]
  __int64 v41; // [rsp+B8h] [rbp-58h]
  __int64 v42; // [rsp+C0h] [rbp-50h]
  __int64 v43; // [rsp+C8h] [rbp-48h]
  __int64 v44; // [rsp+D0h] [rbp-40h]
  __int64 v45; // [rsp+D8h] [rbp-38h]

  v7 = a1;
  if ( !byte_4F99930[0] )
  {
    v23 = sub_2207590(byte_4F99930);
    v7 = a1;
    if ( v23 )
    {
      v24 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v24 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v24;
      sub_2207640(byte_4F99930);
      v7 = a1;
    }
  }
  v10 = *v7;
  v45 = qword_4F99938;
  v30 = 0;
  v11 = sub_15B2320(src, &v30, src[0].m128i_i8, (unsigned __int64)&v38, v10);
  v12 = *a2;
  v31 = v30;
  v13 = sub_15B3A60(src, &v31, v11, (unsigned __int64)&v38, v12);
  v32 = v31;
  v14 = sub_15B2320(src, &v32, v13, (unsigned __int64)&v38, *a3);
  v15 = *a4;
  v33 = v32;
  v16 = sub_15B2130(src, &v33, v14, (unsigned __int64)&v38, v15);
  v17 = *a5;
  v34 = v33;
  v18 = sub_15B2320(src, &v34, v16, (unsigned __int64)&v38, v17);
  v35 = v34;
  v19 = sub_15B2130(src, &v35, v18, (unsigned __int64)&v38, *a6);
  v36 = v35;
  v20 = sub_15B2130(src, &v36, v19, (unsigned __int64)&v38, *a7);
  v21 = v36;
  if ( !v36 )
    return sub_1593600(src, v20 - (__int8 *)src, v45);
  v25 = v20 - (__int8 *)src;
  sub_15AF6E0(src[0].m128i_i8, v20, (char *)&v38);
  sub_1593A20(&v38, src);
  v26 = v38
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v44 ^ v42)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v44 ^ v42)) ^ v44)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v44 ^ v42)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v44 ^ v42)) ^ v44)))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v21 + v25) >> 47) ^ (v21 + v25));
  v27 = 0x9DDFEA08EB382D69LL
      * (v26
       ^ (v40
        - 0x4B6D499041670D8DLL * (v39 ^ (v39 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v43 ^ v41)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v43 ^ v41)) ^ v43)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v43 ^ v41)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v43 ^ v41)) ^ v43)))));
  return 0x9DDFEA08EB382D69LL
       * ((0x9DDFEA08EB382D69LL * ((v27 >> 47) ^ v27 ^ v26)) ^ ((0x9DDFEA08EB382D69LL * ((v27 >> 47) ^ v27 ^ v26)) >> 47));
}
