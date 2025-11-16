// Function: sub_15B5D10
// Address: 0x15b5d10
//
unsigned __int64 __fastcall sub_15B5D10(__int64 *a1, __int64 *a2, int *a3, __int64 *a4, __int64 *a5, __int64 *a6)
{
  __int64 *v6; // r10
  __int64 v9; // r8
  __int8 *v10; // rax
  __int64 v11; // r8
  __int8 *v12; // rax
  __int8 *v13; // rax
  __int64 v14; // r8
  __int8 *v15; // rax
  __int64 v16; // r8
  __int8 *v17; // rax
  __int8 *v18; // rax
  __int64 v19; // r13
  int v21; // eax
  unsigned __int64 v22; // rdx
  signed __int64 v23; // rbx
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // rdx
  __int64 v28; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v29; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v30; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v31; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v32; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+48h] [rbp-B8h] BYREF
  __m128i src[4]; // [rsp+50h] [rbp-B0h] BYREF
  unsigned __int64 v35; // [rsp+90h] [rbp-70h] BYREF
  unsigned __int64 v36; // [rsp+98h] [rbp-68h]
  __int64 v37; // [rsp+A0h] [rbp-60h]
  __int64 v38; // [rsp+A8h] [rbp-58h]
  __int64 v39; // [rsp+B0h] [rbp-50h]
  __int64 v40; // [rsp+B8h] [rbp-48h]
  __int64 v41; // [rsp+C0h] [rbp-40h]
  __int64 v42; // [rsp+C8h] [rbp-38h]

  v6 = a1;
  if ( !byte_4F99930[0] )
  {
    v21 = sub_2207590(byte_4F99930);
    v6 = a1;
    if ( v21 )
    {
      v22 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v22 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v22;
      sub_2207640(byte_4F99930);
      v6 = a1;
    }
  }
  v9 = *v6;
  v42 = qword_4F99938;
  v28 = 0;
  v10 = sub_15B3A60(src, &v28, src[0].m128i_i8, (unsigned __int64)&v35, v9);
  v11 = *a2;
  v29 = v28;
  v12 = sub_15B2320(src, &v29, v10, (unsigned __int64)&v35, v11);
  v30 = v29;
  v13 = sub_15B2130(src, &v30, v12, (unsigned __int64)&v35, *a3);
  v14 = *a4;
  v31 = v30;
  v15 = sub_15B2320(src, &v31, v13, (unsigned __int64)&v35, v14);
  v16 = *a5;
  v32 = v31;
  v17 = sub_15B2320(src, &v32, v15, (unsigned __int64)&v35, v16);
  v33 = v32;
  v18 = sub_15B2320(src, &v33, v17, (unsigned __int64)&v35, *a6);
  v19 = v33;
  if ( !v33 )
    return sub_1593600(src, v18 - (__int8 *)src, v42);
  v23 = v18 - (__int8 *)src;
  sub_15AF6E0(src[0].m128i_i8, v18, (char *)&v35);
  sub_1593A20(&v35, src);
  v24 = v35
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v41 ^ v39)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v41 ^ v39)) ^ v41)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v41 ^ v39)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v41 ^ v39)) ^ v41)))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v19 + v23) >> 47) ^ (v19 + v23));
  v25 = 0x9DDFEA08EB382D69LL
      * (v24
       ^ (v37
        - 0x4B6D499041670D8DLL * (v36 ^ (v36 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v40 ^ v38)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v40 ^ v38)) ^ v40)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v40 ^ v38)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v40 ^ v38)) ^ v40)))));
  return 0x9DDFEA08EB382D69LL
       * ((0x9DDFEA08EB382D69LL * ((v25 >> 47) ^ v25 ^ v24)) ^ ((0x9DDFEA08EB382D69LL * ((v25 >> 47) ^ v25 ^ v24)) >> 47));
}
