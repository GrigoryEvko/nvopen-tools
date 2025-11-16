// Function: sub_15B6BA0
// Address: 0x15b6ba0
//
unsigned __int64 __fastcall sub_15B6BA0(__int64 *a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 *a5)
{
  __int64 *v5; // r10
  __int64 v8; // r8
  __int8 *v9; // rax
  __int64 v10; // r8
  __int8 *v11; // rax
  __int8 *v12; // rax
  __int64 v13; // r8
  __int8 *v14; // rax
  __int64 v15; // r8
  __int8 *v16; // rax
  __int64 v17; // r13
  int v19; // eax
  unsigned __int64 v20; // rdx
  signed __int64 v21; // rbx
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // rdx
  __int64 v25; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v26; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v27; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v28; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v29; // [rsp+38h] [rbp-B8h] BYREF
  __m128i src[4]; // [rsp+40h] [rbp-B0h] BYREF
  unsigned __int64 v31; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 v32; // [rsp+88h] [rbp-68h]
  __int64 v33; // [rsp+90h] [rbp-60h]
  __int64 v34; // [rsp+98h] [rbp-58h]
  __int64 v35; // [rsp+A0h] [rbp-50h]
  __int64 v36; // [rsp+A8h] [rbp-48h]
  __int64 v37; // [rsp+B0h] [rbp-40h]
  __int64 v38; // [rsp+B8h] [rbp-38h]

  v5 = a1;
  if ( !byte_4F99930[0] )
  {
    v19 = sub_2207590(byte_4F99930);
    v5 = a1;
    if ( v19 )
    {
      v20 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v20 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v20;
      sub_2207640(byte_4F99930);
      v5 = a1;
    }
  }
  v8 = *v5;
  v38 = qword_4F99938;
  v25 = 0;
  v9 = sub_15B2320(src, &v25, src[0].m128i_i8, (unsigned __int64)&v31, v8);
  v10 = *a2;
  v26 = v25;
  v11 = sub_15B3A60(src, &v26, v9, (unsigned __int64)&v31, v10);
  v27 = v26;
  v12 = sub_15B3A60(src, &v27, v11, (unsigned __int64)&v31, *a3);
  v13 = *a4;
  v28 = v27;
  v14 = sub_15B3A60(src, &v28, v12, (unsigned __int64)&v31, v13);
  v15 = *a5;
  v29 = v28;
  v16 = sub_15B3A60(src, &v29, v14, (unsigned __int64)&v31, v15);
  v17 = v29;
  if ( !v29 )
    return sub_1593600(src, v16 - (__int8 *)src, v38);
  v21 = v16 - (__int8 *)src;
  sub_15AF6E0(src[0].m128i_i8, v16, (char *)&v31);
  sub_1593A20(&v31, src);
  v22 = v31
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v37 ^ v35)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v37 ^ v35)) ^ v37)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v37 ^ v35)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v37 ^ v35)) ^ v37)))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v17 + v21) >> 47) ^ (v17 + v21));
  v23 = 0x9DDFEA08EB382D69LL
      * (v22
       ^ (v33
        - 0x4B6D499041670D8DLL * (v32 ^ (v32 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v36 ^ v34)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v36 ^ v34)) ^ v36)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v36 ^ v34)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v36 ^ v34)) ^ v36)))));
  return 0x9DDFEA08EB382D69LL
       * ((0x9DDFEA08EB382D69LL * ((v23 >> 47) ^ v23 ^ v22)) ^ ((0x9DDFEA08EB382D69LL * ((v23 >> 47) ^ v23 ^ v22)) >> 47));
}
