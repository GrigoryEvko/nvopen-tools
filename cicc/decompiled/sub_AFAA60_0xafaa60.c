// Function: sub_AFAA60
// Address: 0xafaa60
//
unsigned __int64 __fastcall sub_AFAA60(__int64 *a1, __int64 *a2, int *a3, __int64 *a4, __int64 *a5)
{
  __int64 v8; // r8
  __int8 *v9; // rax
  __int64 v10; // r8
  __int8 *v11; // rax
  __int64 v12; // rcx
  __int8 *v13; // rdi
  int v14; // eax
  char *v15; // r9
  char *v16; // r8
  __int64 v17; // rcx
  __int64 v18; // r8
  __int8 *v19; // rax
  __int64 v20; // r8
  __int8 *v21; // rax
  __int64 v22; // r14
  signed __int64 v24; // rbx
  unsigned __int64 v25; // rbx
  unsigned __int64 v26; // rax
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __int64 v29; // [rsp+0h] [rbp-120h]
  __int64 v30; // [rsp+0h] [rbp-120h]
  char *v31; // [rsp+8h] [rbp-118h]
  __m128i v32; // [rsp+10h] [rbp-110h] BYREF
  __m128i v33; // [rsp+20h] [rbp-100h] BYREF
  __m128i v34; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v35; // [rsp+40h] [rbp-E0h]
  __int64 v36; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v37; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v38; // [rsp+60h] [rbp-C0h] BYREF
  __int64 src; // [rsp+68h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v41; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v42; // [rsp+C0h] [rbp-60h]
  __m128i v43; // [rsp+D0h] [rbp-50h]
  __int64 v44; // [rsp+E0h] [rbp-40h]
  __int64 (__fastcall *v45)(); // [rsp+E8h] [rbp-38h]

  v8 = *a1;
  memset(dest, 0, sizeof(dest));
  v41 = 0u;
  v42 = 0u;
  v43 = 0u;
  v44 = 0;
  v45 = sub_C64CA0;
  v36 = 0;
  v9 = sub_AF8740(dest, &v36, dest[0].m128i_i8, (unsigned __int64)&v41, v8);
  v10 = *a2;
  v37 = v36;
  v11 = sub_AF8740(dest, &v37, v9, (unsigned __int64)&v41, v10);
  v12 = v37;
  v13 = v11;
  v14 = *a3;
  v15 = v13 + 4;
  LODWORD(src) = *a3;
  if ( v13 + 4 <= (__int8 *)&v41 )
  {
    *(_DWORD *)v13 = v14;
  }
  else
  {
    v29 = v37;
    memcpy(v13, &src, (char *)&v41 - v13);
    if ( v29 )
    {
      sub_AC2A10((unsigned __int64 *)&v41, dest);
      v16 = (char *)((char *)&v41 - v13);
      v17 = v29 + 64;
    }
    else
    {
      sub_AC28A0((unsigned __int64 *)&v32, dest[0].m128i_i64, (unsigned __int64)v45);
      v27 = _mm_loadu_si128(&v33);
      v17 = 64;
      v28 = _mm_loadu_si128(&v34);
      v16 = (char *)((char *)&v41 - v13);
      v41 = _mm_loadu_si128(&v32);
      v44 = v35;
      v42 = v27;
      v43 = v28;
    }
    v30 = v17;
    v31 = &dest[0].m128i_i8[4LL - (_QWORD)v16];
    if ( v31 > (char *)&v41 )
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v16, 4LL - (_QWORD)v16);
    v15 = v31;
    v12 = v30;
  }
  v18 = *a4;
  v38 = v12;
  v19 = sub_AF8740(dest, &v38, v15, (unsigned __int64)&v41, v18);
  v20 = *a5;
  src = v38;
  v21 = sub_AF8740(dest, &src, v19, (unsigned __int64)&v41, v20);
  v22 = src;
  if ( !src )
    return sub_AC25F0(dest, v21 - (__int8 *)dest, (__int64)v45);
  v24 = v21 - (__int8 *)dest;
  sub_AF1140(dest[0].m128i_i8, v21, v41.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v41, dest);
  v25 = v41.m128i_i64[0]
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v44 ^ v43.m128i_i64[0]))
          ^ v44
          ^ ((0x9DDFEA08EB382D69LL * (v44 ^ v43.m128i_i64[0])) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v44 ^ v43.m128i_i64[0]))
         ^ v44
         ^ ((0x9DDFEA08EB382D69LL * (v44 ^ v43.m128i_i64[0])) >> 47))))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v22 + v24) >> 47) ^ (v22 + v24));
  v26 = 0x9DDFEA08EB382D69LL
      * (v25
       ^ (0xB492B66FBE98F273LL * (v41.m128i_i64[1] ^ ((unsigned __int64)v41.m128i_i64[1] >> 47))
        + v42.m128i_i64[0]
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1])) >> 47)
            ^ (0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1]))
            ^ v43.m128i_i64[1])) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1])) >> 47)
           ^ (0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1]))
           ^ v43.m128i_i64[1])))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v26 ^ v25 ^ (v26 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v26 ^ v25 ^ (v26 >> 47))));
}
