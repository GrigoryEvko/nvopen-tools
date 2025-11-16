// Function: sub_15B3100
// Address: 0x15b3100
//
unsigned __int64 __fastcall sub_15B3100(int *a1, int *a2, __int64 *a3, __int64 *a4)
{
  int *v4; // r8
  int v7; // r8d
  __int8 *v8; // rax
  __int64 v9; // rcx
  __int8 *v10; // rdi
  int v11; // eax
  char *v12; // r9
  char *v13; // r8
  __int64 v14; // rcx
  __int64 v15; // r8
  __int8 *v16; // rax
  __int64 v17; // r8
  __int8 *v18; // rax
  __int64 v19; // r14
  int v21; // eax
  unsigned __int64 v22; // rax
  signed __int64 v23; // rbx
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // rax
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  __int64 v28; // [rsp+0h] [rbp-120h]
  __int64 v29; // [rsp+0h] [rbp-120h]
  char *v30; // [rsp+8h] [rbp-118h]
  __m128i v31; // [rsp+10h] [rbp-110h] BYREF
  __m128i v32; // [rsp+20h] [rbp-100h] BYREF
  __m128i v33; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v34; // [rsp+40h] [rbp-E0h]
  __int64 v35; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v36; // [rsp+60h] [rbp-C0h] BYREF
  __int64 src; // [rsp+68h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v39; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v40; // [rsp+C0h] [rbp-60h]
  __m128i v41; // [rsp+D0h] [rbp-50h]
  __int64 v42; // [rsp+E0h] [rbp-40h]
  __int64 v43; // [rsp+E8h] [rbp-38h]

  v4 = a1;
  if ( !byte_4F99930[0] )
  {
    v21 = sub_2207590(byte_4F99930);
    v4 = a1;
    if ( v21 )
    {
      v22 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v22 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v22;
      sub_2207640(byte_4F99930);
      v4 = a1;
    }
  }
  v7 = *v4;
  v43 = qword_4F99938;
  v35 = 0;
  v8 = sub_15B2130(dest, &v35, dest[0].m128i_i8, (unsigned __int64)&v39, v7);
  v9 = v35;
  v10 = v8;
  v11 = *a2;
  v12 = v10 + 4;
  LODWORD(src) = *a2;
  if ( v10 + 4 <= (__int8 *)&v39 )
  {
    *(_DWORD *)v10 = v11;
  }
  else
  {
    v28 = v35;
    memcpy(v10, &src, (char *)&v39 - v10);
    if ( v28 )
    {
      sub_1593A20((unsigned __int64 *)&v39, dest);
      v13 = (char *)((char *)&v39 - v10);
      v14 = v28 + 64;
    }
    else
    {
      sub_15938B0((unsigned __int64 *)&v31, dest[0].m128i_i64, v43);
      v26 = _mm_loadu_si128(&v32);
      v14 = 64;
      v27 = _mm_loadu_si128(&v33);
      v13 = (char *)((char *)&v39 - v10);
      v39 = _mm_loadu_si128(&v31);
      v42 = v34;
      v40 = v26;
      v41 = v27;
    }
    v29 = v14;
    v30 = &dest[0].m128i_i8[4LL - (_QWORD)v13];
    if ( v30 > (char *)&v39 )
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v13, 4LL - (_QWORD)v13);
    v12 = v30;
    v9 = v29;
  }
  v15 = *a3;
  v36 = v9;
  v16 = sub_15B2320(dest, &v36, v12, (unsigned __int64)&v39, v15);
  v17 = *a4;
  src = v36;
  v18 = sub_15B2320(dest, &src, v16, (unsigned __int64)&v39, v17);
  v19 = src;
  if ( !src )
    return sub_1593600(dest, v18 - (__int8 *)dest, v43);
  v23 = v18 - (__int8 *)dest;
  sub_15AF6E0(dest[0].m128i_i8, v18, v39.m128i_i8);
  sub_1593A20((unsigned __int64 *)&v39, dest);
  v24 = v39.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v42 ^ v41.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v42 ^ v41.m128i_i64[0]))
         ^ v42))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v42 ^ v41.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v42 ^ v41.m128i_i64[0]))
          ^ v42)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v19 + v23) >> 47) ^ (v19 + v23));
  v25 = 0x9DDFEA08EB382D69LL
      * (v24
       ^ (v40.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v39.m128i_i64[1] ^ ((unsigned __int64)v39.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v41.m128i_i64[1] ^ v40.m128i_i64[1]))
            ^ v41.m128i_i64[1]
            ^ ((0x9DDFEA08EB382D69LL * (v41.m128i_i64[1] ^ v40.m128i_i64[1])) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v41.m128i_i64[1] ^ v40.m128i_i64[1]))
           ^ v41.m128i_i64[1]
           ^ ((0x9DDFEA08EB382D69LL * (v41.m128i_i64[1] ^ v40.m128i_i64[1])) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v25 ^ v24 ^ (v25 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v25 ^ v24 ^ (v25 >> 47))));
}
