// Function: sub_AF71E0
// Address: 0xaf71e0
//
unsigned __int64 __fastcall sub_AF71E0(int *a1, int *a2, __int64 *a3, __int64 *a4, __int8 *a5)
{
  int v8; // r8d
  __int8 *v9; // rax
  int v10; // r8d
  __int8 *v11; // rax
  __int64 v12; // r8
  __int8 *v13; // rax
  __int64 v14; // r8
  __int8 *v15; // rax
  __int64 v16; // r15
  __int8 *v17; // rdi
  __int8 v18; // al
  char *v19; // r14
  char *v20; // rbx
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __m128i v26; // [rsp+10h] [rbp-120h] BYREF
  __m128i v27; // [rsp+20h] [rbp-110h] BYREF
  __m128i v28; // [rsp+30h] [rbp-100h] BYREF
  __int64 v29; // [rsp+40h] [rbp-F0h]
  __int8 src; // [rsp+5Fh] [rbp-D1h] BYREF
  __int64 v31; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v32; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v33; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+78h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v36; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v37; // [rsp+D0h] [rbp-60h]
  __m128i v38; // [rsp+E0h] [rbp-50h]
  __int64 v39; // [rsp+F0h] [rbp-40h]
  __int64 (__fastcall *v40)(); // [rsp+F8h] [rbp-38h]

  v8 = *a1;
  memset(dest, 0, sizeof(dest));
  v36 = 0u;
  v37 = 0u;
  v38 = 0u;
  v39 = 0;
  v40 = sub_C64CA0;
  v31 = 0;
  v9 = sub_AF6D70(dest, &v31, dest[0].m128i_i8, (unsigned __int64)&v36, v8);
  v10 = *a2;
  v32 = v31;
  v11 = sub_AF6D70(dest, &v32, v9, (unsigned __int64)&v36, v10);
  v12 = *a3;
  v33 = v32;
  v13 = sub_AF70F0(dest, &v33, v11, (unsigned __int64)&v36, v12);
  v14 = *a4;
  v34 = v33;
  v15 = sub_AF70F0(dest, &v34, v13, (unsigned __int64)&v36, v14);
  v16 = v34;
  v17 = v15;
  v18 = *a5;
  v19 = v17 + 1;
  src = *a5;
  if ( v17 + 1 <= (__int8 *)&v36 )
  {
    *v17 = v18;
  }
  else
  {
    v20 = (char *)((char *)&v36 - v17);
    memcpy(v17, &src, (char *)&v36 - v17);
    if ( v16 )
    {
      v16 += 64;
      sub_AC2A10((unsigned __int64 *)&v36, dest);
    }
    else
    {
      v16 = 64;
      sub_AC28A0((unsigned __int64 *)&v26, dest[0].m128i_i64, (unsigned __int64)v40);
      v24 = _mm_loadu_si128(&v27);
      v25 = _mm_loadu_si128(&v28);
      v36 = _mm_loadu_si128(&v26);
      v39 = v29;
      v37 = v24;
      v38 = v25;
    }
    v19 = &dest[0].m128i_i8[1LL - (_QWORD)v20];
    if ( v19 > (char *)&v36 )
      BUG();
    memcpy(dest, &src + (_QWORD)v20, 1LL - (_QWORD)v20);
  }
  if ( !v16 )
    return sub_AC25F0(dest, v19 - (char *)dest, (__int64)v40);
  sub_AF1140(dest[0].m128i_i8, v19, v36.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v36, dest);
  v22 = v36.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0]))
         ^ v39))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0]))
          ^ v39)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v16 + v19 - (char *)dest) >> 47) ^ (v16 + v19 - (char *)dest));
  v23 = 0x9DDFEA08EB382D69LL
      * (v22
       ^ (v37.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v36.m128i_i64[1] ^ ((unsigned __int64)v36.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v38.m128i_i64[1] ^ v37.m128i_i64[1])) >> 47)
            ^ (0x9DDFEA08EB382D69LL * (v38.m128i_i64[1] ^ v37.m128i_i64[1]))
            ^ v38.m128i_i64[1])) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v38.m128i_i64[1] ^ v37.m128i_i64[1])) >> 47)
           ^ (0x9DDFEA08EB382D69LL * (v38.m128i_i64[1] ^ v37.m128i_i64[1]))
           ^ v38.m128i_i64[1])))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * ((v23 >> 47) ^ v23 ^ v22)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v23 >> 47) ^ v23 ^ v22)));
}
