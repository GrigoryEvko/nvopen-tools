// Function: sub_AFA420
// Address: 0xafa420
//
unsigned __int64 __fastcall sub_AFA420(__int64 *a1, _QWORD *a2, __int64 *a3, __int64 *a4, int *a5)
{
  __int64 v6; // r8
  __int64 v7; // rbx
  __int64 v8; // rax
  char *v9; // r10
  __int8 *v10; // rdi
  char *v11; // rcx
  __int64 v12; // r8
  __int8 *v13; // rax
  __int8 *v14; // rax
  __int8 *v15; // rax
  __int64 v16; // r13
  signed __int64 v18; // rbx
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // rax
  __m128i v21; // xmm2
  __m128i v22; // xmm3
  __int8 *dest; // [rsp+8h] [rbp-128h]
  char *desta; // [rsp+8h] [rbp-128h]
  char *destb; // [rsp+8h] [rbp-128h]
  __m128i v28; // [rsp+20h] [rbp-110h] BYREF
  __m128i v29; // [rsp+30h] [rbp-100h] BYREF
  __m128i v30; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v31; // [rsp+50h] [rbp-E0h]
  __int64 v32; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v33; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v34; // [rsp+70h] [rbp-C0h] BYREF
  __int64 src; // [rsp+78h] [rbp-B8h] BYREF
  __m128i v36[4]; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v37; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v38; // [rsp+D0h] [rbp-60h]
  __m128i v39; // [rsp+E0h] [rbp-50h]
  __int64 v40; // [rsp+F0h] [rbp-40h]
  __int64 (__fastcall *v41)(); // [rsp+F8h] [rbp-38h]

  v6 = *a1;
  memset(v36, 0, sizeof(v36));
  v37 = 0u;
  v38 = 0u;
  v39 = 0u;
  v40 = 0;
  v41 = sub_C64CA0;
  v32 = 0;
  dest = sub_AF8740(v36, &v32, v36[0].m128i_i8, (unsigned __int64)&v37, v6);
  v7 = v32;
  v8 = sub_C94880(*a2, a2[1]);
  src = v8;
  v9 = dest + 8;
  if ( dest + 8 <= (__int8 *)&v37 )
  {
    *(_QWORD *)dest = v8;
  }
  else
  {
    v10 = dest;
    desta = (char *)((char *)&v37 - dest);
    memcpy(v10, &src, (size_t)desta);
    if ( v7 )
    {
      v7 += 64;
      sub_AC2A10((unsigned __int64 *)&v37, v36);
      v11 = desta;
    }
    else
    {
      v7 = 64;
      sub_AC28A0((unsigned __int64 *)&v28, v36[0].m128i_i64, (unsigned __int64)v41);
      v21 = _mm_loadu_si128(&v29);
      v22 = _mm_loadu_si128(&v30);
      v11 = desta;
      v37 = _mm_loadu_si128(&v28);
      v40 = v31;
      v38 = v21;
      v39 = v22;
    }
    destb = &v36[0].m128i_i8[8LL - (_QWORD)v11];
    if ( destb > (char *)&v37 )
      BUG();
    memcpy(v36, (char *)&src + (_QWORD)v11, 8LL - (_QWORD)v11);
    v9 = destb;
  }
  v12 = *a3;
  v33 = v7;
  v13 = sub_AF70F0(v36, &v33, v9, (unsigned __int64)&v37, v12);
  v34 = v33;
  v14 = sub_AF70F0(v36, &v34, v13, (unsigned __int64)&v37, *a4);
  src = v34;
  v15 = sub_AF6D70(v36, &src, v14, (unsigned __int64)&v37, *a5);
  v16 = src;
  if ( !src )
    return sub_AC25F0(v36, v15 - (__int8 *)v36, (__int64)v41);
  v18 = v15 - (__int8 *)v36;
  sub_AF1140(v36[0].m128i_i8, v15, v37.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v37, v36);
  v19 = v37.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v40 ^ v39.m128i_i64[0]))
         ^ v40
         ^ ((0x9DDFEA08EB382D69LL * (v40 ^ v39.m128i_i64[0])) >> 47)))
       ^ ((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v40 ^ v39.m128i_i64[0]))
          ^ v40
          ^ ((0x9DDFEA08EB382D69LL * (v40 ^ v39.m128i_i64[0])) >> 47))) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v16 + v18) >> 47) ^ (v16 + v18));
  v20 = 0x9DDFEA08EB382D69LL
      * (v19
       ^ (v38.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v37.m128i_i64[1] ^ ((unsigned __int64)v37.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v39.m128i_i64[1] ^ v38.m128i_i64[1])) >> 47)
            ^ (0x9DDFEA08EB382D69LL * (v39.m128i_i64[1] ^ v38.m128i_i64[1]))
            ^ v39.m128i_i64[1])) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v39.m128i_i64[1] ^ v38.m128i_i64[1])) >> 47)
           ^ (0x9DDFEA08EB382D69LL * (v39.m128i_i64[1] ^ v38.m128i_i64[1]))
           ^ v39.m128i_i64[1])))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v20 ^ v19 ^ (v20 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v20 ^ v19 ^ (v20 >> 47))));
}
