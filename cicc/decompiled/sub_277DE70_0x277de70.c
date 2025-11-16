// Function: sub_277DE70
// Address: 0x277de70
//
unsigned __int64 __fastcall sub_277DE70(int *a1, __int64 *a2, __int64 *a3, __int64 *a4)
{
  int v5; // r8d
  __int8 *v6; // rax
  __int64 v7; // r8
  __int8 *v8; // rax
  __int64 v9; // rbx
  __int8 *v10; // rdi
  __int64 v11; // rax
  __m128i *v12; // r15
  char *v13; // r14
  char *v14; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // r15
  __m128i v18; // xmm4
  __m128i v19; // xmm5
  __m128i v20; // xmm6
  char *v21; // r8
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __m128i v27; // [rsp+10h] [rbp-110h] BYREF
  __m128i v28; // [rsp+20h] [rbp-100h] BYREF
  __m128i v29; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v30; // [rsp+40h] [rbp-E0h]
  __int64 v31; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v32; // [rsp+58h] [rbp-C8h] BYREF
  __int64 src; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+68h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v36; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v37; // [rsp+C0h] [rbp-60h]
  __m128i v38; // [rsp+D0h] [rbp-50h]
  __int64 v39; // [rsp+E0h] [rbp-40h]
  void (__fastcall *v40)(__int64, __int64); // [rsp+E8h] [rbp-38h]

  v5 = *a1;
  memset(dest, 0, sizeof(dest));
  v36 = 0u;
  v37 = 0u;
  v38 = 0u;
  v39 = 0;
  v40 = sub_C64CA0;
  v31 = 0;
  v6 = sub_AF6D70(dest, &v31, dest[0].m128i_i8, (unsigned __int64)&v36, v5);
  v7 = *a2;
  v32 = v31;
  v8 = sub_277DD80(dest, &v32, v6, (unsigned __int64)&v36, v7);
  v9 = v32;
  v10 = v8;
  v11 = *a3;
  v12 = (__m128i *)(v10 + 8);
  src = *a3;
  if ( v10 + 8 <= (__int8 *)&v36 )
  {
    *(_QWORD *)v10 = v11;
  }
  else
  {
    v13 = (char *)((char *)&v36 - v10);
    memcpy(v10, &src, (char *)&v36 - v10);
    if ( v9 )
    {
      v9 += 64;
      sub_AC2A10((unsigned __int64 *)&v36, dest);
    }
    else
    {
      v9 = 64;
      sub_AC28A0((unsigned __int64 *)&v27, dest[0].m128i_i64, (unsigned __int64)v40);
      v24 = _mm_loadu_si128(&v28);
      v25 = _mm_loadu_si128(&v29);
      v36 = _mm_loadu_si128(&v27);
      v39 = v30;
      v37 = v24;
      v38 = v25;
    }
    v12 = (__m128i *)((char *)dest + 8LL - (_QWORD)v13);
    if ( v12 > &v36 )
LABEL_5:
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v13, 8LL - (_QWORD)v13);
  }
  v14 = &v12->m128i_i8[8];
  v15 = *a4;
  v34 = *a4;
  if ( &v12->m128i_u64[1] > (unsigned __int64 *)&v36 )
  {
    memcpy(v12, &v34, (char *)&v36 - (char *)v12);
    if ( v9 )
    {
      v9 += 64;
      sub_AC2A10((unsigned __int64 *)&v36, dest);
      v21 = (char *)((char *)&v36 - (char *)v12);
    }
    else
    {
      v9 = 64;
      sub_AC28A0((unsigned __int64 *)&v27, dest[0].m128i_i64, (unsigned __int64)v40);
      v18 = _mm_loadu_si128(&v27);
      v19 = _mm_loadu_si128(&v28);
      v20 = _mm_loadu_si128(&v29);
      v39 = v30;
      v21 = (char *)((char *)&v36 - (char *)v12);
      v36 = v18;
      v37 = v19;
      v38 = v20;
    }
    v14 = &dest[0].m128i_i8[8LL - (_QWORD)v21];
    if ( v14 > (char *)&v36 )
      goto LABEL_5;
    v16 = 8LL - (_QWORD)v21;
    memcpy(dest, (char *)&v34 + (_QWORD)v21, 8LL - (_QWORD)v21);
    if ( !v9 )
      return sub_AC25F0(dest, v16, (__int64)v40);
  }
  else
  {
    v12->m128i_i64[0] = v15;
    v16 = v14 - (char *)dest;
    if ( !v9 )
      return sub_AC25F0(&dest[0], v16, (__int64)v40);
  }
  sub_2778790(dest[0].m128i_i8, v14, v36.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v36, dest);
  v22 = 0x9DDFEA08EB382D69LL
      * ((0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0]))
         ^ v39
         ^ ((0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0])) >> 47)))
       ^ ((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0]))
          ^ v39
          ^ ((0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0])) >> 47))) >> 47))
      + v36.m128i_i64[0]
      - 0x4B6D499041670D8DLL * (((v9 + v16) >> 47) ^ (v9 + v16));
  v23 = 0x9DDFEA08EB382D69LL
      * ((0x9DDFEA08EB382D69LL
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
             ^ v38.m128i_i64[1]))))))
       ^ v22
       ^ ((0x9DDFEA08EB382D69LL
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
              ^ v38.m128i_i64[1])))))) >> 47));
  return 0x9DDFEA08EB382D69LL * ((v23 >> 47) ^ v23);
}
