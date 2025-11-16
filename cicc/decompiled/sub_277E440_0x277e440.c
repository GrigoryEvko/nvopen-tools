// Function: sub_277E440
// Address: 0x277e440
//
unsigned __int64 __fastcall sub_277E440(int *a1, int *a2, __int64 *a3, __int64 *a4)
{
  int v6; // r8d
  __int8 *v7; // rax
  __int64 v8; // rcx
  __int8 *v9; // rdi
  int v10; // eax
  char *v11; // r9
  char *v12; // r15
  __int64 v13; // rcx
  __int64 v14; // r8
  __int8 *v15; // rax
  __int64 v16; // r15
  __int8 *v17; // rdi
  __int64 v18; // rax
  char *v19; // r14
  unsigned __int64 v20; // rbx
  char *v22; // rbx
  __m128i v23; // xmm4
  __m128i v24; // xmm5
  __m128i v25; // xmm6
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  __m128i v28; // xmm2
  __m128i v29; // xmm3
  __int64 v30; // [rsp+8h] [rbp-128h]
  __int64 v31; // [rsp+18h] [rbp-118h]
  __m128i v32; // [rsp+20h] [rbp-110h] BYREF
  __m128i v33; // [rsp+30h] [rbp-100h] BYREF
  __m128i v34; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v35; // [rsp+50h] [rbp-E0h]
  int src; // [rsp+64h] [rbp-CCh] BYREF
  __int64 v37; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v38; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v39; // [rsp+78h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v41; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v42; // [rsp+D0h] [rbp-60h]
  __m128i v43; // [rsp+E0h] [rbp-50h]
  __int64 v44; // [rsp+F0h] [rbp-40h]
  void (__fastcall *v45)(__int64, __int64); // [rsp+F8h] [rbp-38h]

  v6 = *a1;
  v41 = 0u;
  v42 = 0u;
  v43 = 0u;
  v44 = 0;
  v45 = sub_C64CA0;
  v37 = 0;
  memset(dest, 0, sizeof(dest));
  v7 = sub_AF6D70(dest, &v37, dest[0].m128i_i8, (unsigned __int64)&v41, v6);
  v8 = v37;
  v9 = v7;
  v10 = *a2;
  v11 = v9 + 4;
  src = *a2;
  if ( v9 + 4 <= (__int8 *)&v41 )
  {
    *(_DWORD *)v9 = v10;
  }
  else
  {
    v31 = v37;
    v12 = (char *)((char *)&v41 - v9);
    memcpy(v9, &src, (char *)&v41 - v9);
    if ( v31 )
    {
      sub_AC2A10((unsigned __int64 *)&v41, dest);
      v13 = v31 + 64;
    }
    else
    {
      sub_AC28A0((unsigned __int64 *)&v32, dest[0].m128i_i64, (unsigned __int64)v45);
      v28 = _mm_loadu_si128(&v33);
      v13 = 64;
      v29 = _mm_loadu_si128(&v34);
      v41 = _mm_loadu_si128(&v32);
      v44 = v35;
      v42 = v28;
      v43 = v29;
    }
    v30 = v13;
    if ( (__m128i *)((char *)dest + 4LL - (_QWORD)v12) > &v41 )
LABEL_5:
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v12, 4LL - (_QWORD)v12);
    v11 = &dest[0].m128i_i8[4LL - (_QWORD)v12];
    v8 = v30;
  }
  v14 = *a3;
  v38 = v8;
  v15 = sub_277DD80(dest, &v38, v11, (unsigned __int64)&v41, v14);
  v16 = v38;
  v17 = v15;
  v18 = *a4;
  v19 = v17 + 8;
  v39 = *a4;
  if ( v17 + 8 > (__int8 *)&v41 )
  {
    v22 = (char *)((char *)&v41 - v17);
    memcpy(v17, &v39, (char *)&v41 - v17);
    if ( v16 )
    {
      v16 += 64;
      sub_AC2A10((unsigned __int64 *)&v41, dest);
    }
    else
    {
      v16 = 64;
      sub_AC28A0((unsigned __int64 *)&v32, dest[0].m128i_i64, (unsigned __int64)v45);
      v23 = _mm_loadu_si128(&v32);
      v24 = _mm_loadu_si128(&v33);
      v25 = _mm_loadu_si128(&v34);
      v44 = v35;
      v41 = v23;
      v42 = v24;
      v43 = v25;
    }
    v19 = &dest[0].m128i_i8[8LL - (_QWORD)v22];
    if ( v19 > (char *)&v41 )
      goto LABEL_5;
    memcpy(dest, (char *)&v39 + (_QWORD)v22, 8LL - (_QWORD)v22);
    v20 = 8LL - (_QWORD)v22;
    if ( !v16 )
      return sub_AC25F0(dest, v20, (__int64)v45);
  }
  else
  {
    *(_QWORD *)v17 = v18;
    v20 = v19 - (char *)dest;
    if ( !v16 )
      return sub_AC25F0(&dest[0], v20, (__int64)v45);
  }
  sub_2778790(dest[0].m128i_i8, v19, v41.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v41, dest);
  v26 = v41.m128i_i64[0]
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v44 ^ v43.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v44 ^ v43.m128i_i64[0]))
          ^ v44)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v44 ^ v43.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v44 ^ v43.m128i_i64[0]))
         ^ v44)))
      - 0x4B6D499041670D8DLL * (((v16 + v20) >> 47) ^ (v16 + v20));
  v27 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (v26
          ^ (v42.m128i_i64[0]
           - 0x4B6D499041670D8DLL * (v41.m128i_i64[1] ^ ((unsigned __int64)v41.m128i_i64[1] >> 47))
           - 0x622015F714C7D297LL
           * (((0x9DDFEA08EB382D69LL
              * ((0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1]))
               ^ v43.m128i_i64[1]
               ^ ((0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1])) >> 47))) >> 47)
            ^ (0x9DDFEA08EB382D69LL
             * ((0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1]))
              ^ v43.m128i_i64[1]
              ^ ((0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1])) >> 47))))))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (v26
         ^ (v42.m128i_i64[0]
          - 0x4B6D499041670D8DLL * (v41.m128i_i64[1] ^ ((unsigned __int64)v41.m128i_i64[1] >> 47))
          - 0x622015F714C7D297LL
          * (((0x9DDFEA08EB382D69LL
             * ((0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1]))
              ^ v43.m128i_i64[1]
              ^ ((0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1])) >> 47))) >> 47)
           ^ (0x9DDFEA08EB382D69LL
            * ((0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1]))
             ^ v43.m128i_i64[1]
             ^ ((0x9DDFEA08EB382D69LL * (v43.m128i_i64[1] ^ v42.m128i_i64[1])) >> 47)))))))
       ^ v26);
  return 0x9DDFEA08EB382D69LL * ((v27 >> 47) ^ v27);
}
