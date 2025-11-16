// Function: sub_277EFE0
// Address: 0x277efe0
//
unsigned __int64 __fastcall sub_277EFE0(int *a1, int *a2, __int64 *a3, __int64 *a4, __int64 *a5, __int64 *a6)
{
  int v8; // r8d
  __int8 *v9; // rax
  __int64 v10; // rcx
  __int8 *v11; // rdi
  int v12; // eax
  char *v13; // r10
  char *v14; // r8
  __int64 v15; // rcx
  __int8 *v16; // rax
  __int64 v17; // r8
  __int8 *v18; // rax
  __int64 v19; // r8
  __int8 *v20; // rax
  __int8 *v21; // rax
  __int64 v22; // r13
  signed __int64 v24; // rbx
  unsigned __int64 v25; // rbx
  unsigned __int64 v26; // rax
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __int64 v29; // [rsp+0h] [rbp-140h]
  __int64 v30; // [rsp+0h] [rbp-140h]
  char *v31; // [rsp+8h] [rbp-138h]
  __m128i v34; // [rsp+20h] [rbp-120h] BYREF
  __m128i v35; // [rsp+30h] [rbp-110h] BYREF
  __m128i v36; // [rsp+40h] [rbp-100h] BYREF
  __int64 v37; // [rsp+50h] [rbp-F0h]
  __int64 v38; // [rsp+68h] [rbp-D8h] BYREF
  __int64 v39; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v40; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v41; // [rsp+80h] [rbp-C0h] BYREF
  __int64 src; // [rsp+88h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v44; // [rsp+D0h] [rbp-70h] BYREF
  __m128i v45; // [rsp+E0h] [rbp-60h]
  __m128i v46; // [rsp+F0h] [rbp-50h]
  __int64 v47; // [rsp+100h] [rbp-40h]
  void (__fastcall *v48)(__int64, __int64); // [rsp+108h] [rbp-38h]

  v8 = *a1;
  v44 = 0u;
  v45 = 0u;
  v46 = 0u;
  v47 = 0;
  v48 = sub_C64CA0;
  v38 = 0;
  memset(dest, 0, sizeof(dest));
  v9 = sub_AF6D70(dest, &v38, dest[0].m128i_i8, (unsigned __int64)&v44, v8);
  v10 = v38;
  v11 = v9;
  v12 = *a2;
  v13 = v11 + 4;
  LODWORD(src) = *a2;
  if ( v11 + 4 <= (__int8 *)&v44 )
  {
    *(_DWORD *)v11 = v12;
  }
  else
  {
    v29 = v38;
    memcpy(v11, &src, (char *)&v44 - v11);
    if ( v29 )
    {
      sub_AC2A10((unsigned __int64 *)&v44, dest);
      v14 = (char *)((char *)&v44 - v11);
      v15 = v29 + 64;
    }
    else
    {
      sub_AC28A0((unsigned __int64 *)&v34, dest[0].m128i_i64, (unsigned __int64)v48);
      v27 = _mm_loadu_si128(&v35);
      v15 = 64;
      v28 = _mm_loadu_si128(&v36);
      v14 = (char *)((char *)&v44 - v11);
      v44 = _mm_loadu_si128(&v34);
      v47 = v37;
      v45 = v27;
      v46 = v28;
    }
    v30 = v15;
    v31 = &dest[0].m128i_i8[4LL - (_QWORD)v14];
    if ( v31 > (char *)&v44 )
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v14, 4LL - (_QWORD)v14);
    v13 = v31;
    v10 = v30;
  }
  v39 = v10;
  v16 = sub_277DD80(dest, &v39, v13, (unsigned __int64)&v44, *a3);
  v17 = *a4;
  v40 = v39;
  v18 = sub_277DD80(dest, &v40, v16, (unsigned __int64)&v44, v17);
  v19 = *a5;
  v41 = v40;
  v20 = sub_277DD80(dest, &v41, v18, (unsigned __int64)&v44, v19);
  src = v41;
  v21 = sub_277DD80(dest, &src, v20, (unsigned __int64)&v44, *a6);
  v22 = src;
  if ( !src )
    return sub_AC25F0(dest, v21 - (__int8 *)dest, (__int64)v48);
  v24 = v21 - (__int8 *)dest;
  sub_2778790(dest[0].m128i_i8, v21, v44.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v44, dest);
  v25 = v44.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v47 ^ v46.m128i_i64[0]))
         ^ v47
         ^ ((0x9DDFEA08EB382D69LL * (v47 ^ v46.m128i_i64[0])) >> 47)))
       ^ ((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v47 ^ v46.m128i_i64[0]))
          ^ v47
          ^ ((0x9DDFEA08EB382D69LL * (v47 ^ v46.m128i_i64[0])) >> 47))) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v22 + v24) >> 47) ^ (v22 + v24));
  v26 = 0x9DDFEA08EB382D69LL
      * (v25
       ^ (v45.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v44.m128i_i64[1] ^ ((unsigned __int64)v44.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v46.m128i_i64[1] ^ v45.m128i_i64[1])) >> 47)
            ^ (0x9DDFEA08EB382D69LL * (v46.m128i_i64[1] ^ v45.m128i_i64[1]))
            ^ v46.m128i_i64[1])) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v46.m128i_i64[1] ^ v45.m128i_i64[1])) >> 47)
           ^ (0x9DDFEA08EB382D69LL * (v46.m128i_i64[1] ^ v45.m128i_i64[1]))
           ^ v46.m128i_i64[1])))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v26 ^ v25 ^ (v26 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v26 ^ v25 ^ (v26 >> 47))));
}
