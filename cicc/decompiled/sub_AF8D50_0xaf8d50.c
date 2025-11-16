// Function: sub_AF8D50
// Address: 0xaf8d50
//
unsigned __int64 __fastcall sub_AF8D50(
        __int64 *a1,
        __int64 *a2,
        __int64 *a3,
        __int64 *a4,
        int *a5,
        __int64 *a6,
        __int8 *a7,
        __int8 *a8,
        __int64 *a9,
        __int64 *a10)
{
  __int64 v13; // r8
  __int8 *v14; // rax
  __int64 v15; // r8
  __int8 *v16; // rax
  __int64 v17; // r8
  __int8 *v18; // rax
  __int64 v19; // r8
  __int8 *v20; // rax
  int v21; // r8d
  __int8 *v22; // rax
  __int64 v23; // r8
  __int8 *v24; // rax
  __int64 v25; // r15
  __int8 *v26; // rdi
  __m128i *v27; // r14
  __int8 v28; // al
  char *v29; // rbx
  char *v30; // r9
  __int8 v31; // al
  __int8 *v32; // rax
  __int8 *v33; // rax
  __int64 v34; // r14
  __m128i v36; // xmm4
  __m128i v37; // xmm5
  __m128i v38; // xmm6
  char *v39; // rcx
  signed __int64 v40; // rbx
  unsigned __int64 v41; // rbx
  unsigned __int64 v42; // rax
  __m128i v43; // xmm2
  __m128i v44; // xmm3
  char *v46; // [rsp+8h] [rbp-138h]
  __m128i v47; // [rsp+10h] [rbp-130h] BYREF
  __m128i v48; // [rsp+20h] [rbp-120h] BYREF
  __m128i v49; // [rsp+30h] [rbp-110h] BYREF
  __int64 v50; // [rsp+40h] [rbp-100h]
  __int64 v51; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v52; // [rsp+58h] [rbp-E8h] BYREF
  __int64 v53; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v54; // [rsp+68h] [rbp-D8h] BYREF
  __int64 v55; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v56; // [rsp+78h] [rbp-C8h] BYREF
  __int64 src; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v58; // [rsp+88h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v60; // [rsp+D0h] [rbp-70h] BYREF
  __m128i v61; // [rsp+E0h] [rbp-60h]
  __m128i v62; // [rsp+F0h] [rbp-50h]
  __int64 v63; // [rsp+100h] [rbp-40h]
  __int64 (__fastcall *v64)(); // [rsp+108h] [rbp-38h]

  v13 = *a1;
  memset(dest, 0, sizeof(dest));
  v60 = 0u;
  v61 = 0u;
  v62 = 0u;
  v63 = 0;
  v64 = sub_C64CA0;
  v51 = 0;
  v14 = sub_AF70F0(dest, &v51, dest[0].m128i_i8, (unsigned __int64)&v60, v13);
  v15 = *a2;
  v52 = v51;
  v16 = sub_AF8740(dest, &v52, v14, (unsigned __int64)&v60, v15);
  v17 = *a3;
  v53 = v52;
  v18 = sub_AF8740(dest, &v53, v16, (unsigned __int64)&v60, v17);
  v19 = *a4;
  v54 = v53;
  v20 = sub_AF70F0(dest, &v54, v18, (unsigned __int64)&v60, v19);
  v21 = *a5;
  v55 = v54;
  v22 = sub_AF6D70(dest, &v55, v20, (unsigned __int64)&v60, v21);
  v23 = *a6;
  v56 = v55;
  v24 = sub_AF70F0(dest, &v56, v22, (unsigned __int64)&v60, v23);
  v25 = v56;
  v26 = v24;
  v27 = (__m128i *)(v24 + 1);
  v28 = *a7;
  LOBYTE(src) = *a7;
  if ( v27 <= &v60 )
  {
    *v26 = v28;
  }
  else
  {
    v29 = (char *)((char *)&v60 - v26);
    memcpy(v26, &src, (char *)&v60 - v26);
    if ( v25 )
    {
      v25 += 64;
      sub_AC2A10((unsigned __int64 *)&v60, dest);
    }
    else
    {
      v25 = 64;
      sub_AC28A0((unsigned __int64 *)&v47, dest[0].m128i_i64, (unsigned __int64)v64);
      v43 = _mm_loadu_si128(&v48);
      v44 = _mm_loadu_si128(&v49);
      v60 = _mm_loadu_si128(&v47);
      v63 = v50;
      v61 = v43;
      v62 = v44;
    }
    v27 = (__m128i *)((char *)dest + 1LL - (_QWORD)v29);
    if ( v27 > &v60 )
LABEL_5:
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v29, 1LL - (_QWORD)v29);
  }
  v30 = &v27->m128i_i8[1];
  v31 = *a8;
  LOBYTE(v58) = *a8;
  if ( &v27->m128i_i8[1] > (__int8 *)&v60 )
  {
    memcpy(v27, &v58, (char *)&v60 - (char *)v27);
    if ( v25 )
    {
      v25 += 64;
      sub_AC2A10((unsigned __int64 *)&v60, dest);
      v39 = (char *)((char *)&v60 - (char *)v27);
    }
    else
    {
      v25 = 64;
      sub_AC28A0((unsigned __int64 *)&v47, dest[0].m128i_i64, (unsigned __int64)v64);
      v36 = _mm_loadu_si128(&v47);
      v37 = _mm_loadu_si128(&v48);
      v38 = _mm_loadu_si128(&v49);
      v63 = v50;
      v39 = (char *)((char *)&v60 - (char *)v27);
      v60 = v36;
      v61 = v37;
      v62 = v38;
    }
    v46 = &dest[0].m128i_i8[1LL - (_QWORD)v39];
    if ( v46 > (char *)&v60 )
      goto LABEL_5;
    memcpy(dest, (char *)&v58 + (_QWORD)v39, 1LL - (_QWORD)v39);
    v30 = v46;
  }
  else
  {
    v27->m128i_i8[0] = v31;
  }
  src = v25;
  v32 = sub_AF70F0(dest, &src, v30, (unsigned __int64)&v60, *a9);
  v58 = src;
  v33 = sub_AF70F0(dest, &v58, v32, (unsigned __int64)&v60, *a10);
  v34 = v58;
  if ( !v58 )
    return sub_AC25F0(dest, v33 - (__int8 *)dest, (__int64)v64);
  v40 = v33 - (__int8 *)dest;
  sub_AF1140(dest[0].m128i_i8, v33, v60.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v60, dest);
  v41 = v60.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v63 ^ v62.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v63 ^ v62.m128i_i64[0]))
         ^ v63))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v63 ^ v62.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v63 ^ v62.m128i_i64[0]))
          ^ v63)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v34 + v40) >> 47) ^ (v34 + v40));
  v42 = 0x9DDFEA08EB382D69LL
      * (v41
       ^ (v61.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v60.m128i_i64[1] ^ ((unsigned __int64)v60.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v62.m128i_i64[1] ^ v61.m128i_i64[1])) >> 47)
            ^ (0x9DDFEA08EB382D69LL * (v62.m128i_i64[1] ^ v61.m128i_i64[1]))
            ^ v62.m128i_i64[1])) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v62.m128i_i64[1] ^ v61.m128i_i64[1])) >> 47)
           ^ (0x9DDFEA08EB382D69LL * (v62.m128i_i64[1] ^ v61.m128i_i64[1]))
           ^ v62.m128i_i64[1])))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v42 ^ v41 ^ (v42 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v42 ^ v41 ^ (v42 >> 47))));
}
