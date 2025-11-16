// Function: sub_15B5960
// Address: 0x15b5960
//
unsigned __int64 __fastcall sub_15B5960(__int64 *a1, __int64 *a2, int *a3, __int64 *a4, __int64 *a5)
{
  __int64 *v5; // r10
  __int64 v8; // r8
  __int8 *v9; // rax
  __int64 v10; // r8
  __int8 *v11; // rax
  __int64 v12; // rcx
  __int8 *v13; // rdi
  int v14; // eax
  char *v15; // r15
  char *v16; // r8
  __int64 v17; // rcx
  __int8 *v18; // rax
  __int64 v19; // r8
  __int8 *v20; // rax
  __int64 v21; // r14
  int v23; // eax
  unsigned __int64 v24; // rdx
  signed __int64 v25; // rbx
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // rax
  __m128i v28; // xmm1
  __m128i v29; // xmm2
  __int64 v30; // [rsp+8h] [rbp-128h]
  __int64 v31; // [rsp+10h] [rbp-120h]
  __m128i v33; // [rsp+20h] [rbp-110h] BYREF
  __m128i v34; // [rsp+30h] [rbp-100h] BYREF
  __m128i v35; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v36; // [rsp+50h] [rbp-E0h]
  __int64 v37; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v38; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v39; // [rsp+70h] [rbp-C0h] BYREF
  __int64 src; // [rsp+78h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v42; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v43; // [rsp+D0h] [rbp-60h]
  __m128i v44; // [rsp+E0h] [rbp-50h]
  __int64 v45; // [rsp+F0h] [rbp-40h]
  __int64 v46; // [rsp+F8h] [rbp-38h]

  v5 = a1;
  if ( !byte_4F99930[0] )
  {
    v23 = sub_2207590(byte_4F99930);
    v5 = a1;
    if ( v23 )
    {
      v24 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v24 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v24;
      sub_2207640(byte_4F99930);
      v5 = a1;
    }
  }
  v8 = *v5;
  v46 = qword_4F99938;
  v37 = 0;
  v9 = sub_15B3A60(dest, &v37, dest[0].m128i_i8, (unsigned __int64)&v42, v8);
  v10 = *a2;
  v38 = v37;
  v11 = sub_15B3A60(dest, &v38, v9, (unsigned __int64)&v42, v10);
  v12 = v38;
  v13 = v11;
  v14 = *a3;
  v15 = v13 + 4;
  LODWORD(src) = *a3;
  if ( v13 + 4 <= (__int8 *)&v42 )
  {
    *(_DWORD *)v13 = v14;
  }
  else
  {
    v30 = v38;
    memcpy(v13, &src, (char *)&v42 - v13);
    if ( v30 )
    {
      sub_1593A20((unsigned __int64 *)&v42, dest);
      v16 = (char *)((char *)&v42 - v13);
      v17 = v30 + 64;
    }
    else
    {
      sub_15938B0((unsigned __int64 *)&v33, dest[0].m128i_i64, v46);
      v28 = _mm_loadu_si128(&v34);
      v17 = 64;
      v29 = _mm_loadu_si128(&v35);
      v16 = (char *)((char *)&v42 - v13);
      v42 = _mm_loadu_si128(&v33);
      v45 = v36;
      v43 = v28;
      v44 = v29;
    }
    v31 = v17;
    v15 = &dest[0].m128i_i8[4LL - (_QWORD)v16];
    if ( v15 > (char *)&v42 )
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v16, 4LL - (_QWORD)v16);
    v12 = v31;
  }
  v39 = v12;
  v18 = sub_15B3A60(dest, &v39, v15, (unsigned __int64)&v42, *a4);
  v19 = *a5;
  src = v39;
  v20 = sub_15B3A60(dest, &src, v18, (unsigned __int64)&v42, v19);
  v21 = src;
  if ( !src )
    return sub_1593600(dest, v20 - (__int8 *)dest, v46);
  v25 = v20 - (__int8 *)dest;
  sub_15AF6E0(dest[0].m128i_i8, v20, v42.m128i_i8);
  sub_1593A20((unsigned __int64 *)&v42, dest);
  v26 = v42.m128i_i64[0]
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v45 ^ v44.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v45 ^ v44.m128i_i64[0]))
          ^ v45)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v45 ^ v44.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v45 ^ v44.m128i_i64[0]))
         ^ v45)))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v21 + v25) >> 47) ^ (v21 + v25));
  v27 = 0x9DDFEA08EB382D69LL
      * (v26
       ^ (v43.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v42.m128i_i64[1] ^ ((unsigned __int64)v42.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v44.m128i_i64[1] ^ v43.m128i_i64[1]))
            ^ v44.m128i_i64[1]
            ^ ((0x9DDFEA08EB382D69LL * (v44.m128i_i64[1] ^ v43.m128i_i64[1])) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v44.m128i_i64[1] ^ v43.m128i_i64[1]))
           ^ v44.m128i_i64[1]
           ^ ((0x9DDFEA08EB382D69LL * (v44.m128i_i64[1] ^ v43.m128i_i64[1])) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v26 ^ v27 ^ (v27 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v26 ^ v27 ^ (v27 >> 47))));
}
