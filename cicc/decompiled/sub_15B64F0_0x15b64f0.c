// Function: sub_15B64F0
// Address: 0x15b64f0
//
unsigned __int64 __fastcall sub_15B64F0(int *a1, int *a2, __int64 *a3)
{
  int v4; // r8d
  __int8 *v5; // rax
  __int64 v6; // r15
  __int8 *v7; // rdi
  int v8; // eax
  char *v9; // r14
  char *v10; // rcx
  __int64 v11; // r8
  __int8 *v12; // rax
  __int64 v13; // r14
  unsigned __int64 v15; // rax
  signed __int64 v16; // rbx
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rax
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __m128i v21; // [rsp+10h] [rbp-100h] BYREF
  __m128i v22; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v23; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v24; // [rsp+40h] [rbp-D0h]
  __int64 v25; // [rsp+50h] [rbp-C0h] BYREF
  __int64 src; // [rsp+58h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v28; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v29; // [rsp+B0h] [rbp-60h]
  __m128i v30; // [rsp+C0h] [rbp-50h]
  __int64 v31; // [rsp+D0h] [rbp-40h]
  __int64 v32; // [rsp+D8h] [rbp-38h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v15 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v15 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v15;
    sub_2207640(byte_4F99930);
  }
  v4 = *a1;
  v32 = qword_4F99938;
  v25 = 0;
  v5 = sub_15B2130(dest, &v25, dest[0].m128i_i8, (unsigned __int64)&v28, v4);
  v6 = v25;
  v7 = v5;
  v8 = *a2;
  v9 = v7 + 4;
  LODWORD(src) = *a2;
  if ( v7 + 4 <= (__int8 *)&v28 )
  {
    *(_DWORD *)v7 = v8;
  }
  else
  {
    memcpy(v7, &src, (char *)&v28 - v7);
    if ( v6 )
    {
      v6 += 64;
      sub_1593A20((unsigned __int64 *)&v28, dest);
      v10 = (char *)((char *)&v28 - v7);
    }
    else
    {
      v6 = 64;
      sub_15938B0((unsigned __int64 *)&v21, dest[0].m128i_i64, v32);
      v19 = _mm_loadu_si128(&v22);
      v20 = _mm_loadu_si128(&v23);
      v10 = (char *)((char *)&v28 - v7);
      v28 = _mm_loadu_si128(&v21);
      v31 = v24;
      v29 = v19;
      v30 = v20;
    }
    v9 = &dest[0].m128i_i8[4LL - (_QWORD)v10];
    if ( v9 > (char *)&v28 )
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v10, 4LL - (_QWORD)v10);
  }
  v11 = *a3;
  src = v6;
  v12 = sub_15B3A60(dest, &src, v9, (unsigned __int64)&v28, v11);
  v13 = src;
  if ( !src )
    return sub_1593600(dest, v12 - (__int8 *)dest, v32);
  v16 = v12 - (__int8 *)dest;
  sub_15AF6E0(dest[0].m128i_i8, v12, v28.m128i_i8);
  sub_1593A20((unsigned __int64 *)&v28, dest);
  v17 = v28.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v31 ^ v30.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v31 ^ v30.m128i_i64[0]))
         ^ v31))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v31 ^ v30.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v31 ^ v30.m128i_i64[0]))
          ^ v31)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v13 + v16) >> 47) ^ (v13 + v16));
  v18 = 0x9DDFEA08EB382D69LL
      * (v17
       ^ (v29.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v28.m128i_i64[1] ^ ((unsigned __int64)v28.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v30.m128i_i64[1] ^ v29.m128i_i64[1])) >> 47)
            ^ (0x9DDFEA08EB382D69LL * (v30.m128i_i64[1] ^ v29.m128i_i64[1]))
            ^ v30.m128i_i64[1])) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v30.m128i_i64[1] ^ v29.m128i_i64[1])) >> 47)
           ^ (0x9DDFEA08EB382D69LL * (v30.m128i_i64[1] ^ v29.m128i_i64[1]))
           ^ v30.m128i_i64[1])))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v18 ^ v17 ^ (v18 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v18 ^ v17 ^ (v18 >> 47))));
}
