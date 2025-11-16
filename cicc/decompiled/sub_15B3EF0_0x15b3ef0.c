// Function: sub_15B3EF0
// Address: 0x15b3ef0
//
unsigned __int64 __fastcall sub_15B3EF0(__int64 *a1, __int64 *a2, int *a3)
{
  __int64 v4; // r8
  __int8 *v5; // rax
  __int64 v6; // r8
  __int8 *v7; // rax
  __int64 v8; // r15
  __int8 *v9; // rdi
  int v10; // eax
  char *v11; // r14
  char *v12; // rbx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __m128i v19; // [rsp+10h] [rbp-110h] BYREF
  __m128i v20; // [rsp+20h] [rbp-100h] BYREF
  __m128i v21; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v22; // [rsp+40h] [rbp-E0h]
  int src; // [rsp+5Ch] [rbp-C4h] BYREF
  __int64 v24; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v25; // [rsp+68h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v27; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v28; // [rsp+C0h] [rbp-60h]
  __m128i v29; // [rsp+D0h] [rbp-50h]
  __int64 v30; // [rsp+E0h] [rbp-40h]
  __int64 v31; // [rsp+E8h] [rbp-38h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v14 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v14 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v14;
    sub_2207640(byte_4F99930);
  }
  v4 = *a1;
  v31 = qword_4F99938;
  v24 = 0;
  v5 = sub_15B2320(dest, &v24, dest[0].m128i_i8, (unsigned __int64)&v27, v4);
  v6 = *a2;
  v25 = v24;
  v7 = sub_15B3A60(dest, &v25, v5, (unsigned __int64)&v27, v6);
  v8 = v25;
  v9 = v7;
  v10 = *a3;
  v11 = v9 + 4;
  src = *a3;
  if ( v9 + 4 <= (__int8 *)&v27 )
  {
    *(_DWORD *)v9 = v10;
  }
  else
  {
    v12 = (char *)((char *)&v27 - v9);
    memcpy(v9, &src, (char *)&v27 - v9);
    if ( v8 )
    {
      v8 += 64;
      sub_1593A20((unsigned __int64 *)&v27, dest);
    }
    else
    {
      v8 = 64;
      sub_15938B0((unsigned __int64 *)&v19, dest[0].m128i_i64, v31);
      v17 = _mm_loadu_si128(&v20);
      v18 = _mm_loadu_si128(&v21);
      v27 = _mm_loadu_si128(&v19);
      v30 = v22;
      v28 = v17;
      v29 = v18;
    }
    v11 = &dest[0].m128i_i8[4LL - (_QWORD)v12];
    if ( v11 > (char *)&v27 )
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v12, 4LL - (_QWORD)v12);
  }
  if ( !v8 )
    return sub_1593600(dest, v11 - (char *)dest, v31);
  sub_15AF6E0(dest[0].m128i_i8, v11, v27.m128i_i8);
  sub_1593A20((unsigned __int64 *)&v27, dest);
  v15 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v8 + v11 - (char *)dest) >> 47) ^ (v8 + v11 - (char *)dest))
      + v27.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v30 ^ v29.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v30 ^ v29.m128i_i64[0]))
         ^ v30))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v30 ^ v29.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v30 ^ v29.m128i_i64[0]))
          ^ v30)) >> 47));
  v16 = 0x9DDFEA08EB382D69LL
      * (v15
       ^ (v28.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v27.m128i_i64[1] ^ ((unsigned __int64)v27.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v29.m128i_i64[1] ^ v28.m128i_i64[1]))
            ^ v29.m128i_i64[1]
            ^ ((0x9DDFEA08EB382D69LL * (v29.m128i_i64[1] ^ v28.m128i_i64[1])) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v29.m128i_i64[1] ^ v28.m128i_i64[1]))
           ^ v29.m128i_i64[1]
           ^ ((0x9DDFEA08EB382D69LL * (v29.m128i_i64[1] ^ v28.m128i_i64[1])) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * ((v16 >> 47) ^ v16 ^ v15)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v16 >> 47) ^ v16 ^ v15)));
}
