// Function: sub_15B2700
// Address: 0x15b2700
//
unsigned __int64 __fastcall sub_15B2700(__int64 *a1, __int64 *a2, int *a3, int *a4)
{
  __int64 *v4; // r8
  __int64 v7; // r8
  __int8 *v8; // rax
  __int64 v9; // r8
  __int8 *v10; // rax
  int v11; // r8d
  __int8 *v12; // rax
  __int64 v13; // r15
  __int8 *v14; // rdi
  int v15; // eax
  char *v16; // r14
  char *v17; // rbx
  int v19; // eax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __m128i v23; // xmm1
  __m128i v24; // xmm2
  __m128i v25; // [rsp+10h] [rbp-110h] BYREF
  __m128i v26; // [rsp+20h] [rbp-100h] BYREF
  __m128i v27; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v28; // [rsp+40h] [rbp-E0h]
  int src; // [rsp+54h] [rbp-CCh] BYREF
  __int64 v30; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v31; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v32; // [rsp+68h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v34; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v35; // [rsp+C0h] [rbp-60h]
  __m128i v36; // [rsp+D0h] [rbp-50h]
  __int64 v37; // [rsp+E0h] [rbp-40h]
  __int64 v38; // [rsp+E8h] [rbp-38h]

  v4 = a1;
  if ( !byte_4F99930[0] )
  {
    v19 = sub_2207590(byte_4F99930);
    v4 = a1;
    if ( v19 )
    {
      v20 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v20 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v20;
      sub_2207640(byte_4F99930);
      v4 = a1;
    }
  }
  v7 = *v4;
  v38 = qword_4F99938;
  v30 = 0;
  v8 = sub_15B2320(dest, &v30, dest[0].m128i_i8, (unsigned __int64)&v34, v7);
  v9 = *a2;
  v31 = v30;
  v10 = sub_15B2320(dest, &v31, v8, (unsigned __int64)&v34, v9);
  v11 = *a3;
  v32 = v31;
  v12 = sub_15B2130(dest, &v32, v10, (unsigned __int64)&v34, v11);
  v13 = v32;
  v14 = v12;
  v15 = *a4;
  v16 = v14 + 4;
  src = *a4;
  if ( v14 + 4 <= (__int8 *)&v34 )
  {
    *(_DWORD *)v14 = v15;
  }
  else
  {
    v17 = (char *)((char *)&v34 - v14);
    memcpy(v14, &src, (char *)&v34 - v14);
    if ( v13 )
    {
      v13 += 64;
      sub_1593A20((unsigned __int64 *)&v34, dest);
    }
    else
    {
      v13 = 64;
      sub_15938B0((unsigned __int64 *)&v25, dest[0].m128i_i64, v38);
      v23 = _mm_loadu_si128(&v26);
      v24 = _mm_loadu_si128(&v27);
      v34 = _mm_loadu_si128(&v25);
      v37 = v28;
      v35 = v23;
      v36 = v24;
    }
    v16 = &dest[0].m128i_i8[4LL - (_QWORD)v17];
    if ( v16 > (char *)&v34 )
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v17, 4LL - (_QWORD)v17);
  }
  if ( !v13 )
    return sub_1593600(dest, v16 - (char *)dest, v38);
  sub_15AF6E0(dest[0].m128i_i8, v16, v34.m128i_i8);
  sub_1593A20((unsigned __int64 *)&v34, dest);
  v21 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v13 + v16 - (char *)dest) >> 47) ^ (v13 + v16 - (char *)dest))
      + v34.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v37 ^ v36.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v37 ^ v36.m128i_i64[0]))
         ^ v37))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v37 ^ v36.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v37 ^ v36.m128i_i64[0]))
          ^ v37)) >> 47));
  v22 = 0x9DDFEA08EB382D69LL
      * (v21
       ^ (v35.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v34.m128i_i64[1] ^ ((unsigned __int64)v34.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v36.m128i_i64[1] ^ v35.m128i_i64[1]))
            ^ v36.m128i_i64[1]
            ^ ((0x9DDFEA08EB382D69LL * (v36.m128i_i64[1] ^ v35.m128i_i64[1])) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v36.m128i_i64[1] ^ v35.m128i_i64[1]))
           ^ v36.m128i_i64[1]
           ^ ((0x9DDFEA08EB382D69LL * (v36.m128i_i64[1] ^ v35.m128i_i64[1])) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * ((v22 >> 47) ^ v22 ^ v21)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v22 >> 47) ^ v22 ^ v21)));
}
