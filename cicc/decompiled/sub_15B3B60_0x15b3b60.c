// Function: sub_15B3B60
// Address: 0x15b3b60
//
unsigned __int64 __fastcall sub_15B3B60(__int64 *a1, __int64 *a2, __int64 *a3, __int64 *a4, int *a5)
{
  __int64 *v5; // r10
  __int64 v8; // r8
  __int8 *v9; // rax
  __int64 v10; // r8
  __int8 *v11; // rax
  __int8 *v12; // rax
  __int64 v13; // r8
  __int8 *v14; // rax
  __int64 v15; // rcx
  __int8 *v16; // rdi
  int v17; // eax
  char *v18; // r13
  char *v19; // r15
  __int64 v20; // rcx
  int v22; // eax
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rcx
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  __int64 v29; // [rsp+8h] [rbp-128h]
  __int64 v30; // [rsp+8h] [rbp-128h]
  __int64 v31; // [rsp+8h] [rbp-128h]
  __m128i v32; // [rsp+10h] [rbp-120h] BYREF
  __m128i v33; // [rsp+20h] [rbp-110h] BYREF
  __m128i v34; // [rsp+30h] [rbp-100h] BYREF
  __int64 v35; // [rsp+40h] [rbp-F0h]
  int src; // [rsp+5Ch] [rbp-D4h] BYREF
  __int64 v37; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v38; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v39; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+78h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v42; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v43; // [rsp+D0h] [rbp-60h]
  __m128i v44; // [rsp+E0h] [rbp-50h]
  __int64 v45; // [rsp+F0h] [rbp-40h]
  __int64 v46; // [rsp+F8h] [rbp-38h]

  v5 = a1;
  if ( !byte_4F99930[0] )
  {
    v22 = sub_2207590(byte_4F99930);
    v5 = a1;
    if ( v22 )
    {
      v23 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v23 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v23;
      sub_2207640(byte_4F99930);
      v5 = a1;
    }
  }
  v8 = *v5;
  v46 = qword_4F99938;
  v37 = 0;
  v9 = sub_15B2320(dest, &v37, dest[0].m128i_i8, (unsigned __int64)&v42, v8);
  v10 = *a2;
  v38 = v37;
  v11 = sub_15B2320(dest, &v38, v9, (unsigned __int64)&v42, v10);
  v39 = v38;
  v12 = sub_15B3A60(dest, &v39, v11, (unsigned __int64)&v42, *a3);
  v13 = *a4;
  v40 = v39;
  v14 = sub_15B2320(dest, &v40, v12, (unsigned __int64)&v42, v13);
  v15 = v40;
  v16 = v14;
  v17 = *a5;
  v18 = v16 + 4;
  src = *a5;
  if ( v16 + 4 <= (__int8 *)&v42 )
  {
    *(_DWORD *)v16 = v17;
  }
  else
  {
    v29 = v40;
    v19 = (char *)((char *)&v42 - v16);
    memcpy(v16, &src, (char *)&v42 - v16);
    if ( v29 )
    {
      sub_1593A20((unsigned __int64 *)&v42, dest);
      v20 = v29 + 64;
    }
    else
    {
      sub_15938B0((unsigned __int64 *)&v32, dest[0].m128i_i64, v46);
      v26 = _mm_loadu_si128(&v33);
      v20 = 64;
      v27 = _mm_loadu_si128(&v34);
      v42 = _mm_loadu_si128(&v32);
      v45 = v35;
      v43 = v26;
      v44 = v27;
    }
    v30 = v20;
    v18 = &dest[0].m128i_i8[4LL - (_QWORD)v19];
    if ( v18 > (char *)&v42 )
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v19, 4LL - (_QWORD)v19);
    v15 = v30;
  }
  if ( !v15 )
    return sub_1593600(dest, v18 - (char *)dest, v46);
  v31 = v15;
  sub_15AF6E0(dest[0].m128i_i8, v18, v42.m128i_i8);
  sub_1593A20((unsigned __int64 *)&v42, dest);
  v24 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v31 + v18 - (char *)dest) >> 47) ^ (v31 + v18 - (char *)dest))
      + v42.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v45 ^ v44.m128i_i64[0]))
         ^ v45
         ^ ((0x9DDFEA08EB382D69LL * (v45 ^ v44.m128i_i64[0])) >> 47)))
       ^ ((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v45 ^ v44.m128i_i64[0]))
          ^ v45
          ^ ((0x9DDFEA08EB382D69LL * (v45 ^ v44.m128i_i64[0])) >> 47))) >> 47));
  v25 = 0x9DDFEA08EB382D69LL
      * (v24
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
       * (((0x9DDFEA08EB382D69LL * ((v25 >> 47) ^ v25 ^ v24)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v25 >> 47) ^ v25 ^ v24)));
}
