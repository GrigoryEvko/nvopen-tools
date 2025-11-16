// Function: sub_1620E10
// Address: 0x1620e10
//
unsigned __int64 __fastcall sub_1620E10(__int64 *a1)
{
  __int64 v2; // rax
  unsigned int v3; // edx
  __int64 *v4; // rax
  __int64 v5; // rax
  __int64 v7; // r8
  __int8 *v8; // rax
  __int64 v9; // rcx
  __int8 *v10; // rdi
  __int64 v11; // rax
  char *v12; // r12
  char *v13; // r14
  __int64 v14; // rcx
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  __int64 v20; // [rsp+8h] [rbp-108h]
  __int64 v21; // [rsp+8h] [rbp-108h]
  __int64 v22; // [rsp+8h] [rbp-108h]
  __m128i v23; // [rsp+10h] [rbp-100h] BYREF
  __m128i v24; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v25; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v26; // [rsp+40h] [rbp-D0h]
  __int64 v27; // [rsp+50h] [rbp-C0h] BYREF
  __int64 src; // [rsp+58h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v30; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v31; // [rsp+B0h] [rbp-60h]
  __m128i v32; // [rsp+C0h] [rbp-50h]
  __int64 v33; // [rsp+D0h] [rbp-40h]
  __int64 v34; // [rsp+D8h] [rbp-38h]

  if ( *(_BYTE *)*a1 == 1 )
  {
    v2 = *(_QWORD *)(*a1 + 136);
    v3 = *(_DWORD *)(v2 + 32);
    v4 = *(__int64 **)(v2 + 24);
    if ( v3 <= 0x40 )
      v5 = (__int64)((_QWORD)v4 << (64 - (unsigned __int8)v3)) >> (64 - (unsigned __int8)v3);
    else
      v5 = *v4;
    dest[0].m128i_i64[0] = v5;
    return sub_15B15B0(dest, a1 + 1);
  }
  else
  {
    if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
    {
      v17 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v17 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v17;
      sub_2207640(byte_4F99930);
    }
    v7 = *a1;
    v34 = qword_4F99938;
    v27 = 0;
    v8 = sub_15B2320(dest, &v27, dest[0].m128i_i8, (unsigned __int64)&v30, v7);
    v9 = v27;
    v10 = v8;
    v11 = a1[1];
    v12 = v10 + 8;
    src = v11;
    if ( v10 + 8 <= (__int8 *)&v30 )
    {
      *(_QWORD *)v10 = v11;
    }
    else
    {
      v20 = v27;
      v13 = (char *)((char *)&v30 - v10);
      memcpy(v10, &src, (char *)&v30 - v10);
      if ( v20 )
      {
        sub_1593A20((unsigned __int64 *)&v30, dest);
        v14 = v20 + 64;
      }
      else
      {
        sub_15938B0((unsigned __int64 *)&v23, dest[0].m128i_i64, v34);
        v15 = _mm_loadu_si128(&v24);
        v14 = 64;
        v16 = _mm_loadu_si128(&v25);
        v30 = _mm_loadu_si128(&v23);
        v33 = v26;
        v31 = v15;
        v32 = v16;
      }
      v21 = v14;
      v12 = &dest[0].m128i_i8[8LL - (_QWORD)v13];
      if ( v12 > (char *)&v30 )
        abort();
      memcpy(dest, (char *)&src + (_QWORD)v13, 8LL - (_QWORD)v13);
      v9 = v21;
    }
    if ( v9 )
    {
      v22 = v9;
      sub_161D190(dest[0].m128i_i8, v12, v30.m128i_i8);
      sub_1593A20((unsigned __int64 *)&v30, dest);
      v18 = v30.m128i_i64[0]
          - 0x622015F714C7D297LL
          * (((0x9DDFEA08EB382D69LL
             * ((0x9DDFEA08EB382D69LL * (v33 ^ v32.m128i_i64[0]))
              ^ v33
              ^ ((0x9DDFEA08EB382D69LL * (v33 ^ v32.m128i_i64[0])) >> 47))) >> 47)
           ^ (0x9DDFEA08EB382D69LL
            * ((0x9DDFEA08EB382D69LL * (v33 ^ v32.m128i_i64[0]))
             ^ v33
             ^ ((0x9DDFEA08EB382D69LL * (v33 ^ v32.m128i_i64[0])) >> 47))))
          - 0x4B6D499041670D8DLL * (((unsigned __int64)(v12 - (char *)dest + v22) >> 47) ^ (v12 - (char *)dest + v22));
      v19 = 0x9DDFEA08EB382D69LL
          * (v18
           ^ (0xB492B66FBE98F273LL * (v30.m128i_i64[1] ^ ((unsigned __int64)v30.m128i_i64[1] >> 47))
            + v31.m128i_i64[0]
            - 0x622015F714C7D297LL
            * (((0x9DDFEA08EB382D69LL
               * (((0x9DDFEA08EB382D69LL * (v32.m128i_i64[1] ^ v31.m128i_i64[1])) >> 47)
                ^ (0x9DDFEA08EB382D69LL * (v32.m128i_i64[1] ^ v31.m128i_i64[1]))
                ^ v32.m128i_i64[1])) >> 47)
             ^ (0x9DDFEA08EB382D69LL
              * (((0x9DDFEA08EB382D69LL * (v32.m128i_i64[1] ^ v31.m128i_i64[1])) >> 47)
               ^ (0x9DDFEA08EB382D69LL * (v32.m128i_i64[1] ^ v31.m128i_i64[1]))
               ^ v32.m128i_i64[1])))));
      return 0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * ((v19 >> 47) ^ v19 ^ v18))
            ^ ((0x9DDFEA08EB382D69LL * ((v19 >> 47) ^ v19 ^ v18)) >> 47));
    }
    else
    {
      return sub_1593600(dest, v12 - (char *)dest, v34);
    }
  }
}
