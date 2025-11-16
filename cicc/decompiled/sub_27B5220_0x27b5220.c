// Function: sub_27B5220
// Address: 0x27b5220
//
unsigned __int64 __fastcall sub_27B5220(__int64 *a1, int *a2, char *a3, __int64 a4)
{
  __int64 v5; // r8
  char *v6; // rax
  __int64 v7; // rbx
  char *v8; // rdi
  int v9; // eax
  _BYTE *v10; // r8
  char *v11; // r15
  char v12; // al
  __m128i *v13; // r15
  unsigned __int64 v14; // rax
  char *v15; // rax
  __int64 v16; // r14
  char *v18; // r14
  __m128i v19; // xmm4
  __m128i v20; // xmm5
  __m128i v21; // xmm6
  signed __int64 v22; // rbx
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // rax
  __m128i v25; // xmm2
  __m128i v26; // xmm3
  __m128i v28; // [rsp+20h] [rbp-110h] BYREF
  __m128i v29; // [rsp+30h] [rbp-100h] BYREF
  __m128i v30; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v31; // [rsp+50h] [rbp-E0h]
  char v32; // [rsp+6Fh] [rbp-C1h] BYREF
  __int64 v33; // [rsp+70h] [rbp-C0h] BYREF
  __int64 src; // [rsp+78h] [rbp-B8h] BYREF
  _OWORD dest[4]; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v36; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v37; // [rsp+D0h] [rbp-60h]
  __m128i v38; // [rsp+E0h] [rbp-50h]
  __int64 v39; // [rsp+F0h] [rbp-40h]
  void (__fastcall *v40)(__int64, __int64); // [rsp+F8h] [rbp-38h]

  v5 = *a1;
  v36 = 0u;
  v37 = 0u;
  v38 = 0u;
  v39 = 0;
  v40 = sub_C64CA0;
  v33 = 0;
  memset(dest, 0, sizeof(dest));
  v6 = (char *)sub_CA5190((unsigned __int64 *)dest, &v33, dest, (unsigned __int64)&v36, v5);
  v7 = v33;
  v8 = v6;
  v9 = *a2;
  v10 = v8 + 4;
  LODWORD(src) = *a2;
  if ( v8 + 4 <= (char *)&v36 )
  {
    *(_DWORD *)v8 = v9;
  }
  else
  {
    v11 = (char *)((char *)&v36 - v8);
    memcpy(v8, &src, (char *)&v36 - v8);
    if ( v7 )
    {
      v7 += 64;
      sub_AC2A10((unsigned __int64 *)&v36, dest);
    }
    else
    {
      v7 = 64;
      sub_AC28A0((unsigned __int64 *)&v28, (__int64 *)dest, (unsigned __int64)v40);
      v25 = _mm_loadu_si128(&v29);
      v26 = _mm_loadu_si128(&v30);
      v36 = _mm_loadu_si128(&v28);
      v39 = v31;
      v37 = v25;
      v38 = v26;
    }
    if ( (__m128i *)((char *)dest + 4LL - (_QWORD)v11) > &v36 )
LABEL_5:
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v11, 4LL - (_QWORD)v11);
    v10 = (char *)dest + 4LL - (_QWORD)v11;
  }
  v12 = *a3;
  v13 = (__m128i *)(v10 + 1);
  v32 = *a3;
  if ( v10 + 1 > (_BYTE *)&v36 )
  {
    v18 = (char *)((char *)&v36 - v10);
    memcpy(v10, &v32, (char *)&v36 - v10);
    if ( v7 )
    {
      v7 += 64;
      sub_AC2A10((unsigned __int64 *)&v36, dest);
    }
    else
    {
      v7 = 64;
      sub_AC28A0((unsigned __int64 *)&v28, (__int64 *)dest, (unsigned __int64)v40);
      v19 = _mm_loadu_si128(&v28);
      v20 = _mm_loadu_si128(&v29);
      v21 = _mm_loadu_si128(&v30);
      v39 = v31;
      v36 = v19;
      v37 = v20;
      v38 = v21;
    }
    v13 = (__m128i *)((char *)dest + 1LL - (_QWORD)v18);
    if ( v13 > &v36 )
      goto LABEL_5;
    memcpy(dest, &v32 + (_QWORD)v18, 1LL - (_QWORD)v18);
  }
  else
  {
    *v10 = v12;
  }
  src = v7;
  v14 = sub_AC61D0(*(__int64 **)a4, *(_QWORD *)a4 + 4LL * *(_QWORD *)(a4 + 8));
  v15 = (char *)sub_CA5190((unsigned __int64 *)dest, &src, v13, (unsigned __int64)&v36, v14);
  v16 = src;
  if ( !src )
    return sub_AC25F0(dest, v15 - (char *)dest, (__int64)v40);
  v22 = v15 - (char *)dest;
  sub_27AC2B0((char *)dest, v15, v36.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v36, dest);
  v23 = v36.m128i_i64[0]
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0]))
         ^ v39))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v39 ^ v38.m128i_i64[0]))
          ^ v39)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v16 + v22) >> 47) ^ (v16 + v22));
  v24 = 0x9DDFEA08EB382D69LL
      * (v23
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
           ^ v38.m128i_i64[1])))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v24 ^ v23 ^ (v24 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v24 ^ v23 ^ (v24 >> 47))));
}
