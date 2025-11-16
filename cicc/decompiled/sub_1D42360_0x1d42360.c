// Function: sub_1D42360
// Address: 0x1d42360
//
_QWORD *__fastcall sub_1D42360(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned __int8 *v6; // rdx
  const void **v7; // r14
  unsigned int v8; // r15d
  char *v9; // r8
  char *v10; // r11
  size_t v11; // r9
  __int64 v12; // r13
  int *v13; // rdi
  int v14; // esi
  int *v15; // rax
  int v16; // esi
  __int64 v17; // rcx
  __int64 v18; // r8
  int v19; // edx
  int v20; // edi
  const __m128i *v21; // rdx
  __int64 v22; // rsi
  int *v23; // r8
  __int64 v24; // r9
  __m128i v25; // xmm0
  __int64 v26; // r10
  __int64 v27; // r11
  _QWORD *v28; // r14
  int v30; // eax
  __int64 v31; // [rsp+0h] [rbp-B0h]
  char *v32; // [rsp+0h] [rbp-B0h]
  __int64 v33; // [rsp+8h] [rbp-A8h]
  int *v34; // [rsp+10h] [rbp-A0h]
  char *v35; // [rsp+10h] [rbp-A0h]
  __int64 v36; // [rsp+18h] [rbp-98h]
  size_t v38; // [rsp+30h] [rbp-80h]
  __int64 v39; // [rsp+40h] [rbp-70h] BYREF
  int v40; // [rsp+48h] [rbp-68h]
  int *v41; // [rsp+50h] [rbp-60h] BYREF
  __int64 v42; // [rsp+58h] [rbp-58h]
  _BYTE dest[80]; // [rsp+60h] [rbp-50h] BYREF

  v6 = *(unsigned __int8 **)(a2 + 40);
  v7 = (const void **)*((_QWORD *)v6 + 1);
  v8 = *v6;
  LOBYTE(v41) = *v6;
  v42 = (__int64)v7;
  if ( (_BYTE)v41 )
  {
    v9 = *(char **)(a2 + 88);
    v10 = &v9[4 * word_42E7700[(unsigned __int8)((_BYTE)v41 - 14)]];
  }
  else
  {
    v30 = sub_1F58D30(&v41);
    v9 = *(char **)(a2 + 88);
    v10 = &v9[4 * v30];
  }
  v11 = v10 - v9;
  v41 = (int *)dest;
  v42 = 0x800000000LL;
  v12 = (v10 - v9) >> 2;
  if ( (unsigned __int64)(v10 - v9) > 0x20 )
  {
    v38 = v10 - v9;
    v32 = v9;
    v35 = v10;
    sub_16CD150((__int64)&v41, dest, (v10 - v9) >> 2, 4, (int)v9, v11);
    v15 = v41;
    v14 = v42;
    v11 = v38;
    v10 = v35;
    v9 = v32;
    v13 = &v41[(unsigned int)v42];
  }
  else
  {
    v13 = (int *)dest;
    v14 = 0;
    v15 = (int *)dest;
  }
  if ( v10 != v9 )
  {
    memcpy(v13, v9, v11);
    v15 = v41;
    v14 = v42;
  }
  v16 = v12 + v14;
  v17 = 0;
  LODWORD(v42) = v16;
  if ( v16 )
  {
    v18 = (__int64)&v15[v16 - 1 + 1];
    do
    {
      v19 = *v15;
      if ( *v15 >= 0 )
      {
        v20 = v19 - v16;
        if ( v19 < v16 )
          v20 = v19 + v16;
        *v15 = v20;
      }
      ++v15;
    }
    while ( (int *)v18 != v15 );
    v15 = v41;
    v17 = (unsigned int)v42;
  }
  v21 = *(const __m128i **)(a2 + 32);
  v22 = *(_QWORD *)(a2 + 72);
  v23 = v15;
  v24 = v17;
  v25 = _mm_loadu_si128(v21);
  v26 = v21[2].m128i_i64[1];
  v39 = v22;
  v27 = v21[3].m128i_i64[0];
  if ( v22 )
  {
    v31 = v26;
    v33 = v21[3].m128i_i64[0];
    v34 = v15;
    v36 = v17;
    sub_1623A60((__int64)&v39, v22, 2);
    v26 = v31;
    v27 = v33;
    v23 = v34;
    v24 = v36;
  }
  v40 = *(_DWORD *)(a2 + 64);
  v28 = sub_1D41320(
          a1,
          v8,
          v7,
          (__int64)&v39,
          v26,
          v27,
          *(double *)v25.m128i_i64,
          a4,
          a5,
          v25.m128i_i64[0],
          v25.m128i_i64[1],
          v23,
          v24);
  if ( v39 )
    sub_161E7C0((__int64)&v39, v39);
  if ( v41 != (int *)dest )
    _libc_free((unsigned __int64)v41);
  return v28;
}
