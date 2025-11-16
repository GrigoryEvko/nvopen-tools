// Function: sub_E6D520
// Address: 0xe6d520
//
__int64 __fastcall sub_E6D520(__int64 a1, void **a2, char a3, int a4, __int64 a5, int a6)
{
  __int64 v6; // rax
  const char *v7; // r15
  __m128i v10; // xmm0
  char v11; // dl
  __int64 v12; // r15
  _QWORD *v14; // r15
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r14
  size_t *v19; // rsi
  size_t v20; // rdx
  const void *v21; // rsi
  __int64 v22; // rax
  int v25; // [rsp+8h] [rbp-E8h]
  char v26; // [rsp+10h] [rbp-E0h]
  int v27; // [rsp+10h] [rbp-E0h]
  __int64 v28; // [rsp+18h] [rbp-D8h]
  __m128i *v29; // [rsp+18h] [rbp-D8h]
  _BYTE *v30[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v31; // [rsp+30h] [rbp-C0h] BYREF
  __m128i *v32; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v33; // [rsp+48h] [rbp-A8h]
  __m128i v34; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v35; // [rsp+60h] [rbp-90h] BYREF
  int v36; // [rsp+70h] [rbp-80h]
  __m128i v37; // [rsp+80h] [rbp-70h] BYREF
  __m128i v38; // [rsp+90h] [rbp-60h] BYREF
  __m128i v39; // [rsp+A0h] [rbp-50h]
  int v40; // [rsp+B0h] [rbp-40h]
  __int64 v41; // [rsp+B8h] [rbp-38h]

  v6 = 0;
  v7 = byte_3F871B3;
  if ( a5 )
  {
    v7 = 0;
    if ( (*(_BYTE *)(a5 + 8) & 1) != 0 )
    {
      v14 = *(_QWORD **)(a5 - 8);
      v6 = *v14;
      v7 = (const char *)(v14 + 3);
    }
  }
  v28 = v6;
  sub_CA0F50((__int64 *)v30, a2);
  v32 = &v34;
  sub_E62BB0((__int64 *)&v32, v30[0], (__int64)&v30[0][(unsigned __int64)v30[1]]);
  v35.m128i_i64[0] = (__int64)v7;
  v36 = a6;
  v35.m128i_i64[1] = v28;
  v37.m128i_i64[0] = (__int64)&v38;
  if ( v32 == &v34 )
  {
    v38 = _mm_load_si128(&v34);
  }
  else
  {
    v37.m128i_i64[0] = (__int64)v32;
    v38.m128i_i64[0] = v34.m128i_i64[0];
  }
  v10 = _mm_load_si128(&v35);
  v40 = a6;
  v37.m128i_i64[1] = v33;
  v32 = &v34;
  v33 = 0;
  v34.m128i_i8[0] = 0;
  v41 = 0;
  v39 = v10;
  v29 = sub_E6A800((_QWORD *)(a1 + 2120), &v37);
  v26 = v11;
  if ( (__m128i *)v37.m128i_i64[0] != &v38 )
    j_j___libc_free_0(v37.m128i_i64[0], v38.m128i_i64[0] + 1);
  if ( v32 != &v34 )
    j_j___libc_free_0(v32, v34.m128i_i64[0] + 1);
  if ( (__int64 *)v30[0] != &v31 )
    j_j___libc_free_0(v30[0], v31 + 1);
  if ( !v26 )
    return v29[5].m128i_i64[1];
  v15 = v29[2].m128i_i64[0];
  v16 = v29[2].m128i_i64[1];
  v39.m128i_i16[0] = 261;
  v27 = v15;
  v37.m128i_i64[0] = v15;
  v25 = v16;
  v37.m128i_i64[1] = v16;
  v17 = sub_E6BFC0((_DWORD *)a1, (__int64)&v37, 1, 0);
  v18 = v17;
  if ( (*(_BYTE *)(v17 + 8) & 1) != 0 )
  {
    v19 = *(size_t **)(v17 - 8);
    v20 = *v19;
    v21 = v19 + 3;
  }
  else
  {
    v20 = 0;
    v21 = 0;
  }
  *(_QWORD *)(sub_E6B3F0(a1, v21, v20) + 8) = v17;
  *(_DWORD *)(v18 + 32) = 3;
  *(_BYTE *)(v18 + 36) = 1;
  v22 = *(_QWORD *)(a1 + 960);
  *(_QWORD *)(a1 + 1040) += 184LL;
  v12 = (v22 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(a1 + 968) >= (unsigned __int64)(v12 + 184) && v22 )
    *(_QWORD *)(a1 + 960) = v12 + 184;
  else
    v12 = sub_9D1E70(a1 + 960, 184, 184, 3);
  sub_E92760(v12, 4, v27, v25, (unsigned __int8)(a3 - 2) <= 1u, 0, v18);
  *(_QWORD *)(v12 + 152) = a5;
  *(_DWORD *)(v12 + 148) = a6;
  *(_BYTE *)(v12 + 172) = 0;
  *(_QWORD *)v12 = &unk_49E3650;
  *(_QWORD *)(v12 + 160) = 0;
  *(_DWORD *)(v12 + 168) = 0;
  *(_BYTE *)(v12 + 173) = (unsigned __int8)(a3 - 4) <= 0x10u;
  *(_BYTE *)(v12 + 174) = a3 == 0;
  *(_DWORD *)(v12 + 176) = a4;
  v29[5].m128i_i64[1] = v12;
  *(_QWORD *)v18 = sub_E6B260((_QWORD *)a1, v12);
  return v12;
}
