// Function: sub_CA3AE0
// Address: 0xca3ae0
//
__m128i *__fastcall sub_CA3AE0(__m128i *a1, __int64 a2)
{
  int v3; // edi
  __int32 v4; // eax
  __int64 v5; // rdx
  void *v6; // rdx
  void *v7; // rax
  __m128i v8; // xmm2
  __int64 *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __m128i v13; // xmm1
  __int8 v14; // al
  __int64 v15; // rdx
  void *v17[4]; // [rsp+0h] [rbp-110h] BYREF
  __int16 v18; // [rsp+20h] [rbp-F0h]
  _OWORD v19[2]; // [rsp+30h] [rbp-E0h] BYREF
  __int128 v20; // [rsp+50h] [rbp-C0h]
  __int128 v21; // [rsp+60h] [rbp-B0h]
  __int64 v22; // [rsp+70h] [rbp-A0h]
  _QWORD v23[2]; // [rsp+80h] [rbp-90h] BYREF
  __int64 v24; // [rsp+90h] [rbp-80h] BYREF
  __m128i v25; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v26; // [rsp+B0h] [rbp-60h]
  __int64 v27; // [rsp+B8h] [rbp-58h]
  __int64 v28; // [rsp+C0h] [rbp-50h]
  __int64 v29; // [rsp+C8h] [rbp-48h]
  char v30; // [rsp+D0h] [rbp-40h]

  if ( sub_CA3AD0(a2 + 16) )
  {
LABEL_6:
    a1[5].m128i_i8[8] &= ~1u;
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    sub_CA1F00(a1->m128i_i64, *(_BYTE **)(a2 + 16), *(_QWORD *)(a2 + 16) + *(_QWORD *)(a2 + 24));
    v11 = *(_QWORD *)(a2 + 80);
    v12 = *(_QWORD *)(a2 + 72);
    v13 = _mm_loadu_si128((const __m128i *)(a2 + 48));
    a1[3].m128i_i64[0] = *(_QWORD *)(a2 + 64);
    v14 = *(_BYTE *)(a2 + 96);
    a1[4].m128i_i64[0] = v11;
    v15 = *(_QWORD *)(a2 + 88);
    a1[3].m128i_i64[1] = v12;
    a1[4].m128i_i64[1] = v15;
    a1[5].m128i_i8[0] = v14;
    a1[2] = v13;
    return a1;
  }
  v3 = *(_DWORD *)(a2 + 8);
  v22 = 0;
  v20 = 0;
  memset(v19, 0, sizeof(v19));
  HIDWORD(v20) = 0xFFFF;
  v21 = 0;
  v4 = sub_C82AC0(v3, (__int64)v19);
  if ( !v4 )
  {
    v6 = *(void **)(a2 + 16);
    v7 = *(void **)(a2 + 24);
    v18 = 261;
    v17[0] = v6;
    v17[1] = v7;
    sub_CA37D0((__int64)v23, (__int64)v19, v17);
    sub_2240D70(a2 + 16, v23);
    v8 = _mm_loadu_si128(&v25);
    v9 = (__int64 *)v23[0];
    *(_QWORD *)(a2 + 64) = v26;
    v10 = v27;
    *(__m128i *)(a2 + 48) = v8;
    *(_QWORD *)(a2 + 72) = v10;
    *(_QWORD *)(a2 + 80) = v28;
    *(_QWORD *)(a2 + 88) = v29;
    *(_BYTE *)(a2 + 96) = v30;
    if ( v9 != &v24 )
      j_j___libc_free_0(v9, v24 + 1);
    goto LABEL_6;
  }
  a1[5].m128i_i8[8] |= 1u;
  a1->m128i_i32[0] = v4;
  a1->m128i_i64[1] = v5;
  return a1;
}
