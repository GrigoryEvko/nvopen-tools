// Function: sub_2D2C3A0
// Address: 0x2d2c3a0
//
void __fastcall sub_2D2C3A0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9)
{
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __int64 v15; // r9
  __int64 v16; // rsi
  __int64 v17; // r12
  int v18; // eax
  __int64 v19; // r12
  __int64 v20; // rsi
  __int64 v21; // r13
  __int64 v22; // r12
  __int64 v23; // rsi
  unsigned __int64 v24; // rdi
  __int32 v25; // r12d
  int v26; // [rsp+0h] [rbp-80h]
  __int64 v27; // [rsp+10h] [rbp-70h] BYREF
  __int64 v28; // [rsp+18h] [rbp-68h]
  __m128i v29[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v30; // [rsp+40h] [rbp-40h]

  v13 = _mm_loadu_si128((const __m128i *)&a7);
  v14 = _mm_loadu_si128((const __m128i *)&a8);
  v27 = 0;
  v28 = 0;
  v30 = a9;
  v29[0] = v13;
  v29[1] = v14;
  v26 = sub_2D2C1F0((_QWORD *)a1, v29);
  v16 = *a3;
  v27 = v16;
  if ( v16 )
    sub_B96E90((__int64)&v27, v16, 1);
  v28 = a4;
  v17 = *(unsigned int *)(a1 + 136);
  v18 = v17;
  if ( *(_DWORD *)(a1 + 140) <= (unsigned int)v17 )
  {
    v21 = sub_C8D7D0(a1 + 128, a1 + 144, 0, 0x20u, (unsigned __int64 *)v29, v15);
    v22 = v21 + 32LL * *(unsigned int *)(a1 + 136);
    if ( v22 )
    {
      *(_DWORD *)v22 = v26;
      *(_QWORD *)(v22 + 8) = a2;
      v23 = v27;
      *(_QWORD *)(v22 + 16) = v27;
      if ( v23 )
        sub_B96E90(v22 + 16, v23, 1);
      *(_QWORD *)(v22 + 24) = v28;
    }
    sub_2D296B0((unsigned int *)(a1 + 128), v21);
    v24 = *(_QWORD *)(a1 + 128);
    v25 = v29[0].m128i_i32[0];
    if ( a1 + 144 != v24 )
      _libc_free(v24);
    ++*(_DWORD *)(a1 + 136);
    *(_QWORD *)(a1 + 128) = v21;
    *(_DWORD *)(a1 + 140) = v25;
  }
  else
  {
    v19 = *(_QWORD *)(a1 + 128) + 32 * v17;
    if ( v19 )
    {
      *(_DWORD *)v19 = v26;
      *(_QWORD *)(v19 + 8) = a2;
      v20 = v27;
      *(_QWORD *)(v19 + 16) = v27;
      if ( v20 )
        sub_B96E90(v19 + 16, v20, 1);
      *(_QWORD *)(v19 + 24) = v28;
      v18 = *(_DWORD *)(a1 + 136);
    }
    *(_DWORD *)(a1 + 136) = v18 + 1;
  }
  if ( v27 )
    sub_B91220((__int64)&v27, v27);
}
