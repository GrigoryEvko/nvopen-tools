// Function: sub_2D2C570
// Address: 0x2d2c570
//
void __fastcall sub_2D2C570(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9)
{
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  __int64 v13; // rsi
  unsigned __int64 *v14; // rax
  __int64 v15; // r9
  __int64 v16; // r12
  unsigned int *v17; // rbx
  unsigned int v18; // eax
  __int64 v19; // r12
  __int64 v20; // rsi
  __int64 v21; // r13
  __int64 v22; // r12
  __int64 v23; // rsi
  __int32 v24; // r12d
  __int64 v25; // [rsp+8h] [rbp-88h] BYREF
  int v26; // [rsp+10h] [rbp-80h]
  __int64 v27; // [rsp+18h] [rbp-78h]
  __int64 v28; // [rsp+20h] [rbp-70h] BYREF
  __int64 v29; // [rsp+28h] [rbp-68h]
  __m128i v30[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v31; // [rsp+50h] [rbp-40h]

  v11 = _mm_loadu_si128((const __m128i *)&a7);
  v12 = _mm_loadu_si128((const __m128i *)&a8);
  v25 = a2;
  v28 = 0;
  v29 = 0;
  v31 = a9;
  v30[0] = v11;
  v30[1] = v12;
  v27 = a3;
  v26 = sub_2D2C1F0(a1, v30);
  v13 = *a4;
  v28 = v13;
  if ( v13 )
    sub_B96E90((__int64)&v28, v13, 1);
  v29 = a5;
  v14 = sub_2D2B8B0(a1 + 9, &v25);
  v16 = *((unsigned int *)v14 + 2);
  v17 = (unsigned int *)v14;
  v18 = v16;
  if ( v17[3] <= (unsigned int)v16 )
  {
    v21 = sub_C8D7D0((__int64)v17, (__int64)(v17 + 4), 0, 0x20u, (unsigned __int64 *)v30, v15);
    v22 = v21 + 32LL * v17[2];
    if ( v22 )
    {
      *(_DWORD *)v22 = v26;
      *(_QWORD *)(v22 + 8) = v27;
      v23 = v28;
      *(_QWORD *)(v22 + 16) = v28;
      if ( v23 )
        sub_B96E90(v22 + 16, v23, 1);
      *(_QWORD *)(v22 + 24) = v29;
    }
    sub_2D296B0(v17, v21);
    v24 = v30[0].m128i_i32[0];
    if ( v17 + 4 != *(unsigned int **)v17 )
      _libc_free(*(_QWORD *)v17);
    ++v17[2];
    *(_QWORD *)v17 = v21;
    v17[3] = v24;
  }
  else
  {
    v19 = *(_QWORD *)v17 + 32 * v16;
    if ( v19 )
    {
      *(_DWORD *)v19 = v26;
      *(_QWORD *)(v19 + 8) = v27;
      v20 = v28;
      *(_QWORD *)(v19 + 16) = v28;
      if ( v20 )
        sub_B96E90(v19 + 16, v20, 1);
      *(_QWORD *)(v19 + 24) = v29;
      v18 = v17[2];
    }
    v17[2] = v18 + 1;
  }
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
}
