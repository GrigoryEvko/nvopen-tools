// Function: sub_1D4B920
// Address: 0x1d4b920
//
unsigned __int64 __fastcall sub_1D4B920(__int64 *a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v7; // rsi
  int v8; // eax
  const __m128i *v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  const __m128i *v12; // r13
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __m128 *v16; // rdx
  const __m128i *v17; // rax
  __int64 *v18; // rdi
  __int64 v19; // r9
  __int64 *v20; // rax
  __int64 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  unsigned __int64 result; // rax
  __int128 v27; // [rsp-10h] [rbp-A0h]
  unsigned __int64 v28; // [rsp-10h] [rbp-A0h]
  const __m128i *v29; // [rsp+8h] [rbp-88h]
  __int64 v30; // [rsp+10h] [rbp-80h] BYREF
  int v31; // [rsp+18h] [rbp-78h]
  __int64 v32; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v33; // [rsp+28h] [rbp-68h]
  unsigned __int64 v34; // [rsp+30h] [rbp-60h]
  unsigned __int8 v35[8]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v36; // [rsp+48h] [rbp-48h]
  char v37; // [rsp+50h] [rbp-40h]
  __int64 v38; // [rsp+58h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 72);
  v30 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v30, v7, 2);
  v8 = *(_DWORD *)(a2 + 64);
  v9 = *(const __m128i **)(a2 + 32);
  v32 = 0;
  v33 = 0;
  v31 = v8;
  v10 = *(unsigned int *)(a2 + 56);
  v34 = 0;
  v11 = 40 * v10;
  v12 = (const __m128i *)((char *)v9 + v11);
  v13 = 0xCCCCCCCCCCCCCCD0LL * (v11 >> 3);
  if ( v11 )
  {
    v29 = v9;
    v14 = sub_22077B0(0xCCCCCCCCCCCCCCD0LL * (v11 >> 3));
    v32 = v14;
    v15 = v14;
    v34 = v14 + v13;
    if ( v29 != v12 )
    {
      v16 = (__m128 *)v14;
      v17 = v29;
      do
      {
        if ( v16 )
        {
          a3 = _mm_loadu_si128(v17);
          *v16 = (__m128)a3;
        }
        v17 = (const __m128i *)((char *)v17 + 40);
        ++v16;
      }
      while ( v17 != v12 );
      v15 = v15 - 0x3333333333333330LL * ((unsigned __int64)((char *)v17 - (char *)v29 - 40) >> 3) + 16;
    }
  }
  else
  {
    v34 = 0;
    v15 = 0;
  }
  v33 = v15;
  sub_1D4B520(a1, (__int64)&v32, (__int64)&v30, a3, a4, a5);
  v18 = (__int64 *)a1[34];
  v35[0] = 1;
  v36 = 0;
  v37 = 111;
  v38 = 0;
  *((_QWORD *)&v27 + 1) = (__int64)(v33 - v32) >> 4;
  *(_QWORD *)&v27 = v32;
  v20 = sub_1D373B0(v18, 0xC1u, (__int64)&v30, v35, 2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, v19, v27);
  *((_DWORD *)v20 + 7) = -1;
  v21 = (__int64)v20;
  sub_1D444E0(a1[34], a2, (__int64)v20);
  sub_1D49010(v21);
  sub_1D2DC70((const __m128i *)a1[34], a2, v22, v23, v24, v25);
  result = v28;
  if ( v32 )
    result = j_j___libc_free_0(v32, v34 - v32);
  if ( v30 )
    return sub_161E7C0((__int64)&v30, v30);
  return result;
}
