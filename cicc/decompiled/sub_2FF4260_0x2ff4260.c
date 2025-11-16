// Function: sub_2FF4260
// Address: 0x2ff4260
//
_QWORD *__fastcall sub_2FF4260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // r14
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // r8
  unsigned __int64 v13; // rsi
  int v14; // eax
  __m128i *v15; // rdx
  __m128i v16; // xmm1
  __int64 v17; // rdi
  __int64 v19; // rdi
  char *v20; // r14
  _QWORD v21[6]; // [rsp+0h] [rbp-70h] BYREF
  char v22; // [rsp+30h] [rbp-40h]

  v6 = (const __m128i *)v21;
  v21[1] = a3;
  v10 = *(unsigned int *)(a1 + 32);
  v21[5] = a4;
  v11 = *(unsigned int *)(a1 + 36);
  v21[2] = a5;
  v12 = v10 + 1;
  v21[0] = a2;
  v13 = *(_QWORD *)(a1 + 24);
  v21[4] = &unk_4A2D840;
  v14 = v10;
  v21[3] = a6;
  v22 = 1;
  if ( v10 + 1 > v11 )
  {
    v19 = a1 + 24;
    if ( v13 > (unsigned __int64)v21 || (unsigned __int64)v21 >= v13 + 56 * v10 )
    {
      sub_2FF41A0(v19, v10 + 1, v10, v11, v12, a6);
      v10 = *(unsigned int *)(a1 + 32);
      v13 = *(_QWORD *)(a1 + 24);
      v14 = *(_DWORD *)(a1 + 32);
    }
    else
    {
      v20 = (char *)v21 - v13;
      sub_2FF41A0(v19, v10 + 1, v10, v11, v12, a6);
      v13 = *(_QWORD *)(a1 + 24);
      v10 = *(unsigned int *)(a1 + 32);
      v6 = (const __m128i *)&v20[v13];
      v14 = *(_DWORD *)(a1 + 32);
    }
  }
  v15 = (__m128i *)(v13 + 56 * v10);
  if ( v15 )
  {
    v16 = _mm_loadu_si128(v6 + 1);
    *v15 = _mm_loadu_si128(v6);
    v15[1] = v16;
    v15[2].m128i_i64[1] = v6[2].m128i_i64[1];
    v15[3].m128i_i8[0] = v6[3].m128i_i8[0];
    v15[2].m128i_i64[0] = (__int64)&unk_4A2D840;
    v14 = *(_DWORD *)(a1 + 32);
  }
  v17 = *(_QWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 32) = v14 + 1;
  return sub_C52F90(v17, a2, a3);
}
