// Function: sub_2DF98F0
// Address: 0x2df98f0
//
__int64 __fastcall sub_2DF98F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  const __m128i *v13; // rdx
  __m128i *v14; // rax
  unsigned __int64 v16; // r13
  __int64 v17; // rdi
  const void *v18; // rsi
  _QWORD v19[10]; // [rsp+0h] [rbp-50h] BYREF

  v5 = *(_QWORD *)(a2 + 24);
  v6 = *(_QWORD *)(a2 + 8);
  sub_2E88DB0(a2);
  v9 = *(unsigned int *)(a1 + 216);
  v10 = *(unsigned int *)(a1 + 220);
  v19[0] = a2;
  v19[1] = a3;
  v11 = v9 + 1;
  v19[2] = v5;
  if ( v9 + 1 > v10 )
  {
    v16 = *(_QWORD *)(a1 + 208);
    v17 = a1 + 208;
    v18 = (const void *)(a1 + 224);
    if ( v16 > (unsigned __int64)v19 || (unsigned __int64)v19 >= v16 + 24 * v9 )
    {
      sub_C8D5F0(v17, v18, v11, 0x18u, v7, v8);
      v12 = *(_QWORD *)(a1 + 208);
      v9 = *(unsigned int *)(a1 + 216);
      v13 = (const __m128i *)v19;
    }
    else
    {
      sub_C8D5F0(v17, v18, v11, 0x18u, v7, v8);
      v12 = *(_QWORD *)(a1 + 208);
      v9 = *(unsigned int *)(a1 + 216);
      v13 = (const __m128i *)((char *)v19 + v12 - v16);
    }
  }
  else
  {
    v12 = *(_QWORD *)(a1 + 208);
    v13 = (const __m128i *)v19;
  }
  v14 = (__m128i *)(v12 + 24 * v9);
  *v14 = _mm_loadu_si128(v13);
  v14[1].m128i_i64[0] = v13[1].m128i_i64[0];
  ++*(_DWORD *)(a1 + 216);
  return v6;
}
