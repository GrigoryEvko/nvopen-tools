// Function: sub_D53210
// Address: 0xd53210
//
__int64 __fastcall sub_D53210(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdx
  __m128i *v11; // rax
  __m128i *v12; // rsi
  const __m128i *v13; // r8
  const __m128i *v14; // rdx
  const __m128i *v15; // rcx
  __m128i **v16; // rdi
  __int64 v17; // r9
  __int64 result; // rax

  sub_C8CD80(a1, a1 + 32, a2, a4, a5, a6);
  v8 = ((__int64)(*(_QWORD *)(a2 + 144) - *(_QWORD *)(a2 + 152)) >> 5)
     + 16 * (((__int64)(*(_QWORD *)(a2 + 168) - *(_QWORD *)(a2 + 136)) >> 3) - 1);
  v9 = *(_QWORD *)(a2 + 128) - *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  sub_D53100((__int64 *)(a1 + 96), v8 + (v9 >> 5), v10);
  v11 = *(__m128i **)(a1 + 112);
  v12 = *(__m128i **)(a1 + 128);
  v13 = *(const __m128i **)(a2 + 144);
  v14 = *(const __m128i **)(a2 + 112);
  v15 = *(const __m128i **)(a2 + 128);
  v16 = (__m128i **)(*(_QWORD *)(a1 + 136) + 8LL);
  v17 = *(_QWORD *)(a2 + 136);
  while ( v13 != v14 )
  {
    while ( 1 )
    {
      if ( v11 )
      {
        *v11 = _mm_loadu_si128(v14);
        v11[1] = _mm_loadu_si128(v14 + 1);
      }
      v14 += 2;
      if ( v15 == v14 )
      {
        v14 = *(const __m128i **)(v17 + 8);
        v17 += 8;
        v15 = v14 + 32;
      }
      v11 += 2;
      if ( v12 != v11 )
        break;
      v11 = *v16++;
      v12 = v11 + 32;
      if ( v13 == v14 )
        goto LABEL_9;
    }
  }
LABEL_9:
  result = *(unsigned int *)(a2 + 176);
  *(_DWORD *)(a1 + 176) = result;
  return result;
}
