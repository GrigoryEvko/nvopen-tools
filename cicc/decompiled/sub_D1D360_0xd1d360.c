// Function: sub_D1D360
// Address: 0xd1d360
//
_QWORD *__fastcall sub_D1D360(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rax
  __m128i v8; // xmm1
  __m128i v9; // xmm0
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  _QWORD *v17; // rdx
  _QWORD *result; // rax
  __int64 *v19; // rcx
  _QWORD *v20; // rsi
  __int64 v21; // rdi

  v3 = *a2;
  *(_QWORD *)(a1 + 24) = 0;
  v5 = *(_QWORD *)(a1 + 32);
  v6 = a1 + 40;
  *(_QWORD *)(v6 - 40) = v3;
  v7 = a2[3];
  v8 = _mm_loadu_si128((const __m128i *)(v6 - 32));
  v9 = _mm_loadu_si128((const __m128i *)(a2 + 1));
  a2[3] = 0;
  *(_QWORD *)(v6 - 16) = v7;
  v10 = a2[4];
  a2[4] = v5;
  *(_QWORD *)(v6 - 8) = v10;
  *(__m128i *)(v6 - 32) = v9;
  *(__m128i *)(a2 + 1) = v8;
  sub_C8CF70(v6, (void *)(a1 + 72), 8, (__int64)(a2 + 9), (__int64)(a2 + 5));
  *(_BYTE *)(a1 + 136) = 0;
  sub_C8CF70(a1 + 144, (void *)(a1 + 176), 8, (__int64)(a2 + 22), (__int64)(a2 + 18));
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_DWORD *)(a1 + 264) = 0;
  v11 = a2[31];
  v12 = *((_DWORD *)a2 + 66);
  ++a2[30];
  *(_QWORD *)(a1 + 248) = v11;
  v13 = a2[32];
  a2[31] = 0;
  a2[32] = 0;
  *((_DWORD *)a2 + 66) = 0;
  *(_QWORD *)(a1 + 256) = v13;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_DWORD *)(a1 + 296) = 0;
  v14 = a2[35];
  *(_DWORD *)(a1 + 264) = v12;
  v15 = *((_DWORD *)a2 + 74);
  *(_QWORD *)(a1 + 280) = v14;
  v16 = a2[36];
  ++a2[34];
  *(_QWORD *)(a1 + 288) = v16;
  v17 = (_QWORD *)(a1 + 336);
  *(_DWORD *)(a1 + 296) = v15;
  result = a2 + 42;
  a2[35] = 0;
  a2[36] = 0;
  *((_DWORD *)a2 + 74) = 0;
  *(_QWORD *)(a1 + 240) = 1;
  *(_QWORD *)(a1 + 272) = 1;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_DWORD *)(a1 + 328) = 0;
  v19 = (__int64 *)a2[42];
  v20 = (_QWORD *)a2[43];
  v21 = a2[44];
  *(_QWORD *)(a1 + 336) = v19;
  *(_QWORD *)(a1 + 344) = v20;
  *(_QWORD *)(a1 + 352) = v21;
  if ( a2 + 42 == v19 )
  {
    *(_QWORD *)(a1 + 344) = v17;
    *(_QWORD *)(a1 + 336) = v17;
  }
  else
  {
    *v20 = v17;
    *(_QWORD *)(*(_QWORD *)(a1 + 336) + 8LL) = v17;
    a2[43] = (__int64)result;
    a2[42] = (__int64)result;
    a2[44] = 0;
    for ( result = *(_QWORD **)(a1 + 336); v17 != result; result = (_QWORD *)*result )
      result[6] = a1;
  }
  return result;
}
