// Function: sub_1348460
// Address: 0x1348460
//
__int64 __fastcall sub_1348460(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6, const __m128i *a7)
{
  __int64 v11; // rdx

  if ( (unsigned __int8)sub_130AF40(a1 + 176) || (unsigned __int8)sub_130AF40(a1 + 64) )
    return 1;
  *(_QWORD *)(a1 + 56) = a2;
  *(_QWORD *)(a1 + 288) = a4;
  sub_1340B90(a1 + 296, a5);
  sub_134BAC0(a1 + 320);
  *(_DWORD *)(a1 + 5608) = a6;
  *(_QWORD *)(a1 + 5600) = 0;
  *(_QWORD *)(a1 + 5616) = a3;
  *(__m128i *)(a1 + 5624) = _mm_loadu_si128(a7);
  *(__m128i *)(a1 + 5640) = _mm_loadu_si128(a7 + 1);
  v11 = a7[2].m128i_i64[0];
  *(_QWORD *)(a1 + 5664) = 0;
  *(_QWORD *)(a1 + 5656) = v11;
  sub_130B140((__int64 *)(a1 + 5704), &qword_4287F20);
  *(_QWORD *)(a1 + 5672) = 0;
  *(_QWORD *)a1 = sub_1348400;
  *(_QWORD *)(a1 + 8) = sub_13483E0;
  *(_QWORD *)(a1 + 16) = sub_1347270;
  *(_QWORD *)(a1 + 24) = sub_1347280;
  *(_QWORD *)(a1 + 32) = sub_1347C10;
  *(_QWORD *)(a1 + 40) = sub_13479B0;
  *(_QWORD *)(a1 + 48) = sub_1347400;
  *(_QWORD *)(a1 + 5680) = 0;
  *(_QWORD *)(a1 + 5688) = 0;
  *(_QWORD *)(a1 + 5696) = 0;
  return 0;
}
