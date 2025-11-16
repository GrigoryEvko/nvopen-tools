// Function: sub_1D189E0
// Address: 0x1d189e0
//
__int64 __fastcall sub_1D189E0(
        __int64 a1,
        __int16 a2,
        int a3,
        unsigned __int8 **a4,
        __int64 a5,
        int a6,
        __int128 a7,
        __int64 a8)
{
  unsigned __int8 *v11; // rsi
  __int64 v12; // r14
  __m128i v13; // xmm0
  unsigned __int16 v14; // dx
  __int64 result; // rax
  __int64 v16; // [rsp+8h] [rbp-48h]
  _QWORD v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v11 = *a4;
  v12 = a8;
  v17[0] = v11;
  if ( v11 )
  {
    v16 = a5;
    sub_1623A60((__int64)v17, (__int64)v11, 2);
    v11 = (unsigned __int8 *)v17[0];
    a5 = v16;
  }
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_WORD *)(a1 + 24) = a2;
  *(_DWORD *)(a1 + 28) = -1;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = a5;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 60) = a6;
  *(_DWORD *)(a1 + 64) = a3;
  *(_QWORD *)(a1 + 72) = v11;
  if ( v11 )
    sub_1623210((__int64)v17, v11, a1 + 72);
  *(_WORD *)(a1 + 80) &= 0xF000u;
  v13 = _mm_loadu_si128((const __m128i *)&a7);
  *(_WORD *)(a1 + 26) = 0;
  v14 = *(_WORD *)(v12 + 32);
  *(_QWORD *)(a1 + 104) = v12;
  *(__m128i *)(a1 + 88) = v13;
  result = *(_BYTE *)(a1 + 26) & 0x87
         | (((v14 >> 5) & 1) << 6)
         | (32 * ((v14 >> 4) & 1))
         | (8 * ((v14 >> 2) & 1))
         | (16 * ((v14 >> 3) & 1u));
  *(_BYTE *)(a1 + 26) = *(_BYTE *)(a1 + 26) & 0x87
                      | (((v14 & 0x20) != 0) << 6)
                      | (32 * ((v14 & 0x10) != 0))
                      | (8 * ((v14 & 4) != 0))
                      | (16 * ((v14 & 8) != 0));
  return result;
}
