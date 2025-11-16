// Function: sub_21BE780
// Address: 0x21be780
//
__int64 __fastcall sub_21BE780(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  char v6; // al
  unsigned int v7; // ebx
  unsigned int v8; // edx
  __int64 v9; // rsi
  _QWORD *v10; // r14
  __int128 v11; // rax
  __int64 v12; // r9
  __int64 v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v19; // [rsp+0h] [rbp-40h] BYREF
  int v20; // [rsp+8h] [rbp-38h]

  v6 = sub_21BE190(a1);
  v7 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL) + 84LL);
  switch ( v7 )
  {
    case 0u:
    case 6u:
    case 0xFu:
    case 0x10u:
    case 0x16u:
      v7 = 1;
      break;
    case 1u:
    case 0x11u:
      v7 = 0;
      break;
    case 2u:
    case 0x12u:
      v7 = 4;
      break;
    case 3u:
    case 0x13u:
      v7 = 5;
      break;
    case 4u:
    case 0x14u:
      v7 = 2;
      break;
    case 5u:
    case 0x15u:
      v7 = 3;
      break;
    case 7u:
      v7 = 16;
      break;
    case 8u:
      v7 = 17;
      break;
    case 9u:
      v7 = 10;
      break;
    case 0xAu:
      v7 = 14;
      break;
    case 0xBu:
      v7 = 15;
      break;
    case 0xCu:
    case 0xDu:
      break;
    case 0xEu:
      v7 = 11;
      break;
  }
  v8 = v7;
  v9 = *(_QWORD *)(a2 + 72);
  v19 = *(_QWORD *)(a2 + 72);
  if ( v6 )
  {
    BYTE1(v8) = BYTE1(v7) | 1;
    v7 = v8;
  }
  if ( v9 )
    sub_1623A60((__int64)&v19, v9, 2);
  v10 = *(_QWORD **)(a1 + 272);
  v20 = *(_DWORD *)(a2 + 64);
  *(_QWORD *)&v11 = sub_1D38BB0((__int64)v10, v7, (__int64)&v19, 5, 0, 1, a3, a4, a5, 0);
  v13 = sub_1D25B60(
          v10,
          3347,
          (__int64)&v19,
          2,
          0,
          v12,
          2,
          0,
          *(_OWORD *)*(_QWORD *)(a2 + 32),
          *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
          v11);
  sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v13);
  sub_1D49010(v13);
  sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v14, v15, v16, v17);
  if ( v19 )
    sub_161E7C0((__int64)&v19, v19);
  return 1;
}
