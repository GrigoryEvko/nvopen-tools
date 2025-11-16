// Function: sub_36DC890
// Address: 0x36dc890
//
__int64 __fastcall sub_36DC890(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rdi
  __int64 (*v6)(void); // rax
  __int64 v7; // rdi
  char v8; // al
  unsigned int v9; // ebx
  unsigned int v10; // edx
  __int64 v11; // rsi
  __int64 *v12; // r14
  __int128 v13; // rax
  __int64 v14; // r9
  __int64 v15; // r14
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v20; // [rsp+0h] [rbp-40h] BYREF
  int v21; // [rsp+8h] [rbp-38h]

  v5 = *(_QWORD *)(a1 + 1136);
  v6 = *(__int64 (**)(void))(*(_QWORD *)v5 + 144LL);
  if ( (char *)v6 == (char *)sub_3020010 )
    v7 = v5 + 960;
  else
    v7 = v6();
  v8 = sub_3037D80(v7, *(__int64 **)(a1 + 40));
  v9 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) + 96LL);
  switch ( v9 )
  {
    case 1u:
    case 0x11u:
      v9 = 0;
      break;
    case 2u:
    case 0x12u:
      v9 = 4;
      break;
    case 3u:
    case 0x13u:
      v9 = 5;
      break;
    case 4u:
    case 0x14u:
      v9 = 2;
      break;
    case 5u:
    case 0x15u:
      v9 = 3;
      break;
    case 6u:
    case 0x16u:
      v9 = 1;
      break;
    case 7u:
      v9 = 16;
      break;
    case 8u:
      v9 = 17;
      break;
    case 9u:
      v9 = 10;
      break;
    case 0xAu:
      v9 = 14;
      break;
    case 0xBu:
      v9 = 15;
      break;
    case 0xCu:
    case 0xDu:
      break;
    case 0xEu:
      v9 = 11;
      break;
    default:
      BUG();
  }
  v10 = v9;
  v11 = *(_QWORD *)(a2 + 80);
  v20 = *(_QWORD *)(a2 + 80);
  if ( v8 )
  {
    BYTE1(v10) = BYTE1(v9) | 1;
    v9 = v10;
  }
  if ( v11 )
    sub_B96E90((__int64)&v20, v11, 1);
  v12 = *(__int64 **)(a1 + 64);
  v21 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v13 = sub_3400BD0((__int64)v12, v9, (__int64)&v20, 7, 0, 1u, a3, 0);
  v15 = sub_33E6A90(
          v12,
          3532,
          (__int64)&v20,
          2,
          0,
          v14,
          2,
          0,
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
          v13);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v15, v16, v17, v18);
  sub_3421DB0(v15);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v20 )
    sub_B91220((__int64)&v20, v20);
  return 1;
}
