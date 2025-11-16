// Function: sub_213D690
// Address: 0x213d690
//
__int64 *__fastcall sub_213D690(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  __int128 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r12
  unsigned int v10; // edx
  __int64 *v11; // r15
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  unsigned __int8 v14; // bl
  __int64 v15; // r9
  __int64 *v16; // r12
  bool v18; // al
  __int64 v19; // rdx
  __int64 v20; // [rsp+0h] [rbp-70h]
  __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  int v22; // [rsp+28h] [rbp-48h]
  unsigned __int8 v23[8]; // [rsp+30h] [rbp-40h] BYREF
  __int64 v24; // [rsp+38h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 72);
  v21 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v21, v6, 2);
  v22 = *(_DWORD *)(a2 + 64);
  *(_QWORD *)&v7 = sub_2138AD0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = *((_QWORD *)&v7 + 1);
  v9 = sub_1D309E0(
         *(__int64 **)(a1 + 8),
         144,
         (__int64)&v21,
         **(unsigned __int8 **)(a2 + 40),
         *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
         0,
         *(double *)a3.m128i_i64,
         a4,
         *(double *)a5.m128i_i64,
         v7);
  v11 = *(__int64 **)(a1 + 8);
  v12 = v10 | v8 & 0xFFFFFFFF00000000LL;
  v13 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v14 = *(_BYTE *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v23[0] = v14;
  v24 = v15;
  if ( v14 )
  {
    if ( (unsigned __int8)(v14 - 14) <= 0x5Fu )
    {
      switch ( v14 )
      {
        case 0x18u:
        case 0x19u:
        case 0x1Au:
        case 0x1Bu:
        case 0x1Cu:
        case 0x1Du:
        case 0x1Eu:
        case 0x1Fu:
        case 0x20u:
        case 0x3Eu:
        case 0x3Fu:
        case 0x40u:
        case 0x41u:
        case 0x42u:
        case 0x43u:
          v14 = 3;
          break;
        case 0x21u:
        case 0x22u:
        case 0x23u:
        case 0x24u:
        case 0x25u:
        case 0x26u:
        case 0x27u:
        case 0x28u:
        case 0x44u:
        case 0x45u:
        case 0x46u:
        case 0x47u:
        case 0x48u:
        case 0x49u:
          v14 = 4;
          break;
        case 0x29u:
        case 0x2Au:
        case 0x2Bu:
        case 0x2Cu:
        case 0x2Du:
        case 0x2Eu:
        case 0x2Fu:
        case 0x30u:
        case 0x4Au:
        case 0x4Bu:
        case 0x4Cu:
        case 0x4Du:
        case 0x4Eu:
        case 0x4Fu:
          v14 = 5;
          break;
        case 0x31u:
        case 0x32u:
        case 0x33u:
        case 0x34u:
        case 0x35u:
        case 0x36u:
        case 0x50u:
        case 0x51u:
        case 0x52u:
        case 0x53u:
        case 0x54u:
        case 0x55u:
          v14 = 6;
          break;
        case 0x37u:
          v14 = 7;
          break;
        case 0x56u:
        case 0x57u:
        case 0x58u:
        case 0x62u:
        case 0x63u:
        case 0x64u:
          v14 = 8;
          break;
        case 0x59u:
        case 0x5Au:
        case 0x5Bu:
        case 0x5Cu:
        case 0x5Du:
        case 0x65u:
        case 0x66u:
        case 0x67u:
        case 0x68u:
        case 0x69u:
          v14 = 9;
          break;
        case 0x5Eu:
        case 0x5Fu:
        case 0x60u:
        case 0x61u:
        case 0x6Au:
        case 0x6Bu:
        case 0x6Cu:
        case 0x6Du:
          v14 = 10;
          break;
        default:
          v14 = 2;
          break;
      }
      v15 = 0;
    }
  }
  else
  {
    v20 = v15;
    v18 = sub_1F58D20((__int64)v23);
    v15 = v20;
    if ( v18 )
    {
      v14 = sub_1F596B0((__int64)v23);
      v15 = v19;
    }
  }
  v16 = sub_1D3BC50(v11, v9, v12, (__int64)&v21, v14, v15, a3, a4, a5);
  if ( v21 )
    sub_161E7C0((__int64)&v21, v21);
  return v16;
}
