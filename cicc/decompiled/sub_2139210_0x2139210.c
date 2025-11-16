// Function: sub_2139210
// Address: 0x2139210
//
__int64 *__fastcall sub_2139210(__int64 a1, unsigned __int64 a2, __int64 a3, __m128i a4, double a5, __m128i a6)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned __int8 v10; // dl
  __int64 v11; // rax
  __int64 v12; // r14
  unsigned int v13; // edx
  __int64 *v14; // r15
  unsigned __int64 v15; // r12
  unsigned __int8 v16; // bl
  __int64 v17; // r9
  __int64 *v18; // r14
  __int64 v20; // rdx
  unsigned __int8 v21[8]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  int v24; // [rsp+38h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v9 = *(_QWORD *)(a2 + 72);
  v10 = *(_BYTE *)v8;
  v11 = *(_QWORD *)(v8 + 8);
  v23 = v9;
  v21[0] = v10;
  v22 = v11;
  if ( v9 )
    sub_1623A60((__int64)&v23, v9, 2);
  v24 = *(_DWORD *)(a2 + 64);
  v12 = sub_2138AD0(a1, a2, a3);
  v14 = *(__int64 **)(a1 + 8);
  v15 = a3 & 0xFFFFFFFF00000000LL | v13;
  v16 = v21[0];
  if ( v21[0] )
  {
    if ( (unsigned __int8)(v21[0] - 14) > 0x5Fu )
    {
LABEL_5:
      v17 = v22;
      goto LABEL_6;
    }
    switch ( v21[0] )
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
        v16 = 3;
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
        v16 = 4;
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
        v16 = 5;
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
        v16 = 6;
        break;
      case 0x37u:
        v16 = 7;
        break;
      case 0x56u:
      case 0x57u:
      case 0x58u:
      case 0x62u:
      case 0x63u:
      case 0x64u:
        v16 = 8;
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
        v16 = 9;
        break;
      case 0x5Eu:
      case 0x5Fu:
      case 0x60u:
      case 0x61u:
      case 0x6Au:
      case 0x6Bu:
      case 0x6Cu:
      case 0x6Du:
        v16 = 10;
        break;
      default:
        v16 = 2;
        break;
    }
    v17 = 0;
  }
  else
  {
    if ( !sub_1F58D20((__int64)v21) )
      goto LABEL_5;
    v16 = sub_1F596B0((__int64)v21);
    v17 = v20;
  }
LABEL_6:
  v18 = sub_1D3BC50(v14, v12, v15, (__int64)&v23, v16, v17, a4, a5, a6);
  if ( v23 )
    sub_161E7C0((__int64)&v23, v23);
  return v18;
}
